#!/usr/bin/env python3
import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# --------------------
# Defaults
# --------------------
TEXT_EXTS = {".txt", ".text", ".md"}
BLOCK_SIZE = 1024
BATCH_SIZE = 16
NUM_WORKERS = 2
GRAD_ACCUM_STEPS = 4
LR = 2e-5
MAX_EPOCHS = 1
WARMUP_STEPS = 50
SEED = 42
JSON_PATH = "./data/models_plan.json"

# Limit for in-memory token cache (approximate number of tokens to keep)
# tokens * 8 bytes (int64) ~ memory footprint; 4_000_000 ~ 32 MB
TOKEN_CACHE_BUDGET = 4_000_000


def find_text_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS]


class LRUTokenCache:
    """LRU cache for per-file tokenized ids with a global token budget."""
    def __init__(self, token_budget: int):
        self.token_budget = token_budget
        self.cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self.total_tokens = 0

    def get(self, key: Path):
        if key in self.cache:
            val = self.cache.pop(key)
            self.cache[key] = val
            return val
        return None

    def put(self, key: Path, value: torch.Tensor):
        # value is 1D Long tensor of token ids
        toks = int(value.numel())
        # Evict until under budget
        while self.total_tokens + toks > self.token_budget and len(self.cache) > 0:
            k_old, v_old = self.cache.popitem(last=False)
            self.total_tokens -= int(v_old.numel())
        self.cache[key] = value
        self.total_tokens += toks


class TokenChunkDataset(Dataset):
    """
    Map-style dataset that:
      - Precomputes how many `block_size` chunks each file yields (drop remainder)
      - Exposes a flat index mapping item -> (file_idx, chunk_idx)
      - On access, tokenizes the file (cached via LRU) and slices out the chunk
    """
    def __init__(
        self,
        files: List[Path],
        tokenizer: AutoTokenizer,
        block_size: int,
        shuffle_files: bool = False,
        cache_token_budget: int = TOKEN_CACHE_BUDGET,
    ):
        super().__init__()
        assert block_size > 0
        self.files = list(files)
        if shuffle_files:
            random.shuffle(self.files)

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.eos = tokenizer.eos_token or "\n"
        self.cache = LRUTokenCache(cache_token_budget)

        # Build chunk index (pre-pass, no token storage)
        self._file_chunk_counts: List[int] = []
        self._index: List[Tuple[int, int]] = []  # (file_idx, chunk_idx)
        for fi, fp in enumerate(self.files):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                self._file_chunk_counts.append(0)
                continue

            if text and not text.endswith(self.eos):
                text = text + self.eos

            # Tokenize once to compute how many chunks (drop remainder)
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            n_tokens = len(ids)
            n_chunks = n_tokens // self.block_size
            self._file_chunk_counts.append(n_chunks)
            for ci in range(n_chunks):
                self._index.append((fi, ci))

        # If shuffle_files was requested, we still keep chunks in file order.
        # If you want chunk-level shuffling, shuffle self._index here.
        # random.shuffle(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def _tokens_for_file(self, file_idx: int) -> torch.Tensor:
        fp = self.files[file_idx]
        cached = self.cache.get(fp)
        if cached is not None:
            return cached

        # Tokenize and cache
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        if text and not text.endswith(self.eos):
            text = text + self.eos
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        t = torch.tensor(ids, dtype=torch.long)
        self.cache.put(fp, t)
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, chunk_idx = self._index[idx]
        tok = self._tokens_for_file(file_idx)
        start = chunk_idx * self.block_size
        end = start + self.block_size
        x = tok[start:end]
        # Safety (should already be exact size):
        if x.numel() != self.block_size:
            # pad (unlikely unless file changed between pre-pass and now)
            pad = self.block_size - x.numel()
            x = torch.nn.functional.pad(x, (0, pad), value=self.tokenizer.pad_token_id or 0)

        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": x.clone(),
        }


def collate_already_tokenized(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # All items are exactly block_size; just stack
    return {k: torch.stack([ex[k] for ex in batch], dim=0) for k in batch[0].keys()}


class FiniteDataModule(pl.LightningDataModule):
    """Same finite, sized dataset for train and val (consistent behavior/progress bars)."""
    def __init__(
        self,
        train_files: List[Path],
        val_files: List[Path],
        tokenizer: AutoTokenizer,
        block_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        train_shuffle_files: bool = True,
    ):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_shuffle_files = train_shuffle_files

        self._train_ds = None
        self._val_ds = None

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self._train_ds = TokenChunkDataset(
                self.train_files, self.tokenizer, self.block_size, shuffle_files=self.train_shuffle_files
            )
            self._val_ds = TokenChunkDataset(
                self.val_files, self.tokenizer, self.block_size, shuffle_files=False
            )

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            collate_fn=collate_already_tokenized,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,  # dataset already in desired order (chunk-level shuffle optional)
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            collate_fn=collate_already_tokenized,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class HFCLMModule(pl.LightningModule):
    def __init__(
        self,
        repo: str,
        trust_remote_code: bool,
        lr: float,
        warmup_steps: int,
        bf16_if_cuda: bool = True,
        enable_grad_ckpt: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model - let Lightning handle the device management
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.float32,  # MPS doesn't support bf16
            trust_remote_code=trust_remote_code,
        )

        if enable_grad_ckpt and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        self._lr = lr
        self._warmup_steps = warmup_steps

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        val_loss = out.loss
        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self._lr)
        # Scheduler steps depend on trainer.estimated_stepping_batches (works with sized dataloaders)
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else 1000
        sched = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=min(self._warmup_steps, total_steps // 10), num_training_steps=total_steps
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }


def safe_run_name(repo: str) -> str:
    return repo.replace("/", "__")


def run_one_model(
    repo: str,
    parm_count, 
    trust_remote_code: bool,
    train_files: List[Path],
    val_files: List[Path],
    args,
):
    run_name = safe_run_name(f"{repo}_{parm_count}")
    logger = TensorBoardLogger(save_dir=args.log_dir, name=run_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=Path(args.out_dir) / run_name,
        filename="{epoch:02d}-{step:06d}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    # Module first to get its tokenizer for the DataModule
    module = HFCLMModule(
        repo=repo,
        trust_remote_code=trust_remote_code,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        bf16_if_cuda=not args.no_bf16,
        enable_grad_ckpt=not args.no_grad_ckpt,
    )
    
    pin_memory = torch.cuda.is_available()
    dm = FiniteDataModule(
        train_files=train_files,
        val_files=val_files,
        tokenizer=module.tokenizer,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        train_shuffle_files=True,
    )

    # Build trainer with finite epochs → bounded progress bars
    trainer = pl.Trainer(
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices="auto",
        precision="32-true",  # MPS doesn't support bf16
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.grad_accum,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        logger=logger,
        callbacks=[ckpt_cb, lrmon],
        # sanity val is fine now; you can set num_sanity_val_steps=0 to skip
    )

    trainer.fit(module, datamodule=dm)
    print(f"[OK] Finished Lightning run for {repo}. Logs: {logger.log_dir}  Checkpoints: {ckpt_cb.dirpath}")


def main():
    parser = argparse.ArgumentParser(description="Finite-epoch Lightning finetune for multiple HF models")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing 'training' and 'validation' subfolders")
    parser.add_argument("--json", type=str, default=JSON_PATH, help="Path to models_plan.json")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--log_dir", type=str, default="lightning_logs")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16 mixed precision on CUDA")
    parser.add_argument("--no_grad_ckpt", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--token_cache_budget", type=int, default=TOKEN_CACHE_BUDGET, help="Approx token budget for in-memory file token cache")
    args = parser.parse_args()

    # Sanity
    data_dir = Path(args.data_dir).expanduser().resolve()
    train_dir = data_dir / "training"
    val_dir = data_dir / "validation"
    assert train_dir.exists() and train_dir.is_dir(), f"Train folder not found: {train_dir}"
    assert val_dir.exists() and val_dir.is_dir(), f"Val folder not found: {val_dir}"

    with open(args.json, "r", encoding="utf-8") as f:
        plan = json.load(f)
    assert "models" in plan, "JSON has no 'models' key"

    train_files = find_text_files(train_dir)
    val_files = find_text_files(val_dir)
    assert train_files, f"No text files found under {train_dir}"
    assert val_files, f"No text files found under {val_dir}"
    print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")

    # Repro
    random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for entry in plan["models"]:
        assert "hf_repo" in entry, f"Model entry missing 'hf_repo': {entry}"
        repo = entry["hf_repo"]
        size_b = entry["size_b"]
        trust_remote_code = bool(entry.get("trust_remote_code", False))
        print(f"\n=== Lightning run: {repo} ===")
        run_one_model(repo, size_b, trust_remote_code, train_files, val_files, args)


if __name__ == "__main__":
    main()
