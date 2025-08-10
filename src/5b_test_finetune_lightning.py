#!/usr/bin/env python3
import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Iterator, Dict, Any, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset

# --------------------
# Defaults (tweak as needed)
# --------------------
TEXT_EXTS = {".txt", ".text", ".md"}
BLOCK_SIZE = 1024
BATCH_SIZE = 4
NUM_WORKERS = 2
GRAD_ACCUM_STEPS = 4
LR = 2e-5
MAX_STEPS = 100
WARMUP_STEPS = 50
SEED = 42
JSON_PATH = "./data/models_plan.json"


def find_text_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS]



class DocDataset(Dataset):
    """
    Map-style, sized dataset for validation: one item = full file contents.
    Lets Lightning know len() so the progress bar has an end.
    """
    def __init__(self, files: List[Path]):
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> str:
        fp = self.files[idx]
        with open(fp, "r", encoding="utf-8") as f:
            return f.read().strip()
        
#
class LineStream(IterableDataset):
    """Streams non-empty lines from files (low memory)."""
    def __init__(self, files: List[Path], shuffle_files: bool = True):
        self.files = list(files)
        self.shuffle_files = shuffle_files

    def __iter__(self) -> Iterator[str]:
        files = self.files[:]
        if self.shuffle_files:
            random.shuffle(files)
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
            except Exception:
                continue


def make_collate_fn(tokenizer, block_size: int):
    eos = tokenizer.eos_token

    def collate(batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        # Append EOS for stability if available
        if eos:
            texts = [t if t.endswith(eos) else (t + eos) for t in batch_texts]
        else:
            texts = batch_texts

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=block_size,
        )
        enc["labels"] = enc["input_ids"].clone()
        return enc

    return collate


class TokenChunkStream(IterableDataset):
    """
    Streams files, tokenizes on the fly, concatenates into a rolling buffer of token ids,
    and yields fixed-length blocks of size `block_size`.
    """
    def __init__(self, files: List[Path], tokenizer, block_size: int, shuffle_files: bool = True):
        self.files = list(files)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.shuffle_files = shuffle_files

        # Ensure we have an EOS to separate docs; if no eos, use newline.
        self._eos = tokenizer.eos_token or "\n"

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = self.files[:]
        if self.shuffle_files:
            random.shuffle(files)

        # rolling token buffer
        buf = []

        def push_text(txt: str):
            if not txt:
                return
            # Append EOS to separate docs/lines
            if not txt.endswith(self._eos):
                txt = txt + self._eos
            ids = self.tokenizer(txt, add_special_tokens=False)["input_ids"]
            buf.extend(ids)

        for fp in files:
            try:
                # Read whole file (or you can stream by line and call push_text per line)
                with open(fp, "r", encoding="utf-8") as f:
                    push_text(f.read())
            except Exception:
                continue

            # While we have enough tokens, emit fixed-size blocks
            while len(buf) >= self.block_size:
                block = buf[:self.block_size]
                buf = buf[self.block_size:]
                x = torch.tensor(block, dtype=torch.long)
                yield {
                    "input_ids": x,
                    "attention_mask": torch.ones_like(x),
                    "labels": x.clone(),  # CausalLM loss handles shift
                }

        # (Optional) drop remainder or pad it to a full block.
        # For a streaming “infinite” feel, it’s fine to drop remainders here.

def collate_already_tokenized(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Simple stack (no padding), because every sample is exactly `block_size` tokens.
    return {k: torch.stack([ex[k] for ex in batch], dim=0) for k in batch[0].keys()}


class StreamingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_files: List[Path],
        val_files: List[Path],
        tokenizer,
        block_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


    def train_dataloader(self):
        ds = TokenChunkStream(
            self.train_files, tokenizer=self.tokenizer,
            block_size=self.block_size, shuffle_files=True
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=collate_already_tokenized,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        # Use sized, per-document dataset so validation is finite and progress bar shows total
        ds = DocDataset(self.val_files)
        return DataLoader(
            ds,
            batch_size=self.batch_size,              # with 2 files & batch_size=2 => exactly 1 batch
            collate_fn=make_collate_fn(self.tokenizer, self.block_size),
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
        max_steps: int,
        warmup_steps: int,
        bf16_if_cuda: bool = True,
        enable_grad_ckpt: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            # many GPT-2 tokenizers have no PAD; make PAD=EOS so we can pad
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model (Lightning will move it to device)
        dtype = torch.bfloat16 if (torch.cuda.is_available() and bf16_if_cuda) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

        if enable_grad_ckpt and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        self._lr = lr
        self._max_steps = max_steps
        self._warmup_steps = warmup_steps

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self._lr)
        # Scheduler needs total steps; use trainer.max_steps
        total_steps = self._max_steps if self.trainer is None else self.trainer.max_steps
        sched = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=self._warmup_steps, num_training_steps=total_steps
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
    # e.g. "openai-community/gpt2" -> "openai-community__gpt2"
    return repo.replace("/", "__")


def run_one_model(
    repo: str,
    trust_remote_code: bool,
    train_files: List[Path],
    val_files: List[Path],
    args,
):
    # Lightning bits
    run_name = safe_run_name(repo)
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

    # Module + Data
    module = HFCLMModule(
        repo=repo,
        trust_remote_code=trust_remote_code,
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        bf16_if_cuda=not args.no_bf16,
        enable_grad_ckpt=not args.no_grad_ckpt,
    )
    pin_memory = torch.cuda.is_available()
    dm = StreamingDataModule(
        train_files=train_files,
        val_files=val_files,
        tokenizer=module.tokenizer,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if (torch.cuda.is_available() and not args.no_bf16) else "32-true",
        max_steps=args.max_steps,
        accumulate_grad_batches=args.grad_accum,
        val_check_interval=args.val_every,   # validate every N training steps
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        logger=logger,
        callbacks=[ckpt_cb, lrmon],
    )

    trainer.fit(module, datamodule=dm)
    print(f"[OK] Finished Lightning run for {repo}. Logs: {logger.log_dir}  Checkpoints: {ckpt_cb.dirpath}")


def main():
    parser = argparse.ArgumentParser(description="Lightning streaming finetune for multiple HF models")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder with data text files: needs training and validation sub folders")
    parser.add_argument("--json", type=str, default=JSON_PATH, help="Path to models_plan.json")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--val_every", type=int, default=50, help="Validate every N steps")
    parser.add_argument("--log_dir", type=str, default="lightning_logs")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16 mixed precision on CUDA")
    parser.add_argument("--no_grad_ckpt", action="store_true", help="Disable gradient checkpointing")
    args = parser.parse_args()

    # Sanity
    data_dir =  Path(args.data_dir).expanduser().resolve()
    train_dir = Path.joinpath(data_dir, 'training')
    val_dir = Path.joinpath(data_dir, 'validation')
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

    # Loop models
    for entry in plan["models"]:
        assert "hf_repo" in entry, f"Model entry missing 'hf_repo' key: {entry}"
        repo = entry["hf_repo"]
        trust_remote_code = bool(entry.get("trust_remote_code", False))
        print(f"\n=== Lightning run: {repo} ===")
        run_one_model(repo, trust_remote_code, train_files, val_files, args)


if __name__ == "__main__":
    main()
