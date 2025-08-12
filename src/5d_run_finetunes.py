#!/usr/bin/env python3
"""
finetune_eval_l2.py  (Lightning 2.x style)

Finetunes/evaluates one or more small LMs (from a JSON config) on a simple
line-based dataset with context windows. Logs losses, perplexity, LR, and
sample generations to TensorBoard. Performs full validation before training.

Key Lightning 2.x bits:
- Imports from `lightning.pytorch` (or `import lightning as L`)
- `seed_everything` from `lightning.pytorch`
- Precision strings like "16-mixed" / "32-true"
- Tuner for batch-size autoscale

Docs:
- 2.x migration & Trainer/precision docs: lightning.ai/docs/pytorch (see README)
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # safest with DataLoader workers
os.environ["TRANSFORMERS_VERBOSITY"] = "error" 


import sys
import json
import math
import argparse
import logging
import traceback
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.tuner.tuning import Tuner

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# --------------------
# Defaults
# --------------------
TEXT_EXTS = {".txt", ".md", ".text"}
LOG = logging.getLogger("finetune_eval")


# --------------------
# Files & text utils
# --------------------
def find_text_files(root: Path) -> List[Path]:
    files = []
    for ext in TEXT_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def safe_read_lines(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    except Exception as e:
        LOG.warning(f"Failed to read {path}: {e}")
        return []


def build_windows_from_lines(lines: List[str], context: int) -> List[str]:
    if context <= 0:
        return lines[:]
    out = []
    for i in range(context, len(lines)):
        ctx = lines[i - context : i]
        tgt = lines[i]
        out.append("\n".join(ctx + [tgt]))
    return out


# --------------------
# Dataset with LRU-cached tokenization
# --------------------
class WindowTextDataset(Dataset):
    def __init__(self, windows: List[str], tokenizer: AutoTokenizer, block_size: int):
        self.windows = windows
        self.block_size = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.windows)

    @lru_cache(maxsize=50_000)
    def _cached_encode(self, text: str, block_size: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )
        return tuple(enc["input_ids"]), tuple(enc["attention_mask"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.windows[idx]
        input_ids_tup, attn_mask_tup = self._cached_encode(text, self.block_size)
        input_ids = torch.tensor(input_ids_tup, dtype=torch.long)
        attention_mask = torch.tensor(attn_mask_tup, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # standard causal LM loss
        }


# --------------------
# Lightning DataModule
# --------------------
class SimpleDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        tokenizer: AutoTokenizer,
        block_size: int,
        context: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        val_sample_count: int = 3,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.context = context
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_sample_count = val_sample_count

        self._train_ds = None
        self._val_ds = None
        self.val_preview_texts: List[str] = []

    def setup(self, stage: str | None = None):
        train_root = self.data_dir / "training"
        val_root = self.data_dir / "validation"

        # Assertions per spec
        assert train_root.exists(), f"Training directory not found: {train_root}"
        assert val_root.exists(), f"Validation directory not found: {val_root}"

        train_files = find_text_files(train_root)
        val_files = find_text_files(val_root)

        assert len(train_files) > 0, "No training files found"
        assert len(val_files) > 0, "No validation files found"

        train_lines: List[str] = []
        for p in train_files:
            train_lines.extend(safe_read_lines(p))
        val_lines: List[str] = []
        for p in val_files:
            val_lines.extend(safe_read_lines(p))

        assert len(train_lines) > 0, "Training files contained no usable text"
        assert len(val_lines) > 0, "Validation files contained no usable text"

        train_windows = build_windows_from_lines(train_lines, self.context)
        val_windows = build_windows_from_lines(val_lines, self.context)

        self.val_preview_texts = val_windows[: self.val_sample_count]

        self._train_ds = WindowTextDataset(train_windows, self.tokenizer, self.block_size)
        self._val_ds = WindowTextDataset(val_windows, self.tokenizer, self.block_size)

        LOG.info(f"Train samples: {len(self._train_ds)} | Val samples: {len(self._val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


# --------------------
# LightningModule
# --------------------
class CausalLMModule(L.LightningModule):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        max_new_tokens: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_new_tokens = max_new_tokens

        self._param_count = sum(p.numel() for p in self.model.parameters())

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out.loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        val_loss = out.loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def on_train_start(self):
        if isinstance(self.logger, TensorBoardLogger):
            tb = self.logger.experiment
            tb.add_scalar("model/num_parameters", float(self._param_count), self.global_step)
            tb.add_text("model/tokenizer_info", f"vocab_size={self.tokenizer.vocab_size}", self.global_step)

    def on_validation_epoch_end(self):
        # Perplexity from aggregated val_loss
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            try:
                ppl = torch.exp(val_loss)
                self.log("perplexity", ppl, prog_bar=True, sync_dist=True)
                if isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_scalar("val/perplexity", float(ppl), self.global_step)
            except Exception:
                pass

        # Sample generations for a few validation prompts
        dm = self.trainer.datamodule
        if not isinstance(dm, SimpleDataModule) or not dm.val_preview_texts:
            return
        try:
            self.model.eval()
            samples = []
            for i, prompt_text in enumerate(dm.val_preview_texts):
                enc = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=dm.block_size,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                with torch.no_grad():
                    gen_ids = self.model.generate(
                        **enc,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                samples.append(f"### Prompt {i+1}\n{prompt_text}\n\n### Output {i+1}\n{gen_text}\n")

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_text("samples", "\n".join(samples), self.global_step)
        except Exception as e:
            LOG.warning(f"Sample generation failed: {e}")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Estimate total steps the Lightning 2.x way
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None:
            train_batches = len(self.trainer.datamodule.train_dataloader())  # type: ignore[attr-defined]
            total_steps = train_batches * self.trainer.max_epochs

        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "linear_warmup",
            },
        }


# --------------------
# Device / precision selection (Lightning 2.x precision strings)
# --------------------
def select_accel_precision_devices() -> Tuple[str, str, int]:
    """
    Returns (accelerator, precision, devices)
      accelerator: 'gpu' | 'mps' | 'cpu'
      precision:   '16-mixed' for CUDA else '32-true'
      devices:     1
    """
    if torch.cuda.is_available():
        LOG.info("CUDA available: using GPU with mixed precision.")
        return "gpu", "16-mixed", 1
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        LOG.info("MPS available: using Apple MPS with 32-bit precision.")
        return "mps", "32-true", 1
    LOG.info("Falling back to CPU (32-bit precision).")
    return "cpu", "32-true", 1


# --------------------
# Training runner with OOM fallback
# --------------------
def run_for_model(model_cfg: Dict[str, Any], args: argparse.Namespace, global_outdir: Path):
    hf_repo = model_cfg["hf_repo"]
    size_b = model_cfg["size_b"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    safe_name = hf_repo.replace("/", "__")
    outdir = global_outdir / safe_name
    outdir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=str(outdir), name="tb")
    accelerator, precision, devices = select_accel_precision_devices()

    # Tokenizer & model
    LOG.info(f"Loading tokenizer: {hf_repo}")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    LOG.info(f"Loading model: {hf_repo} (size {size_b})")
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_repo, trust_remote_code=trust_remote_code)
    except Exception as e:
        LOG.error(f"Failed to load model {hf_repo}: {e}")
        return

    # Gradient checkpointing
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            if getattr(model.config, "use_cache", None) is not None:
                model.config.use_cache = False
            model.gradient_checkpointing_enable()
            LOG.info("Enabled gradient checkpointing.")
    except Exception as e:
        LOG.warning(f"Could not enable gradient checkpointing: {e}")

    # Data
    num_workers = max(1, (os.cpu_count() or 2) - 1)
    pin_memory = (accelerator == "gpu")  # pin memory is meaningful for CUDA
    datamodule = SimpleDataModule(
        data_dir=Path(args.data_dir),
        tokenizer=tokenizer,
        block_size=args.block_size,
        context=args.context,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_sample_count=3,
    )

    lit_module = CausalLMModule(
        model=model,
        tokenizer=tokenizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_new_tokens=args.sample_max_new_tokens,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", mode="min", patience=args.early_stopping_patience),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=str(outdir / "checkpoints"),
            filename="{epoch:02d}-{val_loss:.4f}",
        ),
    ]

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        max_epochs=args.epochs,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=False,
        enable_progress_bar=True,
        # checkpointing is enabled by default; explicit for clarity
        enable_checkpointing=True,
    )

    # Optional batch size autoscaling via Tuner (Lightning 2.x)
    if args.auto_scale_bs:
        LOG.info("Running batch size auto-scaling (binsearch).")
        tuner = Tuner(trainer)
        new_bs = tuner.scale_batch_size(lit_module, datamodule=datamodule, mode="binsearch")
        LOG.info(f"Auto-scaled batch size: {new_bs}")
        datamodule.batch_size = new_bs

    # Full validation BEFORE training (beyond sanity val)
    LOG.info("Running initial full validation...")
    try:
        trainer.validate(lit_module, datamodule=datamodule, verbose=False)
    except Exception as e:
        LOG.warning(f"Initial validation encountered an issue (continuing): {e}")

    # Train with OOM fallback (halve batch size until it fits)
    bs = datamodule.batch_size
    while bs > 0:
        try:
            LOG.info(f"Starting training with batch_size={bs}")
            trainer.fit(lit_module, datamodule=datamodule)
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                LOG.warning(f"OOM at batch_size={bs}. Reducing by half and retrying...")
                bs //= 2
                if bs < 1:
                    LOG.error("Cannot reduce batch size further. Aborting for this model.")
                    break
                datamodule.batch_size = bs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                LOG.error(f"Runtime error during training: {e}")
                LOG.debug(traceback.format_exc())
                break
        except Exception as e:
            LOG.error(f"Unexpected error during training: {e}")
            LOG.debug(traceback.format_exc())
            break

    LOG.info("Training complete.")
    try:
        ckpt = callbacks[-1].best_model_path  # from ModelCheckpoint
        LOG.info(f"Best checkpoint: {ckpt if ckpt else 'none'}")
    except Exception:
        pass


# --------------------
# Config parsing & validation
# --------------------
def load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    assert config_path.exists(), f"Config not found: {config_path}"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert "models" in cfg and isinstance(cfg["models"], list), 'Config must contain key "models" (list).'
    for m in cfg["models"]:
        assert "hf_repo" in m and "size_b" in m, 'Each model requires "hf_repo" and "size_b"'
        if "trust_remote_code" not in m:
            m["trust_remote_code"] = False
    return cfg


# --------------------
# Main
# --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Finetune/evaluate small LMs with Lightning 2.x.")
    p.add_argument("--config", type=str, required=True, help="Path to JSON config with models list.")
    p.add_argument("--data_dir", type=str, required=True, help="Data directory with training/ and validation/")
    p.add_argument("--out_dir", type=str, default="./runs", help="Output root directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4, help="Starting batch size (may be auto-scaled or reduced on OOM)")
    p.add_argument("--auto_scale_bs", action="store_true", help="Use Lightning Tuner to auto-scale batch size")
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--block_size", type=int, default=512, help="Max tokens per example")
    p.add_argument("--context", type=int, default=3, help="Number of previous lines as context")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--sample_max_new_tokens", type=int, default=64, help="Tokens to generate for sample predictions")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # --- Critical assertions (early exit) ---
    config_path = Path(args.config)
    data_dir = Path(args.data_dir)

    assert config_path.exists(), f"Config not found: {config_path}"
    assert data_dir.exists(), f"Data directory not found: {data_dir}"

    # Check dirs & text file presence up front for better errors
    train_dir = data_dir / "training"
    val_dir = data_dir / "validation"
    train_files = find_text_files(train_dir)
    val_files = find_text_files(val_dir)

    assert train_dir.exists(), f"Training directory not found: {train_dir}"
    assert val_dir.exists(), f"Validation directory not found: {val_dir}"
    assert len(train_files) > 0, "No training files found"
    assert len(val_files) > 0, "No validation files found"

    cfg = load_and_validate_config(config_path)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Seeding (Lightning 2.x)
    seed_everything(args.seed, workers=True)

    # Train/evaluate each model
    for m in cfg["models"]:
        LOG.info(f"=== Model: {m['hf_repo']} (size {m['size_b']}) ===")
        try:
            run_for_model(m, args, out_root)
        except Exception as e:
            LOG.error(f"Fatal error for model {m['hf_repo']}: {e}")
            LOG.debug(traceback.format_exc())
            LOG.info("Continuing to next model...")

    LOG.info("All done.")


if __name__ == "__main__":
    main()
