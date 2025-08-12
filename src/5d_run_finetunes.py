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

# import sys
import json
# import math
import argparse
import logging
import traceback
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch

# Put this near the top of your script (right after importing torch)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")  # or "medium"
    # Optional: for FP32 *convolutions* only
    torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import random
from torch.utils.data import IterableDataset, get_worker_info

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

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100   # ignore pad in CE loss
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": input_ids.clone(),  # standard causal LM loss
        # }

# TrainIterableDataset: change ctor signature
class TrainIterableDataset(IterableDataset):
    def __init__(self, files, tok_name_or_path: str, tok_kwargs: dict,
                 block_size: int, context: int, shuffle_files: bool = True):
        super().__init__()
        self.files = list(files)
        self.tok_name_or_path = tok_name_or_path
        self.tok_kwargs = tok_kwargs
        self.block_size = block_size
        self.context = context
        self.shuffle_files = shuffle_files
        self._tok = None  # lazy per-process

    def _tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.tok_name_or_path, **self.tok_kwargs)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token or self._tok.cls_token
        return self._tok

    def _yield_windows_from_file(self, path: Path):
        lines = safe_read_lines(path)
        for i in range(self.context, len(lines)):
            yield "\n".join(lines[i-self.context:i] + [lines[i]])

    def __iter__(self):
        import random, torch
        rng = random.Random(torch.initial_seed() % (2**32))
        worker = get_worker_info()
        file_iter = self.files if worker is None else self.files[worker.id :: worker.num_workers]
        file_iter = list(file_iter)
        if self.shuffle_files:
            rng.shuffle(file_iter)

        tok = self._tokenizer()
        for p in file_iter:
            for text in self._yield_windows_from_file(p):
                enc = tok(text, truncation=True, max_length=self.block_size, padding="max_length")
                yield {
                    "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(enc["input_ids"], dtype=torch.long),
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

        self._train_files: list[Path] | None = None
        self._train_ds = None          # built lazily in train_dataloader
        self._val_ds: WindowTextDataset | None = None
        self.val_preview_texts: list[str] = []

    def setup(self, stage: str | None = None):
        train_root = self.data_dir / "training"
        val_root   = self.data_dir / "validation"
        assert train_root.exists(), f"Training directory not found: {train_root}"
        assert val_root.exists(),   f"Validation directory not found: {val_root}"

        # Build ONLY what the stage needs:
        if stage in (None, "validate"):
            if self._val_ds is None:
                val_files = find_text_files(val_root)
                assert len(val_files) > 0, "No validation files found"
                val_lines: list[str] = []
                for p in val_files:
                    val_lines.extend(safe_read_lines(p))
                assert val_lines, "Validation files contained no usable text"
                val_windows = build_windows_from_lines(val_lines, self.context)
                self.val_preview_texts = val_windows[: self.val_sample_count]
                self._val_ds = WindowTextDataset(val_windows, self.tokenizer, self.block_size)
                LOG.info(f"Val samples: {len(self._val_ds)}")

        if stage in (None, "fit"):
            if self._train_files is None:
                train_files = find_text_files(train_root)
                assert len(train_files) > 0, "No training files found"
                # store file list only; dataset is created lazily in train_dataloader()
                self._train_files = train_files
                LOG.info(f"Train files: {len(self._train_files)}")
            # Ensure val exists during fit (if validate() wasn’t called earlier)
            if self._val_ds is None:
                self.setup(stage="validate")

    def train_dataloader(self):
        # Build the iterable dataset lazily so we don't pre-load training data
        if self._train_files is None:
            self.setup(stage="fit")
        self._train_ds = TrainIterableDataset(
            files=self._train_files,
            tok_name_or_path=self.tokenizer.name_or_path,   # <- pass name/path, not object
            tok_kwargs={"use_fast": True, "trust_remote_code": getattr(self.tokenizer, "init_kwargs", {}).get("trust_remote_code", False)},
            block_size=self.block_size,
            context=self.context,
            shuffle_files=True,
        )
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=1,  # was 2 by default
        )
    def val_dataloader(self):
        if self._val_ds is None:
            self.setup(stage="validate")
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
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
        # Prefer trainer.max_steps if provided
        total_steps = getattr(self.trainer, "max_steps", None)
        if not total_steps or total_steps < 0:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if not total_steps or total_steps <= 0:
            # last-resort fallback (e.g., user forgot --max_steps). keep small but nonzero.
            total_steps = 1000

        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "linear_warmup"}}


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



def autoscale_batch_size_mps_safe(lit_module, datamodule, accelerator, devices, precision):
    """set a viable batch size using as much VRAM as possible """
    tune_trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        num_sanity_val_steps=0,   # no pre-fit val
        max_epochs=1,             # we only need a few steps
        limit_train_batches=2,    # tiny trial
        limit_val_batches=0,      # skip val during tuning
        enable_checkpointing=False,
        logger=False,
    )
    tuner = Tuner(tune_trainer)
    new_bs = tuner.scale_batch_size(
        lit_module,
        datamodule=datamodule,
        mode="power",             # safer on MPS than "binsearch"
        steps_per_trial=1,        # minimum work per try
        init_val=max(1, datamodule.batch_size // 2),
        max_trials=6,             # cap the search
        batch_arg_name="batch_size",
    )
    # Clear allocator caches after tuning (important for MPS)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return new_bs

def suggested_num_workers(cap: int | None = None) -> int:
    import os
    # 1) What PyTorch uses for the warning: count of allowed logical CPUs (Linux cpuset)
    try:
        max_allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        # macOS / others (no cpuset): fall back to all CPUs
        max_allowed = os.cpu_count() or 2

    # 2) If running under SLURM, cap to CPUs-per-task (often stricter)
    slurm_cpt = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpt:
        try:
            max_allowed = min(max_allowed, int(slurm_cpt))
        except ValueError:
            pass

    # 3) Leave one core for the main process; enforce >=1
    n = max(1, max_allowed - 1)

    # Optional global cap (e.g., avoid silly values on huge nodes)
    if cap is not None:
        n = min(n, cap)
    return n

def get_model_max_len(model, tokenizer, default_cap: int = 4096) -> int:
    # collect candidate caps from config/tokenizer, ignore "very large" sentinel values
    cands = []
    for attr in ("n_positions", "max_position_embeddings", "seq_length", "max_seq_len"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v > 0:
            cands.append(v)
    tm = getattr(tokenizer, "model_max_length", None)
    if isinstance(tm, int) and 0 < tm < 10**9:
        cands.append(tm)
    return min(cands) if cands else default_cap

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


    logger = TensorBoardLogger(save_dir=str(outdir), name=f"size_{size_b}")
    ckpt_dir = Path(logger.log_dir) / "checkpoints" 
    
    accelerator, precision, devices = select_accel_precision_devices()
    # Tokenizer & model
    LOG.info(f"Loading tokenizer: {hf_repo}")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
   
    tokenizer.truncation_side = "left"     # keep the tail
    tokenizer.padding_side = "right"       # usual default



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
    # num_workers = max(1, (os.cpu_count() or 2) - 1)
    num_workers = suggested_num_workers()   # e.g., becomes 1 if SLURM grants 2 CPUs
    pin_memory = (accelerator == "gpu")  # pin memory is meaningful for CUDA
    
    safe_model_ctx = get_model_max_len(model, tokenizer)
    # Clamp your block_size so you never exceed the model’s context
    effective_block_size = min(args.block_size, safe_model_ctx)

    datamodule = SimpleDataModule(
        data_dir=Path(args.data_dir),
        tokenizer=tokenizer,
        block_size=effective_block_size,   # <-- use the clamped value
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
            dirpath=str(ckpt_dir),            # <<— inside version_X
            # dirpath=str(outdir / "checkpoints"),
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
        # max_steps=args.max_steps,  # if set, bypasses step estimation
        deterministic=False,
        enable_progress_bar=True,
        # checkpointing is enabled by default; explicit for clarity
        enable_checkpointing=True,
        num_sanity_val_steps=0,  # skip extra pre-training val
    )

    if args.auto_scale_bs:
        LOG.info(f"Prior to auto-scaling, batch size is {datamodule.batch_size}")
        try:
            new_bs = autoscale_batch_size_mps_safe(lit_module, datamodule, accelerator, devices, precision)
            if new_bs: 
                datamodule.batch_size = new_bs
                LOG.info(f"Auto-scaled batch size (MPS-safe): {new_bs}")
        except RuntimeError as e:
            LOG.warning(f"Batch-size tuning failed on this device: {e}. Will continue with OOM backoff.")
            # datamodule.batch_size = int(datamodule.batch_size / 4)
            # maybe  
     
    if torch.backends.mps.is_available():# always scale to 0.25 as  mps seems over-keen on over allocation
        datamodule.batch_size = int(datamodule.batch_size / 4)
            
    LOG.info(f"Chose batch size {datamodule.batch_size}")

    # Full validation BEFORE training (beyond sanity val)
    LOG.info("Running initial full validation...")
    # try:
    trainer.validate(lit_module, datamodule=datamodule, verbose=False)
    # except Exception as e:
        # LOG.warning(f"Initial validation encountered an issue (continuing): {e}")

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
    p.add_argument("--model", type=str, required=False, default=None, help="only run one model with this HF model name - needs to be in the JSON file")  
    p.add_argument("--data_dir", type=str, required=True, help="Data directory with training/ and validation/")
    p.add_argument("--out_dir", type=str, default="./runs", help="Output root directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4, help="Starting batch size (may be auto-scaled or reduced on OOM)")    
    # p.add_argument("--auto_scale_bs", action="store_true", help="Use Lightning Tuner to auto-scale batch size")
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--block_size", type=int, default=512, help="Max tokens per example")
    p.add_argument("--context", type=int, default=64, help="Number of previous lines as context. in NJAM, that's number of notes")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--sample_max_new_tokens", type=int, default=64, help="Tokens to generate for sample predictions")
   
    p.add_argument("--auto_scale_bs", action="store_true", help="Auto-tune batch size with Lightning Tuner")
    p.add_argument("--bs_mode", type=str, default="binsearch", choices=["power", "binsearch"])
    p.add_argument("--bs_init", type=int, default=None, help="Initial batch size to start search with (defaults to --batch_size)")
    p.add_argument("--bs_max_trials", type=int, default=10, help="Max doublings before binary search terminates")
    p.add_argument("--bs_steps_per_trial", type=int, default=3, help="Steps to try per candidate batch size")

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

    if args.model is not None:
        cfg["models"] = [mdl for mdl in cfg["models"] if mdl["hf_repo"] == args.model]
        assert len(cfg["models"]) > 0, f"Could not find hf model {args.model} in config"
    # Train/evaluate each model

    for m in cfg["models"]:
            
        LOG.info(f"=== Model: {m['hf_repo']} (size {m['size_b']}) ===")
        # try:
        run_for_model(m, args, out_root)
        # except Exception as e:
        #     LOG.error(f"Fatal error for model {m['hf_repo']}: {e}")
        #     LOG.debug(traceback.format_exc())
        #     LOG.info("Continuing to next model...")

    LOG.info("All done.")


if __name__ == "__main__":
    main()
