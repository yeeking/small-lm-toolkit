#!/usr/bin/env python3
"""
finetune_eval_lora.py  — Lightning 2.x + HF Transformers + PEFT (LoRA)

This is a LoRA version of your full-finetune script. Key points:
- Wraps the base model with PEFT LoRA; trains only adapter params
- Keeps your dataset, DataModule, validation preview, TB logging
- Saves adapter weights (not full FP32 model) at the end and "best" val
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import json
import argparse
import logging
import traceback
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple
# import random 
import torch 
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True


from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset, get_worker_info
from torch.optim import AdamW

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    Callback,
)


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# PEFT (LoRA)
from peft import LoraConfig, get_peft_model, TaskType

import shared_utils


LOG = logging.getLogger("finetune_eval_lora")
# TEXT_EXTS = {".txt", ".md", ".text"}

# --------------------
# LightningModule (LoRA)
# --------------------
class CausalLMWithLoRA(L.LightningModule):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        lr: float = 2e-4,             # LoRA often benefits from a slightly higher LR than full FT
        weight_decay: float = 0.0,    # commonly 0 for adapters
        warmup_ratio: float = 0.05,
        max_new_tokens: int = 64,
        save_adapter_dir: Path | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer", "save_adapter_dir"])
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_new_tokens = max_new_tokens
        self.save_adapter_dir = save_adapter_dir
        self._param_count = sum(p.numel() for p in self.model.parameters())
        self._trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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
            tb.add_scalar("model/num_parameters_total", float(self._param_count), self.global_step)
            tb.add_scalar("model/num_parameters_trainable", float(self._trainable_count), self.global_step)
            tb.add_text("model/tokenizer_info", f"vocab_size={self.tokenizer.vocab_size}", self.global_step)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            try:
                ppl = torch.exp(val_loss)
                self.log("perplexity", ppl, prog_bar=True, sync_dist=True)
                if isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_scalar("val/perplexity", float(ppl), self.global_step)
            except Exception:
                pass

        dm = self.trainer.datamodule
        if not isinstance(dm, shared_utils.SimpleDataModule) or not dm.val_preview_texts:
            return
        try:
            self.model.eval()
            samples = []
            for i, prompt_text in enumerate(dm.val_preview_texts):
                enc = self.tokenizer(
                    prompt_text, return_tensors="pt", truncation=True, max_length=dm.block_size
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
        # optimize only trainable (adapter) params
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)

        total_steps = getattr(self.trainer, "max_steps", None)
        if not total_steps or total_steps < 0:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if not total_steps or total_steps <= 0:
            total_steps = 1000
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "linear_warmup"}}

    def on_train_end(self):
        # Always save a final adapter checkpoint
        if self.save_adapter_dir is not None:
            final_dir = Path(self.save_adapter_dir) / "lora_adapter-final"
            final_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(final_dir)
            self.tokenizer.save_pretrained(final_dir)
            LOG.info(f"Saved final LoRA adapter to: {final_dir}")

# Save the current best adapter whenever ModelCheckpoint updates
class SaveBestAdapterCallback(Callback):
    def __init__(self, dirpath: Path, monitor: str = "val_loss", mode: str = "min"):
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        val = float(current.detach().cpu())
        better = (
            self.best is None or
            (self.mode == "min" and val < self.best) or
            (self.mode == "max" and val > self.best)
        )
        if better:
            self.best = val
            out_dir = self.dirpath / "lora_adapter-best"
            out_dir.mkdir(parents=True, exist_ok=True)
            pl_module.model.save_pretrained(out_dir)
            pl_module.tokenizer.save_pretrained(out_dir)
            LOG.info(f"Saved BEST LoRA adapter ({self.monitor}={val:.4f}) to: {out_dir}")


# --------------------
# Runner
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

    accelerator, precision, devices = shared_utils.select_accel_precision_devices()

    tokenizer, base_model = shared_utils.load_model_lora(hf_repo, trust_remote_code, args)
    peft_model =shared_utils.wrap_with_lora(base_model, args)

    num_workers = shared_utils.suggested_num_workers()
    pin_memory = False  # your note about leaked fds/persistent_workers
    safe_model_ctx = shared_utils.get_model_max_len(peft_model, tokenizer)
    effective_block_size = min(args.block_size, safe_model_ctx)

    dm = shared_utils.SimpleDataModule(
        data_dir=Path(args.data_dir),
        tokenizer=tokenizer,
        block_size=effective_block_size,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_sample_count=3, 
        # could specify min and max lines of context here... defaults are ok though
        
    )

    lit_module = CausalLMWithLoRA(
        model=peft_model,
        tokenizer=tokenizer,
        lr=args.lr,  # you can try higher for LoRA (e.g., 5e-4..2e-3)
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_new_tokens=args.sample_max_new_tokens,
        save_adapter_dir=outdir,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", mode="min", patience=args.early_stopping_patience),
        ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1,
            dirpath=str(ckpt_dir), filename="{epoch:02d}-{val_loss:.4f}",
        ),
        SaveBestAdapterCallback(dirpath=outdir, monitor="val_loss", mode="min"),
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
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )

    if args.auto_scale_bs:
        LOG.info(f"Prior to auto-scaling, batch size is {dm.batch_size}")
        try:
            new_bs = shared_utils.autoscale_batch_size_mps_safe(lit_module, dm, accelerator, devices, precision)
            if new_bs:
                dm.batch_size = new_bs
                LOG.info(f"Auto-scaled batch size (MPS-safe): {new_bs}")
        except RuntimeError as e:
            LOG.warning(f"Batch-size tuning failed: {e}. Continuing with OOM backoff.")

    if torch.backends.mps.is_available():
        dm.batch_size = max(1, int(dm.batch_size / 4))
    if torch.cuda.is_available():
        dm.batch_size = max(1, int(dm.batch_size / 2))
    LOG.info(f"Chose batch size {dm.batch_size}")

    LOG.info("Running initial full validation...")
    trainer.validate(lit_module, datamodule=dm, verbose=False)

    bs = dm.batch_size
    while bs > 0:
        try:
            LOG.info(f"Starting LoRA training with batch_size={bs}")
            trainer.fit(lit_module, datamodule=dm, ckpt_path=None)
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                LOG.warning(f"OOM at batch_size={bs}. Reducing by half and retrying...")
                bs //= 2
                if bs < 1:
                    LOG.error("Cannot reduce batch size further. Aborting for this model.")
                    break
                dm.batch_size = bs
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
        ckpt = callbacks[2].best_model_path  # ModelCheckpoint
        LOG.info(f"Best Lightning checkpoint: {ckpt if ckpt else 'none'}")
    except Exception:
        pass

# --------------------
# Config
# --------------------


# --------------------
# Main / args
# --------------------
def parse_args():
    p = argparse.ArgumentParser(description="LoRA finetune/evaluate small LMs with Lightning 2.x.")
    p.add_argument("--config", type=str, required=True, help="Path to JSON config with models list.")
    p.add_argument("--model", type=str, default=None, help="Run only this HF model (must be in the JSON).")
    p.add_argument("--data_dir", type=str, required=True, help="Data dir with training/ and validation/")
    p.add_argument("--out_dir", type=str, default="./runs_lora", help="Output root directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--block_size", type=int, default=512)
    # p.add_argument("--context", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)          # LoRA typical LR
    p.add_argument("--weight_decay", type=float, default=0.0) # adapters often 0
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--sample_max_new_tokens", type=int, default=64)

    # Batch-size tuner (kept from your script)
    p.add_argument("--auto_scale_bs", action="store_true")
    p.add_argument("--bs_mode", type=str, default="power", choices=["power","binsearch"])
    p.add_argument("--bs_init", type=int, default=None)
    p.add_argument("--bs_max_trials", type=int, default=10)
    p.add_argument("--bs_steps_per_trial", type=int, default=3)

    # LoRA hyperparams
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none", choices=["none","all","lora_only"])
    # Either "all-linear" (new PEFT) or list like q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    p.add_argument("--lora_target_modules", type=lambda s: s.split(",") if "," in s else s, default="all-linear",
                   help='Target modules (e.g. "all-linear" or "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")')

    # Optional quantized loading (defaults off)
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4","fp4"])

    return p.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # Critical assertions
    config_path = Path(args.config)
    data_dir = Path(args.data_dir)
    assert config_path.exists(), f"Config not found: {config_path}"
    assert data_dir.exists(), f"Data directory not found: {data_dir}"

    train_dir = data_dir / "training"
    val_dir = data_dir / "validation"
    train_files = shared_utils.find_text_files(train_dir)
    val_files = shared_utils.find_text_files(val_dir)
    assert train_dir.exists(), f"Training directory not found: {train_dir}"
    assert val_dir.exists(), f"Validation directory not found: {val_dir}"
    assert len(train_files) > 0, "No training files found"
    assert len(val_files) > 0, "No validation files found"

    cfg = shared_utils.load_and_validate_config(config_path)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed, workers=True)
    cfg["models"] = shared_utils.sort_models_by_size(cfg["models"])
    for m in cfg["models"]:
        print(f"Found model in cfg {m['hf_repo']} with params {m['size_b']}")

    if args.model is not None:
        cfg["models"] = [mdl for mdl in cfg["models"] if mdl["hf_repo"] == args.model]
        assert len(cfg["models"]) > 0, f"Could not find hf model {args.model} in config"

    for m in cfg["models"]:
        LOG.info(f"=== Model (LoRA): {m['hf_repo']} (size {m['size_b']}) ===")
        run_for_model(m, args, out_root)

    LOG.info("All done (LoRA).")

if __name__ == "__main__":
    import random  # used by TrainIterableDataset
    main()
