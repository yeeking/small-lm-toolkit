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

from huggingface_hub import snapshot_download

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

import shared_utils

# --------------------
# Defaults
# --------------------
# TEXT_EXTS = {".txt", ".md", ".text"}
LOG = logging.getLogger("finetune_eval")


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
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss

    def on_train_start(self):
        if isinstance(self.logger, TensorBoardLogger):
            tb = self.logger.experiment
            tb.add_scalar("model/num_parameters", float(self._param_count), self.global_step)
            tb.add_text("model/tokenizer_info", f"vocab_size={self.tokenizer.vocab_size}", self.global_step)

    def on_validation_epoch_end(self):
        # Perplexity from aggregated val_loss
        LOG.info(f"Validation epoch ended - running additional evaluations")

        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            try:
                ppl = torch.exp(val_loss)
                self.log("perplexity", ppl, prog_bar=True)
                if isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_scalar("val/perplexity", float(ppl), self.global_step)
            except Exception:
                pass

        # Sample generations for a few validation prompts
        dm = self.trainer.datamodule
        if not isinstance(dm, shared_utils.SimpleDataModule) or not dm.val_preview_texts:
            return
        try:
            self.model.eval()
            samples = []
            for i, prompt_text in enumerate(dm.val_preview_texts):
                enc = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=dm.want_ctx_size,
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
        LOG.info(f"Validation epoch ended - additional evaluations complete")


    def configure_optimizers(self):
        """This configuration drives learning rate scheduling using epoch"""
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        warmup_epochs = max(1, int(self.trainer.max_epochs * self.warmup_ratio))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)   # linear warmup
            progress = (epoch - warmup_epochs) / max(1, self.trainer.max_epochs - warmup_epochs)
            return max(0.0, 1.0 - progress)                      # linear decay

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch", "name": "epoch_warmup_linear"}}



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
    
    accelerator, precision, devices = shared_utils.select_accel_precision_devices()
    # Tokenizer & model

    LOG.info(f"Loading tokenizer & model for '{hf_repo}' from cache...")
    # tokenizer, model = load_model_cache(
    #     hf_repo,
    #     trust_remote_code=trust_remote_code,
    #     model_kwargs={
    #         "torch_dtype": (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
    #         "device_map": "auto" if torch.cuda.is_available() else None
    #     }
    # )
    tokenizer, model = shared_utils.load_model_no_cache(hf_repo, size_b, trust_remote_code)
    assert tokenizer is not None, f"Tokenizer did not load correctly."
    assert model is not None, f"Mode did not load correctly."
    
    if args.randomise_weights:
        LOG.info(f"Randomising the model weights for a from-scratch training run")
        shared_utils.reinit_model_weights(model)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    tokenizer.truncation_side = "left"     # keep the tail
    tokenizer.padding_side = "right"       # usual default


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
    num_workers = shared_utils.suggested_num_workers()   # e.g., becomes 1 if SLURM grants 2 CPUs
    # pin_memory = (accelerator == "gpu")  # pin memory is meaningful for CUDA
    pin_memory = False # pinning memory causes leaky file handles so OS.too many files open
    
    safe_model_ctx = shared_utils.get_model_max_len(model, tokenizer)
    LOG.info(f"Model context length is  maxing out at {safe_model_ctx} - you asked for {args.want_ctx_size}")
    # Clamp your available_ctx_size so you never exceed the model’s context
    available_ctx_size = min(args.want_ctx_size, safe_model_ctx)

    datamodule = shared_utils.SimpleDataModule(
        data_dir=Path(args.data_dir),
        tokenizer=tokenizer,
        want_ctx_size=available_ctx_size,   # <-- use the clamped value
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        min_lines_in_context=8, 
        max_lines_in_context=64, 
        # can optionally specify the min and max lines of context here, aka 'previous notes'
    )
    LOG.info(f"Validation batches: {len(datamodule.val_dataloader())}")
    
    # the actual model or at least the lightning wrapper around it 
    smallm_model = CausalLMModule(
        model=model,
        tokenizer=tokenizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_new_tokens=args.sample_max_new_tokens,
    )

    # renders example output from the model during training 
    example_renderer = shared_utils.HFPreviewResponder(smallm_model, shared_utils.PreviewGenConfig(
        max_new_tokens=64,
        do_sample=False,          # or True for sampling previews
        return_full_text=False,   # only show continuation
        truncate_to=1024, #smallm_model.trainer.datamodule.want_ctx_size if hasattr(smallm_model.trainer.datamodule, "want_ctx_size") else 1024,
    ))

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", mode="min", patience=args.early_stopping_patience),
        # Best checkpoint callback
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_loss:.4f}",
        ),
        # Periodic checkpoint callback
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch-{epoch:02d}",
            every_n_epochs=args.save_ckpt_every,
            save_top_k=-1,  # Keep all checkpoints
            save_on_train_epoch_end=True,
        ),
        # generates previews during training 
        shared_utils.PreviewAudioCallback(
            prompts=["what is the capital of paris? "], 
            render_fn=example_renderer, 
            every_n_epochs=1,  
            every_n_steps=1
        )
    ]

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        # max_epochs=args.epochs,
        max_epochs=args.epochs,
        min_epochs=args.epochs,   # run exactly N epochs
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
 	max_steps=args.epochs * 100000, # very high as it hept stopping         
# max_steps=args.max_steps,  # if set, bypasses step estimation
        deterministic=False,
        enable_progress_bar=args.progress_bar,
        # checkpointing is enabled by default; explicit for clarity
        enable_checkpointing=True,
        num_sanity_val_steps=0,  # skip extra pre-training val
    )

    if args.auto_scale_bs:
        LOG.info(f"Prior to auto-scaling, batch size is {datamodule.batch_size}")
        try:
            new_bs = shared_utils.autoscale_batch_size_mps_safe(smallm_model, datamodule, accelerator, devices, precision)
            if new_bs: 
                datamodule.batch_size = new_bs
                LOG.info(f"Auto-scaled batch size (MPS-safe): {new_bs}")
        except RuntimeError as e:
            LOG.warning(f"Batch-size tuning failed on this device: {e}. Will continue with OOM backoff.")
            # datamodule.batch_size = int(datamodule.batch_size / 4)
            # maybe  
     
    if torch.backends.mps.is_available():# always scale to 0.25 as  mps seems over-keen on over allocation
        datamodule.batch_size = int(datamodule.batch_size / 4)
    if torch.cuda.is_available():
        datamodule.batch_size = int(datamodule.batch_size / 2)
    if datamodule.batch_size == 0: datamodule.batch_size = 1
            
    LOG.info(f"Chose batch size {datamodule.batch_size}")

    # Train with OOM fallback (halve batch size until it fits)
    bs = datamodule.batch_size
    if bs > 0:
        LOG.info(f"Starting training with batch_size={bs}")
        trainer.fit(smallm_model, datamodule=datamodule, ckpt_path=None) # no ckpt by default to avoid it finding the ones created in batch size estimation
    else:
        LOG.info(f"Batch size came out at zero for {hf_repo} to not gonna run it")


    LOG.info("Training complete.")
    try:
        ckpt = callbacks[-1].best_model_path  # from ModelCheckpoint
        LOG.info(f"Best checkpoint: {ckpt if ckpt else 'none'}")
    except Exception:
        pass


# --------------------
# Main
# --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Finetune/evaluate small LMs with Lightning 2.x.")
    p.add_argument("--config", type=str, required=True, help="Path to JSON config with models list.")
    p.add_argument("--model", type=str, required=False, default=None, help="only run one model with this HF model name - needs to be in the JSON file")  
    p.add_argument("--data_dir", type=str, required=True, help="Data directory with training/ and validation/")
    p.add_argument("--out_dir", type=str, default="./runs", help="Output root directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4, help="Starting batch size (may be auto-scaled or reduced on OOM)")    
    p.add_argument("--save_ckpt_every", type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--randomise_weights", type=bool, default=False, help="Set to true to randomsze model weights before training")
    p.add_argument("--progress_bar", type=bool, default=False, help="Show a pbar during validation and training")
    
    # p.add_argument("--auto_scale_bs", action="store_true", help="Use Lightning Tuner to auto-scale batch size")
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--want_ctx_size", type=int, default=2048, help="Max tokens per example")
    # p.add_argument("--context", type=int, default=64, help="Number of previous lines as context. in NJAM, that's number of notes")
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
    train_files = shared_utils.find_text_files(train_dir)
    val_files = shared_utils.find_text_files(val_dir)

    assert train_dir.exists(), f"Training directory not found: {train_dir}"
    assert val_dir.exists(), f"Validation directory not found: {val_dir}"
    assert len(train_files) > 0, "No training files found"
    assert len(val_files) > 0, "No validation files found"

    cfg = shared_utils.load_and_validate_config(config_path)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Seeding (Lightning 2.x)
    seed_everything(args.seed, workers=True)
    cfg["models"] = shared_utils.sort_models_by_size(cfg["models"])
    for m in cfg["models"]:
        print(f"Found model in cfg {m['hf_repo']} with params {m['size_b']}")

    if args.model is not None:
        cfg["models"] = [mdl for mdl in cfg["models"] if mdl["hf_repo"] == args.model]
        assert len(cfg["models"]) > 0, f"Could not find hf model {args.model} in config"
    # Train/evaluate each model

    for m in cfg["models"]:
            
        LOG.info(f"=== Model: {m['hf_repo']} (size {m['size_b']}) ===")
        run_for_model(m, args, out_root)

    LOG.info("All done.")


if __name__ == "__main__":
    main()
