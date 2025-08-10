#!/usr/bin/env python3
import os
import sys
import json
import random
from pathlib import Path
from typing import List, Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# --------------------
# Config knobs
# --------------------
JSON_PATH = "./data/models_plan.json"
TEXT_EXTS = {".txt", ".text", ".md"}
BLOCK_SIZE = 1024           # truncate long samples to this many tokens
BATCH_SIZE = 4              # per-device batch size
GRAD_ACCUM_STEPS = 4        # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LR = 2e-5
WARMUP_STEPS = 50
MAX_STEPS = 200             # keep it small for a quick “does this work?” run
SEED = 42
SHUFFLE_FILES_EACH_EPOCH = True


def find_text_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS]


class LineStream(IterableDataset):
    """
    Streams lines of text from a list of files.
    Keeps memory low: reads one file at a time and yields non-empty lines.
    """
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
                # Skip unreadable files
                continue


def make_collate_fn(tokenizer, block_size: int):
    """
    Tokenize on the fly.
    - Truncates to block_size
    - Pads to the longest in the batch
    - Labels = input_ids (causal LM)
    """
    eos = tokenizer.eos_token

    def collate(batch_texts: List[str]):
        # Add EOS for stability if model expects it
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


def main():
    # ----- CLI: dataset folder -----
    if len(sys.argv) < 2:
        print("Usage: python finetune_streaming.py <dataset_folder>")
        sys.exit(2)
    data_dir = Path(sys.argv[1]).expanduser().resolve()
    assert data_dir.exists() and data_dir.is_dir(), f"Folder not found: {data_dir}"

    # ----- JSON plan -----
    assert os.path.exists(JSON_PATH), f"Missing JSON file {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)
    assert "models" in plan, "JSON has no 'models' key"

    # ----- Files -----
    files = find_text_files(data_dir)
    assert files, f"No text files under {data_dir}"
    print(f"Found {len(files)} text files")

    # ----- Seed -----
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ----- Iterate models -----
    for entry in plan["models"]:
        for k in ["hf_repo"]:
            assert k in entry, f"Model entry missing '{k}'"
        repo = entry["hf_repo"]
        trust_remote_code = bool(entry.get("trust_remote_code", False))

        print(f"\n=== Test finetune: {repo} ===")

        # Load tokenizer/model
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)
        # Some GPT-2 family tokenizers lack pad_token; set it to eos to enable padding
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=trust_remote_code,
        )
        model.train()

        # Optional: gradient checkpointing for memory
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        # Dataset / DataLoader
        dataset = LineStream(files, shuffle_files=SHUFFLE_FILES_EACH_EPOCH)
        collate_fn = make_collate_fn(tokenizer, BLOCK_SIZE)

        # Pin memory speeds up host->device copies on CUDA
        dl = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        # Optimizer / Scheduler
        optim = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
        )

        device = next(model.parameters()).device
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.bfloat16))
        # Note: PyTorch AMP supports bf16 autocast on Ampere+ without GradScaler; we keep scaler enabled flag to False for bf16
        use_autocast = device.type == "cuda"

        step = 0
        running_loss = 0.0
        optim.zero_grad(set_to_none=True)

        data_iter = iter(dl)
        while step < MAX_STEPS:
            try:
                batch = next(data_iter)
            except StopIteration:
                # re-start the iterator (IterableDataset can be endless; we reuse)
                data_iter = iter(dl)
                batch = next(data_iter)

            # Move to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                out = model(**batch)
                loss = out.loss / GRAD_ACCUM_STEPS

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if scaler.is_enabled():
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            running_loss += loss.item() * GRAD_ACCUM_STEPS
            if (step + 1) % 20 == 0:
                avg = running_loss / 20
                print(f"Step {step+1}/{MAX_STEPS} - loss {avg:.4f}")
                running_loss = 0.0

            step += 1

        # Quick validation: generate a tiny sample
        model.eval()
        prompt = "The quick brown fox"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=40, do_sample=True, top_p=0.9)
        print("Sample:", tokenizer.decode(gen_ids[0], skip_special_tokens=True))

        # (Optional) Save a small checkpoint per model
        out_dir = Path("checkpoints") / repo.replace("/", "__")
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"[OK] Finished test finetune for {repo} -> {out_dir}")

if __name__ == "__main__":
    main()
