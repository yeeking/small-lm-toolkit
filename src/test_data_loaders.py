#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from math import log2, ceil, floor

# import your dataset & helpers
from shared_utils import TrainIterableDatasetVarCtx, safe_read_lines

def build_schedule(min_ctx: int, max_ctx: int):
    k_start = ceil(log2(max(1, min_ctx)))
    k_end   = floor(log2(max(1, max_ctx)))
    return [1 << k for k in range(k_start, k_end + 1) if (1 << k) >= min_ctx and (1 << k) <= max_ctx]

def verify_var_contexts(
    files,
    tokenizer,
    block_size: int,
    min_lines_in_context: int,
    max_lines_in_context: int,
    samples: int = 64,
):
    ds = TrainIterableDatasetVarCtx(
        files=files,
        tok_name_or_path=tokenizer.name_or_path,
        tok_kwargs={"use_fast": True, "trust_remote_code": getattr(tokenizer, "init_kwargs", {}).get("trust_remote_code", False)},
        want_ctx_size=block_size,
        min_lines_in_context=min_lines_in_context,
        max_lines_in_context=max_lines_in_context,
        shuffle_files=False,  # keep deterministic to see the cycle
    )

    loader = DataLoader(ds, batch_size=1, num_workers=0)

    expected_schedule = build_schedule(min_lines_in_context, max_lines_in_context)
    seen = []
    trunc_count = 0

    for n, batch in enumerate(loader):
        if n >= samples:
            break
        ids = batch["input_ids"][0]
        mask = batch["attention_mask"][0]
        # keep only unpadded tokens before decoding
        ids = ids[mask == 1].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)

        # number of lines in the window = context_lines + 1 (current line)
        # (blank lines shouldn't exist because you strip() in your loader)
        num_lines = text.count("\n") + 1 if text else 0
        ctx_lines = max(0, num_lines - 1)
        seen.append(ctx_lines)

        # quick truncation check: if decoded text length in tokens is == block_size, we *might* be truncating
        if len(ids) == block_size:
            trunc_count += 1

        print(f"[{n:02d}] ctx_lines={ctx_lines:>4}  "
              f"(in schedule? {'YES' if ctx_lines in expected_schedule else 'no '})  "
              f"{repr(text[:120])}{'...' if len(text) > 120 else ''}")

    print("\n--- Summary ---")
    print(f"Requested schedule: {expected_schedule}")
    print(f"Observed unique ctx sizes: {sorted(set(seen))}")
    missing = [s for s in expected_schedule if s not in seen]
    if missing:
        print(f"WARNING: did not observe these scheduled sizes (within first {samples} samples): {missing}")
    if trunc_count:
        print(f"NOTE: {trunc_count}/{samples} samples hit block_size=={block_size}; "
              "older context may be truncated (left-side) due to tokenizer truncation.")

if __name__ == "__main__":
    # --- CONFIG ---
    data_dir = Path("./data/tiny_dataset/training")
    files = sorted(list(data_dir.rglob("*.txt")))
    assert files, f"No training files under {data_dir}"

    hf_repo = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    # pick conservative block_size while testing
    verify_var_contexts(
        files=files,
        tokenizer=tokenizer,
        block_size=512,
        min_lines_in_context=8,
        max_lines_in_context=128,
        samples=48,
    )
