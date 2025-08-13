#!/usr/bin/env python3
import torch
from pathlib import Path
from transformers import AutoTokenizer
# from shared_utils import SimpleDataModule, get_model_max_len
import shared_utils

# ---- CONFIG ----
# model_name_or_path = "gpt2"  # or your HF repo
data_dir = Path("./data/tiny_dataset")  # adjust to match your dataset
block_size = 512
context = 4
batch_size = 2
num_workers = 0
size_b = 0.07
hf_repo = "EleutherAI/pythia-70m"
tokenizer, model = shared_utils.load_model_no_cache(hf_repo, size_b, False)
assert tokenizer is not None, f"Tokenizer did not load correctly."
assert model is not None, f"Mode did not load correctly."
    
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
tokenizer.truncation_side = "left"
tokenizer.padding_side = "right"

# Clamp block_size to model context length
safe_block_size = min(block_size, shared_utils.get_model_max_len(model, tokenizer, default_cap=block_size))

# ---- INIT DATALOADER ----
dm = shared_utils.SimpleDataModule(
    data_dir=data_dir,
    tokenizer=tokenizer,
    want_ctx_size=safe_block_size,
    context=context,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=False,
    val_sample_count=3,
)

dm.setup(stage="fit")
train_loader = dm.train_dataloader()

# ---- FETCH ONE BATCH ----
batch = next(iter(train_loader))

input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]
labels = batch["labels"]

print("=== Raw batch tensors ===")
print("input_ids shape:", input_ids.shape)
print("attention_mask shape:", attention_mask.shape)
print("labels shape:", labels.shape)

# ---- SHOW FULL CONTEXT -> NEXT LABEL FOR EACH STEP ----
PRINT_ALL_STEPS = True       # set False to limit pairs
MAX_STEPS = 24               # only used if PRINT_ALL_STEPS = False

for i in range(min(batch_size, len(input_ids))):
    ids  = input_ids[i].tolist()
    lbls = labels[i].tolist()
    mask = attention_mask[i].tolist()

    # Keep only non-pad positions
    ids_valid  = [t for t, m in zip(ids,  mask) if m == 1]
    lbls_valid = [t for t, m in zip(lbls, mask) if m == 1]

    # Sanity: labels should match inputs at valid positions
    # (HF shifts internally for loss; we shift only for display)
    assert len(ids_valid) == len(lbls_valid)

    # Token strings (useful to see exact merges)
    toks_valid = tokenizer.convert_ids_to_tokens(ids_valid)

    print(f"\n=== Example {i+1} ===")
    print("Full decoded input window:")
    print(repr(tokenizer.decode(ids_valid, skip_special_tokens=False)))

    steps = range(len(ids_valid) - 1)
    if not PRINT_ALL_STEPS:
        steps = list(steps)[:MAX_STEPS]

    for t in steps:
        # Context up to and including position t
        ctx_ids = ids_valid[:t+1]
        # Target is the *next* token at position t+1
        tgt_id  = ids_valid[t+1]

        ctx_text  = tokenizer.decode(ctx_ids, skip_special_tokens=False)
        tgt_text  = tokenizer.decode([tgt_id], skip_special_tokens=False)
        ctx_last_tok = toks_valid[t]
        tgt_tok      = tokenizer.convert_ids_to_tokens([tgt_id])[0]

        print(f"\n[t={t:02d}]")
        print("Context (decoded):")
        print(repr(ctx_text))
        print(f"Next label token id: {tgt_id}  | token: {repr(tgt_tok)}  | decoded: {repr(tgt_text)}")


# # ---- DECODE EXAMPLES ----
# for i in range(min(batch_size, len(input_ids))):
#     ids = input_ids[i].tolist()
#     lbls = labels[i].tolist()
#     mask = attention_mask[i].tolist()

#     decoded_input = tokenizer.decode(ids, skip_special_tokens=False)
#     decoded_labels = tokenizer.decode([t for t in lbls if t != -100], skip_special_tokens=False)

#     print(f"\n--- Example {i+1} ---")
#     # print("Input IDs:", ids)
#     # print("Attention mask:", mask)
#     print("Decoded input:", repr("','".join(decoded_input)))
#     # print("Labels (filtered):", [t for t in lbls if t != -100])
#     print("Decoded labels:", repr("','".join(decoded_labels[1:])))
