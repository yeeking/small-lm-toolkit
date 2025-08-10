#!/usr/bin/env python3
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any
# --- add this near LineStream ---
from torch.utils.data import Dataset

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


JSON_PATH = "./data/models_plan.json"
# File extensions considered "text" – add more if you like
TEXT_EXTS = {".txt", ".text", ".md"}


def find_text_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS]


def safe_model_dir_name(repo: str) -> str:
    # e.g. "openai-community/gpt2" -> "openai-community__gpt2"
    return repo.replace("/", "__")


def load_prepare_func(code: str):
    """
    Exec the model-specific prepare function from JSON and return it.
    The code is expected to define `prepare_text_file(path, ...) -> dict[tensor]`.
    """
    # Provide a clean namespace; exec code can import what it needs
    ns: Dict[str, Any] = {}
    exec(code, ns, ns)  # noqa: S102 (we control the JSON)
    if "prepare_text_file" not in ns or not callable(ns["prepare_text_file"]):
        raise RuntimeError("prepare_text_file_snippet_py did not define a callable prepare_text_file")
    return ns["prepare_text_file"]


def to_model_device(batch: Dict[str, Any], model) -> Dict[str, Any]:
    # Try to detect a device from model parameters (works with accelerate too)
    try:
        first_param = next(model.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device("cpu")
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main():
    # --- CLI arg: input folder ---
    if len(sys.argv) < 2:
        print("Usage: python prepare_corpus.py <folder_of_text_files>")
        sys.exit(2)

    input_dir = Path(sys.argv[1]).expanduser().resolve()
    assert input_dir.exists() and input_dir.is_dir(), f"Folder not found or not a directory: {input_dir}"

    # --- Load JSON plan ---
    assert os.path.exists(JSON_PATH), f"Cannot find JSON data file {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "models" in data, f"JSON has no 'models' key; keys: {list(data.keys())}"
    models = data["models"]

    # --- Gather text files ---
    text_files = find_text_files(input_dir)
    assert len(text_files) > 0, f"No text files found under {input_dir} with extensions {sorted(TEXT_EXTS)}"
    print(f"Found {len(text_files)} text files under {input_dir}")

    # --- Process per-model ---
    for entry in models:
        for k in ["family", "size_b", "hf_repo", "prepare_text_file_snippet_py"]:
            assert k in entry, f"Model entry missing key '{k}'; keys present: {list(entry.keys())}"

        family = entry["family"]
        size_b = entry["size_b"]
        repo = entry["hf_repo"]
        trust_remote_code = bool(entry.get("trust_remote_code", False))
        prep_code = entry["prepare_text_file_snippet_py"]

        print(f"\n=== Preparing for model: {family} ({size_b}B) from {repo} ===")

        # Load HF bits
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=trust_remote_code,
            )
            model.eval()
        except Exception as e:
            print(f"[FAIL] Could not load model/tokenizer for {repo}: {e}")
            traceback.print_exc()
            # continue to next model
            continue

        # Build prepare function from JSON code
        try:
            prepare_text_file = load_prepare_func(prep_code)
        except Exception as e:
            print(f"[FAIL] Could not construct prepare_text_file() for {repo}: {e}")
            traceback.print_exc()
            continue

        # Output directory per model
        model_dir_name = safe_model_dir_name(repo.split(":")[-1])  # handle hf:// or similar
        out_dir = Path("processed_data") / model_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Iterate files
        ok_files = 0
        for src_path in text_files:
            rel = src_path.relative_to(input_dir)
            dst_base = rel.with_suffix(".pt").name  # keep base name, switch to .pt
            dst_path = out_dir / dst_base

            # try:
            # Run the model-specific preparation (tokenization etc.)
            enc = prepare_text_file(str(src_path), model_id=repo)

            assert isinstance(enc, dict) or any(torch.is_tensor(v) for v in enc.values()), f"prepare_text_file() did not return a dict of tensors"

            # Save prepared batch
            torch.save(enc, dst_path)

            # Reload and sanity-check with the model
            enc2 = torch.load(dst_path, weights_only=False)
            enc2 = to_model_device(enc2, model)

            with torch.no_grad():
                outputs = model(**enc2)

            # Basic checks
            assert hasattr(outputs, "logits"), "Model output lacks logits"
            bs = next(iter(enc2.values())).shape[0] if enc2 else 1
            assert outputs.logits.shape[0] == bs, "Mismatch in batch dimension of logits"

            ok_files += 1
            print(f"[OK] {src_path.name} -> {dst_path.name} ({outputs.logits.shape})")

            # except Exception as e:
            #     print(f"[FAIL] {src_path}: {e}")
            #     traceback.print_exc()

        print(f"Summary for {repo}: {ok_files}/{len(text_files)} files prepared and verified.")


if __name__ == "__main__":
    main()
