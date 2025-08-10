#!/usr/bin/env python3
"""
Export a Hugging Face causal LM (repo ID or local folder) to GGUF via llama.cpp converter,
and optionally quantize it.

Usage examples:
  python export_to_gguf.py --model Qwen/Qwen2.5-1.5B --outdir ./out --outtype f16
  python export_to_gguf.py --model ./local_hf_model --outdir ./out --quantize q4_k_m --quantize-bin ./llama.cpp/quantize

Notes:
- You must have llama.cpp's convert_hf_to_gguf.py locally.
  - Set env: LLAMA_CPP_CONVERTER=/path/to/llama.cpp/convert_hf_to_gguf.py
  - Or pass --converter /path/to/convert_hf_to_gguf.py
- Supported families include LLaMA/Llama-2/3, Qwen2/3, GPTNeoX/Pythia, MPT, Phi, Falcon 3, GPT-J, etc.
  (Falcon-H1 hybrid and ERNIE-4.5 are not supported by llama.cpp at time of writing.)
"""

import argparse
import os
import shutil
import subprocess
import sys
import json 
from pathlib import Path

import shared_utils

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except Exception:
    snapshot_download = None  # only needed when downloading a repo

def infer_is_local_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()

def find_converter(converter_arg: str | None) -> Path:
    # Priority: explicit arg, env var, common relative locations
    candidates = []
    if converter_arg:
        candidates.append(Path(converter_arg))
    env = os.getenv("LLAMA_CPP_CONVERTER")
    if env:
        candidates.append(Path(env))
    # common relative guesses
    candidates += [
        Path("./libs/llama.cpp/convert_hf_to_gguf.py"),
        # Path("../llama.cpp/convert_hf_to_gguf.py"),
        # Path("~/llama.cpp/convert_hf_to_gguf.py").expanduser(),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find convert_hf_to_gguf.py. "
        "Pass --converter or set LLAMA_CPP_CONVERTER=/path/to/convert_hf_to_gguf.py"
    )

def run(cmd: list[str], cwd: Path | None = None):
    print(">>", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return proc


def convert_model(hf_repo, model_size_str):
    ap = argparse.ArgumentParser(description="Export HF model to GGUF via llama.cpp")
    # ap.add_argument("--model", required=True,
    #                 help="Hugging Face repo ID (e.g. Qwen/Qwen2.5-1.5B) OR local HF model dir (must contain config.json)")
    ap.add_argument("--revision", default=None, help="Optional HF revision/commit")
    ap.add_argument("--outdir", default="./gguf_out", help="Where to write GGUF (and quant) files")
    ap.add_argument("--outfile-name", default=None, help="Base filename for GGUF (without extension). Default = derived from model name")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16"],
                    help="Base tensor type for GGUF export (pre-quant). Default: f16")
    ap.add_argument("--converter", default="./libs/llama.cpp/convert_hf_to_gguf.py", help="Path to llama.cpp/convert_hf_to_gguf.py (overrides env)")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="If downloading from HF, pass trust_remote_code to some toolchains (not used by converter itself)")
    ap.add_argument("--quantize", default=None,
                    help="Optional quantization type, e.g. q4_k_m, q5_k_m, q8_0. If set, will run quantize after export.")
    ap.add_argument("--quantize-bin", default=None,
                    help="Path to llama.cpp 'quantize' binary. If omitted, tries ./llama.cpp/quantize or in PATH.")
    ap.add_argument("--hf-token", default=None, help="HF token if the repo is gated/private.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve model path: either local dir or download via huggingface_hub
    # model_arg = args.model
    model_arg = hf_repo
    model_dir: Path
    if infer_is_local_model_dir(Path(model_arg)):
        model_dir = Path(model_arg).resolve()
        print(f"[INFO] Using local model dir: {model_dir}")
    else:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")
        print(f"[INFO] Downloading HF repo: {model_arg}")
        model_dir = shared_utils.download_model(model_arg)
        # model_dir = Path(
        #     snapshot_download(
        #         repo_id=model_arg,
        #         revision=args.revision,
        #         allow_patterns=None,  # grab all
        #         token=args.hf_token,
        #         tqdm_class=None,
        #     )
        # )
        print(f"[INFO] Downloaded to: {model_dir}")

    # Find the converter
    converter = find_converter(args.converter)
    print(f"[INFO] Using converter: {converter}")

    # Determine output base name
    if args.outfile_name:
        base = args.outfile_name
    else:
        # Derive from repo-id or folder name
        base = Path(model_arg).name.replace("/", "_").replace(":", "_")

    gguf_out = outdir / f"{base}.{args.outtype}_{model_size_str}b.gguf"

    # Build converter command
    # Basic converter args: --model <hf_dir> --outfile <out.gguf> --outtype f16
    # The converter typically auto-detects architecture from config.json.
    cmd = [
        sys.executable, str(converter),
        "--outfile", str(gguf_out),
        "--outtype", args.outtype,
        str(model_dir),
        
    ]

    # Run conversion
    print(f"[INFO] Converting to GGUF: {gguf_out.name}")
    # try:
    if os.path.exists(gguf_out):
        print(f"[INFO] GGUF already exists... skipping {gguf_out.name}")
        return 
    run(cmd)
    print("[OK]")
    # except:
        # print("[FAILED]")
    # run(cmd)
    assert os.path.exists(gguf_out), f"GGUF file was not rendered for some reason"
    # if not gguf_out.exists():
        # raise FileNotFoundError(f"Expected GGUF not found: {gguf_out}")

    print(f"[OK] Exported GGUF: {gguf_out}")

    # Optional quantization
    if args.quantize:
        # Resolve quantize binary
        quant_bin = args.quantize_bin
        if quant_bin is None:
            # try a couple defaults
            for candidate in [
                "./llama.cpp/quantize",
                "../llama.cpp/quantize",
                "quantize",
            ]:
                p = shutil.which(candidate) if os.path.basename(candidate) == candidate else Path(candidate)
                if (isinstance(p, str) and p) or (isinstance(p, Path) and Path(p).exists()):
                    quant_bin = str(p)
                    break
        if not quant_bin:
            raise FileNotFoundError("quantize binary not found. Pass --quantize-bin /path/to/quantize")

        quant_out = outdir / f"{base}.{args.quantize}.gguf"
        qcmd = [
            quant_bin,
            str(gguf_out),
            str(quant_out),
            args.quantize
        ]
        print(f"[INFO] Quantizing to: {quant_out.name} ({args.quantize})")
        try:
            run(qcmd)
            print("[OK]")
        except:
            print("[FAILED]")

        if not quant_out.exists():
            raise FileNotFoundError(f"Expected quantized GGUF not found: {quant_out}")

        print(f"[OK] Quantized GGUF: {quant_out}")

    print("[DONE]")


def main():
    # Path to your JSON file
    JSON_PATH = "./data/models_plan.json"
    assert os.path.exists(JSON_PATH), f"Cannot find JSON data file {JSON_PATH}"

    # Load the JSON
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "models" in data.keys(), f"JSON data loaded but does not have models key - just {data.keys()}"
    models = data["models"]

    for entry in models:
        for k in ["family", "size_b", "hf_repo"]:
            assert k in entry.keys(), f"Entry missing key {k} has keys  {entry.keys()}"

        family = entry["family"]
        size_b = entry["size_b"]
        repo = entry["hf_repo"]
        trust_remote_code = bool(entry.get("trust_remote_code", False))

        print(f"\n=== Exporting model to GGUF: {family} ({size_b}B) from {repo} ===")
        convert_model(repo, size_b)

if __name__ == "__main__":
    main()
