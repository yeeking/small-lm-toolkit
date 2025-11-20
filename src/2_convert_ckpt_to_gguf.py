#!/usr/bin/env python3
"""
Export a single local checkpoint directory to GGUF via llama.cpp's converter and
optionally quantize it.

Example usage:
  python 2_convert_ckpt_to_gguf.py --ckpt /path/to/checkpoint --outdir ./gguf_out --outtype f16
  python 2_convert_ckpt_to_gguf.py --ckpt /path/to/checkpoint --quantize q4_k_m --quantize-bin ./llama.cpp/quantize
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def infer_is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def validate_model_dir(path: Path) -> list[str]:
    """Return a list of missing required files for llama.cpp conversion."""
    missing = []
    if not (path / "config.json").exists():
        missing.append("config.json")
    if not any((path / name).exists() for name in ("tokenizer.model", "tokenizer.json")):
        missing.append("tokenizer.model or tokenizer.json")
    if not any(path.glob("*.bin")) and not any(path.glob("*.safetensors")):
        missing.append("model weights (*.bin or *.safetensors)")
    return missing


def find_converter(converter_arg: str | None) -> Path:
    # Priority: explicit arg, env var, common relative locations
    candidates = []
    if converter_arg:
        candidates.append(Path(converter_arg))
    env = os.getenv("LLAMA_CPP_CONVERTER")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path("./libs/llama.cpp/convert_hf_to_gguf.py"),
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


def main():
    ap = argparse.ArgumentParser(description="Export a local HF checkpoint directory to GGUF.")
    ap.add_argument("--ckpt", required=True,
                    help="Path to checkpoint directory (must contain config.json) or a checkpoint file whose parent contains config.json.")
    ap.add_argument("--outdir", default="./gguf_out", help="Where to write GGUF (and quantized) files.")
    ap.add_argument("--outfile-name", default=None,
                    help="Base filename for GGUF (no extension). Default uses the checkpoint name.")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16"],
                    help="Base tensor type for GGUF export (pre-quant).")
    ap.add_argument("--size-b", dest="size_b", default=None,
                    help="Optional model size tag (e.g. 1, 1.5). Used only for naming.")
    ap.add_argument("--converter", default="./libs/llama.cpp/convert_hf_to_gguf.py",
                    help="Path to llama.cpp/convert_hf_to_gguf.py (overrides env).")
    ap.add_argument("--quantize", default=None,
                    help="Optional quantization type, e.g. q4_k_m, q5_k_m, q8_0. If set, runs quantize after export.")
    ap.add_argument("--quantize-bin", default=None,
                    help="Path to llama.cpp 'quantize' binary. If omitted, tries ./llama.cpp/quantize or PATH.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="If set, skip conversion when output GGUF already exists.")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if ckpt_path.is_file():
        model_dir = ckpt_path.parent
        print(f"[INFO] CKPT path is a file; using parent directory for converter: {model_dir}")
    else:
        model_dir = ckpt_path

    if not infer_is_model_dir(model_dir):
        raise FileNotFoundError(f"Checkpoint directory must contain config.json for llama.cpp converter: {model_dir}")

    missing = validate_model_dir(model_dir)
    if missing:
        raise FileNotFoundError(
            f"Missing required files in checkpoint directory {model_dir}: {', '.join(missing)}\n"
            "If you created the checkpoint from a loaded HF model, you can usually populate these with:\n"
            "  model.tokenizer.save_pretrained(outdir, legacy_format=True)\n"
            "  model.model.save_pretrained(outdir)\n"
            "  getattr(model, 'llama_config', model.config).save_pretrained(outdir)\n"
        )

    converter = find_converter(args.converter)
    print(f"[INFO] Using converter: {converter}")

    base = args.outfile_name or ckpt_path.stem
    if args.size_b:
        gguf_out = outdir / f"{base}.{args.outtype}_{args.size_b}b.gguf"
    else:
        gguf_out = outdir / f"{base}.{args.outtype}.gguf"

    cmd = [
        sys.executable, str(converter),
        "--outfile", str(gguf_out),
        "--outtype", args.outtype,
        str(model_dir),
    ]

    print(f"[INFO] Converting checkpoint to GGUF: {gguf_out.name}")
    if gguf_out.exists() and args.skip_existing:
        print(f"[INFO] GGUF already exists and --skip-existing set; skipping: {gguf_out}")
    else:
        run(cmd)
        if not gguf_out.exists():
            raise FileNotFoundError(f"Expected GGUF not found after conversion: {gguf_out}")
        print(f"[OK] Exported GGUF: {gguf_out}")

    if args.quantize:
        quant_bin = args.quantize_bin
        if quant_bin is None:
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
            args.quantize,
        ]
        print(f"[INFO] Quantizing GGUF: {quant_out.name} ({args.quantize})")
        run(qcmd)
        if not quant_out.exists():
            raise FileNotFoundError(f"Expected quantized GGUF not found: {quant_out}")
        print(f"[OK] Quantized GGUF: {quant_out}")

    print("[DONE]")


if __name__ == "__main__":
    main()
