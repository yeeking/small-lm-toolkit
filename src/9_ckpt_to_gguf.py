"""
Load a Lightning checkpoint, restore the HF model/tokenizer, save to a HF-style
directory, and optionally export to GGUF via llama.cpp's converter.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from shared_utils import CausalLMModule


def find_converter(converter_arg: str | None) -> Path:
    """Locate llama.cpp's convert_hf_to_gguf.py."""
    candidates = []
    if converter_arg:
        candidates.append(Path(converter_arg))
    env = os.getenv("LLAMA_CPP_CONVERTER")
    if env:
        candidates.append(Path(env))
    candidates.append(Path("./libs/llama.cpp/convert_hf_to_gguf.py"))

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find convert_hf_to_gguf.py. "
        "Pass --gguf-converter or set LLAMA_CPP_CONVERTER=/path/to/convert_hf_to_gguf.py"
    )


def run(cmd: list[str], cwd: Path | None = None):
    print(">>", " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, check=False, capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc


def runMain():
    ap = argparse.ArgumentParser(description="Restore a checkpoint, save HF format, optionally export GGUF.")
    ap.add_argument("--ckpt", required=True, help="Path to the Lightning checkpoint (.ckpt).")
    ap.add_argument(
        "--hf-repo",
        required=True,
        dest="hf_repo",
        help="Base HF repo/architecture the checkpoint was trained from (e.g. EleutherAI/pythia-70m).",
    )
    ap.add_argument(
        "--hf-outdir",
        default=None,
        help="Where to save the HF-formatted model/tokenizer. Defaults to ./models/exported/<ckpt_stem>.",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not hit the network when loading the base HF model/tokenizer.",
    )
    ap.add_argument(
        "--no-gguf",
        action="store_true",
        help="Skip GGUF export (just save HF format).",
    )
    ap.add_argument("--gguf-outdir", default="./gguf_out", help="Directory to write GGUF (and quant) files.")
    ap.add_argument(
        "--gguf-outtype",
        default="f16",
        choices=["f32", "f16", "bf16"],
        help="Base precision for GGUF export before optional quantization.",
    )
    ap.add_argument(
        "--gguf-converter",
        default="./libs/llama.cpp/convert_hf_to_gguf.py",
        help="Path to llama.cpp's convert_hf_to_gguf.py (or set LLAMA_CPP_CONVERTER).",
    )
    ap.add_argument(
        "--gguf-name",
        default=None,
        help="Base filename (no extension) for GGUF. Default uses HF output directory name.",
    )
    ap.add_argument(
        "--gguf-quantize",
        default=None,
        help="Optional quantization type (e.g. q4_k_m, q5_k_m, q8_0). If set, runs quantize after export.",
    )
    ap.add_argument(
        "--gguf-quantize-bin",
        default=None,
        help="Path to llama.cpp 'quantize' binary. If omitted, tries ./llama.cpp/quantize, ../llama.cpp/quantize, or PATH.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip GGUF export if the target GGUF already exists.",
    )
    ap.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save HF weights as safetensors instead of pytorch_model.bin when supported.",
    )

    args = ap.parse_args()

    # GGUF export
    converter = find_converter(args.gguf_converter)


    ckpt_path = Path(args.ckpt).expanduser().resolve()
    hf_outdir = (
        Path(args.hf_outdir).expanduser().resolve()
        if args.hf_outdir
        else Path("models/exported") / ckpt_path.stem
    )
    hf_outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    lit_module = CausalLMModule.load_from_checkpoint_auto(
        ckpt_path=ckpt_path,
        force_hf_repo=args.hf_repo,
        local_files_only=args.local_files_only,
    )
    model = lit_module.model
    tokenizer = lit_module.tokenizer

    print(f"[INFO] Saving HF model to {hf_outdir}")
    model.save_pretrained(hf_outdir, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(hf_outdir)
    # Ensure config is present (model.save_pretrained usually writes this)
    if hasattr(model, "config"):
        model.config.save_pretrained(hf_outdir)

    if args.no_gguf:
        print("[INFO] Skipping GGUF export (--no-gguf set). Done.")
        return

    base_name = args.gguf_name or hf_outdir.name
    gguf_outdir = Path(args.gguf_outdir).expanduser().resolve()
    gguf_outdir.mkdir(parents=True, exist_ok=True)
    gguf_out = gguf_outdir / f"{base_name}.{args.gguf_outtype}.gguf"

    print(f"[INFO] Exporting to GGUF: {gguf_out}")
    if gguf_out.exists() and args.skip_existing:
        print(f"[INFO] GGUF already exists and --skip-existing set; skipping: {gguf_out}")
    else:
        cmd = [
            sys.executable,
            str(converter),
            "--outfile",
            str(gguf_out),
            "--outtype",
            args.gguf_outtype,
            str(hf_outdir),
        ]
        run(cmd)
        if not gguf_out.exists():
            raise FileNotFoundError(f"Expected GGUF not found after export: {gguf_out}")
        print(f"[OK] Exported GGUF: {gguf_out}")

    if args.gguf_quantize:
        quant_bin = args.gguf_quantize_bin
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
            raise FileNotFoundError("quantize binary not found. Pass --gguf-quantize-bin /path/to/quantize")

        quant_out = gguf_outdir / f"{base_name}.{args.gguf_quantize}.gguf"
        qcmd = [quant_bin, str(gguf_out), str(quant_out), args.gguf_quantize]
        print(f"[INFO] Quantizing GGUF: {quant_out.name} ({args.gguf_quantize})")
        run(qcmd)
        if not quant_out.exists():
            raise FileNotFoundError(f"Expected quantized GGUF not found: {quant_out}")
        print(f"[OK] Quantized GGUF: {quant_out}")

    print("[DONE]")


if __name__ == "__main__":
    runMain()
