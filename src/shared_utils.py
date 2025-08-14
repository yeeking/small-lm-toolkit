## standard imports

from pathlib import Path
import os 
import json 
from typing import List, Dict, Any, Tuple, Iterable, Optional
# from typing import Iterable, List, Optional, Dict, Any
import random 
import math

## LLM imports
import torch 
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

from llama_cpp import Llama
from huggingface_hub import snapshot_download, hf_hub_download

from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner.tuning import Tuner

from peft import LoraConfig, get_peft_model, TaskType

from functools import lru_cache


TEXT_EXTS = {".txt", ".md", ".text"}



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

def get_model_max_len(model, tokenizer, default_cap: int = 4096) -> int:
    cands = []
    for attr in ("n_positions", "max_position_embeddings", "seq_length", "max_seq_len"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v > 0:
            cands.append(v)
    tm = getattr(tokenizer, "model_max_length", None)
    if isinstance(tm, int) and 0 < tm < 10**9:
        cands.append(tm)
    return min(cands) if cands else default_cap

def suggested_num_workers(cap: int | None = None) -> int:
    try:
        max_allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        max_allowed = os.cpu_count() or 2
    slurm_cpt = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpt:
        try:
            max_allowed = min(max_allowed, int(slurm_cpt))
        except ValueError:
            pass
    n = max(1, max_allowed - 1)
    if cap is not None:
        n = min(n, cap)
    return n



def download_model(hf_repo):
    """download the sent hf model (if not already downloaded) and return its location on the local filesystem"""
    model_dir = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=None,  # grab all
            tqdm_class=None,
        )
    )
    return model_dir 

def get_model_gguf(hf_repo, model_size_str, outtype="f16", outdir="./gguf_out"):

    base = Path(hf_repo).name.replace("/", "_").replace(":", "_")

    gguf_out = os.path.join(outdir, f"{base}.{outtype}_{model_size_str}b.gguf")
    return gguf_out

def load_gguf_model(
    gguf_file,
    *,
    n_ctx=2048,
    n_threads=None,
    n_gpu_layers=0,
    vocab_only=False,
    verbose=False
):
    """
    Load a GGUF model with llama.cpp's Python bindings and return the Llama object.

    Args:
        gguf_file (str | Path): Path to the .gguf model file.
        n_ctx (int): Context length to allocate.
        n_threads (int | None): CPU threads to use. Defaults to os.cpu_count() inside llama.cpp if None.
        n_gpu_layers (int): Number of layers to offload to GPU (0 = CPU only).
        vocab_only (bool): If True, load only vocab/metadata (fast sanity check).
        verbose (bool): If False (default), suppress most llama.cpp log output.

    Returns:
        llama_cpp.Llama: Loaded model handle.
    """
    gguf_file = Path(gguf_file)
    if not gguf_file.exists():
        raise FileNotFoundError(f"GGUF not found: {gguf_file}")

    llm = Llama(
        model_path=str(gguf_file),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        vocab_only=vocab_only,
        logits_all=False,
        embedding=False,
        verbose=verbose
    )
    return llm


def tokenize_detokenize(test_str, llamacpp_model, *, add_bos=False, special=True):
    """
    Tokenize and then detokenize a string using a loaded llama.cpp model.

    Args:
        test_str (str): Input text.
        llamacpp_model (llama_cpp.Llama): Model returned by load_gguf_model.
        add_bos (bool): Whether to prepend BOS during tokenization.
        special (bool): Whether to allow/encode special tokens.

    Returns:
        str: The detokenized string reconstructed from tokens.
    """
    if not isinstance(test_str, str):
        raise TypeError("test_str must be a Python str")

    token_ids = llamacpp_model.tokenize(
        test_str.encode("utf-8"),
        add_bos=add_bos,
        special=special,
    )

    detok_bytes = llamacpp_model.detokenize(token_ids)
    return detok_bytes.decode("utf-8", errors="replace")


def describe_model(llamacpp_model):
    """
    Prints key info about the loaded llama.cpp model:
      - Model architecture
      - Context length
      - Embedding & vocab size
      - Number of layers, heads, parameters
      - Quantization type
      - File type & size
    """
    meta = llamacpp_model.metadata  # dict from GGUF
    print("=== Model Description ===")

    # Architecture and context length
    arch = meta.get("general.architecture", "unknown")
    ctx_len = meta.get("llama.context_length", meta.get("general.context_length", "unknown"))
    print(f"Architecture:      {arch}")
    print(f"Context length:    {ctx_len}")

    # Embedding and vocab
    emb_size = meta.get("llama.embedding_length", "unknown")
    vocab_size = meta.get("tokenizer.ggml.tokens", meta.get("llama.vocab_size", "unknown"))
    print(f"Embedding size:    {emb_size}")
    print(f"Vocab size:        {vocab_size}")

    # Layers and heads
    n_layer = meta.get("llama.block_count", "unknown")
    n_head = meta.get("llama.attention.head_count", "unknown")
    n_head_kv = meta.get("llama.attention.head_count_kv", "unknown")
    print(f"Layers:            {n_layer}")
    print(f"Attention heads:   {n_head} (KV heads: {n_head_kv})")

    # Parameters (total count)
    try:
        param_count = llamacpp_model.n_params()
    except AttributeError:
        param_count = "unknown"
    print(f"Parameter count:   {param_count:,}" if isinstance(param_count, int) else f"Parameter count:   {param_count}")

    # Quantization / file type
    file_type = meta.get("general.file_type", "unknown")
    qnt_type = meta.get("general.quantization_type", file_type)
    print(f"Quantization:      {qnt_type}")

    # File size if path is available
    model_path = getattr(llamacpp_model, "_model_path", None)
    if model_path:
        try:
            import os
            size_gb = os.path.getsize(model_path) / (1024**3)
            print(f"File size:         {size_gb:.2f} GB")
        except OSError:
            pass

    print("========================")

def do_test_infer(llamacpp_model, prompt="Which is better, macos or linux", max_tokens=100):
    """
    Run autoregressive inference using the sent prompt and model.

    Args:
        llamacpp_model (llama_cpp.Llama): Loaded GGUF model.
        prompt (str): Prompt to feed the model.
        max_tokens (int): Maximum new tokens to generate.

    Returns:
        str: The generated text from the model (excluding the original prompt).
    """
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string")

    # Run inference
    output = llamacpp_model(
        prompt,
        max_tokens=max_tokens,
        stop=["</s>", "###"],  # common EOS / stop markers
        echo=False,            # don't repeat the prompt in the output text
        temperature=0.7,       # some creativity, but deterministic enough for tests
    )

    # llama.cpp returns a dict with choices[0].text holding the generated string
    try:
        result_text = output["choices"][0]["text"]
    except (KeyError, IndexError):
        result_text = ""

    return result_text.strip()


# --------------------
# Files & text utils
# --------------------
def find_text_files(root: Path) -> List[Path]:
    # assert type(root) == Path, f"find_text_files expects a Path not a {type(root)}"
    files = []
    for ext in TEXT_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)

def sort_models_by_size(models_json):
    if not isinstance(models_json, list):
        raise ValueError("Expected a list of model configs")
    for item in models_json:
        if "size_b" not in item:
            raise ValueError(f"Missing 'size_b' in entry: {item}")
    return sorted(models_json, key=lambda m: m["size_b"])

def safe_read_lines(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    except Exception as e:
        # LOG.warning(f"Failed to read {path}: {e}")
        return []

def powers_of_two(min_ctx: int, max_ctx: int):
    k0 = math.ceil(math.log2(max(1, min_ctx)))
    k1 = math.floor(math.log2(max(1, max_ctx)))
    return [1 << k for k in range(k0, k1 + 1)]



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
# Utilities
# --------------------
def select_accel_precision_devices():
    if torch.cuda.is_available():
        # LOG.info("CUDA available: using GPU with mixed precision.")
        return "gpu", "16-mixed", 1
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        # LOG.info("MPS available: using Apple MPS with 32-bit precision.")
        return "mps", "32-true", 1
    # LOG.info("Falling back to CPU (32-bit precision).")
    return "cpu", "32-true", 1
import tempfile

def autoscale_batch_size_mps_safe(lit_module, datamodule, accelerator, devices, precision):
    """figures out an ideal batch size 
        works in a temporary directory so ckpts that get generated are automatically deleted afterwards
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tune_trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            default_root_dir=tmpdir,   # <— temp sandbox for .scale_batch_size*.ckpt
            num_sanity_val_steps=0,
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=0,
            enable_checkpointing=False,
            logger=False,
            # optionally: barebones=True  # trims more features; not required
        )
        tuner = Tuner(tune_trainer)
        new_bs = tuner.scale_batch_size(
            lit_module,
            datamodule=datamodule,
            mode="power",
            steps_per_trial=1,
            init_val=max(1, datamodule.batch_size // 2),
            max_trials=6,
            batch_arg_name="batch_size",
        )
    # tmpdir (and the .ckpt inside) is removed here
    return new_bs

# def autoscale_batch_size_mps_safe(lit_module, datamodule, accelerator, devices, precision):
#     tune_trainer = Trainer(
#         accelerator=accelerator,
#         devices=devices,
#         precision=precision,
#         num_sanity_val_steps=0,
#         max_epochs=1,
#         limit_train_batches=2,
#         limit_val_batches=0,
#         enable_checkpointing=False,
#         logger=False,
#     )
#     tuner = Tuner(tune_trainer)
#     new_bs = tuner.scale_batch_size(
#         lit_module, datamodule=datamodule, mode="power", steps_per_trial=1,
#         init_val=max(1, datamodule.batch_size // 2), max_trials=6, batch_arg_name="batch_size",
#     )
#     if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
#         torch.mps.empty_cache()
#     return new_bs

def load_model_no_cache(hf_repo, size_b, trust_remote_code):
    # LOG.info(f"Loading tokenizer: {hf_repo}")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, use_fast=True, trust_remote_code=trust_remote_code)

    # LOG.info(f"Loading model: {hf_repo} (size {size_b})")
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_repo, trust_remote_code=trust_remote_code)
    except Exception as e:
        # LOG.error(f"Failed to load model {hf_repo}: {e}")
        return None, None
    
    return tokenizer, model

def load_model_lora(hf_repo: str, trust_remote_code: bool, args) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    # LOG.info(f"Loading tokenizer: {hf_repo}")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    # Optional quantized loading (off by default to avoid extra deps)
    if args.load_in_8bit or args.load_in_4bit:
        try:
            import bitsandbytes as bnb  # noqa: F401
        except Exception as e:
            raise RuntimeError("bitsandbytes is required for 8/4-bit loading. pip install bitsandbytes") from e
        if args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if args.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model_kwargs["bnb_4bit_quant_type"] = args.bnb_4bit_quant_type
        # Avoid device_map="auto" to keep Lightning in charge unless you know you need it
        # model_kwargs["device_map"] = "auto"

    # LOG.info(f"Loading model: {hf_repo}")
    model = AutoModelForCausalLM.from_pretrained(hf_repo, trust_remote_code=trust_remote_code, **model_kwargs)

    # Disable cache for training + enable gradient checkpointing if available
    try:
        if getattr(model.config, "use_cache", None) is not None:
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            # LOG.info("Enabled gradient checkpointing.")
    except Exception as e:
        pass 
        # LOG.warning(f"Gradient checkpointing toggle failed: {e}")

    return tokenizer, model

def wrap_with_lora(model, args):
    # Flexible target_modules:
    target_modules = args.lora_target_modules
    # PEFT allows "all-linear" shorthand on newer versions; else provide list like:
    # ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","W_pack"]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
        inference_mode=False,
    )
    peft_model = get_peft_model(model, lora_cfg)
    # Log trainable parameter count:
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    # LOG.info(f"LoRA trainable params: {trainable:,} / {total:,} "
    #          f"({100.0*trainable/total:.2f}% of total)")
    return peft_model




class MiddleP2WindowDataset(Dataset):
    """
    Validation dataset builder:
      For each file:
        - read lines
        - n = len(lines)//2 (middle index)
        - create windows starting at n with lengths in powers-of-two from min_p2 to max_p2
          (e.g., 4, 8, 16, ..., 256), but only those that fit within the file
        - join with '\n' and append to dataset

    If tokenizer is provided, returns a dict with input_ids/attention_mask/labels;
    otherwise returns the raw concatenated string.

    Args:
        files: iterable of file paths (str or Path)
        min_p2: smallest power-of-two window length (inclusive), e.g., 4
        max_p2: largest power-of-two window length (inclusive), e.g., 256
        tokenizer: optional HF tokenizer
        block_size: max token length if tokenizing
        pad_to_block: if True, pad to block_size (default True when tokenizing)
    """
    def __init__(
        self,
        files: Iterable[str | Path],
        min_p2: int = 4,
        max_p2: int = 256,
        tokenizer=None,
        block_size: Optional[int] = None,
        pad_to_block: bool = True,
    ):
        super().__init__()
        self.files = [Path(p) for p in files]
        self.min_p2 = int(min_p2)
        self.max_p2 = int(max_p2)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_to_block = pad_to_block if tokenizer is not None else False

        if self.min_p2 < 1 or self.max_p2 < self.min_p2:
            raise ValueError("Invalid min/max powers-of-two range.")

        self._lengths = powers_of_two(self.min_p2, self.max_p2)
        if not self._lengths:
            raise ValueError("No powers-of-two in the requested range.")

        # Precompute samples deterministically
        self.samples: List[str] = []
        for fp in self.files:
            lines = safe_read_lines(fp)
            if not lines:
                continue
            n = len(lines) // 2  # middle index
            for L in self._lengths:
                end = n + L
                if end <= len(lines):
                    text = "\n".join(lines[n:end])
                    self.samples.append(text)
                # if it doesn't fit, skip; we only want windows starting at middle

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize(self, text: str) -> Dict[str, Any]:
        if self.tokenizer is None:
            return {"text": text}

        enc = self.tokenizer(
            text,
            truncation=True if self.block_size is not None else False,
            max_length=self.block_size,
            padding=("max_length" if (self.block_size and self.pad_to_block) else False),
            return_tensors="pt",
        )
        # Flatten to 1D tensors
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Standard causal LM labels = input_ids with pads masked to -100
        labels = input_ids.clone()
        if "attention_mask" in enc:
            labels = labels.masked_fill(attention_mask == 0, -100)

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "labels": labels.long(),
        }

    def __getitem__(self, idx: int):
        text = self.samples[idx]
        if self.tokenizer is None:
            return text
        return self._tokenize(text)

# --------------------
# Dataset
# --------------------
class WindowTextDataset(Dataset):
    def __init__(self, windows: List[str], tokenizer: AutoTokenizer, want_ctx_size: int):
        self.windows = windows
        self.want_ctx_size = want_ctx_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.windows)

    @lru_cache(maxsize=50_000)
    def _cached_encode(self, text: str, want_ctx_size: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        enc = self.tokenizer(text, truncation=True, max_length=want_ctx_size, padding="max_length")
        return tuple(enc["input_ids"]), tuple(enc["attention_mask"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.windows[idx]
        input_ids_tup, attn_mask_tup = self._cached_encode(text, self.want_ctx_size)
        input_ids = torch.tensor(input_ids_tup, dtype=torch.long)
        attention_mask = torch.tensor(attn_mask_tup, dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class TrainIterableDataset(IterableDataset):
    """
    An IterableDataset for training language models on windowed text data.

    Args:
        files (Iterable[str or Path]): List of file paths containing text data.
        tok_name_or_path (str): Name or path of the tokenizer to use.
        tok_kwargs (dict): Additional keyword arguments for tokenizer initialization.
        want_ctx_size (int): Maximum sequence length for tokenization.
        context (int): Number of previous lines to include as context for each window.
        shuffle_files (bool, optional): Whether to shuffle the file order for each epoch. Defaults to True.

    Yields:
        dict: A dictionary containing:
            - "input_ids" (torch.LongTensor): Token IDs for the input sequence.
            - "attention_mask" (torch.LongTensor): Attention mask for the input sequence.
            - "labels" (torch.LongTensor): Token IDs used as labels for training.

    Notes:
        - Each yielded sample consists of `context` previous lines and the current line, joined together.
        - Tokenization is performed with truncation and padding to `want_ctx_size`.
        - Supports multi-worker data loading via PyTorch DataLoader.
    """
    """"""
    def __init__(self, files, tok_name_or_path: str, tok_kwargs: dict,
                 want_ctx_size: int, context: int, shuffle_files: bool = True):
        super().__init__()
        self.files = list(files)
        self.tok_name_or_path = tok_name_or_path
        self.tok_kwargs = tok_kwargs
        self.want_ctx_size = want_ctx_size
        self.context = context
        self.shuffle_files = shuffle_files
        self._tok = None

    def _tokenizer(self):
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.tok_name_or_path, **self.tok_kwargs)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token or self._tok.cls_token
        return self._tok

    def _yield_windows_from_file(self, path: Path):
        lines = safe_read_lines(path)
        for i in range(self.context, len(lines)):
            yield "\n".join(lines[i-self.context:i] + [lines[i]])

    def __iter__(self):
        rng = random.Random(torch.initial_seed() % (2**32))
        worker = get_worker_info()
        file_iter = self.files if worker is None else self.files[worker.id :: worker.num_workers]
        file_iter = list(file_iter)
        if self.shuffle_files:
            rng.shuffle(file_iter)
        tok = self._tokenizer()
        for p in file_iter:
            for text in self._yield_windows_from_file(p):
                enc = tok(text, truncation=True, max_length=self.want_ctx_size, padding="max_length")
                yield {
                    "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(enc["input_ids"], dtype=torch.long),
                }




def make_lm_collate(tokenizer, want_ctx_size: int, pad_to_max: bool = False):
    """
    Batch-tokenise a list of strings. Builds labels = input_ids with pads masked to -100.
    pad_to_max=False => pad to longest in batch (faster, less padding).
    """
    def collate(batch_texts):
        enc = tokenizer(
            batch_texts,
            padding=("max_length" if pad_to_max else "longest"),
            truncation=True,
            max_length=want_ctx_size,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attn == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    return collate


class TrainIterableDatasetVarCtx(IterableDataset):
    """
    Cycles through powers-of-two context sizes (in lines/events) between
    min_lines_in_context and max_lines_in_context (inclusive), e.g. [16,32,64,...].

    Yields raw '\n'-joined strings. Tokenisation is handled by the DataLoader's collate_fn.
    """
    def __init__(
        self,
        files: Iterable[str | Path],
        tok_name_or_path: str,   # kept so you can still seed/tokeniser-config in workers if needed
        tok_kwargs: dict,
        want_ctx_size: int,      # token max length (used by collate_fn)
        max_lines_in_context: int,
        min_lines_in_context: int = 16,
        shuffle_files: bool = True,
    ):
        super().__init__()
        self.files = list(files)
        self.tok_name_or_path = tok_name_or_path
        self.tok_kwargs = tok_kwargs
        self.want_ctx_size = int(want_ctx_size)
        self.max_lines_in_context = int(max_lines_in_context)
        self.min_lines_in_context = int(min_lines_in_context)
        self.shuffle_files = shuffle_files

        self._context_schedule = self._build_context_schedule(
            self.min_lines_in_context, self.max_lines_in_context
        )
        if not self._context_schedule:
            raise ValueError(
                f"No powers-of-two between min={self.min_lines_in_context} and max={self.max_lines_in_context}."
            )

    def _build_context_schedule(self, min_ctx: int, max_ctx: int) -> List[int]:
        if min_ctx > max_ctx:
            min_ctx, max_ctx = max_ctx, min_ctx
        k0 = math.ceil(math.log2(max(1, min_ctx)))
        k1 = math.floor(math.log2(max(1, max_ctx)))
        return [1 << k for k in range(k0, k1 + 1) if (1 << k) >= min_ctx and (1 << k) <= max_ctx]

    @staticmethod
    def _floor_pow2(n: int) -> int:
        return (1 << (n.bit_length() - 1)) if n >= 1 else 0

    def __iter__(self):
        rng = random.Random(torch.initial_seed() % (2**32))
        worker = get_worker_info()

        file_iter = self.files if worker is None else self.files[worker.id :: worker.num_workers]
        file_iter = list(file_iter)
        if self.shuffle_files:
            rng.shuffle(file_iter)

        schedule = self._context_schedule
        sched_idx = rng.randrange(len(schedule))

        for p in file_iter:
            lines = safe_read_lines(Path(p))
            if not lines:
                continue

            n_lines = len(lines)
            i = self.min_lines_in_context
            while i < n_lines:
                want_ctx = schedule[sched_idx]
                sched_idx = (sched_idx + 1) % len(schedule)

                ctx_cap = self._floor_pow2(i)
                if ctx_cap == 0:
                    i += 1
                    continue
                ctx = want_ctx if want_ctx <= ctx_cap else ctx_cap

                text = "\n".join(lines[i - ctx : i] + [lines[i]])
                yield text
                i += 1


# --------------------
# DataModule
# --------------------
class SimpleDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for language modeling tasks using text files.

    This module handles loading, preprocessing, and batching of training and validation data.
    It supports windowed text datasets for validation and iterable datasets for training, with
    tokenization and context windowing.

    Args:
        data_dir (Path): Root directory containing 'training' and 'validation' subdirectories with text files.
        tokenizer (AutoTokenizer): Tokenizer instance for encoding text.
        want_ctx_size (int): Maximum sequence length for model inputs.
        context (int): Number of lines to use as context for windowed validation samples.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): Whether to use pinned memory in data loaders.
        val_sample_count (int, optional): Number of validation samples to preview. Defaults to 3.

    Attributes:
        _train_files (list[Path] | None): List of training file paths.
        _train_ds: Training dataset instance.
        _val_ds (WindowTextDataset | None): Validation dataset instance.
        val_preview_texts (list[str]): List of preview validation texts.

    Methods:
        setup(stage: str | None): Prepares datasets for training or validation.
        train_dataloader(): Returns a DataLoader for training.
        val_dataloader(): Returns a DataLoader for validation.

    Raises:
        AssertionError: If required directories or files are missing, or if validation files contain no usable text.
    """
    """ what does data module do?> """
    def __init__(
        self,
        data_dir: Path,
        tokenizer: AutoTokenizer,
        want_ctx_size: int,
        # context: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        val_sample_count: int = 3,
        max_lines_in_context: int = 64,   # max lines of context to feed, where in njam, one line is one musical event
        min_lines_in_context: int = 8,
     
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.want_ctx_size = want_ctx_size
        # self.context = context
        self.max_lines_in_context=max_lines_in_context
        self.min_lines_in_context=min_lines_in_context
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_sample_count = val_sample_count

        self._train_files: list[Path] | None = None
        self._train_ds = None
        self._val_ds: MiddleP2WindowDataset | None = None
        self.val_preview_texts: list[str] = []

        # now sanity check the requested context max and min
        test_folder   = self.data_dir / "validation"
        files = find_text_files(test_folder)
        ctx_stats = SimpleDataModule.get_context_stats_for_file(files[0], self.tokenizer, self.want_ctx_size)
        # print(f"{ctx_stats}")
        def floor_power_of_two(n: int) -> int:
            if n < 1:
                raise ValueError("n must be >= 1")
            return 1 << (n.bit_length() - 1)
        # ensure that the max lines of context is not greater than 
        # the 'possible lines of context given typical token length of a line' 
        if ctx_stats["lines_can_fit_safely_in_ctx"] < self.max_lines_in_context:
            self.max_lines_in_context = floor_power_of_two(ctx_stats["lines_can_fit_safely_in_ctx"])
        if self.min_lines_in_context > self.max_lines_in_context:
            # go down to the next power of two below self.max_lines_in_context
            self.min_lines_in_context = floor_power_of_two(self.max_lines_in_context - 1)


    def setup(self, stage: str | None = None):
        train_root = self.data_dir / "training"
        val_root   = self.data_dir / "validation"
        assert train_root.exists(), f"Training directory not found: {train_root}"
        assert val_root.exists(),   f"Validation directory not found: {val_root}"

        # def powers_of_two(min_ctx: int, max_ctx: int):
        #     k0 = math.ceil(math.log2(max(1, min_ctx)))
        #     k1 = math.floor(math.log2(max(1, max_ctx)))
        #     return [1 << k for k in range(k0, k1 + 1)]
        

        ## some notes on the validation dataset
        ## I wanted to have different sizes of 'context' input
        ## in the validation set and I want to do that over a few different files
        if stage in (None, "validate"):
            if self._val_ds is None:
                val_files = find_text_files(val_root)
                assert len(val_files) > 0, "No validation files found"

                # Use the middlep2window dataset which is designed for validation datasets
                self._val_ds = MiddleP2WindowDataset(val_files, block_size=self.want_ctx_size, tokenizer=self.tokenizer,  min_p2=self.min_lines_in_context, 
                                                     max_p2=self.max_lines_in_context)


        if stage in (None, "fit"):
            if self._train_files is None:
                train_files = find_text_files(train_root)
                assert len(train_files) > 0, "No training files found"
                self._train_files = train_files
                # LOG.info(f"Train files: {len(self._train_files)}")
            if self._val_ds is None:
                self.setup(stage="validate")


    def train_dataloader(self):
        if self._train_files is None:
            self.setup(stage="fit")

        ds = TrainIterableDatasetVarCtx(
            files=self._train_files,
            tok_name_or_path=self.tokenizer.name_or_path,
            tok_kwargs={
                "use_fast": True,
                "trust_remote_code": getattr(self.tokenizer, "init_kwargs", {}).get("trust_remote_code", False),
            },
            want_ctx_size=self.want_ctx_size,          # or whatever you pass in today
            max_lines_in_context=self.max_lines_in_context,
            min_lines_in_context=self.min_lines_in_context,
            shuffle_files=True,
        )

        # Ensure tokenizer padding/truncation sides are set once
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.cls_token
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=make_lm_collate(self.tokenizer, want_ctx_size=self.want_ctx_size, pad_to_max=False),
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
    
    @staticmethod    
    def get_context_stats_for_file(file_path, tokenizer, model_ctx_len):
        """
        Compute token statistics per line ("event") in a text file.

        Args:
            file_path (str or Path): Path to the text file.
            tokenizer: Hugging Face tokenizer instance.
            model (optional): Hugging Face model instance (used only to print max input length).

        Returns:
            dict: {
                "mean": float,
                "min": int,
                "max": int,
                "total_lines": int,
                "model_max_length": int or None
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"No such file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        token_counts = [len(tokenizer.encode(line, add_special_tokens=False)) for line in lines]

        stats = {
            "mean_tokens_per_line": sum(token_counts) / len(token_counts) if token_counts else 0,
            "min_tokens_per_line": min(token_counts) if token_counts else 0,
            "max_tokens_per_line": max(token_counts) if token_counts else 0,
            "total_lines": len(token_counts),
            "model_ctx_length": model_ctx_len, 
            "lines_can_fit_safely_in_ctx": int(model_ctx_len / max(token_counts)) 
        }

        return stats

