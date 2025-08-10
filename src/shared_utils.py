from pathlib import Path
import os 
from llama_cpp import Llama


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



