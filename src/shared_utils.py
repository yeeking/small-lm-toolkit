## standard imports

from pathlib import Path
import os 
import json 
from typing import List, Dict, Any, Tuple, Iterable, Optional, Callable
# from typing import Iterable, List, Optional, Dict, Any
import random 
import math
import os, shutil
from pathlib import Path

## LLM imports
import torch 
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

from torch.optim import AdamW

from llama_cpp import Llama
from huggingface_hub import snapshot_download, hf_hub_download

from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # get_linear_schedule_with_warmup,
    pipeline
)

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.loggers import TensorBoardLogger


from peft import LoraConfig, get_peft_model, TaskType

from functools import lru_cache
import torchaudio
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import pretty_midi  # if you choose MIDI rendering; requires fluidsynth/soundfont installed
# import pypianoroll
import librosa
from dataclasses import dataclass

from neural_jammer_lib import NeuralJammerLanguageV2
from MIDI import score2midi as score_to_midi

TEXT_EXTS = {".txt", ".md", ".text"}





def up_to_last(s: str, sub: str) -> str:
    """returns sub string of s up to last occrence of sub """
    idx = s.rfind(sub)
    if idx == -1:
        return s 
    return s[:idx]


def get_model_folder(hf_repo, size_b):
    """returns a folder name based on the repo and size used to save logs, retrieve saved models etc."""
    safe_name = hf_repo.replace("/", "__")
    fname = f"{safe_name}/{size_b}/"
    return fname
           

def njam_to_midi(njam_text:str, outfile:str):
    """convert the sent njam string into midi events and save to a file"""
    print(f"Writing a midi file to {outfile}")
    score = NeuralJammerLanguageV2.njam_to_score(njam_text.split('\n'))
    midi_raw = score_to_midi(score)
    with open(outfile, 'wb') as f:
        f.write(midi_raw)


def njam_to_midi_test(njam_text:str, outfile:str):
    """test version of njam to midi that generates a simple midi file"""
    # Create a PrettyMIDI object
    cello_c_chord = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    cello = pretty_midi.Instrument(program=cello_program)
    # Iterate over note names, which will be converted to note number later
    for note_name in ['C5', 'E5', 'G5']:
        # Retrieve the MIDI note number for this note name
        note_number = pretty_midi.note_name_to_number(note_name)
        # Create a Note instance, starting at 0s and ending at .5s
        note = pretty_midi.Note(
            velocity=100, pitch=note_number, start=0, end=.5)
        # Add it to our cello instrument
        cello.notes.append(note)
    # Add the cello instrument to the PrettyMIDI object
    cello_c_chord.instruments.append(cello)
    # Write out the MIDI data
    cello_c_chord.write(outfile)

def midi_to_pretty_midi(midifile:str):
    """ load sent midi file and convert to pretty_midi object """
    assert os.path.exists(midifile), f"Cannot find midi file {midifile}"
    pm = pretty_midi.PrettyMIDI(midifile)
    return pm 

def pretty_midi_to_fig(pm:pretty_midi.PrettyMIDI):
    """convert sent pretty midi object into a pyplot fig"""
    # your code to plot the piano roll, as we discussed
    fig = plt.figure(figsize=(8,3))
    # mt = pypianoroll.from_pretty_midi(pm)
    # pypianoroll.plot(mt, preset="full")
    fig.tight_layout()
    fs = 100 # 1/fs steps per second
    start_pitch = 24
    end_pitch = 96
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

    return fig

def pretty_midi_to_audio(pm:pretty_midi.PrettyMIDI, sr:int=16000):
    """render sent pretty_midi object into wonderful music and return raw audio samples"""
    print(f"pretty_midi_to_audio rendering a midi file with {len(pm.get_onsets())} notes ")
    audio = pm.fluidsynth(fs=sr)  # numpy [T]
    wave = torch.from_numpy(audio).unsqueeze(0)  # [1, T], float
    return wave, sr


def extract_prompts_from_batch(
    batch,
    tokenizer,
    k: int = 4,
    mode: str = "auto",                      # "auto" | "labels" | "markers" | "full"
    assistant_markers: Optional[List[str]] = None,  # e.g. ["<|assistant|>", "### Assistant:"]
) -> List[str]:
    """
    Return up to k prompt strings from a LM batch.
    - Always trims trailing pad via attention_mask/pad_token_id.
    - If mode == "labels" (or auto-detected), uses labels==-100 (prompt masking at START) to cut before target.
    - If mode == "markers", splits by the earliest occurrence of any marker string.
    - Else returns the full trimmed input.
    """
    input_ids = batch["input_ids"]
    attn = batch.get("attention_mask")
    labels = batch.get("labels")
    B = input_ids.size(0)

    texts = []
    for b in range(min(B, k)):
        ids = input_ids[b]

        # 1) Trim to real tokens
        if attn is not None:
            seq_len = int(attn[b].sum().item())
        else:
            pad_id = tokenizer.pad_token_id
            if pad_id is not None:
                nonpad = (ids != pad_id).nonzero(as_tuple=True)[0]
                seq_len = int(nonpad[-1].item()) + 1 if len(nonpad) > 0 else 0
            else:
                seq_len = ids.size(0)
        ids = ids[:seq_len]

        # 2) Try to get "prompt-only"
        used_prompt_cut = False
        if (mode in ("auto", "labels")) and (labels is not None):
            lab = labels[b][:seq_len]
            # Convention A (prompt masked): labels start with -100 and become valid at first target token
            if lab.numel() > 0 and lab[0].item() == -100 and (lab != -100).any():
                tgt_start = int((lab != -100).nonzero(as_tuple=True)[0][0].item())
                ids = ids[:tgt_start]
                used_prompt_cut = True

        if not used_prompt_cut and (mode in ("auto", "markers") and assistant_markers):
            # Decode with specials preserved so markers can be found in raw text
            raw = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
            cut = _find_first_marker(raw, assistant_markers)
            if cut is not None:
                # Re-tokenize only the prompt part to stay consistent
                prompt_ids = tokenizer(raw[:cut], add_special_tokens=False)["input_ids"]
                ids = ids.new_tensor(prompt_ids)

        texts.append(tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip())
    return texts

def _find_first_marker(text: str, markers: List[str]) -> Optional[int]:
    idxs = [text.find(m) for m in markers]
    idxs = [i for i in idxs if i != -1]
    return min(idxs) if idxs else None


class PreviewAudioCallback(Callback):
    """Training callback that passes a set of prompts to the model, autoregresses a few steps then converts output to MIDI and audio
    which is saved to the  previews folder for this run
    """
    def __init__(
        self,
        prompt_files,
        max_prompt_len:int = 1024,  
        every_n_steps:int = 0, 
        every_n_epochs: int = 1,
        max_secs: float = 8.0,
        pl_module= None,
        max_new_tokens: int = 128,
        do_sample: bool = True,           # set True for stochastic decoding
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        # num_beams: int = 1,
        repetition_penalty: float = 1.1,   # >1.0 discourages repeats
        # truncate_to: Optional[int] = 1024, # max prompt length tokens (None = no extra truncation)
        # return_full_text: bool = False,    # if False, only return the continuation (common for previews)
        audio_sr: int = 44100,
        max_audio_secs: float = 8.0       # keep TB event files lean

    ):
        super().__init__()

        
        prompts = []
        for fname in prompt_files:
            assert os.path.exists(fname), f"Trying to setup training output previews but {fname} does not exist"
            print(f"PreviewAudioCallback loading file {fname} for len: {max_prompt_len}")
            with open(fname) as f:
                prompt = f.read()[0:max_prompt_len] # trim prompt to max length
                prompt = up_to_last(prompt, '\n') # make sure prompt ends after a complete njam message
                prompts.append(prompt)
        
        self.prompts = prompts
        self.every_n_epochs = every_n_epochs
        self.max_secs = max_secs
        self.every_n_steps = every_n_steps

        ### stuff needed to do the renders

        self.pl_module = pl_module
        self.model = pl_module.model
        self.tokenizer = pl_module.tokenizer
        self.device = pl_module.device
        self.max_audio_secs = max_audio_secs
        self.audio_sr = audio_sr
        self.max_new_tokens = max_new_tokens
        self.max_prompt_len = max_prompt_len
        # # Reasonable fallbacks for PAD/EOS to avoid generate() complaints.
        # # (Some decoder-only models don't have pad_token_id set.)
        # if self.tokenizer.pad_token_id is None:
        #     # safest fallback is EOS as PAD for decoder-only LMs
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.eos_id = self.tokenizer.eos_token_id
        # self.pad_id = self.tokenizer.pad_token_id

        # if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generator_pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=do_sample,           # set False for greedy / deterministic
            temperature=temperature,
            top_p=top_p,
            top_k=top_k, 
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )



    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called after every training batch """
        # if "train" not in self.run_on: return
        #print(f"Preview gen callback triggered {trainer.global_step}")
        if self.every_n_steps <= 0: return
        if trainer.global_step % self.every_n_steps != 0: return
        print(f"Preview gen callback rendering... {trainer.global_step}")
        # print(self.prompts)
        self._log_preview(trainer, pl_module, tag_prefix="train", prompts=self.prompts)


    @torch.inference_mode()
    def render_previews(self, include_prompt_in_midi_render=True):
        """
        feeds each of self,prompts[0:self.max_prompt_len] to the model and autoregresses for up to self.max_new_tokens tokens
        returns array of these: 
        {"audio":wave.cpu(), "samplerate":sr, "prompt": prompt, "gen_text": gen_text, "midi_file":midipath, "pretty_midi_obj":pm}

        """
        prompts = self.prompts
        # print(f"HFPreviewResponder __call... prompt lens {[len(p) for p in prompts]}")
        assert len(prompts) == 1, f"You sent more than one prompt but currently only support 1"
        if not prompts:
            return []

        result = self.generator_pipeline(prompts[0], return_full_text=False)
        outs = [r["generated_text"] for r in result]

        previews = []
        for prompt, gen_text in zip(prompts, outs):
            print(f"Length of prompt {len(prompt)} len of output {len(gen_text)}")
            # print(f"here's the output... \n{gen_text}")
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                # print(f"HFPreviewResponder:__call__ about to render... prompt: {prompt} \n\n result: {gen_text}")
                midipath = tmp.name
                try:
                    if include_prompt_in_midi_render: njam_to_midi(prompt + "\n" + gen_text, midipath)
                    else: njam_to_midi(gen_text, midipath)
                    pm = midi_to_pretty_midi(midipath)
                except:
                    pm = None
                    midipath = None
                # render audio 
                #wave, sr = pretty_midi_to_audio(pm, sr=self.audio_sr)
                # crop to keep TB small
                #max_len = int(self.max_audio_secs * self.audio_sr)
                #wave = wave[..., :max_len]
                #  in case fluidsynth no worky
                wave = [0] 
                sr = 44100

#                preview = {"audio":wave.cpu(), "samplerate":sr, "prompt": prompt, "gen_text": gen_text, "midi_file":midipath, "pretty_midi_obj":pm}
                preview = {"audio":wave, "samplerate":sr, "prompt": prompt, "gen_text": gen_text, "midi_file":midipath, "pretty_midi_obj":pm}
 
                previews.append(preview)
                # previews.append((wave.cpu(), sr, {"prompt": prompt, "gen": gen_text, "midi_file":midipath}, pm))
        return previews

    def _log_preview(self, trainer:Trainer, pl_module, tag_prefix: str, prompts:list):
        """generate a preview into tensorboard's log and save the generated midi file out"""
        # logger = getattr(trainer, "logger", None)
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        writer = logger.experiment
        run_dir = getattr(logger, "log_dir", None) or getattr(writer, "log_dir", None)
        global_step = trainer.global_step
        preview_out_dir = os.path.join('./', run_dir, "previews")
        preview_out_dir = Path(preview_out_dir)
        preview_out_dir.mkdir(parents=True, exist_ok=True)

        pl_module.eval()

        with torch.no_grad():
            # previews = self.render_fn(prompts)
            previews = self.render_previews()

        # make some assertions about what is in the previews
        assert len(previews) > 0, f"Previews from render fn look bad..."
        want_keys = ["audio", "samplerate", "prompt", "gen_text", "midi_file", "pretty_midi_obj"]
        for p in previews:
            for k in want_keys: assert k in p.keys(), f"Preview missing key {k}: {p.keys()}"
                
        global_step = trainer.global_step
        # for i, (wave, sr, txt, maybe_pm) in enumerate(previews):
        for i, p in enumerate(previews):
            # log audio & text as before...
            #writer.add_audio(f"preview/{i+1}/audio", p["audio"], global_step=global_step, sample_rate=p["samplerate"])
            writer.add_text(f"preview/{i+1}/text", p['gen_text'], global_step=global_step)
            # if maybe_pm is not None:
            out_txt_file = os.path.join(preview_out_dir, f"preview_step_{global_step}_{i}.txt")
            with open(out_txt_file, 'w') as f:
                f.write(f"{p['prompt']} \n\n {p['gen_text']}") 
                
            if p['pretty_midi_obj'] is not None: # midi conversion crashes sometimes
                fig = pretty_midi_to_fig(p["pretty_midi_obj"])
                writer.add_figure(f"preview/{i+1}/pianoroll", fig, global_step=global_step)
                plt.close(fig)
                source_midi_file = p['midi_file']
                # , f"step_{global_step:12d}"
                out_midi_file = os.path.join(preview_out_dir, f"preview_step_{global_step}_{i}.mid")
                shutil.copy2(source_midi_file, out_midi_file)
                # we could save 'wave' to an audio file with 
                # librosa here too ... 


def load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """Check model config file which should contain multiple model descriptions in json format """
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
    # print(f"get_model_max_len got candidates {cands}")
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
def find_files(root: Path, ext) -> List[Path]:
    # assert type(root) == Path, f"find_text_files expects a Path not a {type(root)}"
    files = []
    files.extend(root.rglob(f"*{ext}"))
    return sorted(files)

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


def load_model_no_cache(hf_repo, size_b, trust_remote_code):
    # LOG.info(f"Loading tokenizer: {hf_repo}")

    # LOG.info(f"Loading model: {hf_repo} (size {size_b})")
    # first try loading from cache
    model_dir = os.path.join('models', 'saved', get_model_folder(hf_repo, size_b))

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        print(f"load_model_no_cache: loaded model from filesystem yay")

        return tokenizer, model 
    except Exception as e:
        # LOG.error(f"Failed to load model fr{hf_repo}: {e}")
        print(f"load_model_no_cache: can't load from an offline save. going via HF tried {model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_repo, use_fast=True, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(hf_repo, trust_remote_code=trust_remote_code)
        print(f"load_model_no_cache: loaded model from HF")
        return tokenizer, model

    except Exception as e:
        # LOG.error(f"Failed to load model fr{hf_repo}: {e}")
        return None, None
    
    return tokenizer, model

def reinit_model_weights(model):
    """
    Reinitialize all model parameters to random values using the model's initialization scheme.
    """
    def _init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Reinitialize linear and embedding weights from normal distribution
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            # Initialize LayerNorm weights to 1.0 and bias to 0.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Apply initialization to all model parameters
    model.apply(_init_weights)
    return model

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
        want_lines_in_ctx: list, 
        shuffle_files: bool = True,
    ):
        super().__init__()
        self.files = list(files)
        self.tok_name_or_path = tok_name_or_path
        self.tok_kwargs = tok_kwargs
        self.want_ctx_size = int(want_ctx_size)
        self.want_lines_in_ctx = want_lines_in_ctx
        self.shuffle_files = shuffle_files

        # # self._context_schedule = self._build_context_schedule(
        # #     self.min_lines_in_context, self.max_lines_in_context
        # # )
        # self._context_schedule = 

        # if not self._context_schedule:
        #     raise ValueError(
        #         f"No powers-of-two between min={self.min_lines_in_context} and max={self.max_lines_in_context}."
        #     )

    # def _build_context_schedule(self, min_ctx: int, max_ctx: int) -> List[int]:
    #     if min_ctx > max_ctx:
    #         min_ctx, max_ctx = max_ctx, min_ctx
    #     k0 = math.ceil(math.log2(max(1, min_ctx)))
    #     k1 = math.floor(math.log2(max(1, max_ctx)))
    #     return [1 << k for k in range(k0, k1 + 1) if (1 << k) >= min_ctx and (1 << k) <= max_ctx]

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

        # schedule = self._context_schedule
        schedule = self.want_lines_in_ctx
        
        sched_idx = rng.randrange(len(schedule))

        for p in file_iter:
            # print(f"TrainIterableDatasetVarCtx reading file {p}")
            lines = safe_read_lines(Path(p))
            if not lines:
                continue

            n_lines = len(lines)

            want_ctx_lines = schedule[sched_idx]
            sched_idx = (sched_idx + 1) % len(schedule)
            # select a random start point from 0 to (n_lines) - want_ctx_lines
            if want_ctx_lines > n_lines:# tiny file
                range_start = 0
                range_end = len(lines)
            else:
                range_start = random.randint(0,  n_lines - want_ctx_lines)
                range_end = range_start + want_ctx_lines

            text ="\n".join(lines[range_start:range_end]) 
            if len(text) > self.want_ctx_size: text = text[0:self.want_ctx_size]
            
            # print(f"Dataset read a file with {n_lines}. My job is to select a chunk from that file of size {want_ctx_lines}")
            # print(f"Selected chunk from {range_start} to {range_end}")            
            # print(text)
            yield text

    


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

        self._train_files: list[Path] | None = None
        self._train_ds = None
        self._val_ds: MiddleP2WindowDataset | None = None
        self.val_preview_texts: list[str] = []

        # now sanity check the requested context max and min
        test_folder   = self.data_dir / "validation"
        files = find_text_files(test_folder)
        ctx_stats = SimpleDataModule.get_context_stats_for_file(files[0], self.tokenizer, self.want_ctx_size)
        print(f"SimpleDataModule: analysed model you need you nyou need to manually create the training and validation data folderseed to manually create the training and validation data foldersto manually create the training and validation data foldersto find max ctx sentences. Heres the stats: {ctx_stats}")
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
        # now set up the list of 'lines in ctx' we can do with min and mac
        lines_in_ctx = []
        val = self.min_lines_in_context
        while val <= self.max_lines_in_context:
            lines_in_ctx.append(val)
            val *= 2
        print(f"SipleDatamodule chose lines in ctx: {lines_in_ctx} where max is {self.max_lines_in_context}")
        self.lines_in_ctx = lines_in_ctx

    def setup(self, stage: str | None = None):
        train_root = self.data_dir / "training"
        val_root   = self.data_dir / "validation"
        assert train_root.exists(), f"Training directory not found: {train_root} you need to manually create the training and validation data folders"
        assert val_root.exists(),   f"Validation directory not found: {val_root} you need to manually create the training and validation data folders"

        ## some notes on the validation dataset
        ## I wanted to have different sizes of 'context' input
        ## in the validation set and I want to do that over a few different files
        if stage in (None, "validate"):
            if self._val_ds is None:
                val_files = find_text_files(val_root)
                self.val_files = val_files # so we can query them later...
                assert len(val_files) > 0, "No validation files found"

                # Use the middlep2window dataset which is designed for validation datasets
                self._val_ds = MiddleP2WindowDataset(val_files, block_size=self.want_ctx_size, tokenizer=self.tokenizer,  min_p2=self.min_lines_in_context, 
                                                     max_p2=self.max_lines_in_context)


        if stage in (None, "fit"):
            if self._train_files is None:
                train_files = find_text_files(train_root)
                # If you are here and you are having problems getting it to run with a given dataset
                # I wanted a minimal dataset to check my full epoch code and that lead me to investigate this!
                # I found that 47 files was the magic number. No idea why. and no time to find out. life is strange. 
                assert len(train_files) > 46, "You have too few training files in your training folder. Weirdly, I want 47 or more"
                self._train_files = train_files # at least 47 needed. 
                print(f"SimpleDataModule:: setup about to train - using {len(train_files)} training data files")
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
            want_lines_in_ctx=self.lines_in_ctx, 
            # max_lines_in_context=self.max_lines_in_context,
            # min_lines_in_context=self.min_lines_in_context,
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


# --------------------
# LightningModule
# --------------------

class CausalLMModule(L.LightningModule):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        hf_repo_name: str | None = None,   
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        max_new_tokens: int = 64,
    ):
        super().__init__()

        # If not explicitly given, try to infer from the model
        if hf_repo_name is None:
            # HF typically sets these:
            hf_repo_name = getattr(model, "name_or_path", None)
            if hf_repo_name is None and hasattr(model, "config"):
                hf_repo_name = getattr(model.config, "_name_or_path", None)

        self.hf_repo_name = hf_repo_name

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
        print(f"Validation step called, val_loss={val_loss.item()}")
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

    @classmethod    
    def load_from_checkpoint_auto(cls, ckpt_path: str, force_hf_repo:str = None, local_files_only: bool = False):
        """Attempt to load from a lightning checkpoint. Note it expects hf_repo_name in the ckpt hyper params
        as it first 
        """
        # Peek into the checkpoint to find the repo/path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        hparams = ckpt.get("hyper_parameters", {})
        if force_hf_repo is not None: repo = force_hf_repo
        else: repo = hparams.get("hf_repo_name", None)
        assert repo is not None, f"no hf repo in the ckpt or passed as force_hf_repo so cannot load"
        print(f"Loading from HF...")
        
        tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(repo, local_files_only=local_files_only)
        
        print(f"Tokenizer and model loaded successfully. Now importing state dict and lightning status from checkpoint at {ckpt_path} ")
        # lit_module = CausalLMModule(
        #     model=model,
        #     tokenizer=tokenizer
        # )
        # lit_module.load_from_checkpoint(ckpt_path)
        # return lit_module
        try:
            result = cls.load_from_checkpoint(
                ckpt_path,
                model=model,
                tokenizer=tokenizer,
            )
            return result 
        except:
            print(f"Could not import from your checkpoint at {ckpt_path} - probably your checkpoint is not actually {repo} type")
            return None


# class CausalLMModule(L.LightningModule):
#     def __init__(
#         self,
#         model: AutoModelForCausalLM,
#         tokenizer: AutoTokenizer,
#         lr: float = 2e-5,
#         weight_decay: float = 0.01,
#         warmup_ratio: float = 0.05,
#         max_new_tokens: int = 64,
#     ):
#         super().__init__()
#         self.save_hyperparameters(ignore=["model", "tokenizer"])
#         self.model = model
#         self.tokenizer = tokenizer
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.warmup_ratio = warmup_ratio
#         self.max_new_tokens = max_new_tokens

#         self._param_count = sum(p.numel() for p in self.model.parameters())

#     def forward(self, **batch):
#         return self.model(**batch)

#     def training_step(self, batch, batch_idx):
#         out = self.model(**batch)
#         loss = out.loss
#         self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         out = self.model(**batch)
#         val_loss = out.loss
#         self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
#         return val_loss

#     def on_train_start(self):
#         if isinstance(self.logger, TensorBoardLogger):
#             tb = self.logger.experiment
#             tb.add_scalar("model/num_parameters", float(self._param_count), self.global_step)
#             tb.add_text("model/tokenizer_info", f"vocab_size={self.tokenizer.vocab_size}", self.global_step)

#     def on_validation_epoch_end(self):
#         # Perplexity from aggregated val_loss
#         # LOG.info(f"Validation epoch ended - running additional evaluations")

#         val_loss = self.trainer.callback_metrics.get("val_loss")
#         if val_loss is not None:
#             try:
#                 ppl = torch.exp(val_loss)
#                 self.log("perplexity", ppl, prog_bar=True)
#                 if isinstance(self.logger, TensorBoardLogger):
#                     self.logger.experiment.add_scalar("val/perplexity", float(ppl), self.global_step)
#             except Exception:
#                 pass

#         # Sample generations for a few validation prompts
#         dm = self.trainer.datamodule
#         if not isinstance(dm, SimpleDataModule) or not dm.val_preview_texts:
#             return
#         # try:
#         self.model.eval()
#         samples = []
#         for i, prompt_text in enumerate(dm.val_preview_texts):
#             enc = self.tokenizer(
#                 prompt_text,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=dm.want_ctx_size,
#             )
#             enc = {k: v.to(self.device) for k, v in enc.items()}

#             with torch.no_grad():
#                 gen_ids = self.model.generate(
#                     **enc,
#                     max_new_tokens=self.max_new_tokens,
#                     do_sample=False,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                     eos_token_id=self.tokenizer.eos_token_id,
#                 )
#             gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
#             samples.append(f"### Prompt {i+1}\n{prompt_text}\n\n### Output {i+1}\n{gen_text}\n")

#         if isinstance(self.logger, TensorBoardLogger):
#             self.logger.experiment.add_text("samples", "\n".join(samples), self.global_step)
#         # except Exception as e:
#             # LOG.warning(f"Sample generation failed: {e}")
#         # LOG.info(f"Validation epoch ended - additional evaluations complete")


#     def configure_optimizers(self):
#         """This configuration drives learning rate scheduling using epoch"""
#         opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

#         warmup_epochs = max(1, int(self.trainer.max_epochs * self.warmup_ratio))
#         def lr_lambda(epoch):
#             if epoch < warmup_epochs:
#                 return float(epoch + 1) / float(warmup_epochs)   # linear warmup
#             progress = (epoch - warmup_epochs) / max(1, self.trainer.max_epochs - warmup_epochs)
#             return max(0.0, 1.0 - progress)                      # linear decay

#         sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
#         return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch", "name": "epoch_warmup_linear"}}

