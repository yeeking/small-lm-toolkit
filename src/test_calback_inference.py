## Tests the shared_utils sample renderer

import json
import os
import shared_utils

# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import shared_utils
from transformers import pipeline


def main(prompt):
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

        print(f"\n=== Testing model: {family} ({size_b}B) from {repo} ===")
        # try:
        tokenizer, model = shared_utils.load_model_no_cache(repo, size_b, trust_remote_code=True)

        
        # Assertions to verify objects exist and look sane
        assert tokenizer is not None, f"'tokenizer' not created for {repo}"
        assert model is not None, f"'model' not created for {repo}"
        # light sanity: tokenizer must have eos_token or vocab size
        assert getattr(tokenizer, "vocab_size", None) or tokenizer.get_vocab(), "Tokenizer seems empty"
        # model must have some parameters
        assert sum(p.numel() for p in model.parameters()) > 0, "Model has zero parameters?"

        # Some models don’t have a pad token; set it to eos to avoid warnings/errors
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        smallm_model = shared_utils.CausalLMModule(
            model=model,
            tokenizer=tokenizer
        )

        # [PosixPath("data/pijama-mini/validation/'Round Midnight - Live At Maybeck Recital Hall, Berkeley, CA  April 1990.txt")]
        prompt_files=["data/pijama-mini/validation/'Round Midnight - Live At Maybeck Recital Hall, Berkeley, CA  April 1990.txt"]
        prompts = []
        for fname in prompt_files:
            assert os.path.exists(fname), f"Trying to setup training output previews but {fname} does not exist"
            with open(fname) as f:
                prompts.append(f.read()[0:1024])

        # renders example output from the model during training 
        audio_preview_render_fn = shared_utils.HFPreviewResponder(smallm_model, 
            max_new_tokens=512,  # autoregress for this many tokens 
            do_sample=True,          # or True for sampling previews
        )

        previews = audio_preview_render_fn(prompts=prompts)

        print(f"[OK] Successfully loaded {repo}")
        break
    # except Exception as e:
        #     print(f"[FAIL] Error loading {repo}: {e}")
        #     # Uncomment to stop on first failure:
        #     # raise

if __name__ == "__main__":


    # main(prompt)
    main("ignore_me")
    

## load model config

## iterate over loading models

## generate output from a prompt and print it out 




