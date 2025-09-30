## Tests the shared_utils sample renderer

import json
import os, shutil

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
        
        render_callback = shared_utils.PreviewAudioCallback(
            prompt_files=prompt_files,
            max_prompt_len=1024,  
            pl_module=smallm_model,
            max_new_tokens= 128,
            do_sample = True,           # set True for stochastic decoding
            temperature = 0.8,
            top_p = 0.95,
            top_k = 50,
            repetition_penalty = 1.1   # >1.0 discourages repeats
        )
        

        previews = render_callback.render_previews()
        print(previews)
        # now what you gonna do with them there previews bud?
        # how about this brother?
        preview_out_dir = 'test_preview'
        for i, p in enumerate(previews):
            # log audio & text as before...
            source_midi_file = p['midi_file']
            # , f"step_{global_step:12d}"
            if not os.path.exists(preview_out_dir): os.makedirs(preview_out_dir, exist_ok=True)
            filestub = f"{repo.replace('/', '_')}-{size_b}"

            midifile = os.path.join(preview_out_dir, f"{filestub}.mid")
            print(f"Saving file to {midifile}")
            shutil.copy2(source_midi_file, midifile)
            # we could save 'wave' to an audio file with 
            # librosa here too ...  


        print(f"[OK] Successfully rendered with {repo}")
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




