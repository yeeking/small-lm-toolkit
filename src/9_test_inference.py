## Quick script to test inference on a simple input prompt



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
        try:
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

            generate = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                do_sample=True,           # set False for greedy / deterministic
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                max_new_tokens=200,
            )

            # prompt = "Explain transformers to a musician in one paragraph."
            out = generate(prompt)[0]["generated_text"]
            print(f"{prompt}\n\nMODEL OUTPUT\n\n")
            print(out)

            print(f"[OK] Successfully loaded {repo}")
            # break
        except Exception as e:
            print(f"[FAIL] Error loading {repo}: {e}")
            # Uncomment to stop on first failure:
            # raise


if __name__ == "__main__":
    prompt = """
    p_63 c_0 v_59 d_240 w_120
    p_70 c_0 v_80 d_1840 w_240
    p_44 c_0 v_52 d_1280 w_560
    p_53 c_0 v_54 d_1280 w_0
    p_61 c_0 v_75 d_1360 w_0
    p_68 c_0 v_78 d_720 w_600
    p_54 c_0 v_64 d_3520 w_680
    p_45 c_0 v_59 d_3560 w_0
    p_59 c_0 v_70 d_5520 w_0 
    """

    main(prompt)

## load model config

## iterate over loading models

## generate output from a prompt and print it out 




