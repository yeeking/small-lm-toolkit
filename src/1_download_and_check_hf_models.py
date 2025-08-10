import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

        print(f"\n=== Testing model: {family} ({size_b}B) from {repo} ===")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo,
                trust_remote_code=trust_remote_code,
            )
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=trust_remote_code,
            )
            model.eval()

            # Assertions to verify objects exist and look sane
            assert tokenizer is not None, f"'tokenizer' not created for {repo}"
            assert model is not None, f"'model' not created for {repo}"
            # light sanity: tokenizer must have eos_token or vocab size
            assert getattr(tokenizer, "vocab_size", None) or tokenizer.get_vocab(), "Tokenizer seems empty"
            # model must have some parameters
            assert sum(p.numel() for p in model.parameters()) > 0, "Model has zero parameters?"

            print(f"[OK] Successfully loaded {repo}")
        except Exception as e:
            print(f"[FAIL] Error loading {repo}: {e}")
            # Uncomment to stop on first failure:
            # raise


if __name__ == "__main__":
    main()
