

import json 
import os 
import argparse
from pathlib import Path
from llama_cpp import Llama

import shared_utils


from pathlib import Path


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
        print(f"\n=== Verifying GGUF model {family} ({size_b}B) from {repo} ===")

        gguf_fpath = shared_utils.get_model_gguf(repo, size_b)

        assert os.path.exists(gguf_fpath), f"Cannot find gguf file {gguf_fpath} - have you run the gguf generator script?"

        llm = shared_utils.load_gguf_model(gguf_fpath, n_ctx=512, vocab_only=False)  # quick load

        print(f"[INFO] Printing model info")

        shared_utils.describe_model(llm)

        print(f"[INFO] Doing test token/ detoken!")

        in_str = "Hello world!"
        out_str = shared_utils.tokenize_detokenize(in_str,  llm)
        assert in_str == out_str, f"Input '{in_str}' != output '{out_str}'"

        print(f"[INFO] Doing test infer ")

        out_str = shared_utils.do_test_infer(llm)
        print(f"Infer result: {out_str}")


        


if __name__ == "__main__":
    main()
