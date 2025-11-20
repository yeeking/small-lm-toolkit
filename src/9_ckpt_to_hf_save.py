# load a model from a ckpt file 
# and a repo name 

import os 
import json
import sys
import argparse
from shared_utils import CausalLMModule

def runMain():
    # use argparse to pull in a ckpt arg and an hf-repo arg
    ap = argparse.ArgumentParser(description="Load a checkpoint and save it in HF format.")
    ap.add_argument("--ckpt", required=True, help="Path to the checkpoint file or directory.")
    ap.add_argument("--hf-repo", required=True, dest="hf_repo", help="Target Hugging Face repo ID or name.")
    args = ap.parse_args()

    # Placeholder: replace with actual load/save logic using args.ckpt and args.hf_repo
    print(f"Attempting to create {args.hf_repo} type model using ckpt {args.ckpt}")
    model = CausalLMModule.load_from_checkpoint_auto(ckpt_path=args.ckpt, force_hf_repo = args.hf_repo)
    


if __name__ == "__main__":
    runMain()
