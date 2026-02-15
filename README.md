# small-lm-toolkit
Scripts and such to process and evaluate small language models

## Converting a checkpoint into a gguf 

```
python src/4_convert_ckpt_to_hf.py --ckpt  models/ext_hd/training_runs/202510_from_scratch_njam/EleutherAI__pythia-70m/size_0.07/version_0/checkpoints/best-epoch\=82-val_loss\=0.7829.ckpt --base-model-path models/ext_hd/local_copies_hf_models/hf/EleutherAI__pythia-70m/0.07/  --outdir ./test/ --llama-cpp-dir ./libs/llama.cpp/ --outtype q8_0 

```