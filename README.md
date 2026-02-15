# small-lm-toolkit
Scripts and such to process and evaluate small language models

<<<<<<< HEAD
## Converting a checkpoint into a gguf 

```
python src/4_convert_ckpt_to_hf.py --ckpt  models/ext_hd/training_runs/202510_from_scratch_njam/EleutherAI__pythia-70m/size_0.07/version_0/checkpoints/best-epoch\=82-val_loss\=0.7829.ckpt --base-model-path models/ext_hd/local_copies_hf_models/hf/EleutherAI__pythia-70m/0.07/  --outdir ./test/ --llama-cpp-dir ./libs/llama.cpp/ --outtype q8_0 

```
=======

## Training 

Fine tune with a dataset: 
```
python ./src/5_run_finetunes.py --config ./data/models_plan.json --data_dir ./data/small-lm-pijama/ --epochs 100 --auto_scale_bs 
```

## Exporting your trained checkpoint to a GGUF

To do the GGUF export you'll need to clone llama cpp:

```
cd libs
git clone git@github.com:ggml-org/llama.cpp.git
```

Then:
```
python src/9_ckpt_to_hf_save.py --ckpt ./models/trained/EleutherAI__pythia-70m_from_scratch_99.ckpt --hf-repo EleutherAI/pythia-70m
```

>>>>>>> 22bfdd704d3ac82ea236dd1d268ab7d27aa952a3
