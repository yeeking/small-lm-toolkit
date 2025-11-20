# small-lm-toolkit
Scripts and such to process and evaluate small language models


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

