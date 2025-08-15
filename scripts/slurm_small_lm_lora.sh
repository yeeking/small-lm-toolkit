#!/bin/bash
#SBATCH --time 10080          # time in minutes to reserve
#SBATCH --cpus-per-task 8  # number of cpu cores
#SBATCH --mem 32G           # memory pool for all cores
#SBATCH --gres gpu:nvidia_a30:1       # number of gpu cores
#SBATCH --nodelist ben  # bill be broke for now
#SBATCH  -o small-lm-lora-test.log      # write output to log file

srun   -l python ./src/7_run_lora_finetunes.py  --config ./data/models_plan_sorted.json   --data_dir ./data/small-lm-pijama   --out_dir ./runs_lora   --epochs 2   --auto_scale_bs   --lora_r 16 --lora_alpha 32 --lora_dropout 0.05   --lora_target_modules all-linear
