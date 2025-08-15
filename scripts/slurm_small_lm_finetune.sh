#!/bin/bash
#SBATCH --time 10080          # time in minutes to reserve
#SBATCH --cpus-per-task 8  # number of cpu cores
#SBATCH --mem 32G           # memory pool for all cores
#SBATCH --gres gpu:nvidia_a30:1       # number of gpu cores
#SBATCH --nodelist ben  # bill be broke for now
#SBATCH  -o small-lm-finetune-test.log      # write output to log file

srun   -l  python ./src/6_run_finetunes_separate.py

