import json
import os
import subprocess
import shlex

data_folder = './data'
json_file = os.path.join(data_folder, 'models_plan_sorted.json')
pycmd = f"python {os.path.join(data_folder, '../src', '5_run_finetunes.py')}"
pycmd += f" --config {json_file} --data_dir {os.path.join(data_folder, 'tiny_dataset')}"
pycmd += f" --epochs 1 --auto_scale_bs --model "

with open(json_file) as f:
    jdata = json.load(f)

for model in jdata['models']:
    cmd_str = f"{pycmd}{model['hf_repo']}"
    print(f"Running: {cmd_str}")
    result = subprocess.run(shlex.split(cmd_str))
    if result.returncode != 0:
        print(f"⚠ Command failed for {model['hf_repo']} (exit code {result.returncode})")
