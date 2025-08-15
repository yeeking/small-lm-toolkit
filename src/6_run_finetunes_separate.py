import json
import os
import subprocess
import shlex

data_folder = './data/small-lm-pijama/'
json_file = './data/models_plan.json'
src_folder = "./src/"
# data_folder = '../../src/small-lm-toolkit/data'
# json_file = os.path.join(data_folder, 'models_plan_sorted.json')
pycmd = f"python {os.path.join(src_folder, '5_run_finetunes.py')}"
pycmd += f" --config {json_file} --data_dir {data_folder})}"
pycmd += f" --epochs 5 --auto_scale_bs --model "

with open(json_file) as f:
    jdata = json.load(f)

for model in jdata['models']:
    cmd_str = f"{pycmd}{model['hf_repo']}"
    print(f"Running: {cmd_str}")
    result = subprocess.run(shlex.split(cmd_str))
    if result.returncode != 0:
        print(f"⚠ Command failed for {model['hf_repo']} (exit code {result.returncode})")
