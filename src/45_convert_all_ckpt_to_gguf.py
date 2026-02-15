import os 
import numpy as np

out_folder=  "test"
ckpt_folder = 'models/ext_hd/training_runs/202510_from_scratch_njam/'
hf_folder = 'models/ext_hd/local_copies_hf_models/hf/'

assert os.path.exists(out_folder)
assert os.path.exists(ckpt_folder)
assert os.path.exists(hf_folder)
outtype = 'q8_0'


good_names = []

# python src/4_convert_ckpt_to_gguf.py --ckpt  models/ext_hd/training_runs/202510_from_scratch_njam/EleutherAI__pythia-70m/size_0.07/version_0/checkpoints/best-epoch\=82-val_loss\=0.7829.ckpt \
#                                      --base-model-path models/ext_hd/local_copies_hf_models/hf/EleutherAI__pythia-70m/0.07/  \
#                                      --outdir ./test/ \
#                                      --llama-cpp-dir ./libs/llama.cpp/ \
#                                      --outtype q8_0 \
#                                      --outfile test/EleutherAI__pythia-70m 


params = {"--ckpt":[], "--base-model-path":[], "--outfile":[]}

for model_name in [f for f in os.listdir(ckpt_folder) if os.path.isdir(ckpt_folder + f)]:
    # print(model_name)
    # now find this model name in the saved models
    for model_name_b in [f for f in os.listdir(hf_folder) if os.path.isdir(hf_folder + f)]:
        if model_name == model_name_b: 
            ckpt_sub_folder = os.listdir(os.path.join(ckpt_folder, model_name))[0]
            ckpt_sub_folder = os.path.join(ckpt_folder, model_name, ckpt_sub_folder, 'version_0', 'checkpoints')
            hf_sub_folder = os.listdir(os.path.join(hf_folder, model_name))[0]
            hf_sub_folder = os.path.join(hf_folder, model_name, hf_sub_folder)
            if not os.path.exists(ckpt_sub_folder): 
                continue
            if not os.path.exists(hf_sub_folder): 
                continue
            
            # assert os.path.exists(ckpt_sub_folder), f"checkpoint not found for {model_name} : should be {ckpt_sub_folder}"
            # assert os.path.exists(hf_sub_folder), f'HF save folder not found for {model_name} : should be {hf_sub_folder}'
            # find the 'best' checkpoint
            # assert len( os.listdir(ckpt_sub_folder)) > 0, f"no ckpts found for {model_name} in {ckpt_sub_folder}"
            best_ckpt_file = [f for f in os.listdir(ckpt_sub_folder) if f.startswith('best')][-1]
            all_epochs = [f for f in os.listdir(ckpt_sub_folder) if f.startswith('epoch')]
            all_epochs = np.sort(all_epochs)
            last_ckpt_file = all_epochs[-1]
            # print(f"{model_name}, best: {best_ckpt_file}, last: {last_ckpt_file}")

            full_best_ckpt_path = os.path.join(ckpt_sub_folder, best_ckpt_file)
            full_last_ckpt_path = os.path.join(ckpt_sub_folder, last_ckpt_file)
            # params = {"--ckpt":[], "--base-model-path":[], "--outfile":[]}
            params["--ckpt"].append(full_best_ckpt_path)
            params["--ckpt"].append(full_last_ckpt_path)

            params["--base-model-path"].append(hf_sub_folder)
            params["--base-model-path"].append(hf_sub_folder)

            params["--outfile"].append("test/" + model_name + "-best-" + outtype)
            params["--outfile"].append("test/" + model_name + "-last-" + outtype)



for ind,outfile in enumerate(params['--outfile']):
# python src/4_convert_ckpt_to_gguf.py --ckpt  models/ext_hd/training_runs/202510_from_scratch_njam/EleutherAI__pythia-70m/size_0.07/version_0/checkpoints/best-epoch\=82-val_loss\=0.7829.ckpt \
#                                      --base-model-path models/ext_hd/local_copies_hf_models/hf/EleutherAI__pythia-70m/0.07/  \
#                                      --outdir ./test/ \
#                                      --llama-cpp-dir ./libs/llama.cpp/ \
#                                      --outtype q8_0 \
#                                      --outfile test/EleutherAI__pythia-70m 
    print(f"#{outfile}")    

    cmd = "python src/4_convert_ckpt_to_gguf.py "
    for k in params.keys():
        cmd += f" {k} {params[k][ind]} "

    cmd += f" --outtype {outtype}"
    cmd += " --llama-cpp-dir ./libs/llama.cpp"
    cmd += " --outdir ./test/"
    
    print(cmd)

# print(params)

