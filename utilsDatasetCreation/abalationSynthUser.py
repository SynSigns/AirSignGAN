import os
import shutil
import shlex
import subprocess

g_models = ['g_basiclstm', 'g_basicgru', 'g_basiccnn1d', 'g_attentionlstm', 'g_attentioncnn1d']
d_models = ['d_basiclstm', 'd_basicgru', 'd_basiccnn1d', 'd_attentionlstm', 'd_attentioncnn1d']

actualDataFolder    = "/mnt/MIG_store/Datasets/air-signatures/AirSigns/AirSignsShuffledBothBalls/Combined"
syntheticDataFolder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/Abalation/basiclstm"
models_folder       = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/resultsAbalation/g_basiclstm_d_basiclstm"
main_py_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/main.py"
eval_log_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/evalLogs/Abalation/basiclstm"
gen_model   = g_models[0]  
dis_model   = d_models[0]  


os.makedirs(syntheticDataFolder, exist_ok=True)
os.makedirs(eval_log_path, exist_ok=True)

# finding all the people whose models are present in the models folder
signature_paths = []
for model_name in os.listdir(models_folder):
    if model_name.startswith("."):  # skip hidden files like .DS_Store
        continue

    model_path = os.path.join(models_folder, model_name)
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"{model_path} is not a directory")

    # split at first underscore
    person_name = model_name.split("_", 1)[0]
    signature_paths.append((person_name, model_name))

print(f"Found {len(signature_paths)} models in {models_folder}... ", flush=True)


for idx, (person_name, model_name) in enumerate(signature_paths):
    model_dir = os.path.join(models_folder, model_name)

    model_to_load = os.path.join(model_dir, f"g_multilstm.pth")
    #model_to_load = os.path.join(model_dir, f"{gen_model}.pth")
    amps_to_load  = os.path.join(model_dir, "amps.pth")
    person_data_dir = os.path.join(actualDataFolder, person_name)
    out_dir = os.path.join(syntheticDataFolder, person_name)
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.isdir(person_data_dir):
        print(f"Skipping {person_name}: data folder not found at {person_data_dir}", flush=True)
        continue  # Skip if not a directory
        #raise NotADirectoryError(f"{person_name}: data folder not found at {person_data_dir}", flush=True)
    if not os.path.exists(model_to_load):
            raise FileNotFoundError(f"Generator checkpoint not found: {model_to_load}")
    if not os.path.exists(amps_to_load):
        raise FileNotFoundError(f"Amps checkpoint not found: {amps_to_load}")

    safe_folder_path = shlex.quote(person_data_dir)
    safe_model_path = shlex.quote(model_to_load)
    safe_amps_path  = shlex.quote(amps_to_load)

    command = f"""
                python {main_py_path} --root {safe_folder_path} --evaluation \
                --dir-name {shlex.quote(out_dir)} \
                --model-to-load {safe_model_path} \
                --gen-model {gen_model} --dis-model {dis_model} \
                --amps-to-load {safe_amps_path} \
                --noise-weight 0.001 \
                --num-synthetic-users 5 \
                --num-steps 1 --batch-size 1 > {eval_log_path}/{person_name}.log 2>&1
            """
    subprocess.run(command, shell=True, check=True)

    print(f"Finished processing {person_name}...", flush=True)

print("All models processed successfully.", flush=True)

