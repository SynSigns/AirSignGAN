import subprocess
import os
import shlex

# Generator and Discriminator models
gen_model = ["g_multilstm"]
dis_model = ["d_lstm"]
#generatorName = "g_multilstm.pth"
generatorName = "g_multilstm.pth"

train_data_folder = "../TrainData"
#models_folder = "./resultsComplete"
models_folder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/resultsAbalation/g_attentioncnn1d_d_attentioncnn1d"
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

print(f"Found {len(signature_paths)} models in {models_folder}... Starting Generation", flush=True)

for i, (person_name, model_name) in enumerate(signature_paths):
    model_dir = os.path.join(models_folder, model_name)
    model_to_load = os.path.join(model_dir, "g_multilstm.pth")
    amps_to_load  = os.path.join(model_dir, "amps.pth")
    data_path     = os.path.join(train_data_folder, person_name + ".txt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(model_to_load):
        raise FileNotFoundError(f"Generator checkpoint not found: {model_to_load}")
    if not os.path.exists(amps_to_load):
        raise FileNotFoundError(f"Amps checkpoint not found: {amps_to_load}")
    
    safe_data_path    = shlex.quote(data_path)
    safe_model_path   = shlex.quote(model_to_load)
    safe_amps_path    = shlex.quote(amps_to_load)
    
    command = f"""
    python main.py --root {safe_data_path} --evaluation \
    --dir-name synGen2/{person_name} \
    --model-to-load {safe_model_path} \
    --gen-model {gen_model[0]} --dis-model {dis_model[0]} \
    --amps-to-load {safe_amps_path} \
    --noise-weight 0.001 \
    --num-steps 60 --batch-size 8 > evalLogs/synGen2/{person_name}.log 2>&1
    """

    print(f"[{i+1}/{len(signature_paths)}] Executing:\n{command}", flush=True)
    #proc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    subprocess.run(command, shell=True, check=True)




# command = f"""
#     nohup python main.py --root {safe_data_path} --evaluation \
#     --dir-name {signName}Results \
#     --model-to-load {model_to_load} \
#     --gen-model {gen_model[0]} --dis-model {dis_model[0]} \
#     --amps-to-load {amps_to_load} \
#     --noise-weight 0.01 \
#     --num-steps 1 --batch-size 8 > evalLogs/{signName}.log 2>&1
#     """


   # command = f"nohup python main.py --root {safe_data_path} --noise-weight 0.01 \
    # --dir-name {signName} --gen-model {gen_model[0]} --dis-model {dis_model[0]} \
    # --seed 42 --num-steps 3000 > trainLogs/{signName}.log 2>&1"

# safe_data_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainData/Train/abhijit/abhijit_file-0-aug-0-1.txt"
# os.makedirs("./realAirSignsLayers", exist_ok=True)
# command = f"python main.py --root {safe_data_path} --noise-weight 0.01 \
#     --dir-name HEHEHE --gen-model {gen_model[0]} --dis-model {dis_model[0]} \
#     --seed 42 --num-steps 1 > trainLogs/HEHEHE.log 2>&1"

# subprocess.run(command, shell=True, check=True)
