import os
import shutil
import shlex
import subprocess

# actualDataFolder    = "/home/alpha/Workbenches/Bora/AirSignatures/AirSigns/AirSignsShuffledBothBalls/CombinedAll3NoAug"
# syntheticDataFolder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/synGenNew2"
# models_folder       = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/resultsComplete"
# main_py_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/main.py"
# eval_log_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/evalLogs/synGenNew2"
# gen_model   = "g_multilstm"
# dis_model   = "d_lstm"

# actualDataFolder    = "/mnt/MIG_store/Datasets/air-signatures/AirSigns/AirSignsShuffledBothBalls/Combined"
# syntheticDataFolder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/synGenLstmFinal"
# models_folder       = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/resultsComplete"
# main_py_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/main.py"
# eval_log_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/evalLogs/synGenLstmFinal"
# gen_model   = "g_basiclstm"
# dis_model   = "d_basiclstm"

actualDataFolder    = "/mnt/MIG_store/Datasets/air-signatures/AirSigns/AirSignsShuffledBothBalls/Combined"
syntheticDataFolder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/synGenLSTMAttn"
models_folder       = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/resultsAbalation/g_attentionlstm_d_attentionlstm"
main_py_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/main.py"
eval_log_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/evalLogs/synGenLSTMAttn"
gen_model   = "g_attentionlstm"
dis_model   = "d_attentionlstm"


# actualDataFolder    = "/mnt/MIG_store/Datasets/air-signatures/AirSigns/Forged/6_columns"
# syntheticDataFolder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/synGenForgery"
# models_folder       = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/resultsComplete"
# main_py_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/main.py"
# eval_log_path = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/evalLogs/synGenForgery"
# gen_model   = "g_basiclstm"
# dis_model   = "d_basiclstm"

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
    model_dir = os.path.join(models_folder, model_name)
    model_to_load = os.path.join(model_dir, f"{gen_model}.pth")
    if not os.path.exists(model_to_load):
        print(f"Skipping {person_name}: The model is not complete", flush=True)
        continue
    signature_paths.append((person_name, model_name))

print(f"Found {len(signature_paths)} models in {models_folder}... ", flush=True)


for idx, (person_name, model_name) in enumerate(signature_paths):
    model_dir = os.path.join(models_folder, model_name)
    model_to_load = os.path.join(model_dir, f"{gen_model}.pth")
    #model_to_load = os.path.join(model_dir, f"g_multilstm.pth")
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

    # command = f"""
    #             python {main_py_path} --root {safe_folder_path} --evaluation \
    #             --dir-name {shlex.quote(out_dir)} \
    #             --model-to-load {safe_model_path} \
    #             --gen-model {gen_model} --dis-model {dis_model} \
    #             --amps-to-load {safe_amps_path} \
    #             --noise-weight 0.001 \
    #             --num-synthetic-users 5 \
    #             --num-steps 1 --batch-size 1 > {eval_log_path}/{person_name}.log 2>&1
    #         """
    command = f"""
                python {main_py_path} --root {safe_folder_path} --evaluation \
                --dir-name {shlex.quote(out_dir)} \
                --model-to-load {safe_model_path} \
                --gen-model {gen_model} --dis-model {dis_model} \
                --amps-to-load {safe_amps_path} \
                --noise-weight 0.01 \
                --num-synthetic-users 5 \
                --num-steps 1 --batch-size 1 > {eval_log_path}/{person_name}.log 2>&1
            """
    subprocess.run(command, shell=True, check=True)

    print(f"Finished processing {person_name}...", flush=True)

print("All models processed successfully.", flush=True)



# for idx, (person_name, model_name) in enumerate(signature_paths):
#     model_dir = os.path.join(models_folder, model_name)
#     model_to_load = os.path.join(model_dir, f"{gen_model}.pth")
#     amps_to_load  = os.path.join(model_dir, "amps.pth")

#     # building the person's data directory path in the actual data folder
#     person_data_dir = os.path.join(actualDataFolder, person_name)
#     if not os.path.isdir(person_data_dir):
#         raise NotADirectoryError(f"{person_name}: data folder not found at {person_data_dir}", flush=True)
    
#     if not os.path.exists(model_to_load):
#             raise FileNotFoundError(f"Generator checkpoint not found: {model_to_load}")
#     if not os.path.exists(amps_to_load):
#         raise FileNotFoundError(f"Amps checkpoint not found: {amps_to_load}")

#     safe_model_path = shlex.quote(model_to_load)
#     safe_amps_path  = shlex.quote(amps_to_load)

#     # Loop through every file inside the person's directory
#     for file_name in os.listdir(person_data_dir):
#         data_path = os.path.join(person_data_dir, file_name)
#         if not os.path.isfile(data_path):
#             continue  # Skip if not a file

#         if not os.path.exists(data_path):
#             raise FileNotFoundError(f"Data file not found: {data_path}")
        
#         safe_data_path  = shlex.quote(data_path)
        
#         # Define output subfolder (absolute path for the synthetic data folder)
#         out_dir = os.path.join(syntheticDataFolder, person_name)
#         os.makedirs(out_dir, exist_ok=True)  # ensure the output directory exists
#         base_out_filename = file_name  # base file name

#         # Run 5 iterations for each file, with N_VALUE from 1 to 5
#         for N_VALUE in range(1, 6):
#             out_filename = f"N{N_VALUE}_{base_out_filename}"

            
#             command = f"""
#                 python {main_py_path} --root {safe_data_path} --evaluation \
#                 --dir-name {shlex.quote(out_dir)} \
#                 --model-to-load {safe_model_path} \
#                 --save-name {shlex.quote(out_filename)} \
#                 --gen-model {gen_model} --dis-model {dis_model} \
#                 --amps-to-load {safe_amps_path} \
#                 --noise-weight 0.001 \
#                 --seed {N_VALUE} \
#                 --num-steps 1 --batch-size 1 > {eval_log_path}/{person_name}.log 2>&1
#             """
#             #print(f"[{idx+1}/{len(signature_paths)}] Executing command for {person_name} on file {file_name} with seed {N_VALUE}:\n{command}", flush=True)
#             subprocess.run(command, shell=True, check=True)
    
#     print(f"Finished processing {person_name}...", flush=True)