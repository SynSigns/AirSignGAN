import subprocess
import os
from multiprocessing import Process
import shlex


gen_model = ['g_basiclstm', 'g_basicgru', 'g_basiccnn1d', 'g_attentionlstm', 'g_attentioncnn1d']
dis_model = ['d_basiclstm', 'd_basicgru', 'd_basiccnn1d', 'd_attentionlstm', 'd_attentioncnn1d']

train_data_folder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/TrainData"


signature_paths = []
for person_name in sorted(os.listdir(train_data_folder)):
    if person_name.startswith("."):
        continue
    person_path = os.path.join(train_data_folder, person_name)
    signature_paths.append((person_name.split(".",1)[0], person_path))
print(f"Found {len(signature_paths)} signature files in {train_data_folder}.", flush=True)

def run_model(i, currGenModel, currDisModel):
    deviceId = i % 4
    models_folder = f"./resultsAbalation/{currGenModel}_{currDisModel}"
    os.makedirs(models_folder, exist_ok=True)
    log_folder    = f"trainLogs/{currGenModel}"
    os.makedirs(log_folder, exist_ok=True)

    for person_name, person_path in signature_paths:
        if not os.path.exists(person_path):
            raise FileNotFoundError(f"Data file not found: {person_path}")
        model_person_folder = f"{models_folder}/{person_name}"
        os.makedirs(model_person_folder, exist_ok=True)

        cmd = (
            f"nohup python main.py"
            f"  --root {shlex.quote(person_path)}"
            f"  --noise-weight 0.01"
            f"  --device-ids {deviceId}"
            f"  --dir-name {shlex.quote(model_person_folder)}"
            f"  --gen-model {currGenModel}"
            f"  --dis-model {currDisModel}"
            f"  --seed 42"
            f"  --num-steps 3000"
            f" > {log_folder}/{person_name}.log 2>&1"
        )
        print(f"[model {i}] {currGenModel}/{currDisModel} → {person_name} on GPU {deviceId}", flush=True)
        subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    procs = []
    for i in range(1, len(gen_model)):
        print(f"Starting process for model {i} on device {i%4}: {gen_model[i]} and {dis_model[i]}", flush=True)
        p = Process(target=run_model, args=(i, gen_model[i], dis_model[i]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()




# import subprocess
# import os
# import shlex

# # Generator and Discriminator models
# gen_model = ['g_basiclstm', 'g_basicgru', 'g_basiccnn1d', 'g_attentionlstm', 'g_attentioncnn1d']
# dis_model = ['d_basiclstm', 'd_basicgru', 'd_basiccnn1d', 'd_attentionlstm', 'd_attentioncnn1d']


# train_data_folder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/TrainData"


# signature_paths = []

# #go through the traindatafolder and get a list of all the training data files
# for person_name in sorted(os.listdir(train_data_folder)):
#     if person_name.startswith("."):  # skip hidden files like .DS_Store
#         continue
#     person_path = os.path.join(train_data_folder, person_name)
#     person_name = person_name.split(".", 1)[0]
#     signature_paths.append((person_name, person_path))

# print(f"Found {len(signature_paths)} signature files in {train_data_folder}.", flush=True)

# for i in range(1, len(gen_model)):
#     currGenModel = gen_model[i]
#     currDisModel = dis_model[i]
#     deviceId = i % 4
#     models_folder = f"./resultsAbalation/{currGenModel}_{currDisModel}"
#     os.makedirs(models_folder, exist_ok=True)

#     for(person_name, person_path) in signature_paths:
#         model_person_folder = f"{models_folder}/{person_name}"
#         log_folder = f"trainLogs/{currGenModel}"

#         os.makedirs(model_person_folder, exist_ok=True)
#         os.makedirs(log_folder, exist_ok=True)

#         if not os.path.exists(person_path):
#             raise FileNotFoundError(f"Data file not found: {person_path}")
#         command = (
#         f"nohup python main.py --root {person_path} --noise-weight 0.01 --device-ids {deviceId} "
#         f"--dir-name {model_person_folder} --gen-model {currGenModel} --dis-model {currDisModel} "
#         f"--seed 42 --num-steps 3000 > {log_folder}/{person_name}.log 2>&1"
#         )
#         print(f"[{i}th iteration] Executing:{command}", flush=True)
#         subprocess.run(command, shell=True, check=True)

