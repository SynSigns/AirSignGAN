import os
import shlex
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True, help="Start index (inclusive)")
parser.add_argument('--end', type=int, required=True, help="End index (exclusive)")
args = parser.parse_args()

gen_model = ["g_multilstm"]
dis_model = ["d_lstm"]

train_data_folder = "../TrainData"

signature_paths = []
for filename in os.listdir(train_data_folder):
    if filename.endswith(".txt"):
        full_path = os.path.join(train_data_folder, filename)
        signature_paths.append(full_path)


signature_paths.sort()

for i, data_path in enumerate(signature_paths):
    if i < args.start or i >= args.end:
        continue

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    safe_data_path = shlex.quote(data_path)
    signName = os.path.splitext(os.path.basename(data_path))[0]

    command = (
        f"nohup python main.py --root {safe_data_path} --noise-weight 0.01 "
        f"--dir-name {signName} --gen-model {gen_model[0]} --dis-model {dis_model[0]} "
        f"--seed 42 --num-steps 3000 > trainLogs/{signName}.log 2>&1"
    )

    print(f"Executing: {command}", flush=True)
    subprocess.run(command, shell=True, check=True)
