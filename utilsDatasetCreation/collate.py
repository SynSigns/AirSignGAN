import os
import re
import shutil

# syntheticDataFolder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/Abalation/basiclstm"
# outputRoot = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitDataAbalation/basiclstm" 

syntheticDataFolder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/synGenLSTMAttn"
outputRoot = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/attentionlstm/CombinedNoAug" 


# syntheticDataFolder = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/generation/results/synGenLstmFinal"
# outputRoot = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainData/CombinedNoAug"

# Clear the outputRoot directory if it exists
if os.path.exists(outputRoot):
    shutil.rmtree(outputRoot)
os.makedirs(outputRoot, exist_ok=True)

# Regex to capture the batch (N1–N5) and person name
# e.g. "N1_person-1_file-0_Validation.csv"
pattern = re.compile(r'^(N[1-5])_(.+)$')

for person_name in os.listdir(syntheticDataFolder):
    if person_name.startswith('.'):
        continue
    person_dir = os.path.join(syntheticDataFolder, person_name)
    if not os.path.isdir(person_dir):
        continue

    for root, dirs, files in os.walk(person_dir):
        for fname in files:
            m = pattern.match(fname)
            if not m:
                continue

            batch, rest = m.groups()  
            # batch = "N1", rest = "person-1_file-0_Validation.csv"

            # Destination subfolder: X/person-1_N1
            dest_subdir = os.path.join(outputRoot, f"{person_name}_{batch}")
            os.makedirs(dest_subdir, exist_ok=True)

            src_path = os.path.join(root, fname)
            dest_fname = rest   # drop the "N1_" prefix
            dest_path = os.path.join(dest_subdir, dest_fname)

            shutil.copy2(src_path, dest_path)

num_folders = len([name for name in os.listdir(outputRoot)
                   if os.path.isdir(os.path.join(outputRoot, name))])

print(f"Number of folders in '{outputRoot}': {num_folders}")
print("Done.")
