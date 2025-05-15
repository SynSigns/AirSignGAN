import os
import random
import shutil

# 1) Paths and split ratios
# src_root    = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainData/CombinedAug"
# output_root = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainData"

# src_root    = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainDataNoAug/CombinedNoAug"
# output_root = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainDataNoAug"


src_root    = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/attentionlstm/CombinedAug"
output_root = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/attentionlstm"


splits      = ["Train", "Validation", "Test"]
ratios      = [0.7, 0.15, 0.15]

for split in splits:
    split_dir = os.path.join(output_root, split)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir)

for class_name in sorted(os.listdir(src_root)):
    class_src = os.path.join(src_root, class_name)
    if class_src.startswith('.'):
        continue
    if not os.path.isdir(class_src):
        continue

    # 3a) Gather all files in this class
    files = [
        os.path.join(class_src, f)
        for f in os.listdir(class_src)
        if os.path.isfile(os.path.join(class_src, f))
    ]
    random.shuffle(files)

    # 3b) Compute split boundaries
    n_total = len(files)
    n_train = int(ratios[0] * n_total)
    n_val   = int(ratios[1] * n_total)
    train_files = files[:n_train]
    val_files   = files[n_train : n_train + n_val]
    test_files  = files[n_train + n_val :]

    split_files = {
        "Train":      train_files,
        "Validation": val_files,
        "Test":       test_files,
    }

    # 4) For each split, make a class subfolder and symlink the files
    for split, file_list in split_files.items():
        dest_class_dir = os.path.join(output_root, split, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)

        for src_path in file_list:
            dst_path = os.path.join(dest_class_dir, os.path.basename(src_path))
            # Create symlink if it doesn't already exist
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)

print("Done! Splitting with Symlinks.")
