import os
import shutil

parent_path = "/mnt/MIG_store/Datasets/air-signatures/AirSigns/AirSignsShuffledBothBalls"

src_folders = ['Train', 'Test', 'Validation']
combined_path = os.path.join(parent_path, 'Combined')
os.makedirs(combined_path, exist_ok=True)

for split in src_folders:
    split_path = os.path.join(parent_path, split)
    
    for person_name in sorted(os.listdir(split_path)):
        person_src_path = os.path.join(split_path, person_name)
        person_dest_path = os.path.join(combined_path, person_name)
        os.makedirs(person_dest_path, exist_ok=True)

        for file_name in sorted(os.listdir(person_src_path)):
            src_file = os.path.join(person_src_path, file_name)
            dest_file = os.path.join(person_dest_path, file_name)
            
            # Avoid overwriting if same filename exists in multiple sets
            if os.path.exists(dest_file):
                base, ext = os.path.splitext(file_name)
                count = 1
                while os.path.exists(dest_file):
                    dest_file = os.path.join(person_dest_path, f"{base}_{count}{ext}")
                    count += 1

            shutil.copy2(src_file, dest_file)
