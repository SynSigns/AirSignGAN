import os
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from tqdm import tqdm
import sys

main_dir = '/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainData/Train'
output_csv = "visualisation.csv"

person_dirs = sorted([d for d in os.listdir(main_dir)
                      if os.path.isdir(os.path.join(main_dir, d))])

all_files = []  
person_map = []  

for i, person in enumerate(person_dirs):
    #print(person)
    person_folder = os.path.join(main_dir, person)
    for file in os.listdir(person_folder):
        if file.lower().endswith('.txt'):
            file_path = os.path.join(person_folder, file)
            all_files.append(file_path)
            #person_map.append(person)
            person_map.append(i)

n_samples = len(all_files)
print(f"Found {n_samples} signature samples.", flush=True)

features = []

use_tqdm = sys.stdout.isatty()
pbar = tqdm(total=n_samples, desc="Processing signatures", disable=not use_tqdm)

for file, person in zip(all_files, person_map):
    pbar.set_description(f"Processing {person}")
    try:
        data = pd.read_csv(file, header=None, sep=",").to_numpy()
    except Exception as e:
        print(f"Error reading {file}: {e}")
        pbar.update(1)
        continue
    
    # flattening (500,6) to a (3000,)
    flat_vector = data.flatten()
    features.append(flat_vector)
    pbar.update(1)

pbar.close()

features = np.array(features)
labels = np.array(person_map)
print("Shape of feature matrix:", features.shape, flush=True)

#computing the silhouette score on the raw features.
sample_sil = silhouette_samples(features, labels)
overall_sil = np.mean(sample_sil)
print(f"Overall silhouette score (on raw features): {overall_sil:.4f}", flush=True)

# # tsne reducing dim from 3000 to 3.
# tsne = TSNE(n_components=3, random_state=42)
# features_3d = tsne.fit_transform(features)
# print("Reduced shape after TSNE:", features_3d.shape, flush=True)
# tsne reducing dim from 3000 to 2.
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)
print("Reduced shape after TSNE:", features_2d.shape, flush=True)


df = pd.DataFrame(features_2d, columns=['TSNE1', 'TSNE2'])
df.insert(0, 'Person', person_map)  
df.insert(1, 'Silhouette', sample_sil)  

df.to_csv(output_csv, index=False)
print(f"Saved visualization data to {output_csv}", flush=True)
