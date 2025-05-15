import os
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, pairwise_distances
from tqdm import tqdm
import sys

options = ["attentioncnn","attentionlstm", "basiccnn", "basiclstm", "basicgru"]
#currModel = options[3]
currModel = "baselstmtiponly"

#main_dir = f'/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitDataAbalation/{currModel}'
#main_dir = f'/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/attentionlstm/Train'
main_dir = f"/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainData/Train"
output_csv = f"visualisation{currModel}.csv"
metrics_file = f"metrics{currModel}.txt"

person_dirs = sorted([d for d in os.listdir(main_dir)
                      if os.path.isdir(os.path.join(main_dir, d))])
print(f"Found {len(person_dirs)} people in {main_dir}...", flush=True)
all_files = []   # full path of each signature file
person_map = []  # corresponding person label (index)

for i, person in enumerate(person_dirs):
    #print(person)
    person_folder = os.path.join(main_dir, person)
    for file in os.listdir(person_folder):
        if (file.lower().endswith('.csv') or file.lower().endswith('.txt')):
            file_path = os.path.join(person_folder, file)
            all_files.append(file_path)
            #person_map.append(person)
            person_map.append(i)

n_samples = len(all_files)
print(f"Found {n_samples} signature samples.", flush=True)

features = [] #for the flattened signature vectors

use_tqdm = sys.stdout.isatty()
pbar = tqdm(total=n_samples, desc="Processing signatures", disable=not use_tqdm)

for file, person in zip(all_files, person_map):
    pbar.set_description(f"Processing {person}")
    try:
        data = pd.read_csv(file, header=None, sep=",",usecols=[0, 1, 2]).to_numpy()
    except Exception as e:
        print(f"Error reading {file}: {e}")
        pbar.update(1)
        continue
    
    features.append(data.flatten()) #flattening the (500,6) data into a 3000-dimensional vector
    pbar.update(1)

pbar.close()

features = np.array(features)
labels = np.array(person_map)
print("Shape of feature matrix:", features.shape, flush=True)

##################### COMPUTING THE CLUSTERING METRICS #####################

#computing the silhouette score on the raw features.
sample_sil = silhouette_samples(features, labels)
overall_sil = np.mean(sample_sil)
print(f"Overall silhouette score (on raw features): {overall_sil:.4f}", flush=True)

#davies-bouldin index
dbi = davies_bouldin_score(features, labels)
print(f"Davies-Bouldin index: {dbi:.4f}", flush=True)

#the dunn index
unique_lbls = np.unique(labels)

max_diam = 0.0
for L in unique_lbls:
    feats_L = features[labels == L]
    n_L = len(feats_L)
    if n_L <= 1:
        continue

    # pairwise distances *within* this cluster
    dists = pairwise_distances(feats_L, metric='euclidean')
    diam_L = dists.max()
    max_diam = max(max_diam, diam_L)

min_inter = np.inf
for i, Li in enumerate(unique_lbls):
    feats_i = features[labels == Li]
    for Lj in unique_lbls[i+1:]:
        feats_j = features[labels == Lj]
        
        # only between-cluster distances
        dists_ij = pairwise_distances(feats_i, feats_j, metric='euclidean')
        min_dist = dists_ij.min()
        if min_dist < min_inter:
            min_inter = min_dist

dunn_index = min_inter / max_diam
print(f"Dunn index: {dunn_index:.4f}",flush=True)

with open(metrics_file, 'w') as mf:
    mf.write(f"Overall silhouette score (raw features): {overall_sil:.4f}\n")
    mf.write(f"Davies-Bouldin index: {dbi:.4f}\n")
    mf.write(f"Dunn index: {dunn_index:.4f}\n")
print(f"Saved clustering metrics to {metrics_file}", flush=True)


# Use t-SNE to reduce dimensionality from 3000 to 3.
tsne = TSNE(n_components=3, random_state=42)
features_3d = tsne.fit_transform(features)
print("Reduced shape after TSNE:", features_3d.shape, flush=True)

df = pd.DataFrame(features_3d, columns=['TSNE1', 'TSNE2', 'TSNE3'])
df.insert(0, 'Person', person_map) 
df.insert(1, 'Silhouette', sample_sil)
df.to_csv(output_csv, index=False)
print(f"Saved visualization data to {output_csv}", flush=True)