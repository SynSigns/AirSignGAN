import os
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, pairwise_distances
from tqdm import tqdm
import sys
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import pairwise_distances


def compute_dunn_index(features: np.ndarray, labels: np.ndarray, pca_components: int = None) -> float:
    """
    Compute the Dunn Index for a set of feature vectors with known cluster labels.
    
    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        The high-dimensional data points (e.g., flattened time-series).
    labels : np.ndarray, shape (n_samples,)
        Integer cluster labels for each point.
    pca_components : int or None
        If not None, apply PCA to reduce features to this many dimensions before computing distances.
    
    Returns
    -------
    dunn : float
        The Dunn Index: (minimum inter-cluster distance) / (maximum intra-cluster diameter).
    """
    #Optional dimensionality reduction to stabilize distances in high-D
    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        features = pca.fit_transform(features)
    
    print("Shape of feature matrix after PCA:", features.shape, flush=True)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Precompute masks and cluster member arrays
    clusters = [features[labels == lbl] for lbl in unique_labels]
    
    # 1) Compute maximum intra-cluster diameter Δ = max_k Δ(C_k)
    max_diameter = 0.0
    for pts in clusters:
        if len(pts) < 2:
            continue
        # pairwise distances within cluster
        D = pairwise_distances(pts, metric='euclidean')
        diameter = np.max(D)
        if diameter > max_diameter:
            max_diameter = diameter
    
    # 2) Compute minimum inter-cluster distance δ = min_{i<j} δ(C_i, C_j)
    min_inter = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # distances between cluster i and j
            D_ij = pairwise_distances(clusters[i], clusters[j], metric='euclidean')
            d_ij = np.min(D_ij)
            if d_ij < min_inter:
                min_inter = d_ij
    
    if max_diameter == 0:
        raise ValueError("All clusters have diameter zero (singleton clusters?). Cannot compute Dunn Index.")
    
    dunn_index = min_inter / max_diameter
    return dunn_index

# Example usage:
# dunn = compute_dunn_index(features, labels, pca_components=50)
# print(f"Dunn Index: {dunn:.4f}")


main_dir = '/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainDataNew/Train'
# output_csv = "visualisation.csv"
# metrics_file = "metrics.txt"

person_dirs = sorted([d for d in os.listdir(main_dir)
                      if os.path.isdir(os.path.join(main_dir, d))])

all_files = []   # full path of each signature file
person_map = []  # corresponding person label (index)

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

features = [] #for the flattened signature vectors

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
    
    features.append(data.flatten()) #flattening the (500,6) data into a 3000-dimensional vector
    pbar.update(1)

pbar.close()

features = np.array(features)
labels = np.array(person_map)
print("Shape of feature matrix:", features.shape, flush=True)

dunn_Index = compute_dunn_index(features, labels, pca_components=50)
print(f"Dunn Index: {dunn_Index:.4f}", flush=True)
