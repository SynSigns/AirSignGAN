import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
import warnings

def load_3d_signature(file_path):
    # --- Keep original implementation ---
    try:
        data = pd.read_csv(file_path, header=None, dtype=np.float64, on_bad_lines='warn')
        if data.shape[1] < 6:
            raise ValueError(f"File {file_path} has {data.shape[1]} columns, expected at least 6.")
        tip = data.iloc[:, :3].values
        tail = data.iloc[:, 3:6].values
        if np.isnan(tip).any() or np.isnan(tail).any():
            return None, None
        return tip, tail
    except ValueError as ve:
        return None, None
    except Exception as e:
        return None, None

def normalize_data(data):
    # --- Keep original implementation ---
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1e-8 # Avoid division by zero
    return (data - mean) / std

def compute_velocity(data):
    # --- Keep original implementation ---
    if data.shape[0] < 2: return np.array([])
    diffs = np.diff(data, axis=0)
    velocity = np.linalg.norm(diffs, axis=1)
    return velocity

def compute_acceleration(velocity):
    # --- Keep original implementation ---
    if velocity.size < 2: return np.array([])
    return np.diff(velocity)

def compute_curvature(data):
    # --- Keep original implementation ---
    if data.shape[0] < 3: return 0.0
    vectors = np.diff(data, axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid_indices = (norms > 1e-9).flatten()
    if not np.any(valid_indices): return 0.0
    vectors = vectors[valid_indices, :]
    norms = norms[valid_indices, :]
    if vectors.shape[0] < 2: return 0.0
    unit_vectors = vectors / norms
    dot_products = np.clip(np.sum(unit_vectors[1:] * unit_vectors[:-1], axis=1), -1.0, 1.0)
    angles = np.arccos(dot_products)
    total_curvature = np.sum(np.abs(angles))
    return total_curvature

def compute_histograms(a, b, bins=100):
    # --- Keep original implementation ---
    if a.size == 0 or b.size == 0:
        nan_arr = np.full(bins, np.nan)
        return nan_arr, nan_arr
    a_finite = a[np.isfinite(a)]
    b_finite = b[np.isfinite(b)]
    if a_finite.size == 0 or b_finite.size == 0:
        nan_arr = np.full(bins, np.nan)
        return nan_arr, nan_arr
    min_val = min(a_finite.min(), b_finite.min())
    max_val = max(a_finite.max(), b_finite.max())
    if np.isclose(min_val, max_val):
        min_val -= 0.5
        max_val += 0.5
    edges = np.linspace(min_val, max_val, bins + 1)
    pa, _ = np.histogram(a_finite, bins=edges, density=False)
    qa, _ = np.histogram(b_finite, bins=edges, density=False)
    pa_sum = pa.sum()
    qa_sum = qa.sum()
    pa_norm = pa / pa_sum if pa_sum > 0 else np.full_like(pa, np.nan, dtype=float)
    qa_norm = qa / qa_sum if qa_sum > 0 else np.full_like(qa, np.nan, dtype=float)
    return pa_norm, qa_norm

def kl_divergence(p, q):
    # --- Keep original implementation ---
    if np.any(np.isnan(p)) or np.any(np.isnan(q)): return np.nan
    epsilon = 1e-10
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p /= p.sum()
    q /= q.sum()
    kl_div = np.sum(p * np.log2(p / q))
    return kl_div # Can return inf

def jensen_shannon(p, q):
    # --- Keep original implementation ---
    if np.any(np.isnan(p)) or np.any(np.isnan(q)): return np.nan
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_sum = p.sum(); q_sum = q.sum()
    if not np.isclose(p_sum, 1.0): p = p / p_sum if p_sum > 0 else p
    if not np.isclose(q_sum, 1.0): q = q / q_sum if q_sum > 0 else q
    m = 0.5 * (p + q)
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    if np.isinf(kl_pm) or np.isinf(kl_qm): return np.nan
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return np.clip(jsd, 0.0, 1.0)

def is_outlier_velocity(velocity, z_thresh=3.0):
    # --- Keep original implementation ---
    if velocity.size < 2: return False
    mean_vel = np.mean(velocity)
    std_vel = np.std(velocity)
    if std_vel < 1e-8: return False
    z_scores = (velocity - mean_vel) / std_vel
    return np.any(np.abs(z_scores) > z_thresh)

# --- MODIFIED: Bounding Box Function ---
def compute_bounding_box_volume(points):
    """Calculates the volume of the axis-aligned bounding box containing the given points."""
    if points is None or points.shape[0] == 0:
        return np.nan

    try:
        # Ensure points is 2D (N x 3)
        if points.ndim != 2 or points.shape[1] != 3:
             # print(f"Warning: Invalid shape for bounding box points: {points.shape}")
             return np.nan # Expecting N x 3

        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Check if min/max calculation resulted in NaNs (if input had NaNs)
        if np.any(np.isnan(min_coords)) or np.any(np.isnan(max_coords)):
             return np.nan

        dimensions = max_coords - min_coords

        # If any dimension is zero or very small, volume is effectively zero
        if np.any(dimensions < 1e-9):
            return 0.0

        volume = np.prod(dimensions)
        # Check for NaN/inf just in case (e.g., if coords were infinite)
        if not np.isfinite(volume):
            return np.nan
        return volume
    except Exception as e:
        # print(f"Error calculating bounding box: {e}") # Optional debug
        return np.nan # Catch potential errors during calculation


# --- compute_shannon_entropy (Keep as before) ---
def compute_shannon_entropy(data, bins=100):
    """Compute Shannon entropy of a 1D array by binning. Handles NaNs."""
    data_finite = data[np.isfinite(data)]
    if data_finite.size == 0: return np.nan
    min_val = data_finite.min()
    max_val = data_finite.max()
    if np.isclose(min_val, max_val):
        return 0.0
    edges = np.linspace(min_val, max_val, bins + 1)
    counts, _ = np.histogram(data_finite, bins=edges, density=False)
    counts_sum = counts.sum()
    if counts_sum == 0: return 0.0
    p = counts / counts_sum
    p = p[p > 0]
    if p.size == 0: return 0.0
    return -np.sum(p * np.log2(p))
