import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# Assuming sign_stats_utils.py contains these functions and they work as expected
from sign_stats_utils import (
    load_3d_signature, normalize_data, compute_velocity,
    compute_acceleration, compute_curvature, kl_divergence,
    jensen_shannon, is_outlier_velocity, compute_bounding_box_volume,
    compute_shannon_entropy # Make sure this handles empty/NaN arrays
)

# Suppress ConvergenceWarning from KMeans
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Function Definitions ---
# (load_3d_signature, normalize_data, compute_velocity, compute_acceleration,
# compute_curvature, kl_divergence, jensen_shannon, is_outlier_velocity,
# compute_bounding_box_volume, compute_shannon_entropy - Keep as they are)

# --- collect_normalized_stats Function ---
# (Keep exactly as in the previous version)
def collect_normalized_stats(root_dir):
    # Lists for the 8 primary stats + 1 for combined bbox used only for filtering
    avg_tip_velocities = []
    avg_tip_accelerations = []
    avg_tail_velocities = []
    avg_tail_accelerations = []
    all_tip_curvatures = []
    all_tail_curvatures = []
    all_tip_bbox_volumes = []
    all_tail_bbox_volumes = []
    all_combined_bbox_volumes = [] # For filtering only

    file_ids = []
    skipped_files_loading = 0
    skipped_files_short = 0
    skipped_files_velocity = 0
    skipped_files_accel = 0
    skipped_files_filter = 0 # Generic threshold/NaN filter skips
    skipped_files_curvature = 0
    skipped_files_tip_bbox_nan = 0
    skipped_files_tail_bbox_nan = 0
    skipped_files_comb_bbox_nan = 0
    processed_files = 0

    VEL_THRESH = 5.0
    ACC_THRESH = 0.5

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                tip, tail = load_3d_signature(file_path)
                if tip is None or tail is None:
                    skipped_files_loading += 1
                    continue

                # --- Calculate all 3 BBox Volumes ---
                tip_bbox_volume = compute_bounding_box_volume(tip)
                tail_bbox_volume = compute_bounding_box_volume(tail)
                combined_bbox_volume = np.nan
                if tip.shape[0] > 0 and tail.shape[0] > 0:
                    try:
                         all_points = np.vstack((tip, tail))
                         combined_bbox_volume = compute_bounding_box_volume(all_points)
                    except ValueError:
                         combined_bbox_volume = np.nan

                # --- Existing Checks and Calculations ---
                if tip.shape[0] < 2 or tail.shape[0] < 2:
                    skipped_files_short += 1; continue
                tip_velocity = compute_velocity(tip)
                tail_velocity = compute_velocity(tail)
                if tip_velocity.size < 1 or tail_velocity.size < 1:
                    skipped_files_velocity += 1; continue
                if tip_velocity.size < 2 or tail_velocity.size < 2:
                    skipped_files_accel += 1; continue
                tip_acceleration = compute_acceleration(tip_velocity)
                tail_acceleration = compute_acceleration(tail_velocity)
                if tip_acceleration.size < 1 or tail_acceleration.size < 1:
                    skipped_files_accel += 1; continue
                if tip.shape[0] < 3 or tail.shape[0] < 3:
                    skipped_files_curvature += 1; continue # Need at least 3 points for curvature
                tip_curvature = compute_curvature(tip)
                tail_curvature = compute_curvature(tail)

                # Calculate means
                mean_tip_vel = np.nanmean(tip_velocity) if tip_velocity.size > 0 else np.nan
                mean_tail_vel = np.nanmean(tail_velocity) if tail_velocity.size > 0 else np.nan
                mean_tip_acc = np.nanmean(tip_acceleration) if tip_acceleration.size > 0 else np.nan
                mean_tail_acc = np.nanmean(tail_acceleration) if tail_acceleration.size > 0 else np.nan

                # --- Check ALL stats needed for analysis + combined_bbox for filtering ---
                if not np.isfinite(combined_bbox_volume):
                     skipped_files_comb_bbox_nan += 1; continue
                if not np.isfinite(mean_tip_vel): skipped_files_filter += 1; continue
                if not np.isfinite(mean_tip_acc): skipped_files_filter += 1; continue
                if not np.isfinite(mean_tail_vel): skipped_files_filter += 1; continue
                if not np.isfinite(mean_tail_acc): skipped_files_filter += 1; continue
                # Allow NaN curvature, handle later
                if not np.isfinite(tip_bbox_volume): skipped_files_tip_bbox_nan += 1; continue
                if not np.isfinite(tail_bbox_volume): skipped_files_tail_bbox_nan += 1; continue

                # Apply velocity/acceleration thresholds
                if abs(mean_tip_vel) > VEL_THRESH or abs(mean_tail_vel) > VEL_THRESH or \
                   abs(mean_tip_acc) > ACC_THRESH or abs(mean_tail_acc) > ACC_THRESH:
                    skipped_files_filter += 1; continue

                # --- If all checks pass, append all 9 stats ---
                avg_tip_velocities.append(mean_tip_vel)
                avg_tip_accelerations.append(mean_tip_acc)
                avg_tail_velocities.append(mean_tail_vel)
                avg_tail_accelerations.append(mean_tail_acc)
                all_tip_curvatures.append(tip_curvature if np.isfinite(tip_curvature) else np.nan)
                all_tail_curvatures.append(tail_curvature if np.isfinite(tail_curvature) else np.nan)
                all_tip_bbox_volumes.append(tip_bbox_volume)
                all_tail_bbox_volumes.append(tail_bbox_volume)
                all_combined_bbox_volumes.append(combined_bbox_volume)
                file_ids.append(os.path.basename(file_path))
                processed_files += 1

    total_skipped = (skipped_files_loading + skipped_files_short + skipped_files_velocity +
                     skipped_files_accel + skipped_files_curvature +
                     skipped_files_tip_bbox_nan + skipped_files_tail_bbox_nan + skipped_files_comb_bbox_nan +
                     skipped_files_filter)

    print(f"  Processed (passed initial checks): {processed_files}, Skipped: {total_skipped} ")
    # Optional: Keep detailed skip breakdown if useful

    if processed_files == 0:
        empty_array = np.array([])
        return (empty_array,) * 9 + ([],)

    return (
        np.array(avg_tip_velocities), np.array(avg_tip_accelerations),
        np.array(avg_tail_velocities), np.array(avg_tail_accelerations),
        np.array(all_tip_curvatures), np.array(all_tail_curvatures),
        np.array(all_tip_bbox_volumes), np.array(all_tail_bbox_volumes),
        np.array(all_combined_bbox_volumes), file_ids
    )


# --- plot_comparison (Keep as before) ---
def plot_comparison(original, generated, label, save_path, xlims, ylims):
    """Plots comparison histograms with FIXED axis ranges and NO numeric labels."""
    orig_finite = original[np.isfinite(original)]
    gen_finite = generated[np.isfinite(generated)]

    if orig_finite.size == 0 and gen_finite.size == 0: return
    plt.figure(figsize=(10, 6))

    if xlims and np.isfinite(xlims[0]) and np.isfinite(xlims[1]) and xlims[0] <= xlims[1]:
        common_bins = np.linspace(xlims[0], xlims[1], 101) if not np.isclose(xlims[0], xlims[1]) else np.array([xlims[0] - 0.5, xlims[1] + 0.5])
    else:
        all_finite = np.concatenate((orig_finite, gen_finite))
        if all_finite.size > 0: common_bins = 100
        else: plt.close(); return

    if orig_finite.size > 0: plt.hist(orig_finite, bins=common_bins, alpha=0.6, label='Original', density=True)
    if gen_finite.size > 0: plt.hist(gen_finite, bins=common_bins, alpha=0.6, label='Generated', density=True)

    plt.xlabel(label); plt.ylabel('Density')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    if xlims and np.isfinite(xlims[0]) and np.isfinite(xlims[1]): plt.xlim(xlims)
    if ylims and np.isfinite(ylims[0]) and np.isfinite(ylims[1]) and ylims[1] > ylims[0]: plt.ylim(ylims)
    plt.tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False, labelbottom=False, labelleft=False)

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout(); plt.savefig(save_path)
    except Exception as e: print(f"Error saving plot {save_path}: {e}")
    finally: plt.close()


# --- MODIFIED: calculate_comparison_metrics_and_plot Function ---
# Calculates KL & Silhouette, generates plots. Returns DICTIONARY with results.
def calculate_comparison_metrics_and_plot(
    orig_stats_filtered, gen_stats_filtered,
    generated_data_id, output_plot_dir, global_plot_ranges
):
    """
    Compares FILTERED datasets (8 stats), calculates KL & Silhouette metrics,
    and generates plots. Returns a dictionary with KL and Silhouette scores.
    """
    print(f"\n--- Calculating Comparison Metrics & Plotting vs: {generated_data_id} ---")

    # Unpack EIGHT filtered stats
    (orig_tip_vel, orig_tip_acc, orig_tail_vel, orig_tail_acc,
     orig_tip_curv, orig_tail_curv,
     orig_tip_bbox_vol, orig_tail_bbox_vol) = orig_stats_filtered
    (gen_tip_vel, gen_tip_acc, gen_tail_vel, gen_tail_acc,
     gen_tip_curv, gen_tail_curv,
     gen_tip_bbox_vol, gen_tail_bbox_vol) = gen_stats_filtered

    num_orig_samples = orig_tip_vel.size
    num_gen_samples = gen_tip_vel.size
    comparison_metrics = {"Generated_Data_ID": generated_data_id} # Key results by Gen ID

    # --- Silhouette Score Calculation (using 8 filtered features) ---
    silhouette_k_values = [2, 5, 10]
    for k in silhouette_k_values:
        comparison_metrics[f"Silhouette_k={k}"] = np.nan # Initialize

    min_samples_needed = max(silhouette_k_values) if silhouette_k_values else 2
    # Check if we have enough samples *overall* and *in each group* after potential NaN removal
    if num_orig_samples > 1 and num_gen_samples > 1 and (num_orig_samples + num_gen_samples) >= min_samples_needed :
        try:
            original_features = np.stack(orig_stats_filtered, axis=-1)
            generated_features = np.stack(gen_stats_filtered, axis=-1)
            combined_features = np.vstack((original_features, generated_features))

            if combined_features.shape[0] > 1 and combined_features.shape[1] == 8:
                finite_mask_combined = np.all(np.isfinite(combined_features), axis=1)
                num_finite_combined = np.sum(finite_mask_combined)
                # Also check if enough finite samples remain *within each group* for clustering validity
                num_finite_orig = np.sum(np.all(np.isfinite(original_features), axis=1))
                num_finite_gen = np.sum(np.all(np.isfinite(generated_features), axis=1))


                if num_finite_combined >= min_samples_needed and num_finite_orig > 0 and num_finite_gen > 0:
                    combined_features_finite = combined_features[finite_mask_combined,:]
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(combined_features_finite)
                    n_samples_for_clustering = scaled_features.shape[0]

                    for k in silhouette_k_values:
                        # Ensure k < number of samples *and* k >= 2
                        if k >= 2 and k < n_samples_for_clustering:
                            try:
                                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                cluster_labels = kmeans.fit_predict(scaled_features)
                                # Ensure at least 2 unique clusters were found
                                if len(np.unique(cluster_labels)) >= 2:
                                     score = silhouette_score(scaled_features, cluster_labels)
                                     comparison_metrics[f"Silhouette_k={k}"] = score
                                # else: print(f"    Sil k={k}: Only 1 cluster found.")
                            except Exception as e: print(f"    Error calculating Silhouette Score for k={k}: {e}")
                        # else: print(f"    Sil k={k}: Insufficient samples ({n_samples_for_clustering}) or invalid k.")
                # else: print(f"    Sil: Insufficient finite samples (Combined:{num_finite_combined}, Orig:{num_finite_orig}, Gen:{num_finite_gen}).")
            # else: print(f"    Sil: Incorrect combined shape ({combined_features.shape}) or samples.")
        except ValueError as ve: print(f"    Error during Silhouette prep (stacking/scaling): {ve}")
        except Exception as e: print(f"    Unexpected error during Silhouette calculation: {e}")
    # else: print("    Sil: Not enough initial samples (Orig:{num_orig_samples}, Gen:{num_gen_samples}).")


    # --- KL Divergence Calculation (using 8 filtered stats) ---
    metrics_map = { # Plot Label -> (orig_array, gen_array)
        "Tip Velocity": (orig_tip_vel, gen_tip_vel), "Tip Acceleration": (orig_tip_acc, gen_tip_acc),
        "Tail Velocity": (orig_tail_vel, gen_tail_vel), "Tail Acceleration": (orig_tail_acc, gen_tail_acc),
        "Tip Curvature": (orig_tip_curv, gen_tip_curv), "Tail Curvature": (orig_tail_curv, gen_tail_curv),
        "Tip Bounding Box Volume": (orig_tip_bbox_vol, gen_tip_bbox_vol),
        "Tail Bounding Box Volume": (orig_tail_bbox_vol, gen_tail_bbox_vol),
    }
    metric_names_base = [ # Base names for dict keys/CSV cols
        "Tip_Velocity", "Tip_Acceleration", "Tail_Velocity", "Tail_Acceleration",
        "Tip_Curvature", "Tail_Curvature", "Tip_Bounding_Box_Volume", "Tail_Bounding_Box_Volume"
    ]

    for name in metric_names_base:
        comparison_metrics[f"{name}_KL"] = np.nan # Initialize KL divergence results

    for (plot_label, (o_arr, g_arr)), base_name in zip(metrics_map.items(), metric_names_base):
        o_arr_finite = o_arr[np.isfinite(o_arr)]
        g_arr_finite = g_arr[np.isfinite(g_arr)]

        if o_arr_finite.size > 0 and g_arr_finite.size > 0:
            kl_bins = 100
            kl_range = global_plot_ranges.get(plot_label, {}).get('xlims')
            p, q = np.full(kl_bins, np.nan), np.full(kl_bins, np.nan)

            if kl_range and np.isfinite(kl_range[0]) and np.isfinite(kl_range[1]) and kl_range[0] <= kl_range[1]:
                 edges = np.linspace(kl_range[0], kl_range[1], kl_bins + 1) if not np.isclose(kl_range[0], kl_range[1]) else np.array([kl_range[0] - 0.5, kl_range[1] + 0.5])
            else:
                 combined_finite = np.concatenate((o_arr_finite, g_arr_finite))
                 min_val, max_val = np.min(combined_finite), np.max(combined_finite)
                 edges = np.linspace(min_val, max_val, kl_bins + 1) if not np.isclose(min_val, max_val) else np.array([min_val - 0.5, max_val + 0.5])

            pa_counts, _ = np.histogram(o_arr_finite, bins=edges, density=False)
            qa_counts, _ = np.histogram(g_arr_finite, bins=edges, density=False)
            pa_sum, qa_sum = pa_counts.sum(), qa_counts.sum()
            if pa_sum > 0: p = pa_counts / pa_sum
            if qa_sum > 0: q = qa_counts / qa_sum

            if not (np.any(np.isnan(p)) or np.any(np.isnan(q))):
                kl = kl_divergence(p, q)
                comparison_metrics[f"{base_name}_KL"] = kl if np.isfinite(kl) else np.inf # Store inf or finite value
        # else: print(f"    KL Skipped for {plot_label}: Insufficient finite data.")


    # --- Plot comparisons (8 plots) ---
    plot_dir_hist = os.path.join(output_plot_dir, "histograms") # Specific subdir for histograms
    os.makedirs(plot_dir_hist, exist_ok=True)
    plot_configs = [ # (Plot Label, Orig Data, Gen Data, Filename)
        ("Tip Velocity", orig_tip_vel, gen_tip_vel, "tip_velocity.png"),
        ("Tip Acceleration", orig_tip_acc, gen_tip_acc, "tip_acceleration.png"),
        ("Tail Velocity", orig_tail_vel, gen_tail_vel, "tail_velocity.png"),
        ("Tail Acceleration", orig_tail_acc, gen_tail_acc, "tail_acceleration.png"),
        ("Tip Curvature", orig_tip_curv, gen_tip_curv, "tip_curvature.png"),
        ("Tail Curvature", orig_tail_curv, gen_tail_curv, "tail_curvature.png"),
        ("Tip Bounding Box Volume", orig_tip_bbox_vol, gen_tip_bbox_vol, "tip_bounding_box_volume.png"),
        ("Tail Bounding Box Volume", orig_tail_bbox_vol, gen_tail_bbox_vol, "tail_bounding_box_volume.png"),
    ]

    print(f"  Generating {len(plot_configs)} comparison plots for {generated_data_id}...")
    for label, o_data, g_data, fname in plot_configs:
        ranges = global_plot_ranges.get(label, {})
        plot_comparison(o_data, g_data, label, os.path.join(plot_dir_hist, fname), xlims=ranges.get('xlims'), ylims=ranges.get('ylims'))

    return comparison_metrics # Return dict with KL and Silhouette scores

# --- print_min_max_stats Function ---
# (Keep exactly as in the previous version)
def print_min_max_stats(dataset_name, stats_tuple):
    """Prints the min and max values for each of the 8 statistics in the filtered tuple."""
    metric_names = [ # Now 8 stats
        "Tip Velocity", "Tip Acceleration", "Tail Velocity",
        "Tail Acceleration", "Tip Curvature", "Tail Curvature",
        "Tip Bounding Box Volume", "Tail Bounding Box Volume"
    ]
    print(f"\n--- Min/Max Stats for: {dataset_name} (Filtered Data) ---")
    if not stats_tuple or len(stats_tuple) != len(metric_names):
        print(f"  Error: Invalid stats tuple provided (length {len(stats_tuple) if stats_tuple else 0}). Expected 8 stats.")
        print("-" * (len(dataset_name) + 25))
        return

    for i, name in enumerate(metric_names):
        arr = stats_tuple[i]
        if isinstance(arr, np.ndarray) and arr.size > 0:
             finite_arr = arr[np.isfinite(arr)]
             if finite_arr.size > 0:
                 min_val, max_val = np.min(finite_arr), np.max(finite_arr)
                 print(f"  {name}: Min={min_val:.4f}, Max={max_val:.4f}")
             else: print(f"  {name}: No finite values after filtering.")
        else: print(f"  {name}: No data points or invalid array.")
    print("-" * (len(dataset_name) + 25))


# ==========================
# === Main Execution Block ===
# ==========================

# === Customize paths here ===
# original_data_dir = "/home/alpha/Workbenches/Bora/AirSignatures/AirSigns/AirSignsShuffledBothBalls/Train"
# comparisons_to_run = [
#     ("/home/alpha/Workbenches/Bora/AirSignatures/Gen_Latest/SliTCNN_DataGen_VAE_skip_conn_txt", "SliTCNN_DataGen_VAE_skip_conn"),
#     ("/home/alpha/Workbenches/Bora/AirSignatures/Gen_Latest/SliTCNN_DataGen_VAE_Sep_Conv_skip_conn_txt", "SliTCNN_DataGen_VAE_Sep_Conv_skip_conn"),
#     ("/home/alpha/Workbenches/Bora/AirSignatures/Gen_Latest/SliTCNN_DataGen_VAE_txt", "SliTCNN_DataGen_VAE")
# ]
# output_dir_base = "/home/alpha/Workbenches/Bora/AirSignatures/ComparisonPlots/3D_TipTail_Stats_RefactoredCSV_v2" # New base dir

original_data_dir = "/mnt/MIG_store/Datasets/air-signatures/AirSigns/AirSignsShuffledBothBalls/Train"
comparisons_to_run = [
    #("/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/attentioncnn/Test", "AirSignGAN"),
    #("/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/attentionlstm/Test", "AirSignGAN"),
    #("/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/basiccnn/Test", "AirSignGAN"),
    ("/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/basicgru/Test", "AirSignGAN")
]
output_dir_base = "/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/ComparisonPlots/basicgru"


# Define specific output CSV filenames - MORE GRANULAR
output_entropy_csv = os.path.join(output_dir_base, "summary_entropy.csv")
output_kl_divergence_csv = os.path.join(output_dir_base, "summary_kl_divergence.csv") # NEW KL CSV
output_silhouette_scores_csv = os.path.join(output_dir_base, "summary_silhouette_scores.csv") # NEW SILHOUETTE CSV
output_avg_stats_csv = os.path.join(output_dir_base, "summary_average_stats.csv")
# ---

os.makedirs(output_dir_base, exist_ok=True)

# --- STEP 1: Load ALL Raw Stats Data ---
print("--- STEP 1: Loading Raw Stats Data ---")
all_raw_stats_data = {}
data_dirs_to_process = [original_data_dir] + [gen_dir for gen_dir, _ in comparisons_to_run]
dataset_ids = {}
for data_dir in data_dirs_to_process:
    data_id = 'Original' if data_dir == original_data_dir else os.path.basename(data_dir)
    dataset_ids[data_dir] = data_id
    print(f"Loading stats from: {data_id} ({data_dir})")
    stats_tuple = collect_normalized_stats(data_dir)
    all_raw_stats_data[data_dir] = stats_tuple

# --- STEP 2: Determine Global BBox Percentiles (Using COMBINED BBox) ---
# (Keep Step 2 exactly as in the previous version - determines lower/upper bounds)
print("\n--- STEP 2: Determining BBox Percentile Filter (Using Combined BBox Volume) ---")
all_combined_bbox_volumes_list = []
for data_dir, stats_tuple in all_raw_stats_data.items():
    if stats_tuple and len(stats_tuple) > 8 and stats_tuple[8] is not None:
        combined_bbox_vol_arr = stats_tuple[8]
        finite_combined_bbox = combined_bbox_vol_arr[np.isfinite(combined_bbox_vol_arr)]
        if finite_combined_bbox.size > 0:
            all_combined_bbox_volumes_list.append(finite_combined_bbox)

if not all_combined_bbox_volumes_list: exit("ERROR: No valid COMBINED bounding box volumes found. Cannot filter.")
all_combined_bbox_volumes_combined = np.concatenate(all_combined_bbox_volumes_list)
if all_combined_bbox_volumes_combined.size == 0: exit("ERROR: Combined bounding box volumes array is empty after removing NaNs.")

lower_bound, upper_bound = -np.inf, np.inf
if all_combined_bbox_volumes_combined.size >= 20:
    lower_bound = np.percentile(all_combined_bbox_volumes_combined, 0)
    upper_bound = np.percentile(all_combined_bbox_volumes_combined, 90)
    print(f"Global COMBINED BBox 5th percentile: {lower_bound:.4f}")
    print(f"Global COMBINED BBox 95th percentile: {upper_bound:.4f}")
    if np.isclose(lower_bound, upper_bound): print("Warning: COMBINED BBox 5th/95th percentiles are very close.")
    if lower_bound > upper_bound: print(f"Warning: Lower bound > Upper bound. Using full range."); lower_bound, upper_bound = -np.inf, np.inf
else: print("Warning: Not enough COMBINED bounding box data (< 20). Using full range.")


# --- STEP 3: Create Filtered Datasets (Applying Combined BBox Filter) ---
# (Keep Step 3 exactly as in the previous version - creates all_filtered_stats_data)
print("\n--- STEP 3: Applying Combined BBox Filter to Datasets ---")
all_filtered_stats_data = {}
for data_dir, raw_stats_tuple in all_raw_stats_data.items():
    dataset_name = dataset_ids[data_dir]
    if not raw_stats_tuple or len(raw_stats_tuple) < 10:
        print(f"  Warning: Invalid raw stats tuple for {dataset_name}. Skipping filtering.")
        all_filtered_stats_data[data_dir] = tuple([np.array([])] * 8)
        continue

    combined_bbox_vol_arr = raw_stats_tuple[8]
    if combined_bbox_vol_arr is None or combined_bbox_vol_arr.size == 0:
         print(f"  {dataset_name}: No combined BBox data for filtering. Result empty.")
         all_filtered_stats_data[data_dir] = tuple([np.array([])] * 8); continue

    num_before_filter = combined_bbox_vol_arr.size
    finite_mask = np.isfinite(combined_bbox_vol_arr)
    bounds_mask = (combined_bbox_vol_arr >= lower_bound) & (combined_bbox_vol_arr <= upper_bound)
    combined_mask = np.zeros_like(combined_bbox_vol_arr, dtype=bool)
    combined_mask[finite_mask] = bounds_mask[finite_mask]

    filtered_stat_arrays = []
    valid_lengths = True
    raw_stat_arrays_for_analysis = raw_stats_tuple[:8]
    for i, arr in enumerate(raw_stat_arrays_for_analysis):
        if isinstance(arr, np.ndarray) and arr.size == num_before_filter:
            filtered_stat_arrays.append(arr[combined_mask])
        else:
            print(f"  ERROR/Warning: Mismatch/Invalid array for {dataset_name} (Stat {i}). Appending empty.")
            filtered_stat_arrays.append(np.array([]))
            if isinstance(arr, np.ndarray) and arr.size != num_before_filter: valid_lengths = False

    num_after_filter = filtered_stat_arrays[0].size if filtered_stat_arrays and isinstance(filtered_stat_arrays[0], np.ndarray) else 0
    print(f"  {dataset_name}: Kept {num_after_filter} samples out of {num_before_filter}")
    if not valid_lengths: print(f"  WARNING: Length mismatch occurred for {dataset_name}.")
    all_filtered_stats_data[data_dir] = tuple(filtered_stat_arrays)


# --- STEP 3.5: Print Min/Max for ALL Filtered Datasets ---
# (Keep Step 3.5 exactly as in the previous version - calls print_min_max_stats)
print("\n--- STEP 3.5: Printing Min/Max values for all datasets (Filtered Data, 8 Stats) ---")
for data_dir, filtered_stats_tuple in all_filtered_stats_data.items():
    dataset_name = dataset_ids[data_dir]
    print_min_max_stats(dataset_name, filtered_stats_tuple)


# --- STEP 4: Calculate Entropy & Average Stats for ALL Filtered Datasets ---
# (Keep Step 4 exactly as in the previous version - calculates all_entropy_results, all_average_results)
print("\n--- STEP 4: Calculating Entropy and Average Stats (Filtered Data, 8 Stats) ---")
metric_names_base = [
    "Tip_Velocity", "Tip_Acceleration", "Tail_Velocity", "Tail_Acceleration",
    "Tip_Curvature", "Tail_Curvature", "Tip_Bounding_Box_Volume", "Tail_Bounding_Box_Volume"
]
all_entropy_results = {}
all_average_results = {}

for data_dir, filtered_stats_tuple in all_filtered_stats_data.items():
    dataset_id = dataset_ids[data_dir]
    print(f"Calculating for: {dataset_id}")
    entropy_dict = {'Dataset_ID': dataset_id}
    average_dict = {'Dataset_ID': dataset_id}

    if not filtered_stats_tuple or len(filtered_stats_tuple) != 8:
        print(f"  Warning: Invalid filtered stats for {dataset_id}. Storing NaNs.")
        for name in metric_names_base:
            entropy_dict[f"{name}_Entropy"] = np.nan
            average_dict[name] = np.nan
    else:
        for i, name in enumerate(metric_names_base):
            arr = filtered_stats_tuple[i]
            if isinstance(arr, np.ndarray):
                 entropy_dict[f"{name}_Entropy"] = compute_shannon_entropy(arr)
                 finite_arr = arr[np.isfinite(arr)]
                 average_dict[name] = np.mean(finite_arr) if finite_arr.size > 0 else np.nan
            else:
                 print(f"  Warning: Data for {name} in {dataset_id} not array. Storing NaNs.")
                 entropy_dict[f"{name}_Entropy"] = np.nan
                 average_dict[name] = np.nan

    all_entropy_results[dataset_id] = entropy_dict
    all_average_results[dataset_id] = average_dict


# --- STEP 5: Pre-calculate Global Plot Ranges (using FILTERED data, 8 stats) ---
# (Keep Step 5 exactly as in the previous version - calculates final_global_plot_ranges)
print("\n--- STEP 5: Pre-calculating Global Plot Ranges (from Filtered Data, 8 Stats) ---")
metric_plot_labels = [
    "Tip Velocity", "Tip Acceleration", "Tail Velocity", "Tail Acceleration",
    "Tip Curvature", "Tail Curvature", "Tip Bounding Box Volume", "Tail Bounding Box Volume"
]
global_plot_ranges = {label: {'min': float('inf'), 'max': float('-inf'), 'max_density': 0.0} for label in metric_plot_labels}
num_bins_for_range_calc = 100

print("Calculating global min/max for x-axes...")
for i, plot_label in enumerate(metric_plot_labels):
    all_values_filtered_list = []
    for filtered_stats_tuple in all_filtered_stats_data.values():
        if filtered_stats_tuple and i < len(filtered_stats_tuple):
            arr = filtered_stats_tuple[i]
            if isinstance(arr, np.ndarray):
                 finite_arr = arr[np.isfinite(arr)]
                 if finite_arr.size > 0: all_values_filtered_list.append(finite_arr)
    if not all_values_filtered_list:
        print(f"  Warning: No finite data for '{plot_label}'.")
        global_plot_ranges[plot_label]['min'], global_plot_ranges[plot_label]['max'] = np.nan, np.nan
        continue
    combined_values_filtered = np.concatenate(all_values_filtered_list)
    if combined_values_filtered.size > 1:
        min_val, max_val = np.percentile(combined_values_filtered, 1), np.percentile(combined_values_filtered, 99)
        padding = 0.5 if np.isclose(min_val, max_val) else (max_val - min_val) * 0.05
        calc_min, calc_max = min_val - padding, max_val + padding
        if plot_label in ["Tip Bounding Box Volume", "Tail Bounding Box Volume"]: calc_min = max(0.0, calc_min)
        global_plot_ranges[plot_label]['min'], global_plot_ranges[plot_label]['max'] = calc_min, calc_max
        if global_plot_ranges[plot_label]['min'] > global_plot_ranges[plot_label]['max']: global_plot_ranges[plot_label]['min'], global_plot_ranges[plot_label]['max'] = np.nan, np.nan
        if np.isfinite(global_plot_ranges[plot_label]['min']) and np.isfinite(global_plot_ranges[plot_label]['max']): print(f"  {plot_label}: Range [{global_plot_ranges[plot_label]['min']:.4f}, {global_plot_ranges[plot_label]['max']:.4f}]")
        else: print(f"  {plot_label}: Could not determine finite range.")
    elif combined_values_filtered.size == 1:
        val = combined_values_filtered[0]; calc_min, calc_max = val - 0.5, val + 0.5
        if plot_label in ["Tip Bounding Box Volume", "Tail Bounding Box Volume"]: calc_min = max(0.0, calc_min)
        global_plot_ranges[plot_label]['min'], global_plot_ranges[plot_label]['max'] = calc_min, calc_max
        print(f"  {plot_label}: Range [{global_plot_ranges[plot_label]['min']:.4f}, {global_plot_ranges[plot_label]['max']:.4f}] (single point)")
    else: global_plot_ranges[plot_label]['min'], global_plot_ranges[plot_label]['max'] = np.nan, np.nan

print("Calculating global max densities for y-axes...")
for i, plot_label in enumerate(metric_plot_labels):
    xlims = (global_plot_ranges[plot_label]['min'], global_plot_ranges[plot_label]['max'])
    if not (np.isfinite(xlims[0]) and np.isfinite(xlims[1]) and xlims[0] <= xlims[1]):
        global_plot_ranges[plot_label]['max_density'] = 1.0; continue
    hist_bins = np.linspace(xlims[0], xlims[1], num_bins_for_range_calc + 1) if not np.isclose(xlims[0], xlims[1]) else np.array([xlims[0] - 0.5, xlims[1] + 0.5])
    max_density_found = 0.0
    for filtered_stats_tuple in all_filtered_stats_data.values():
        if filtered_stats_tuple and i < len(filtered_stats_tuple):
            data_arr = filtered_stats_tuple[i]
            if isinstance(data_arr, np.ndarray):
                 finite_data = data_arr[np.isfinite(data_arr)]
                 if finite_data.size > 0:
                    try:
                        hist_counts, _ = np.histogram(finite_data, bins=hist_bins, density=True)
                        max_density_found = max(max_density_found, np.max(hist_counts) if hist_counts.size > 0 else 0.0)
                    except Exception as e: print(f"  Error calculating histogram for {plot_label}: {e}")
    global_plot_ranges[plot_label]['max_density'] = max(max_density_found * 1.1, 1e-9) # Add padding, ensure > 0
    print(f"  {plot_label}: Max Density ~{global_plot_ranges[plot_label]['max_density']:.4f}")

final_global_plot_ranges = {}
for plot_label in metric_plot_labels:
    ranges = global_plot_ranges.get(plot_label, {})
    min_x, max_x = ranges.get('min', np.nan), ranges.get('max', np.nan)
    max_y = ranges.get('max_density', 1.0)
    final_xlims = (min_x, max_x) if (np.isfinite(min_x) and np.isfinite(max_x) and min_x <= max_x) else None
    final_ylims = (0, max_y) if (final_xlims is not None and np.isfinite(max_y) and max_y > 1e-9) else None
    final_global_plot_ranges[plot_label] = {'xlims': final_xlims, 'ylims': final_ylims}
print("--- Global Plot Range Calculation Complete ---")


# --- STEP 6: Run Comparisons (KL/Silhouette) using Filtered Data ---
# MODIFIED: Now collects KL and Silhouette scores into SEPARATE dictionaries
print("\n--- STEP 6: Calculating Comparison Metrics (KL/Silhouette) & Plotting ---")
all_kl_results = {}        # Stores KL results keyed by generated ID
all_silhouette_results = {} # Stores Silhouette results keyed by generated ID

filtered_orig_stats = all_filtered_stats_data.get(original_data_dir)
if not filtered_orig_stats or len(filtered_orig_stats) != 8:
    exit(f"FATAL ERROR: Filtered original data not found/invalid for {original_data_dir}.")
if not any(isinstance(arr, np.ndarray) and arr.size > 0 for arr in filtered_orig_stats):
    print("WARNING: Original dataset appears empty after filtering.")

# Iterate through the generated datasets to compare against original
for gen_dir, output_subdir_name in comparisons_to_run:
    generated_id = dataset_ids[gen_dir]
    plot_dir_for_gen = os.path.join(output_dir_base, output_subdir_name)
    os.makedirs(plot_dir_for_gen, exist_ok=True) # Ensure plot subdir exists

    filtered_gen_stats = all_filtered_stats_data.get(gen_dir)
    kl_entry = {'Generated_Data_ID': generated_id} # Prepare dict for KL
    sil_entry = {'Generated_Data_ID': generated_id} # Prepare dict for Silhouette

    if not filtered_gen_stats or len(filtered_gen_stats) != 8:
        print(f"WARNING: Filtered data invalid for {generated_id}. Skipping comparison.")
        # Store NaNs if skipping
        for name in metric_names_base: kl_entry[f"{name}_KL"] = np.nan
        for k in [2, 5, 10]: sil_entry[f"Silhouette_k={k}"] = np.nan
        all_kl_results[generated_id] = kl_entry
        all_silhouette_results[generated_id] = sil_entry
        continue

    if not any(isinstance(arr, np.ndarray) and arr.size > 0 for arr in filtered_gen_stats):
         print(f"WARNING: Generated dataset {generated_id} appears empty after filtering.")

    # Call comparison function - returns dict with BOTH KL and Silhouette
    comparison_results = calculate_comparison_metrics_and_plot(
        filtered_orig_stats, filtered_gen_stats,
        generated_id, plot_dir_for_gen, final_global_plot_ranges
    )

    # --- Extract KL scores ---
    for name in metric_names_base:
        kl_entry[f"{name}_KL"] = comparison_results.get(f"{name}_KL", np.nan)
    all_kl_results[generated_id] = kl_entry

    # --- Extract Silhouette scores ---
    for k in [2, 5, 10]:
        sil_entry[f"Silhouette_k={k}"] = comparison_results.get(f"Silhouette_k={k}", np.nan)
    all_silhouette_results[generated_id] = sil_entry


# --- STEP 7: Assemble and Save Final CSVs ---
print("\n--- STEP 7: Assembling and Saving Results ---")

# --- Save Entropy CSV --- (Unchanged from previous version)
entropy_data_list = list(all_entropy_results.values())
if entropy_data_list:
    print(f"Saving Entropy results to {output_entropy_csv}...")
    entropy_df = pd.DataFrame(entropy_data_list)
    entropy_cols = ['Dataset_ID'] + [f"{name}_Entropy" for name in metric_names_base]
    for col in entropy_cols:
        if col not in entropy_df.columns: entropy_df[col] = np.nan
    entropy_df = entropy_df[entropy_cols]
    try:
        entropy_df['sort_key'] = entropy_df['Dataset_ID'].apply(lambda x: 0 if x == 'Original' else 1)
        entropy_df = entropy_df.sort_values(by=['sort_key', 'Dataset_ID']).drop(columns='sort_key')
        entropy_df.to_csv(output_entropy_csv, index=False, na_rep='NaN', float_format='%.6f')
        print(f"Entropy CSV file saved successfully.")
    except Exception as e: print(f"Error saving Entropy CSV: {e}")
else: print("No Entropy results generated.")

# --- Save KL Divergence CSV --- (NEW)
kl_data_list = list(all_kl_results.values())
if kl_data_list:
    print(f"Saving KL Divergence results to {output_kl_divergence_csv}...")
    kl_df = pd.DataFrame(kl_data_list)
    kl_cols = ['Generated_Data_ID'] + [f"{name}_KL" for name in metric_names_base]
    for col in kl_cols:
         if col not in kl_df.columns: kl_df[col] = np.nan
    kl_df = kl_df[kl_cols] # Select/reorder
    try:
        kl_df = kl_df.sort_values(by=['Generated_Data_ID'])
        kl_df = kl_df.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN
        kl_df.to_csv(output_kl_divergence_csv, index=False, na_rep='NaN', float_format='%.6f')
        print(f"KL Divergence CSV file saved successfully.")
    except Exception as e: print(f"Error saving KL Divergence CSV: {e}")
else: print("No KL Divergence results generated.")

# --- Save Silhouette Score CSV --- (NEW)
sil_data_list = list(all_silhouette_results.values())
if sil_data_list:
    print(f"Saving Silhouette Score results to {output_silhouette_scores_csv}...")
    sil_df = pd.DataFrame(sil_data_list)
    sil_cols = ['Generated_Data_ID', 'Silhouette_k=2', 'Silhouette_k=5', 'Silhouette_k=10']
    for col in sil_cols:
        if col not in sil_df.columns: sil_df[col] = np.nan
    sil_df = sil_df[sil_cols] # Select/reorder
    try:
        sil_df = sil_df.sort_values(by=['Generated_Data_ID'])
        sil_df.to_csv(output_silhouette_scores_csv, index=False, na_rep='NaN', float_format='%.6f')
        print(f"Silhouette Score CSV file saved successfully.")
    except Exception as e: print(f"Error saving Silhouette Score CSV: {e}")
else: print("No Silhouette Score results generated.")


# --- Save Average Stats CSV --- (Unchanged from previous version)
avg_data_list = list(all_average_results.values())
if avg_data_list:
    print(f"Saving Average Stat results to {output_avg_stats_csv}...")
    avg_stats_df = pd.DataFrame(avg_data_list)
    avg_cols = ['Dataset_ID'] + metric_names_base
    for col in avg_cols:
        if col not in avg_stats_df.columns: avg_stats_df[col] = np.nan
    avg_stats_df = avg_stats_df[avg_cols]
    try:
        avg_stats_df['sort_key'] = avg_stats_df['Dataset_ID'].apply(lambda x: 0 if x == 'Original' else 1)
        avg_stats_df = avg_stats_df.sort_values(by=['sort_key', 'Dataset_ID']).drop(columns='sort_key')
        avg_stats_df.to_csv(output_avg_stats_csv, index=False, na_rep='NaN', float_format='%.6f')
        print(f"Average Stats CSV file saved successfully.")
    except Exception as e: print(f"Error saving Average Stats CSV: {e}")
else: print("No average stat results were generated.")


print("\n--- Script Finished ---")
