# %%
# !pip install -e ..
# !pip install numpy pandas matplotlib scikit-learn pymbar openTSNE umap-learn scipy
# ==============================================================================
# INITIALIZATION AND SETUP
# ==============================================================================
import os
import sys
import time
import tarfile
import gzip
import shutil
import platform
from itertools import product
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

# Detect Google Colab environment
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Setup directories and paths
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    base_dir = '/content/drive/MyDrive/Au_experiment'
    sys.path.append(base_dir)
    data_dir = base_dir
    results_dir_umap = os.path.join(base_dir, "outputs", "umap_results")
    results_dir_fitsne = os.path.join(base_dir, "outputs", "fitsne_results")
    results_dir_sketch = os.path.join(base_dir, "outputs", "sketchmap_results")
    results_dir_mbar = os.path.join(base_dir, "outputs", "mbar_results")
    plots_dir = os.path.join(base_dir, "outputs", "plots")
else:
    base_dir = r"g:\mp\Decoding-Au-clusters\Au13_study"
    reproduce_dir = os.path.join(base_dir, "reproduce_paper")
    data_dir = os.path.join(reproduce_dir, "data")
    results_dir_umap = os.path.join(data_dir, "umap_results")
    results_dir_fitsne = os.path.join(data_dir, "fitsne_results")
    results_dir_sketch = os.path.join(data_dir, "sketchmap_results")
    results_dir_mbar = os.path.join(data_dir, "mbar_results")
    plots_dir = os.path.join(reproduce_dir, "plots")

# Ensure required directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(results_dir_umap, exist_ok=True)
os.makedirs(results_dir_fitsne, exist_ok=True)
os.makedirs(results_dir_sketch, exist_ok=True)
os.makedirs(results_dir_mbar, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Now import dimredpy framework components
from dimredpy.umap_embed import umap_embed
from dimredpy.fitsne import fit_sne
from dimredpy.sketchmap import select_landmarks, sketch_map, project_out_of_sample
from dimredpy.shared.transfer import make_transfer
from dimredpy.mbar import mbar_free_energy_surface
from dimredpy.shared.io import load_spatial_coordinates
from dimredpy.shared.descriptors import compute_trajectory_descriptors

print(f"Environment Initialized. In Colab: {IN_COLAB}, OS: {platform.system()}")

# %%
# ==============================================================================
# STEP 0: EXTRACT AND PREPARE DATA
# ==============================================================================
print("\n=== STEP 0: Extracting Data and Preparing Dataset ===")
tar_path = os.path.join(base_dir, "full_trajectory.xyz.CN_for_SketchMap.tar 1.gz")
extracted_txt_path = os.path.join(data_dir, "full_trajectory.xyz.CN_for_SketchMap")
npy_path = os.path.join(data_dir, "dataset_high_dim.npy")

# Extraction with checkpointing
if os.path.exists(extracted_txt_path):
    print(f"-> Coordinates text file already exists at: {extracted_txt_path}")
else:
    print(f"-> Extracting archive: {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("-> Extraction successful via tarfile module.")
    except Exception as e:
        print(f"-> Tarfile extraction failed: {e}. Trying fallback gzip extraction...")
        with gzip.open(tar_path, 'rb') as f_in:
            with open(extracted_txt_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("-> Extraction successful via gzip fallback.")

# Dataset conversion with checkpointing
if os.path.exists(npy_path):
    print(f"-> Dataset NumPy file already exists at: {npy_path}. Loading...")
    X_global = np.load(npy_path)
else:
    print("-> Converting text coordinate file into NumPy binary...")
    df = pd.read_csv(extracted_txt_path, delim_whitespace=True, header=None, usecols=range(13))
    X_global = df.values.astype(np.float32)
    np.save(npy_path, X_global)
    print(f"-> Saved high-dimensional dataset of shape {X_global.shape} to {npy_path}")

# Copy auxiliary thermodynamic inputs if running locally
if not IN_COLAB:
    su_dir = os.path.join(base_dir, "papaer_code", "su")
    mbar_files = ["temperatures", "replica-indices", "potencial-energies_minus", "potencial-energies_neutral", "potencial-energies_plus"]
    copied_count = 0
    for f in mbar_files:
        src = os.path.join(su_dir, f)
        dst = os.path.join(data_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied_count += 1
    if copied_count > 0:
        print(f"-> Copied {copied_count} MBAR thermodynamic files from {su_dir} to {data_dir}")

print(f"-> STEP 0 Completed. Dataset shape: {X_global.shape}")

# %%
# ==============================================================================
# STEP 1: UMAP CROSS VALIDATION
# ==============================================================================
print("\n=== STEP 1: UMAP Cross Validation ===")

def calcular_stress_amostral(X_high, X_low, sample_size=10000):
    n = X_high.shape[0]
    idx = np.arange(n) if n <= sample_size else np.random.choice(n, sample_size, replace=False)
    D_high = pairwise_distances(X_high[idx], metric="euclidean")
    D_low = pairwise_distances(X_low[idx], metric="euclidean")
    D_high = (D_high - D_high.min()) / (D_high.max() - D_high.min())
    D_low = (D_low - D_low.min()) / (D_low.max() - D_low.min())
    return np.sqrt(np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2))

def trustworthiness_sampled(X_high, X_low, sample_size=10000, n_neighbors=5, random_state=42):
    rng = check_random_state(random_state)
    n = X_high.shape[0]
    idx = np.arange(n) if n <= sample_size else rng.choice(n, sample_size, replace=False)
    return trustworthiness(X_high[idx], X_low[idx], n_neighbors=n_neighbors)

neigh_values = range(5, 51, 5)
dist_values = [0.001, 0.005, 0.01, 0.05, 0.1]
amostra_stress = 10000

def process_umap_combination(args):
    neigh, dis, amostra_stress_val, X = args
    pasta = os.path.join(results_dir_umap, f"neigh_{neigh}_dist_{dis}")
    os.makedirs(pasta, exist_ok=True)
    output_file = os.path.join(pasta, "embedding.dat")
    metrics_file = os.path.join(pasta, "metrics.txt")

    # Skip already completed combinations
    if os.path.exists(output_file) and os.path.exists(metrics_file):
        print(f"-> Skipping UMAP (n_neighbors={neigh}, min_dist={dis}) - already completed.")
        return

    temp_output_file = output_file + ".tmp"
    temp_metrics_file = metrics_file + ".tmp"

    import warnings
    warnings.filterwarnings("ignore", message="divide by zero encountered in power")
    warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")
    
    print(f"-> Running UMAP (n_neighbors={neigh}, min_dist={dis})...")
    start = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = umap_embed(data=X, n_neighbors=neigh, min_dist=dis, n_components=2, metric='mahalanobis', seed=42, use_gpu=True, spread=0.1)
        X_umap = res["embedding"]
        elapsed = time.time() - start

        pd.DataFrame(X_umap).to_csv(temp_output_file, sep='\t', header=False, index=False, float_format='%.6f')
        trust = trustworthiness_sampled(X, X_umap, sample_size=amostra_stress_val, n_neighbors=neigh)
        stress = calcular_stress_amostral(X, X_umap, sample_size=amostra_stress_val)

        with open(temp_metrics_file, "w") as f:
            f.write(f"n_neighbors = {neigh}\nmin_dist = {dis}\nTempo (s) = {elapsed:.2f}\n")
            f.write(f"Trustworthiness = {trust:.6f}\nStress = {stress:.6f}\n")
            
        os.replace(temp_output_file, output_file)
        os.replace(temp_metrics_file, metrics_file)
        print(f"   Success! Time: {elapsed:.2f}s | Trust: {trust:.4f} | Stress: {stress:.4f}")
    except Exception as e:
        print(f"   Failed running UMAP for (neigh={neigh}, dist={dis}): {e}")

umap_combinations = list(product(neigh_values, dist_values))
umap_args_list = [(neigh, dis, amostra_stress, X_global) for neigh, dis in umap_combinations]

# GPU tasks (UMAP, FIt-SNE) are memory heavy. Limit concurrent workers to avoid OOM.
is_windows = platform.system() == "Windows"
num_gpu_workers = 1 if is_windows else (4 if IN_COLAB else max(1, multiprocessing.cpu_count() // 2))

if num_gpu_workers <= 1:
    for args in umap_args_list:
        process_umap_combination(args)
else:
    print(f"-> Running UMAP parallel search using {num_gpu_workers} concurrent GPU workers (Threads)...")
    from multiprocessing.pool import ThreadPool
    with ThreadPool(processes=num_gpu_workers) as pool:
        pool.map(process_umap_combination, umap_args_list)

print("-> STEP 1 Completed.")

# %%
# ==============================================================================
# STEP 2: FIT-SNE CROSS VALIDATION
# ==============================================================================
print("\n=== STEP 2: FIt-SNE Cross Validation ===")
perp_values = [200, 250]
k_values = [150, 200, 250, 300]
max_int_values = [1000, 1500]
ee_coef_values = [1000, 1500]

# --- Phase 2a: GPU cross-validation (cuML, df=1.0) to find best combo fast ---
def process_fitsne_combination(args):
    perp, max_inter, k, ee, X = args
    pasta = os.path.join(results_dir_fitsne, f"perp{perp}_max{max_inter}_k{k}_ee{ee}")
    os.makedirs(pasta, exist_ok=True)
    output_file = os.path.join(pasta, "embedding.dat")

    # Skip already completed combinations
    if os.path.exists(output_file):
        print(f"-> Skipping FIt-SNE (perp={perp}, max_iter={max_inter}, K={k}, ee={ee}) - already completed.")
        return

    temp_output_file = output_file + ".tmp"

    print(f"-> Running FIt-SNE (perp={perp}, max_iter={max_inter}, K={k}, ee={ee})...")
    start = time.time()
    try:
        res = fit_sne(
            data=X, perplexity=perp, n_iter=max_inter, early_exaggeration_iter=800,
            learning_rate="auto", early_exaggeration=ee, min_num_intervals=50,
            use_gpu=True,
            # cuML-specific: use FFT method
            method='fft',
        )
        X_tsne = res["embedding"]
        pd.DataFrame(X_tsne).to_csv(temp_output_file, sep='\t', header=False, index=False, float_format='%.6f')
        os.replace(temp_output_file, output_file)
        elapsed = time.time() - start
        print(f"   Success! Time: {elapsed:.2f}s")
    except Exception as e:
        print(f"   Failed running FIt-SNE: {e}")

fitsne_combinations = list(product(perp_values, max_int_values, k_values, ee_coef_values))
fitsne_args_list = [(perp, max_inter, k, ee, X_global) for perp, max_inter, k, ee in fitsne_combinations]

if num_gpu_workers <= 1:
    for args in fitsne_args_list:
        process_fitsne_combination(args)
else:
    print(f"-> Running FIt-SNE parallel search using {num_gpu_workers} concurrent GPU workers (Threads)...")
    from multiprocessing.pool import ThreadPool
    with ThreadPool(processes=num_gpu_workers) as pool:
        pool.map(process_fitsne_combination, fitsne_args_list)

# --- Phase 2b: CPU final run (openTSNE, df=0.1) for exact paper parity ---
# The paper's best combo: perp=250, max_iter=1000, K=250, ee=1000
best_perp, best_max, best_k, best_ee = 250, 1000, 250, 1000
best_pasta = os.path.join(results_dir_fitsne, f"perp{best_perp}_max{best_max}_k{best_k}_ee{best_ee}")
best_final_file = os.path.join(best_pasta, "embedding_df01.dat")

if os.path.exists(best_final_file):
    print(f"-> Skipping FIt-SNE CPU re-run (df=0.1) - already completed: {best_final_file}")
else:
    print(f"\n-> Re-running BEST FIt-SNE combo on CPU with dof=0.1 for exact paper parity...")
    print(f"   Params: perp={best_perp}, max_iter={best_max}, K={best_k}, ee={best_ee}, dof=0.1")
    print(f"   Strategy: Train on 5% subset, then project full dataset in memory-safe chunks.")
    start = time.time()
    try:
        import gc
        temp_best = best_final_file + ".tmp"
        
        # 1. Extract a 5% random subset (108K points — more than enough for thermodynamics)
        print("   -> Extracting 5% training subset...")
        np.random.seed(42)
        subset_size = int(X_global.shape[0] * 0.05)
        subset_idx = np.random.choice(X_global.shape[0], size=subset_size, replace=False)
        X_subset = X_global[subset_idx].copy()  # .copy() so it owns its own memory
        n_total = X_global.shape[0]
        
        # 2. FREE X_global from RAM to give openTSNE maximum breathing room
        print(f"   -> Freeing main dataset from RAM ({X_global.nbytes / 1e9:.2f} GB) to make room for training...")
        del X_global
        del subset_idx
        gc.collect()

        # 3. Train openTSNE on the small subset with exact paper dof=0.1
        print(f"   -> Training openTSNE on {subset_size:,} points (dof=0.1)... This may take a while.")
        res_cpu = fit_sne(
            data=X_subset, perplexity=best_perp, n_iter=best_max,
            early_exaggeration_iter=800, early_exaggeration=best_ee,
            learning_rate="auto", min_num_intervals=50,
            use_gpu=False,  # Force CPU / openTSNE
            dof=0.1,        # Exact paper df=0.1
        )
        model = res_cpu["model"]
        
        # Free training data
        del X_subset, res_cpu
        gc.collect()
        
        # 4. Reload full dataset from .npy and project in memory-safe chunks
        print("   -> Reloading dataset from disk and projecting in chunks...")
        X_reload = np.load(npy_path, mmap_mode='r')  # memory-mapped: zero RAM cost!
        
        batch_size = 50000
        X_tsne_cpu_list = []
        
        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)
            print(f"      Projecting chunk {start_idx:,} to {end_idx:,}...")
            batch_data = np.array(X_reload[start_idx:end_idx])  # copy chunk into RAM
            batch_embedded = model.transform(batch_data)
            X_tsne_cpu_list.append(batch_embedded)
            del batch_data, batch_embedded
            gc.collect()
            
        X_tsne_cpu = np.vstack(X_tsne_cpu_list)
        del X_tsne_cpu_list
        gc.collect()

        pd.DataFrame(X_tsne_cpu).to_csv(temp_best, sep='\t', header=False, index=False, float_format='%.6f')
        os.replace(temp_best, best_final_file)
        del X_tsne_cpu
        gc.collect()
        
        elapsed = time.time() - start
        print(f"   Success! CPU FIt-SNE (df=0.1) completed in {elapsed:.2f}s")
        
        # 5. Reload X_global so subsequent steps (Sketch-map, MBAR) can use it
        X_global = np.load(npy_path)
        
    except Exception as e:
        print(f"   Failed CPU FIt-SNE re-run: {e}")
        # Make sure X_global is reloaded even on failure
        if 'X_global' not in dir():
            X_global = np.load(npy_path)

print("-> STEP 2 Completed.")



# %%
# ==============================================================================
# STEP 3: SKETCH-MAP CROSS VALIDATION
# ==============================================================================
print("\n=== STEP 3: Sketch-Map Cross Validation ===")
landmark_points = 1000
sigmas = [1.0, 1.1, 1.2]
As = [10.5, 9.5]
Bs = [1.0, 2.0]
a_s = [1.0, 2.0]
b_s = [1.0, 3.0]

def process_sketchmap_combination(args):
    sigma, A, B, a, b, X = args
    pasta = os.path.join(results_dir_sketch, f"s{sigma}_A{A}_B{B}_a{a}_b{b}")
    os.makedirs(pasta, exist_ok=True)
    output_file = os.path.join(pasta, "embedding.dat")
    lock_file = os.path.join(pasta, ".lock")
    
    # Skip already completed combinations or those currently being processed
    if os.path.exists(output_file):
        print(f"-> Skipping Sketch-map (sigma={sigma}, A={A}, B={B}, a={a}, b={b}) - already completed.")
        return
    if os.path.exists(lock_file):
        print(f"-> Skipping Sketch-map (sigma={sigma}, A={A}, B={B}, a={a}, b={b}) - currently being processed by another device.")
        return

    # Claim this task for this device
    open(lock_file, 'a').close()
    
    temp_output_file = output_file + ".tmp"

    print(f"-> Running Sketch-map (sigma={sigma}, A={A}, B={B}, a={a}, b={b})...")
    start = time.time()
    try:
        landmarks_hd = select_landmarks(data=X, n_landmarks=landmark_points, mode='random', seed=42, batch_size=50000)["landmarks"]
        fun_hd = (sigma, A, B)
        fun_ld = (sigma, a, b)
        
        landmarks_ld = sketch_map(data=landmarks_hd, fun_hd=fun_hd, fun_ld=fun_ld, n_components=2, preopt_steps=1000)["embedding"]
        X_ld = project_out_of_sample(samples=X, landmarks_hd=landmarks_hd, landmarks_ld=landmarks_ld, fun_hd=fun_hd, fun_ld=fun_ld, use_gpu=True, cg_steps=100, verbose=True)["embedding"]
        
        pd.DataFrame(X_ld).to_csv(temp_output_file, sep='\t', header=False, index=False, float_format='%.6f')
        os.replace(temp_output_file, output_file)
        elapsed = time.time() - start
        print(f"   Success! Time: {elapsed:.2f}s")
    except Exception as e:
        print(f"   Failed running Sketch-map: {e}")

combinations = list(product(sigmas, As, Bs, a_s, b_s))
args_list = [(s, A, B, a, b, X_global) for (s, A, B, a, b) in combinations]

# Determine parallel worker allocation (disabled on Windows to avoid infinite loop on imports)
is_windows = platform.system() == "Windows"
num_cores = 1 if is_windows else max(1, multiprocessing.cpu_count() - 1)

if IN_COLAB or num_cores == 1:
    for args in args_list:
        process_sketchmap_combination(args)
else:
    print(f"-> Running Sketch-map parallel search using {num_cores} workers...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(process_sketchmap_combination, args_list)

print("-> STEP 3 Completed.")

# %%
# ==============================================================================
# DYNAMIC EVALUATION: FIND OPTIMAL EMBEDDINGS
# ==============================================================================
print("\n=== DYNAMIC EVALUATION: Finding Optimal Embeddings ===")
optimal_dir = os.path.join(base_dir, "outputs", "optimal")
os.makedirs(optimal_dir, exist_ok=True)

def evaluate_method(method_name, results_dir, prefix, data_filename="embedding.dat"):
    print(f"\n-> Dynamically evaluating {method_name} combinations...")
    try:
        np.random.seed(42)
        sample_indices = np.random.choice(X_global.shape[0], size=5000, replace=False)
        X_raw_sample = X_global[sample_indices]
        raw_distances = pairwise_distances(X_raw_sample, metric='euclidean', squared=True)

        folders = [f for f in os.listdir(results_dir) if f.startswith(prefix)]
        scores = []
        print(f"   Evaluating {len(folders)} combinations...")

        for folder in folders:
            # Special case for FIt-SNE fallback
            emb_path = os.path.join(results_dir, folder, data_filename)
            fallback_path = os.path.join(results_dir, folder, "embedding.dat")
            
            if not os.path.exists(emb_path) and os.path.exists(fallback_path):
                emb_path = fallback_path
                
            if os.path.exists(emb_path):
                emb_data = pd.read_csv(emb_path, sep='\t', header=None).values
                if len(emb_data) == X_global.shape[0]:
                    emb_sample = emb_data[sample_indices]
                    trust = trustworthiness(raw_distances, emb_sample, n_neighbors=10, metric='precomputed')
                    scores.append({'Combination': folder, 'Trustworthiness': trust, 'Path': emb_path})

        if not scores:
            print(f"   No valid embeddings found for {method_name}.")
            return None

        scores_df = pd.DataFrame(scores).sort_values(by='Trustworthiness', ascending=False)
        print(f"\n=== TOP 3 {method_name} COMBINATIONS ===")
        print(scores_df[['Combination', 'Trustworthiness']].head(3).to_string(index=False))
        
        best_path = scores_df.iloc[0]['Path']
        best_folder = scores_df.iloc[0]['Combination']
        print(f"   => Selected {best_folder} as optimal!")
        
        # Save to optimal folder
        with open(os.path.join(optimal_dir, f"{method_name}_optimal.txt"), "w") as f:
            f.write(best_path)
            
        return best_path
    except Exception as e:
        print(f"   Failed to calculate {method_name} metrics: {e}")
        return None

# Evaluate all three
opt_sketch = evaluate_method("Sketch-map", results_dir_sketch, "s", "embedding.dat")
opt_fitsne = evaluate_method("FIt-SNE", results_dir_fitsne, "perp", "embedding_df01.dat")
opt_umap   = evaluate_method("UMAP", results_dir_umap, "neigh", "embedding.dat")

# ==============================================================================
# STEP 4: MBAR WEIGHTING (CALCULATION ONLY)
# ==============================================================================
print("\n=== STEP 4: MBAR Weighting (Calculations) ===")
kB = 8.6173324e-5
target_temperature = 300
nbins_per_axes = 200
niterations = 60000

def read_temperatures():
    t_path = os.path.join(data_dir, "temperatures")
    if not os.path.exists(t_path):
        raise FileNotFoundError(f"Missing temperatures file at {t_path}")
    with open(t_path, "r") as f:
        return np.array([float(x) for x in f.readlines()[0].split()])

def read_potential_energies(charge_state, K, T):
    pe_path = os.path.join(data_dir, f"potencial-energies_{charge_state}")
    if not os.path.exists(pe_path):
        raise FileNotFoundError(f"Missing potential energies file for {charge_state} at {pe_path}")
    U_kt = np.zeros([K, T])
    with open(pe_path, "r") as f:
        lines = f.readlines()
        for t in range(T):
            elements = lines[t].split()
            for k in range(K):
                U_kt[k, t] = float(elements[k])
    return U_kt


# ─── Data layout (from get_coords_dimred.sh) ─────────────────────────────────
# Full embedding has 2,160,000 rows ordered as:
#   rows       1 –   720,000 → charge: minus
#   rows 720,001 – 1,440,000 → charge: neutral
#   rows 1,440,001 – 2,160,000 → charge: plus
# Each charge block: 3 morphologies × 12 temperatures × 20,000 snapshots
#   = 240,000 rows per morphology, 720,000 per charge
# Per-temperature replica files {k}.x / {k}.y each have 60,000 lines
#   (3 morphologies × 20,000 snapshots merged in temperature order)
# K = 12 temperatures, T = niterations = 60,000 snapshots per temperature
# ──────────────────────────────────────────────────────────────────────────────

ROWS_PER_CHARGE = 720_000   # rows 0:720000 = minus, 720000:1440000 = neutral, etc.
ROWS_PER_MORPH  = 240_000   # 3 morphologies per charge
ROWS_PER_TEMP   = 20_000    # snapshots per temperature per morphology
N_MORPH = 3
N_TEMPS = 12                 # K in the paper

CHARGE_SLICE = {
    "minus":   (0,           ROWS_PER_CHARGE),
    "neutral": (ROWS_PER_CHARGE,   2 * ROWS_PER_CHARGE),
    "plus":    (2 * ROWS_PER_CHARGE, 3 * ROWS_PER_CHARGE),
}

def split_embedding_to_xkt_ykt(full_embedding, charge_state):
    """Reproduce get_coords_dimred.sh in Python.
    Returns x_kt, y_kt arrays of shape (K=12, T=60000) for the given charge.
    Each column k contains the 60000 snapshots belonging to temperature replica k,
    assembled from all 3 morphologies (3 × 20000 rows per temp).
    """
    start, end = CHARGE_SLICE[charge_state]
    block = full_embedding[start:end]          # (720000, 2)

    # split into 3 morphologies
    morphs = [block[m * ROWS_PER_MORPH:(m + 1) * ROWS_PER_MORPH] for m in range(N_MORPH)]

    # for each morphology, rows are laid out as: temp0 (20000 rows), temp1 (20000 rows), ...
    x_kt = np.zeros((N_TEMPS, niterations))   # niterations = 60000
    y_kt = np.zeros((N_TEMPS, niterations))
    for k in range(N_TEMPS):
        rows_for_k = []
        for morph in morphs:
            rows_for_k.append(morph[k * ROWS_PER_TEMP:(k + 1) * ROWS_PER_TEMP])
        combined = np.vstack(rows_for_k)       # (60000, 2)
        x_kt[k] = combined[:, 0]
        y_kt[k] = combined[:, 1]
    return x_kt, y_kt


def run_mbar_for_charge_method(charge_state, method_name, embedding_file, output_fes_file):
    if os.path.exists(output_fes_file):
        print(f"-> FES calculation already exists for {method_name} - {charge_state}. Skipping.")
        return

    print(f"-> Running MBAR for {charge_state} using {method_name} embedding...")
    start_time = time.time()
    try:
        temperatures = read_temperatures()
        K = len(temperatures)                         # 12
        T = niterations                               # 60000
        beta_k = 1.0 / (kB * temperatures)

        # Load the full flat embedding and split by charge + temperature replica
        full_emb = pd.read_csv(embedding_file, sep='\t', header=None).values
        x_kt, y_kt = split_embedding_to_xkt_ykt(full_emb, charge_state)

        U_kt = read_potential_energies(charge_state, K, T)

        # Assemble flat x_n array of shape (K*T, 2)
        x_n = np.zeros([K * T, 2])
        Ntot = 0
        for k in range(K):
            for n in range(T):
                x_n[Ntot, 0] = x_kt[k, n]
                x_n[Ntot, 1] = y_kt[k, n]
                Ntot += 1

        # Axis limits from data (paper uses x_min/x_max from ranges_axes.sh)
        x_min_val = np.min(x_n[:, 0])
        x_max_val = np.max(x_n[:, 0])
        y_min_val = np.min(x_n[:, 1])
        y_max_val = np.max(x_n[:, 1])

        # ---------------------------------------------------------
        # Execute Unified MBAR Workflow
        # ---------------------------------------------------------
        sample_assignments = np.repeat(np.arange(K), T)
        
        # We want to match the paper's exact binning:
        # Paper uses np.linspace(x_min, x_max, 201) to get 200 bins
        surf = mbar_free_energy_surface(
            energies=U_kt.flatten(),
            temperatures=temperatures,
            collective_vars=x_n,
            target_temperature=target_temperature,
            sample_assignments=sample_assignments,
            n_bins=nbins_per_axes,
            extent=(x_min_val, x_max_val, y_min_val, y_max_val),
            kde=False, # match the exact histogram logic of the paper
            mbar_kwargs={"solver": "robust"} # try robust solver first
        )
        
        prob_surface = surf["probability"]
        centers_x = surf["bin_centers_x"]
        centers_y = surf["bin_centers_y"]

        # Write output in the same format as the paper's mbar.py:
        #   x  y  f_i  df_i
        # Empty bins get SENTINEL_VALUE=1000.0, df=0.0
        SENTINEL_VALUE = 1000.0
        with open(output_fes_file, "w") as fout:
            for i in range(nbins_per_axes):
                for j in range(nbins_per_axes):
                    xc = centers_x[i]
                    yc = centers_y[j]
                    p = prob_surface[i, j]
                    if p > 0:
                        fi = -np.log(p)
                        df = 0.0
                    else:
                        fi  = SENTINEL_VALUE
                        df  = 0.0
                    fout.write(f"{xc:.6f}  {yc:.6f}  {fi:.6f} {df:.6f}\n")

        elapsed = time.time() - start_time
        print(f"   Success! Saved FES file to {output_fes_file} in {elapsed:.2f}s")
    except Exception as e:
        print(f"   Failed MBAR calculation for {charge_state} ({method_name}): {e}")

# MBAR task scheduler — find the best (optimal) embedding for each method.
# The paper's optimal hyperparameters:
#   UMAP:      n_neighbors=50, min_dist=0.001
#   FIt-SNE:   perp=250, max_iter=1000, K=250, ee=1000 (with df=0.1 from CPU run)
optimal_embeddings = {}
for method in ["Sketch-map", "FIt-SNE", "UMAP"]:
    opt_file = os.path.join(optimal_dir, f"{method}_optimal.txt")
    if os.path.exists(opt_file):
        with open(opt_file, "r") as f:
            optimal_embeddings[method] = f.read().strip()
    else:
        print(f"-> WARNING: No optimal embedding found for {method} (missing {opt_file})")

# Define paper's actual preferred embeddings
paper_embeddings = {
    "Sketch-map": os.path.join(results_dir_sketch, "s1.2_A10.5_B1.0_a1.0_b1.0", "embedding.dat"),
    "FIt-SNE": os.path.join(results_dir_fitsne, "perp250_max1000_k250_ee1000", "embedding_df01.dat"),
    "UMAP": os.path.join(results_dir_umap, "neigh_50_dist_0.001", "embedding.dat")
}

fitsne_gpu_fallback = os.path.join(results_dir_fitsne, "perp250_max1000_k250_ee1000", "embedding.dat")
if not os.path.exists(optimal_embeddings.get("FIt-SNE", "")) and os.path.exists(fitsne_gpu_fallback):
    print("-> Note: FIt-SNE CPU (df=0.1) embedding not found, using GPU embedding as fallback for optimal.")
    optimal_embeddings["FIt-SNE"] = fitsne_gpu_fallback
if not os.path.exists(paper_embeddings["FIt-SNE"]) and os.path.exists(fitsne_gpu_fallback):
    paper_embeddings["FIt-SNE"] = fitsne_gpu_fallback

run_any = False
for variant_name, embeddings_dict in [("math", optimal_embeddings), ("paper", paper_embeddings)]:
    print(f"\n--- Running MBAR for {variant_name.upper()} embeddings ---")
    for method_name, emb_file in embeddings_dict.items():
        if not os.path.exists(emb_file):
            print(f"-> Skipping MBAR for {method_name} ({variant_name}) - embedding not found: {emb_file}")
            continue
        for charge in ["minus", "neutral", "plus"]:
            fes_file = os.path.join(results_dir_mbar, f"fes_{method_name}_{variant_name}_{charge}.txt")
            run_mbar_for_charge_method(charge, method_name, emb_file, fes_file)
            run_any = True

if not run_any:
    print("-> Notice: No optimal embedding files found yet. Complete Steps 1-3 first.")

print("-> STEP 4 Completed.")

# %%
# ==============================================================================
# STEP 5: VISUALIZATIONS AND PLOTTING
# ==============================================================================
print("\n=== STEP 5: Visualizations and Plotting ===")

# %%
# --- FIGURE 3: SCATTER PROJECTIONS (SKETCH-MAP, FIT-SNE, UMAP) ---
print("\n--- Figure 3: Scatter Projections ---")

from matplotlib.lines import Line2D

# Colors from paper
CHARGE_COLORS = {"anionic": "#d6182a", "neutral": "#f4a518", "cationic": "#2166ac"}
CHARGE_LABELS = {"anionic": r"Au$_{13}^{(-1)}$", "neutral": r"Au$_{13}^{(0)}$", "cationic": r"Au$_{13}^{(+1)}$"}

for variant_name, embeddings_dict in [("math", optimal_embeddings), ("paper", paper_embeddings)]:
    print(f"\n-> Generating Figure 3 for {variant_name.upper()}...")
    
    embeddings = {}
    for method, emb_file in embeddings_dict.items():
        if os.path.exists(emb_file):
            embeddings[method] = pd.read_csv(emb_file, sep='\t', header=None).values
    
    if len(embeddings) > 0:
        fig, axes = plt.subplots(len(embeddings), 1, figsize=(7, 6 * len(embeddings)))
        if len(embeddings) == 1: axes = [axes]
        fig.subplots_adjust(hspace=0.32)
        
        for ax, method in zip(axes, embeddings.keys()):
            emb_full = embeddings[method]
            
            # 1. Plot full trajectory as a grey cloud underneath (density topology)
            ax.scatter(emb_full[:, 0], emb_full[:, 1],
                       s=0.02, c="#999999", alpha=0.25,
                       linewidths=0, rasterized=True, zorder=1)
                       
            # 2. Add colored dots for specific charge states subsampled
            for charge, color in [("plus", CHARGE_COLORS["cationic"]),
                                  ("neutral", CHARGE_COLORS["neutral"]),
                                  ("minus", CHARGE_COLORS["anionic"])]:
                start, end = CHARGE_SLICE[charge]
                sub = emb_full[start:end][::4000]
                ax.scatter(sub[:, 0], sub[:, 1],
                           s=12, c=color, alpha=0.90,
                           linewidths=0, zorder=3, rasterized=True)
    
            ax.set_title(f"{method} coordinates ({variant_name})", fontsize=12, pad=10, loc="center")
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
    
        handles = [
            Line2D([0],[0], marker="o", color="w", markerfacecolor="#b0b0b0", markersize=5, label="Full projection"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor=CHARGE_COLORS["cationic"], markersize=10, label=CHARGE_LABELS["cationic"]),
            Line2D([0],[0], marker="o", color="w", markerfacecolor=CHARGE_COLORS["neutral"], markersize=10, label=CHARGE_LABELS["neutral"]),
            Line2D([0],[0], marker="o", color="w", markerfacecolor=CHARGE_COLORS["anionic"], markersize=10, label=CHARGE_LABELS["anionic"]),
        ]
        axes[0].legend(handles=handles, fontsize=9, frameon=False, loc="upper left", ncol=1, handletextpad=0.2)
        
        out_path = os.path.join(plots_dir, f"Fig3_nldr_projections_{variant_name}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor='white')
        print(f"   Saved Figure 3 to {out_path}")
        plt.show()
        plt.close(fig)


# %%
# --- FIGURES 4, 5, 6: MBAR PROBABILITY SURFACES AT 300K ---
print("\n--- Figures 4, 5, 6: MBAR Probability Surfaces ---")

from scipy.interpolate import griddata as scipy_griddata
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

PAPER_COLORS = [(0.00, "white"), (0.30, "cornflowerblue"), (0.66, "indigo"), (0.95, "orange"), (1.00, "yellow")]
PAPER_CMAP = mcolors.LinearSegmentedColormap.from_list("paper_cmap", PAPER_COLORS)
SENTINEL_VALUE = 1000.0
SHIFT_PARAM    = 2.5

def plot_mbar_combined(fig_num, method_name, method_key, variant_name):
    # Important: If you re-ran MBAR on the correct embeddings, they should be in the mbar directory!
    fes_files = {charge: os.path.join(results_dir_mbar, f"fes_{method_key}_{variant_name}_{charge}.txt") for charge in ["plus", "minus"]}
    
    if not os.path.exists(fes_files["plus"]) or not os.path.exists(fes_files["minus"]):
        print(f"  WARNING: Missing MBAR FES files for {method_name} ({variant_name}). Make sure MBAR ran on the correct embeddings!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.12)
    
    for ax, charge, lbl, first in zip(axes, ["plus", "minus"], ["(a)", "(b)"], [True, False]):
        data = np.loadtxt(fes_files[charge])
        x_raw, y_raw, f_raw = data[:, 0], data[:, 1], data[:, 2]

        valid_mask = f_raw != SENTINEL_VALUE
        f_proc = np.where(valid_mask, f_raw, f_raw[valid_mask].max()) + SHIFT_PARAM
        log_prob = np.log10(SHIFT_PARAM / f_proc)
        z_min = log_prob.min()

        xi = np.linspace(x_raw.min(), x_raw.max(), 2000)
        yi = np.linspace(y_raw.min(), y_raw.max(), 2000)
        xi_g, yi_g = np.meshgrid(xi, yi)
        zi = scipy_griddata((x_raw, y_raw), log_prob, (xi_g, yi_g), method='cubic')

        im = ax.imshow(zi, extent=[x_raw.min(), x_raw.max(), y_raw.min(), y_raw.max()], origin='lower',
                       cmap=PAPER_CMAP, vmin=z_min, vmax=0, aspect='auto', interpolation='bicubic')
        
        ax.contour(xi_g, yi_g, zi, levels=np.arange(z_min + 0.1, -0.1, 0.08), colors='black', linewidths=0.2, alpha=0.5)

        charge_map = {"plus": "cationic", "minus": "anionic"}
        ax.set_title(f"{lbl} {CHARGE_LABELS[charge_map[charge]]}", fontsize=12, pad=10)
        ax.set_xticks([]); ax.set_yticks([])

        cbar_ax = ax.inset_axes([0.70, 0.04, 0.28, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r"Log(P)", fontsize=9, labelpad=2)
        cbar.ax.tick_params(labelsize=8)

    out_path = os.path.join(plots_dir, f"Fig{fig_num}_fes_{method_name.lower()}_{variant_name}_300k.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"-> Saved Figure {fig_num} ({method_name} - {variant_name}) to {out_path}")
    plt.show()
    plt.close(fig)

for variant in ["math", "paper"]:
    plot_mbar_combined(4, "Sketch-map", "Sketch-map", variant)
    plot_mbar_combined(5, "FIt-SNE", "FIt-SNE", variant)
    plot_mbar_combined(6, "UMAP", "UMAP", variant)

# %%
# --- FIGURES 7 & 8: STRUCTURAL DESCRIPTORS (ECN, Rg, HCM, εd) ---
print("\n--- Structural Descriptors (Figures 7 & 8) ---")
print("-> Note: Calculating ECN, Rg, and HCM requires the 3D atomic coordinates trajectory (.xyz).")
print("   Please place trajectory_minus.xyz, trajectory_neutral.xyz, and trajectory_plus.xyz in the data folder.")
# for charge in ["minus", "neutral", "plus"]:
#     xyz_path = os.path.join(data_dir, f"trajectory_{charge}.xyz")
#     emb_path = os.path.join(results_dir_sketch, f"embedding_{charge}.dat")
#     
#     if os.path.exists(xyz_path) and os.path.exists(emb_path):
#         print(f"-> Computing descriptors for {charge}...")
#         
#         # 1. Load 3D coordinates using dimredpy IO (Shape: T, N_atoms, 3)
#         trajectory = load_spatial_coordinates(xyz_path, format="xyz", label_filter="Au")
#         
#         # 2. Compute descriptors natively via dimredpy
#         descriptors = compute_trajectory_descriptors(trajectory, cutoff=3.4)
#         ecn = descriptors["ecn"]
#         rg = descriptors["rg"]
#         hcm = descriptors["hcm"]
#         
#         # 3. Load sketch-map embedding to map properties onto the 2D space
#         X = pd.read_csv(emb_path, sep='\t', header=None).values
#         
#         # Subsample for plotting (e.g. at 250K replica)
#         T_steps = 60000 
#         X_sub = X[:T_steps:50]
#         ecn_sub = ecn[:T_steps:50]
#         rg_sub = rg[:T_steps:50]
#         hcm_sub = hcm[:T_steps:50]
#         
#         # --- Plot Figure 7 (ECN & Rg) ---
#         plt.figure(figsize=(8, 6))
#         sc = plt.scatter(X_sub[:, 0], X_sub[:, 1], c=ecn_sub, cmap='plasma', 
#                          s=(rg_sub - rg_sub.min() + 0.1)*50, alpha=0.7, edgecolors='none')
#         plt.colorbar(sc, label='Effective Coordination Number (ECN)')
#         plt.title(f"Figure 7: ECN and Rg for {CHARGE_LABELS[charge]} at 250K")
#         plt.xlabel("Sketch-map X")
#         plt.ylabel("Sketch-map Y")
#         plt.tight_layout()
#         out_7 = os.path.join(plots_dir, f"Fig7_{charge}.png")
#         plt.savefig(out_7, dpi=300)
#         print(f"-> Saved Figure 7 ({charge}) to {out_7}")
#         plt.show()
#         plt.close()
#         
#         # --- Plot Figure 8 (HCM & epsilon_d) ---
#         plt.figure(figsize=(8, 6))
#         sc = plt.scatter(X_sub[:, 0], X_sub[:, 1], c=ecn_sub, cmap='viridis', 
#                          s=(hcm_sub)*500 + 10, alpha=0.7, edgecolors='none')
#         plt.colorbar(sc, label='d-band Center (εd) - PLACEHOLDER')
#         plt.title(f"Figure 8: HCM and εd for {CHARGE_LABELS[charge]} at 250K")
#         plt.xlabel("Sketch-map X")
#         plt.ylabel("Sketch-map Y")
#         plt.tight_layout()
#         out_8 = os.path.join(plots_dir, f"Fig8_{charge}.png")
#         plt.savefig(out_8, dpi=300)
#         print(f"-> Saved Figure 8 ({charge}) to {out_8}")
#         plt.show()
#         plt.close()
#     else:
#         print(f"-> Skipping Figures 7 & 8 for {charge}: missing 3D trajectory or optimal Sketch-map embedding.")

print("\n=== STEP 5 Completed. All plotting routines processed successfully. ===")
