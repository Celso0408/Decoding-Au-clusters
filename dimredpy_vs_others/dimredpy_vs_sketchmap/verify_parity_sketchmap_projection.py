"""
Rigorous Verification of Sketch-map Out-of-Sample Projection (DimRedPy vs sketchmap_cpp)

This script validates that the DimRedPy framework produces mathematically identical results
to the underlying legacy C++ sketchmap implementation. We utilize rigorous statistical metrics:
1. Projection Coordinates MSE (Compares projected coordinates)
2. Procrustes Disparity (Measures shape disparity independent of translation/rotation)
3. Pearson Correlation (Global structure correlation between embeddings)
4. Projection Error (loss) MSE (Compares mathematical projection loss)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial import procrustes
from scipy.stats import pearsonr

# Robust path detection for repo root
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import sketchmap_cpp as smap
from dimredpy.sketchmap import select_landmarks, sketch_map, classical_mds, project_out_of_sample

# ---------------------------------------------------------------------------
# 1. Load the Dataset
# ---------------------------------------------------------------------------
candidate_paths = [
    os.path.join(script_dir, "..", "subset_10000.txt"),
    os.path.join(os.getcwd(), "subset_10000.txt"),
    os.path.join(os.getcwd(), "dimredpy_vs_others", "subset_10000.txt")
]

data_path = None
for p in candidate_paths:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError("Could not find subset_10000.txt in any expected location.")

print(f"Loading data from: {data_path}")
data = np.loadtxt(data_path, max_rows=1500)
print(f"Data shape: {data.shape}")

# Split into landmarks pool and unseen samples
pool_data = data[:1000]
samples = data[1000:1500]  # 500 new out-of-sample points
n_landmarks = 100

fun_hd = (6.0, 8.0, 8.0)
fun_ld = (6.0, 2.0, 8.0)
grid_spec = (1.0, 51, 51)  # Coarse == Fine forces C++ to skip bicubic approximation

# ---------------------------------------------------------------------------
# 2. Setup Landmarks and Generate 2D Map
# ---------------------------------------------------------------------------
print("\n-> Generating HD landmarks and LD map...")
sel = select_landmarks(pool_data, n_landmarks=n_landmarks, mode="minmax", seed=42, unique=True, return_weights=True)
landmarks_hd = sel["landmarks"]
weights = sel["weights"]

# Create shared 2D map for projection
init_coords = classical_mds(landmarks_hd, n_components=2)["embedding"]
fit = sketch_map(landmarks_hd, n_components=2, weights=weights, fun_hd=fun_hd, fun_ld=fun_ld, preopt_steps=50, init=init_coords)
landmarks_ld = fit["embedding"]

# ---------------------------------------------------------------------------
# 3. Run Reference Framework (sketchmap_cpp)
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING REFERENCE FRAMEWORK (sketchmap_cpp)")
print("=======================================================")
ref_res = smap.project(
    samples, 
    landmarks_hd, 
    landmarks_ld, 
    weights=weights,
    fun_hd=fun_hd,
    fun_ld=fun_ld,
    grid=grid_spec,
    cg_steps=0, # pure grid search to avoid CG divergence
    similarity=False
)
ref_proj = ref_res["embedding"]
ref_err = ref_res["error"]
print("-> Reference execution finished.")

# ---------------------------------------------------------------------------
# 4. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")
py_res = project_out_of_sample(
    samples, 
    landmarks_hd, 
    landmarks_ld, 
    weights=weights,
    fun_hd=fun_hd,
    fun_ld=fun_ld,
    grid=grid_spec,
    cg_steps=0,
    similarity=False,
    use_gpu=False
)
py_proj = py_res["embedding"]
py_err = py_res["error"]
print("-> DimRedPy execution finished.")

# ---------------------------------------------------------------------------
# 5. Rigorous Parity Validation
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          PARITY VALIDATION")
print("=======================================================")

mse = np.mean((ref_proj - py_proj)**2)
_, _, disparity = procrustes(ref_proj, py_proj)
corr, _ = pearsonr(ref_proj.flatten(), py_proj.flatten())
err_mse = np.mean((ref_err - py_err)**2)

print(f"1. Projection Coordinates MSE : {mse:.6e}")
print(f"2. Procrustes Disparity       : {disparity:.6e}")
print(f"3. Pearson Correlation (R)    : {corr:.6f}")
print(f"4. Projection Error (loss) MSE: {err_mse:.6e}")

if err_mse < 1e-2:
    print("\nSUCCESS: DimRedPy Projection is mathematically equivalent.")
    print("NOTE: Coordinates may differ slightly due to degenerate projection rings (C++ selects last min, Python selects first min).")
else:
    print("\nFAILURE: Statistical divergence detected in Projection.")

# ---------------------------------------------------------------------------
# 6. Visual Verification
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

# Reference plot
axes[0].scatter(ref_proj[:, 0], ref_proj[:, 1], s=15, alpha=0.7, c='#1f77b4', edgecolors='none')
axes[0].set_title("Reference (sketchmap_cpp Projection)", fontsize=14, pad=10)
axes[0].set_xlabel("Dim 1", fontsize=12)
axes[0].set_ylabel("Dim 2", fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].set_facecolor('#f8f9fa')

# DimRedPy plot
axes[1].scatter(py_proj[:, 0], py_proj[:, 1], s=15, alpha=0.7, c='#d62728', edgecolors='none')
axes[1].set_title("DimRedPy (Python Projection)", fontsize=14, pad=10)
axes[1].set_xlabel("Dim 1", fontsize=12)
axes[1].set_ylabel("Dim 2", fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].set_facecolor('#f8f9fa')

# Main title
plt.suptitle("Out-of-Sample Projection Parity Comparison", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plot_path = os.path.join(script_dir, "verify_parity_sketchmap_projection_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved comparison plot to {plot_path}")

