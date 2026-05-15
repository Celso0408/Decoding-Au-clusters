"""
Rigorous Verification of Sketch-map (DimRedPy vs sketchmap_cpp)

This script validates that the DimRedPy framework produces mathematically identical results
to the underlying legacy C++ sketchmap implementation. We utilize rigorous statistical metrics:
1. Mean Squared Error (MSE)
2. Procrustes Disparity (Measures shape disparity independent of translation/rotation)
3. Pearson Correlation (Global structure correlation between embeddings)
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
from dimredpy.sketchmap import sketch_map

# ---------------------------------------------------------------------------
# 1. Load the Dataset (1,000 Samples)
# ---------------------------------------------------------------------------
# We load a statistically significant subset of 1,000 frames from the Au13 trajectory.
# We limit to 1000 to keep the O(N^2) SMACOF optimization perfectly manageable.

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
data = np.loadtxt(data_path, max_rows=1000)
print(f"Data shape: {data.shape}")

# ---------------------------------------------------------------------------
# 2. Define Shared Hyperparameters
# ---------------------------------------------------------------------------
# To ensure absolute parity, we must pass the exact same seed and algorithmic parameters.

np.random.seed(42)
weights = np.random.rand(len(data))

fun_hd = (6.0, 8.0, 8.0) # sigma, A, B for High-D
fun_ld = (6.0, 2.0, 8.0) # sigma, a, b for Low-D
preopt_steps = 100

print(f"High-D Function : {fun_hd}")
print(f"Low-D Function  : {fun_ld}")
print(f"Optimization Iters: {preopt_steps}")

# ---------------------------------------------------------------------------
# 3. Run Reference Framework (sketchmap_cpp)
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING REFERENCE FRAMEWORK (sketchmap_cpp)")
print("=======================================================")

try:
    fit_ref = smap.sketch_map(
        data,
        lowdim=2,
        weights=weights,
        fun_hd=fun_hd,
        fun_ld=fun_ld,
        preopt_steps=preopt_steps,
        grid=None, 
    )
    emb_ref = fit_ref["embedding"]
    print("-> Reference execution finished.")
except Exception as e:
    print(f"Note: Reference framework failed (expected if not compiled/Dockerized): {e}")
    # Fallback to random array for demonstration of script structure
    emb_ref = np.random.rand(len(data), 2)
    print("-> Used fallback dummy reference for script validation.")

# ---------------------------------------------------------------------------
# 4. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")

fit_dimredpy = sketch_map(
    data,
    n_components=2,
    weights=weights,
    sigma_hd=fun_hd[0],
    a_hd=fun_hd[1],
    b_hd=fun_hd[2],
    sigma_ld=fun_ld[0],
    a_ld=fun_ld[1],
    b_ld=fun_ld[2],
    max_iter=preopt_steps,
    tol=1e-6
)
emb_dimredpy = fit_dimredpy["embedding"]
print("-> DimRedPy execution finished.")

# ---------------------------------------------------------------------------
# 5. Rigorous Parity Validation
# ---------------------------------------------------------------------------
# We employ three distinct statistical metrics to prove identity.
print("\n=======================================================")
print("          PARITY VALIDATION")
print("=======================================================")

# 1. Mean Squared Error (MSE)
mse = np.mean((emb_ref - emb_dimredpy) ** 2)

# 2. Procrustes Disparity (Measures shape disparity)
mtx1, mtx2, disparity = procrustes(emb_ref, emb_dimredpy)

# 3. Pearson Correlation (flattened coordinates)
corr, _ = pearsonr(emb_ref.flatten(), emb_dimredpy.flatten())

print(f"1. Mean Squared Error (MSE): {mse:.6e}")
print(f"2. Procrustes Disparity    : {disparity:.6e}")
print(f"3. Pearson Correlation (R) : {corr:.6f}")

if mse < 1e-10 and disparity < 1e-10 and corr > 0.99999:
    print("\nSUCCESS: DimRedPy Sketch-map is mathematically perfect.")
else:
    print("\nFAILURE: Statistical divergence detected.")

# ---------------------------------------------------------------------------
# 6. Visual Verification
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

axes[0].scatter(emb_ref[:, 0], emb_ref[:, 1], s=5, alpha=0.6, c='#1f77b4', edgecolors='none')
axes[0].set_title("Reference (sketchmap_cpp)", fontsize=14, pad=10)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['left'].set_visible(False)

axes[1].scatter(emb_dimredpy[:, 0], emb_dimredpy[:, 1], s=5, alpha=0.6, c='#d62728', edgecolors='none')
axes[1].set_title("DimRedPy (Sketch-map)", fontsize=14, pad=10)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)
axes[1].spines['left'].set_visible(False)

plt.suptitle(f"Sketch-map Rigorous Verification (N=1000)\n"
             f"Procrustes Disparity: {disparity:.2e} | Pearson R: {corr:.5f} | MSE: {mse:.2e}", 
             fontsize=16, y=1.05, fontweight='bold')

plt.tight_layout()
out_plot = os.path.join(script_dir, "sketchmap_rigorous_comparison.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved comparison plot to {out_plot}")
