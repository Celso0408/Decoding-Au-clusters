"""
Rigorous Verification of Supervised UMAP (DimRedPy vs umap-learn)

This script validates that the DimRedPy framework produces mathematically identical results
to the underlying umap-learn reference implementation for supervised dimensionality reduction.
We utilize rigorous statistical metrics:
1. Mean Squared Error (MSE)
2. Procrustes Disparity (Measures shape disparity independent of translation/rotation)
3. Pearson Correlation (Global structure correlation between embeddings)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import umap
from scipy.spatial import procrustes
from scipy.stats import pearsonr

# Robust path detection for repo root
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dimredpy.umap_embed import umap_embed

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
data = np.loadtxt(data_path, max_rows=1000)
print(f"Data shape: {data.shape}")

# Generate a synthetic categorical target for supervised UMAP
np.random.seed(42)
y = np.random.randint(0, 3, size=len(data))

# ---------------------------------------------------------------------------
# 2. Define Shared Hyperparameters
# ---------------------------------------------------------------------------
metric = "mahalanobis"
n_neighbors = 15
min_dist = 0.1
random_state = 42

print(f"Metric: {metric}")
print(f"Nearest Neighbors: {n_neighbors}")
print(f"Minimum Distance: {min_dist}")

# ---------------------------------------------------------------------------
# 3. Run Reference Framework (umap-learn)
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING REFERENCE FRAMEWORK (umap-learn)")
print("=======================================================")

X_ref = np.asarray(data, dtype=float)
cov = np.cov(X_ref, rowvar=False)
cov += np.eye(cov.shape[0]) * 1e-6

reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric=metric,
    metric_kwds={"V": cov},
    random_state=random_state,
    n_jobs=1,
)

emb_ref = reducer.fit_transform(data, y=y)
print("-> Reference execution finished.")


# ---------------------------------------------------------------------------
# 4. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")

res_dimredpy = umap_embed(
    data,
    n_components=2,
    metric=metric,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    y=y,
    seed=random_state,
    n_jobs=1,
    use_gpu=False, # Force CPU to match umap-learn exactly
)
emb_dimredpy = res_dimredpy["embedding"]
print("-> DimRedPy execution finished.")


# ---------------------------------------------------------------------------
# 5. Rigorous Parity Validation
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          PARITY VALIDATION")
print("=======================================================")

mse = np.mean((emb_ref - emb_dimredpy) ** 2)
mtx1, mtx2, disparity = procrustes(emb_ref, emb_dimredpy)
corr, _ = pearsonr(emb_ref.flatten(), emb_dimredpy.flatten())

print(f"1. Mean Squared Error (MSE): {mse:.6e}")
print(f"2. Procrustes Disparity    : {disparity:.6e}")
print(f"3. Pearson Correlation (R) : {corr:.6f}")

if mse < 1e-10 and disparity < 1e-10 and corr > 0.99999:
    print("\nSUCCESS: DimRedPy Supervised UMAP is mathematically perfect.")
else:
    print("\nFAILURE: Statistical divergence detected.")

# ---------------------------------------------------------------------------
# 6. Visual Verification
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

axes[0].scatter(emb_ref[:, 0], emb_ref[:, 1], s=5, alpha=0.6, c=y, cmap='viridis', edgecolors='none')
axes[0].set_title("Reference (umap-learn)", fontsize=14, pad=10)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['left'].set_visible(False)

axes[1].scatter(emb_dimredpy[:, 0], emb_dimredpy[:, 1], s=5, alpha=0.6, c=y, cmap='viridis', edgecolors='none')
axes[1].set_title("DimRedPy (UMAP)", fontsize=14, pad=10)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)
axes[1].spines['left'].set_visible(False)

plt.suptitle(f"Supervised UMAP Rigorous Verification (N=1000)\n"
             f"Procrustes Disparity: {disparity:.2e} | Pearson R: {corr:.5f} | MSE: {mse:.2e}", 
             fontsize=16, y=1.05, fontweight='bold')

plt.tight_layout()
out_plot = os.path.join(script_dir, "verify_parity_umap_supervised_plot.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved comparison plot to {out_plot}")
