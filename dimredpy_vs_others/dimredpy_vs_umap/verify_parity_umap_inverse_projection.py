"""
Rigorous Verification of UMAP Inverse Projection (DimRedPy vs umap-learn)
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

from umap import UMAP
from dimredpy.umap_embed import umap_embed, umap_project, umap_inverse_project

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
data = np.loadtxt(data_path, max_rows=500)  # Use smaller dataset for inverse transform (computationally heavy)

train_data = data[:400]
test_data = data[400:]

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape : {test_data.shape}")

# ---------------------------------------------------------------------------
# 2. Define Shared Hyperparameters
# ---------------------------------------------------------------------------
n_neighbors = 15
min_dist = 0.1
n_epochs = 200
seed = 42

# ---------------------------------------------------------------------------
# 3. Run Reference Framework (umap-learn)
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING REFERENCE FRAMEWORK (umap-learn)")
print("=======================================================")

umap_ref = UMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_epochs=n_epochs,
    metric="euclidean",
    random_state=seed,
    n_jobs=1,
)

emb_train_ref = umap_ref.fit(train_data)
# Projection
emb_test_ref = umap_ref.transform(test_data)
# Inverse Projection
recon_test_ref = umap_ref.inverse_transform(emb_test_ref)

print("-> Reference inverse projection finished.")

# ---------------------------------------------------------------------------
# 4. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")

res = umap_embed(
    train_data,
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_epochs=n_epochs,
    metric="euclidean",
    seed=seed,
    n_jobs=1,
    use_gpu=False,
)
# Projection
emb_test_dimredpy = umap_project(res, test_data)
# Inverse Projection
recon_test_dimredpy = umap_inverse_project(res, emb_test_dimredpy)

print("-> DimRedPy inverse projection finished.")

# ---------------------------------------------------------------------------
# 5. Parity Validation
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          PARITY VALIDATION")
print("=======================================================")

mse = np.mean((recon_test_ref - recon_test_dimredpy) ** 2)
mtx1, mtx2, disparity = procrustes(recon_test_ref, recon_test_dimredpy)
corr, _ = pearsonr(recon_test_ref.flatten(), recon_test_dimredpy.flatten())

print(f"1. Mean Squared Error (MSE): {mse:.6e}")
print(f"2. Procrustes Disparity    : {disparity:.6e}")
print(f"3. Pearson Correlation (R) : {corr:.6f}")

if mse < 1e-5 and disparity < 1e-5 and corr > 0.9999:
    print("\nSUCCESS: DimRedPy UMAP Inverse Projection is mathematically perfect.")
else:
    print("\nFAILURE: Statistical divergence detected.")

# ---------------------------------------------------------------------------
# 6. Visual Verification
# ---------------------------------------------------------------------------
# We will just plot a 2D projection of the reconstructed data (like a scatter of the first two HD dims)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

axes[0].scatter(recon_test_ref[:, 0], recon_test_ref[:, 1], s=15, alpha=0.8, c='#1f77b4', edgecolors='none')
axes[0].set_title("Reference (umap-learn inverse_transform)", fontsize=14, pad=10)
axes[0].axis('off')

axes[1].scatter(recon_test_dimredpy[:, 0], recon_test_dimredpy[:, 1], s=15, alpha=0.8, c='#d62728', edgecolors='none')
axes[1].set_title("DimRedPy (umap_inverse_project)", fontsize=14, pad=10)
axes[1].axis('off')

plt.suptitle(f"UMAP Inverse Projection Parity (N={len(test_data)})\n"
             f"Procrustes Disparity: {disparity:.2e} | Pearson R: {corr:.5f} | MSE: {mse:.2e}", 
             fontsize=16, y=1.05, fontweight='bold')

plt.tight_layout()
out_plot = os.path.join(script_dir, "verify_parity_umap_inverse_projection_plot.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved comparison plot to {out_plot}")
