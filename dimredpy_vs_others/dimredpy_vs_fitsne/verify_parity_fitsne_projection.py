"""
Rigorous Verification of FIt-SNE Projection (DimRedPy vs openTSNE)
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

from openTSNE import TSNE
from dimredpy.fitsne import fit_sne, fitsne_project

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

train_data = data[:800]
test_data = data[800:]

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape : {test_data.shape}")

# ---------------------------------------------------------------------------
# 2. Define Shared Hyperparameters
# ---------------------------------------------------------------------------
perplexity = 30
early_exag = 12.0
seed = 42

# ---------------------------------------------------------------------------
# 3. Run Reference Framework (openTSNE)
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING REFERENCE FRAMEWORK (openTSNE)")
print("=======================================================")

tsne_ref = TSNE(
    n_components=2,
    perplexity=perplexity,
    early_exaggeration=early_exag,
    early_exaggeration_iter=250,
    n_iter=500,
    learning_rate=max(200.0, len(train_data)/early_exag),
    metric="euclidean",
    min_num_intervals=50,
    negative_gradient_method="fft",
    random_state=seed,
    n_jobs=1,
)

emb_train_ref = tsne_ref.fit(train_data)
# Projection
emb_test_ref = np.asarray(emb_train_ref.transform(test_data))
print("-> Reference projection finished.")

# ---------------------------------------------------------------------------
# 4. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")

res = fit_sne(
    train_data,
    n_components=2,
    perplexity=perplexity,
    early_exaggeration=early_exag,
    n_iter=500,
    learning_rate="auto",
    metric="euclidean",
    seed=seed,
    n_jobs=1,
    use_gpu=False,
)
# Projection
emb_test_dimredpy = fitsne_project(res, test_data)
print("-> DimRedPy projection finished.")

# ---------------------------------------------------------------------------
# 5. Parity Validation
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          PARITY VALIDATION")
print("=======================================================")

mse = np.mean((emb_test_ref - emb_test_dimredpy) ** 2)
mtx1, mtx2, disparity = procrustes(emb_test_ref, emb_test_dimredpy)
corr, _ = pearsonr(emb_test_ref.flatten(), emb_test_dimredpy.flatten())

print(f"1. Mean Squared Error (MSE): {mse:.6e}")
print(f"2. Procrustes Disparity    : {disparity:.6e}")
print(f"3. Pearson Correlation (R) : {corr:.6f}")

if mse < 1e-5 and disparity < 1e-5 and corr > 0.9999:
    print("\nSUCCESS: DimRedPy FIt-SNE Projection is mathematically perfect.")
else:
    print("\nFAILURE: Statistical divergence detected.")

# ---------------------------------------------------------------------------
# 6. Visual Verification
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

axes[0].scatter(emb_test_ref[:, 0], emb_test_ref[:, 1], s=15, alpha=0.8, c='#1f77b4', edgecolors='none')
axes[0].set_title("Reference (openTSNE transform)", fontsize=14, pad=10)
axes[0].axis('off')

axes[1].scatter(emb_test_dimredpy[:, 0], emb_test_dimredpy[:, 1], s=15, alpha=0.8, c='#d62728', edgecolors='none')
axes[1].set_title("DimRedPy (fitsne_project)", fontsize=14, pad=10)
axes[1].axis('off')

plt.suptitle(f"FIt-SNE Projection Parity (N={len(test_data)})\n"
             f"Procrustes Disparity: {disparity:.2e} | Pearson R: {corr:.5f} | MSE: {mse:.2e}", 
             fontsize=16, y=1.05, fontweight='bold')

plt.tight_layout()
out_plot = os.path.join(script_dir, "verify_parity_fitsne_projection_plot.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved comparison plot to {out_plot}")
