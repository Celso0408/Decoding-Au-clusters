"""
Rigorous Verification of FIt-SNE (DimRedPy vs openTSNE)
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
from dimredpy.fitsne import fit_sne

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

# ---------------------------------------------------------------------------
# 2. Define Shared Hyperparameters
# ---------------------------------------------------------------------------
perplexity = 30
early_exag = 12.0
n_iter = 500
seed = 42

print(f"Perplexity: {perplexity}")
print(f"Early Exag: {early_exag}")

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
    n_iter=n_iter,
    learning_rate=max(200.0, len(data)/early_exag),
    metric="euclidean",
    min_num_intervals=50,
    negative_gradient_method="fft",
    random_state=seed,
    n_jobs=1,
)

emb_ref = tsne_ref.fit(data)
emb_ref = np.asarray(emb_ref)
print("-> Reference execution finished.")

# ---------------------------------------------------------------------------
# 4. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")

res = fit_sne(
    data,
    n_components=2,
    perplexity=perplexity,
    early_exaggeration=early_exag,
    n_iter=n_iter,
    learning_rate="auto",
    metric="euclidean",
    seed=seed,
    n_jobs=1,
    use_gpu=False,
)
emb_dimredpy = res["embedding"]
print("-> DimRedPy execution finished.")

# ---------------------------------------------------------------------------
# 5. Parity Validation
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

if mse < 1e-5 and disparity < 1e-5 and corr > 0.9999:
    print("\nSUCCESS: DimRedPy FIt-SNE Reduction is mathematically perfect.")
else:
    print("\nFAILURE: Statistical divergence detected.")

# ---------------------------------------------------------------------------
# 6. Visual Verification
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

axes[0].scatter(emb_ref[:, 0], emb_ref[:, 1], s=5, alpha=0.6, c='#1f77b4', edgecolors='none')
axes[0].set_title("Reference (openTSNE)", fontsize=14, pad=10)
axes[0].axis('off')

axes[1].scatter(emb_dimredpy[:, 0], emb_dimredpy[:, 1], s=5, alpha=0.6, c='#d62728', edgecolors='none')
axes[1].set_title("DimRedPy (FIt-SNE)", fontsize=14, pad=10)
axes[1].axis('off')

plt.suptitle(f"FIt-SNE Reduction Parity (N={len(data)})\n"
             f"Procrustes Disparity: {disparity:.2e} | Pearson R: {corr:.5f} | MSE: {mse:.2e}", 
             fontsize=16, y=1.05, fontweight='bold')

plt.tight_layout()
out_plot = os.path.join(script_dir, "verify_parity_fitsne_reduction_plot.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved comparison plot to {out_plot}")
