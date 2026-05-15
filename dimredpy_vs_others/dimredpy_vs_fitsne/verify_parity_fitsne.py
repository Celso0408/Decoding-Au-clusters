"""
Rigorous Verification of FIt-SNE (DimRedPy vs openTSNE)

This script validates that the DimRedPy framework produces mathematically identical results
to the underlying openTSNE reference implementation. We utilize rigorous statistical metrics:
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

from openTSNE import TSNE
from dimredpy.fitsne import fit_sne

# ---------------------------------------------------------------------------
# 1. Load the Dataset (10,000 Samples)
# ---------------------------------------------------------------------------
# We load a statistically significant subset of 10,000 frames from the Au13 trajectory.

candidate_paths = [
    os.path.join(script_dir, "subset_10000.txt"),
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
data = np.loadtxt(data_path)
print(f"Data shape: {data.shape}")

# ---------------------------------------------------------------------------
# 2. Define Shared Hyperparameters
# ---------------------------------------------------------------------------
# To ensure absolute parity, we must pass the exact same seed and algorithmic parameters.

perplexity = 250
early_exag = 12.0
n_iter = 1000
random_state = 42

print(f"Perplexity: {perplexity}")
print(f"Early Exaggeration: {early_exag}")
print(f"Iterations: {n_iter}")

# ---------------------------------------------------------------------------
# 3. Run Reference Framework (openTSNE)
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING REFERENCE FRAMEWORK (openTSNE)")
print("=======================================================")

tsne_ref = TSNE(
    perplexity=perplexity,
    early_exaggeration=early_exag,
    early_exaggeration_iter=250,
    n_iter=n_iter,
    learning_rate=max(200, data.shape[0] / early_exag), # dimredpy's auto learning rate
    metric="euclidean",
    negative_gradient_method="fft",
    min_num_intervals=50,
    random_state=random_state,
    n_jobs=1,
)

emb_ref = tsne_ref.fit(data)
print("-> Reference execution finished.")


# ---------------------------------------------------------------------------
# 4. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")

emb_dimredpy = fit_sne(
    data,
    perplexity=perplexity,
    early_exaggeration=early_exag,
    n_iter=n_iter,
    seed=random_state,
    n_jobs=1,
    use_gpu=False, # Force CPU to match openTSNE exactly
)
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
# A disparity of 0.0 means the shapes are identical.
mtx1, mtx2, disparity = procrustes(emb_ref, emb_dimredpy)

# 3. Pearson Correlation (flattened coordinates)
corr, _ = pearsonr(emb_ref.flatten(), emb_dimredpy.flatten())

print(f"1. Mean Squared Error (MSE): {mse:.6e}")
print(f"2. Procrustes Disparity    : {disparity:.6e}")
print(f"3. Pearson Correlation (R) : {corr:.6f}")

if mse < 1e-10 and disparity < 1e-10 and corr > 0.99999:
    print("\nSUCCESS: DimRedPy FIt-SNE is mathematically perfect.")
else:
    print("\nFAILURE: Statistical divergence detected.")

# ---------------------------------------------------------------------------
# 6. Visual Verification
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

axes[0].scatter(emb_ref[:, 0], emb_ref[:, 1], s=1, alpha=0.6, c='#1f77b4', edgecolors='none')
axes[0].set_title("Reference (openTSNE)", fontsize=14, pad=10)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['left'].set_visible(False)

axes[1].scatter(emb_dimredpy[:, 0], emb_dimredpy[:, 1], s=1, alpha=0.6, c='#d62728', edgecolors='none')
axes[1].set_title("DimRedPy (FIt-SNE)", fontsize=14, pad=10)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)
axes[1].spines['left'].set_visible(False)

plt.suptitle(f"FIt-SNE Rigorous Verification (N=10000)\n"
             f"Procrustes Disparity: {disparity:.2e} | Pearson R: {corr:.5f} | MSE: {mse:.2e}", 
             fontsize=16, y=1.05, fontweight='bold')

plt.tight_layout()
out_plot = os.path.join(script_dir, "fitsne_rigorous_comparison.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved comparison plot to {out_plot}")
