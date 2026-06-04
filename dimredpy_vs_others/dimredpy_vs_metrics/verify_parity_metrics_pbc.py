"""
Rigorous Verification of Metric Parity: PBC Metric
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Robust path detection for repo root
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dimredpy.shared.metrics import PBCMetric

# ---------------------------------------------------------------------------
# 1. Generate Synthetic Data
# ---------------------------------------------------------------------------
print("Generating synthetic coordinates for PBC Metric parity...")
np.random.seed(42)
N = 100
D = 3
period = np.array([10.0, 15.0, 20.0])
X = np.random.uniform(0, 25.0, size=(N, D)) # Extends beyond period

# ---------------------------------------------------------------------------
# 2. Run Reference Framework (Explicit Loop)
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING REFERENCE FRAMEWORK (Explicit Loop)")
print("=======================================================")

ref_dist = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        delta = X[j] - X[i]
        # Minimum image convention explicitly
        delta_wrapped = delta - period * np.round(delta / period)
        ref_dist[i, j] = np.sqrt(np.sum(delta_wrapped**2))

print("-> Explicit toroidal loop finished.")

# ---------------------------------------------------------------------------
# 3. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")

metric = PBCMetric(period=period)
dimred_dist = metric.pairwise(X)
print("-> DimRedPy PBCMetric finished.")

# ---------------------------------------------------------------------------
# 4. Parity Validation
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          PARITY VALIDATION")
print("=======================================================")

mse = np.mean((ref_dist - dimred_dist) ** 2)
print(f"MSE on PBC Distance Matrix: {mse:.6e}")

if mse < 1e-10:
    print("\nSUCCESS: DimRedPy PBCMetric perfectly matches explicit reference.")
else:
    print("\nFAILURE: Divergence detected in PBC Metric parity.")

# ---------------------------------------------------------------------------
# 5. Visual Verification
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='white')
fig.suptitle(f"PBC Metric Parity\nMSE: {mse:.2e}", fontweight='bold', y=1.05)

vmax = np.max(ref_dist)

im0 = axes[0].imshow(ref_dist, aspect='auto', cmap='viridis', vmax=vmax)
axes[0].set_title("Explicit Loop Distances")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(dimred_dist, aspect='auto', cmap='viridis', vmax=vmax)
axes[1].set_title("DimRedPy Distances")
fig.colorbar(im1, ax=axes[1])

diff = np.abs(ref_dist - dimred_dist)
im2 = axes[2].imshow(diff, aspect='auto', cmap='Reds')
axes[2].set_title("Absolute Difference")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plot_path = os.path.join(script_dir, "verify_parity_metrics_pbc_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved comparison plot to {plot_path}")
