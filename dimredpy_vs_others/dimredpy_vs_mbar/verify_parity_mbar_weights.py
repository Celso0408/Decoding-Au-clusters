"""
Rigorous Verification of MBAR Parity: Weights and Free Energies
"""

import numpy as np
import os
import sys

# Robust path detection for repo root
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pymbar import MBAR
from pymbar.testsystems import HarmonicOscillatorsTestCase
from dimredpy.mbar import run_mbar

# ---------------------------------------------------------------------------
# 1. Generate Synthetic Data
# ---------------------------------------------------------------------------
print("Generating harmonic oscillator dataset for MBAR parity...")
testcase = HarmonicOscillatorsTestCase(O_k=[0, 1, 2], K_k=[1, 1, 1])
x_n, u_kn, N_k, s_n = testcase.sample(N_k=[500, 500, 500])

print(f"u_kn shape: {u_kn.shape}")
print(f"N_k       : {N_k}")

# ---------------------------------------------------------------------------
# 2. Run Reference Framework (pymbar)
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING REFERENCE FRAMEWORK (pymbar)")
print("=======================================================")

mbar_ref = MBAR(u_kn, N_k, solver_protocol="default")

print("-> Reference MBAR weights finished.")

# ---------------------------------------------------------------------------
# 3. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")

res_dimredpy = run_mbar(u_kn, N_k, solver="default")

print("-> DimRedPy MBAR weights finished.")

# ---------------------------------------------------------------------------
# 4. Parity Validation
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          PARITY VALIDATION")
print("=======================================================")

mse_f_k = np.mean((mbar_ref.f_k - res_dimredpy["f_k"]) ** 2)
mse_weights = np.mean((mbar_ref.W_nk.T - res_dimredpy["weights"]) ** 2)

print(f"1. MSE on Reduced Free Energies (f_k) : {mse_f_k:.6e}")
print(f"2. MSE on MBAR Weights Matrix         : {mse_weights:.6e}")

total_mse = mse_f_k + mse_weights

# ---------------------------------------------------------------------------
# 5. Plotting Parity
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

r_val, _ = pearsonr(mbar_ref.W_nk.T.flatten(), res_dimredpy["weights"].flatten())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"MBAR Weights Matrix Parity\nPearson R: {r_val:.5f} | MSE: {mse_weights:.2e}", 
             fontsize=14, fontweight='bold', y=1.05)

im1 = axes[0].matshow(mbar_ref.W_nk.T, cmap='viridis', aspect='auto')
axes[0].set_title("Reference (pymbar)", pad=10)
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].matshow(res_dimredpy["weights"], cmap='viridis', aspect='auto')
axes[1].set_title("DimRedPy (MBAR)", pad=10)
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plot_path = os.path.join(script_dir, "verify_parity_mbar_weights_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {plot_path}")

if total_mse < 1e-10:
    print("\nSUCCESS: DimRedPy MBAR Weights perfectly match pymbar reference.")
else:
    print("\nFAILURE: Statistical divergence detected in MBAR parity.")
