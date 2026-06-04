"""
Rigorous Verification of MBAR Parity: Free Energy Differences
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
from dimredpy.mbar import run_mbar, mbar_free_energy_differences

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

# Free Energy Differences
ref_diff = mbar_ref.compute_free_energy_differences(compute_uncertainty=True)
if isinstance(ref_diff, dict):
    # Depending on pymbar version, keys might be Deltaf_ij or Delta_f
    ref_Deltaf = ref_diff.get("Delta_f", ref_diff.get("Deltaf_ij"))
    ref_dDeltaf = ref_diff.get("dDelta_f", ref_diff.get("dDeltaf_ij"))
else:
    ref_Deltaf = ref_diff[0]
    ref_dDeltaf = ref_diff[1]

print("-> Reference MBAR differences finished.")

# ---------------------------------------------------------------------------
# 3. Run DimRedPy Framework
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          RUNNING DIMREDPY FRAMEWORK")
print("=======================================================")

res_dimredpy = run_mbar(u_kn, N_k, solver="default")

diff_dimredpy = mbar_free_energy_differences(res_dimredpy, compute_uncertainty=True)
dimredpy_Deltaf = diff_dimredpy.get("Delta_f", diff_dimredpy.get("Deltaf_ij"))
dimredpy_dDeltaf = diff_dimredpy.get("dDelta_f", diff_dimredpy.get("dDeltaf_ij"))

print("-> DimRedPy MBAR differences finished.")

# ---------------------------------------------------------------------------
# 4. Parity Validation
# ---------------------------------------------------------------------------
print("\n=======================================================")
print("          PARITY VALIDATION")
print("=======================================================")

mse_delta_f = np.mean((ref_Deltaf - dimredpy_Deltaf) ** 2)
mse_ddelta_f = np.mean((ref_dDeltaf - dimredpy_dDeltaf) ** 2)

print(f"1. MSE on Delta f_ij                  : {mse_delta_f:.6e}")
print(f"2. MSE on dDelta f_ij (Uncertainty)   : {mse_ddelta_f:.6e}")

total_mse = mse_delta_f + mse_ddelta_f

# ---------------------------------------------------------------------------
# 5. Plotting Parity
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

r_val, _ = pearsonr(ref_Deltaf.flatten(), dimredpy_Deltaf.flatten())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"MBAR Free Energy Differences Parity\nPearson R: {r_val:.5f} | MSE: {mse_delta_f:.2e}", 
             fontsize=14, fontweight='bold', y=1.05)

im1 = axes[0].matshow(ref_Deltaf, cmap='viridis')
axes[0].set_title("Reference (pymbar)", pad=10)
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].matshow(dimredpy_Deltaf, cmap='viridis')
axes[1].set_title("DimRedPy (MBAR)", pad=10)
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plot_path = os.path.join(script_dir, "verify_parity_mbar_differences_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {plot_path}")

if total_mse < 1e-10:
    print("\nSUCCESS: DimRedPy MBAR Free Energy Differences perfectly match pymbar reference.")
else:
    print("\nFAILURE: Statistical divergence detected in MBAR parity.")
