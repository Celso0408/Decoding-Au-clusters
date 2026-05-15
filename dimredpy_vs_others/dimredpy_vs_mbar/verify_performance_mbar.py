"""
Fair Time Benchmark for MBAR (DimRedPy vs pymbar)
"""

import numpy as np
import time
import os
import sys

# Robust path detection for repo root
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pymbar
from dimredpy.mbar import mbar_free_energy_surface

# ---------------------------------------------------------------------------
# 1. Define Benchmark Functions
# ---------------------------------------------------------------------------
def generate_mbar_data(frames_per_temp):
    np.random.seed(42)
    K = 3
    energies = np.random.randn(K * frames_per_temp) * 0.1
    cvs = np.random.randn(K * frames_per_temp, 2)
    temps = np.array([300.0, 350.0, 400.0])
    return energies, cvs, temps, K

def run_reference(energies, cvs, temps, K, frames_per_temp):
    kB_CONST = 8.617333262e-5
    beta_k = 1.0 / (kB_CONST * temps)
    N_k = np.array([frames_per_temp] * K)
    
    U_kn = np.zeros((K, K * frames_per_temp))
    for k in range(K):
        U_kn[k, :] = energies * beta_k[k]

    mbar_ref = pymbar.MBAR(U_kn, N_k)
    weights_ref = mbar_ref.W_nk.T
    
    w_300K_ref = weights_ref[0] / weights_ref[0].sum()
    prob_ref, _, _ = np.histogram2d(
        cvs[:, 0], cvs[:, 1], bins=50, weights=w_300K_ref
    )
    prob_ref = prob_ref / prob_ref.sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        fe_ref = -np.log(np.where(prob_ref > 0, prob_ref, np.nan))
        fe_ref -= np.nanmin(fe_ref)

def run_dimredpy(energies, cvs, temps):
    mbar_free_energy_surface(
        energies=energies,
        temperatures=temps,
        collective_vars=cvs,
        target_temperature=300.0,
        decorrelate=False,
        kde=False,
        n_bins=50
    )

# ---------------------------------------------------------------------------
# 2. Warm-up Phase
# ---------------------------------------------------------------------------
# Forces JIT compilation and library loading to ensure fairness.
print("==================================================")
print("      FAIR TIME BENCHMARK: REFERENCE VS DIMREDPY  ")
print("==================================================")
print("Running Warm-up phase (10 frames) to compile JIT caches...")
e_w, c_w, t_w, K_w = generate_mbar_data(10)
run_reference(e_w, c_w, t_w, K_w, 10)
run_dimredpy(e_w, c_w, t_w)
print("Warm-up complete.\n")

# ---------------------------------------------------------------------------
# 3. Benchmark Phase
# ---------------------------------------------------------------------------
frames = 10000
iterations = 5
print(f"Running Benchmark phase ({frames} frames/temp, {frames*3} total samples)")
print(f"Executing {iterations} iterations interleaved to average out OS spikes...\n")

e, c, t, K_val = generate_mbar_data(frames)

ref_times = []
dim_times = []

for i in range(iterations):
    # Run Reference
    t0 = time.perf_counter()
    run_reference(e, c, t, K_val, frames)
    t1 = time.perf_counter()
    ref_times.append(t1 - t0)
    
    # Run DimRedPy
    t0 = time.perf_counter()
    run_dimredpy(e, c, t)
    t1 = time.perf_counter()
    dim_times.append(t1 - t0)
    
    print(f"Iter {i+1}/{iterations} | Ref: {ref_times[-1]:.4f}s | DimRedPy: {dim_times[-1]:.4f}s")
    
avg_ref = np.mean(ref_times)
avg_dim = np.mean(dim_times)

print("\n==================================================")
print("               FINAL FAIR RESULTS                 ")
print("==================================================")
print(f"Average Reference Time : {avg_ref:.4f} seconds")
print(f"Average DimRedPy Time  : {avg_dim:.4f} seconds")

diff = avg_dim - avg_ref
perc = (diff / avg_ref) * 100

if abs(perc) < 5.0:
    print("\nConclusion: The execution times are statistically identical.")
    print("DimRedPy introduces negligible overhead while providing a superior API.")
elif diff > 0:
    print(f"\nConclusion: DimRedPy is {perc:.1f}% slower (adds overhead).")
else:
    print(f"\nConclusion: DimRedPy is {abs(perc):.1f}% faster.")
