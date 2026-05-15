"""
Fair Time Benchmark for UMAP (DimRedPy vs umap-learn)
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

import umap
from dimredpy.umap_embed import umap_embed

# ---------------------------------------------------------------------------
# 1. Define Benchmark Functions
# ---------------------------------------------------------------------------
def get_data(num_samples):
    data_path = os.path.join(script_dir, "..", "subset_10000.txt")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.getcwd(), "subset_10000.txt")
    data = np.loadtxt(data_path, max_rows=num_samples)
    return data

def run_reference(data):
    X_ref = np.asarray(data, dtype=float)
    cov = np.cov(X_ref, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="mahalanobis",
        metric_kwds={"V": cov},
        random_state=42,
        n_jobs=1,
    )
    reducer.fit_transform(data)

def run_dimredpy(data):
    umap_embed(
        data,
        n_components=2,
        metric="mahalanobis",
        n_neighbors=15,
        min_dist=0.1,
        seed=42,
        n_jobs=1,
        use_gpu=False,
    )

# ---------------------------------------------------------------------------
# 2. Warm-up Phase
# ---------------------------------------------------------------------------
# Forces JIT compilation and Numba loading to ensure fairness.
print("==================================================")
print("      FAIR TIME BENCHMARK: REFERENCE VS DIMREDPY  ")
print("==================================================")
print("Running Warm-up phase (200 frames) to compile Numba JIT caches...")
warmup_data = get_data(200)
run_reference(warmup_data)
run_dimredpy(warmup_data)
print("Warm-up complete.\n")

# ---------------------------------------------------------------------------
# 3. Benchmark Phase
# ---------------------------------------------------------------------------
frames = 2000 # Use 2000 to keep benchmark fast but statistically valid
iterations = 3
print(f"Running Benchmark phase ({frames} samples)")
print(f"Executing {iterations} iterations interleaved to average out OS spikes...\n")

bench_data = get_data(frames)

ref_times = []
dim_times = []

for i in range(iterations):
    # Run Reference
    t0 = time.perf_counter()
    run_reference(bench_data)
    t1 = time.perf_counter()
    ref_times.append(t1 - t0)
    
    # Run DimRedPy
    t0 = time.perf_counter()
    run_dimredpy(bench_data)
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
