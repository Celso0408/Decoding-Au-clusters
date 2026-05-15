import numpy as np
import os
from pathlib import Path

# Robust path detection
script_dir = Path(__file__).parent.absolute()
repo_root = script_dir.parent

# Paths
source_file = repo_root / "experiments" / "full_trajectory.xyz.CN_for_SketchMap"
target_file = script_dir / "subset_10000.txt"
cache_file = repo_root / "experiments" / "results" / "data_cache.npy"

print(f"Looking for data at {source_file}...")

if not source_file.exists():
    print(f"Error: Could not find {source_file}")
    print("Trying to extract from archive if it exists...")
    if cache_file.exists():
        print(f"Found cache file {cache_file}, loading from there.")
        data = np.load(cache_file)
    else:
        raise FileNotFoundError(f"Data file not found at {source_file} or {cache_file}")
else:
    print(f"Loading first 10,000 rows from {source_file}...")
    data = np.loadtxt(source_file, max_rows=10000)

if data.shape[0] >= 10000:
    subset = data[:10000]
else:
    subset = data

print(f"Saving subset of shape {subset.shape} to {target_file}")
np.savetxt(target_file, subset, fmt='%.6f')
print("Done.")
