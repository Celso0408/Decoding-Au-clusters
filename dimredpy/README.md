# DimRedPy — Framework Technical Manual

**DimRedPy** is a unified, high-performance Python framework for Nonlinear Dimensionality Reduction (NLDR), Out-of-Sample Projection, and MBAR thermodynamic analysis. 

It is designed to be **domain-agnostic**, providing a consistent API for complex scientific data analysis while leveraging modern backends like PyTorch, openTSNE, umap-learn, and NVIDIA RAPIDS.

---

## 🏛 Framework Overview

DimRedPy is organized into several specialized sub-packages, all accessible via the `dimredpy` namespace.

| Sub-package | Core Algorithm | Primary Imports |
|---|---|---|
| `dimredpy.sketchmap` | Sketch-map / MDS | `sketch_map`, `select_landmarks`, `project_out_of_sample` |
| `dimredpy.fitsne` | FIt-SNE | `fit_sne` |
| `dimredpy.umap_embed` | UMAP | `umap_embed` |
| `dimredpy.mbar` | MBAR Analysis | `mbar_free_energy_surface`, `decorrelate_timeseries` |
| `dimredpy.shared` | Utilities | `descriptors`, `metrics`, `transfer`, `io` |

---

## 📉 Dimensionality Reduction Modules

### 1. FIt-SNE (`dimredpy.fitsne`)
FFT-accelerated interpolation-based t-SNE. Scales to millions of points with $O(N \log N)$ complexity by using the FFT-based negative gradient method.

#### **Function Signature**
```python
from dimredpy.fitsne import fit_sne

embedding = fit_sne(data, n_components=2, perplexity=30.0, n_iter=1000, 
                    early_exaggeration=12.0, early_exaggeration_iter=250, 
                    learning_rate="auto", metric="euclidean", 
                    min_num_intervals=50, negative_gradient_method="fft", 
                    seed=42, n_jobs=-1, verbose=False, use_gpu=True, **kwargs)
```

#### **Parameters**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`data`** | `np.ndarray` | *Required* | High-dimensional input array of shape $(N, D)$. |
| **`n_components`** | `int` | `2` | Dimensionality of the target embedding space. |
| **`perplexity`** | `float` | `30.0` | Effective neighbors. Larger values emphasize global topology. |
| **`n_iter`** | `int` | `1000` | Number of gradient descent iterations. |
| **`early_exaggeration`** | `float` | `12.0` | Affinity multiplier for the early phase. Larger = tighter clusters. |
| **`early_exaggeration_iter`** | `int` | `250` | Duration of the early exaggeration phase. |
| **`learning_rate`** | `float/str` | `"auto"` | Step size. `"auto"` uses $max(200, N / early\_exag)$ for stability. |
| **`metric`** | `str` | `"euclidean"` | Distance metric used in the high-dimensional space. |
| **`min_num_intervals`** | `int` | `50` | Grid resolution for FFT force interpolation. |
| **`negative_gradient_method`** | `str` | `"fft"` | Algorithm for repulsive forces: `"fft"` (FIt-SNE) or `"bh"` (Barnes-Hut). |
| **`seed`** | `int` | `42` | Random seed for reproducible results. |
| **`n_jobs`** | `int` | `-1` | Number of CPU threads for `openTSNE` (-1 = all cores). |
| **`verbose`** | `bool` | `False` | If True, prints iteration logs and backend selection details. |
| **`use_gpu`** | `bool` | `True` | Attempts to use `cuml.TSNE` for massive GPU acceleration. |
| **`**kwargs`** | - | - | Additional arguments passed to the underlying TSNE constructor. |

#### **Returns**
| Name | Type | Description |
|:---|:---:|:---|
| **`embedding`** | `np.ndarray` | Projected coordinates of shape $(N, n\_components)$. |

---

### 2. UMAP (`dimredpy.umap_embed`)
Uniform Manifold Approximation and Projection. Features specialized support for the **Mahalanobis** metric on both CPU and GPU backends.

#### **Function Signature**
```python
from dimredpy.umap_embed import umap_embed

umap_embed(data, n_components=2, metric="euclidean", n_neighbors=15, 
           min_dist=0.1, seed=42, n_jobs=-1, verbose=False, 
           use_gpu=True, **kwargs)
```

#### **Parameters**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`data`** | `np.ndarray` | *Required* | High-dimensional input array of shape $(N, D)$. |
| **`n_components`** | `int` | `2` | Dimensionality of the target embedding space. |
| **`metric`** | `str` | `"euclidean"` | Distance metric. Supports `"mahalanobis"`, `"cosine"`, etc. |
| **`n_neighbors`** | `int` | `15` | Local neighborhood size. Larger = more global structure. |
| **`min_dist`** | `float` | `0.1` | Minimum distance in embedding space (0.001 to 0.5). |
| **`seed`** | `int` | `42` | Random seed for reproducibility. |
| **`n_jobs`** | `int` | `-1` | Number of CPU threads to use (-1 = all cores). |
| **`verbose`** | `bool` | `False` | If True, prints progress and backend selection. |
| **`use_gpu`** | `bool` | `True` | Attempts to use `cuml.UMAP` for massive GPU acceleration. |
| **`**kwargs`** | - | - | Additional arguments passed directly to the UMAP constructor. |

#### **Returns**
| Name | Type | Description |
|:---|:---:|:---|
| **`embedding`** | `np.ndarray` | Projected coordinates of shape $(N, n\_components)$. |


---

### 3. Sketch-map (`dimredpy.sketchmap`)
A comprehensive implementation of the Sketch-map algorithm, optimized for high-performance Python and PyTorch. This module handles the full lifecycle: selection, embedding, and out-of-sample projection.

#### **A. Landmark Selection: `select_landmarks`**
Selects a representative subset of points (landmarks) from a large dataset.

```python
from dimredpy.sketchmap import select_landmarks

result = select_landmarks(data, n_landmarks, mode="minmax", metric=None, 
                          input_weights=None, seed=12345, first=-1, 
                          unique=False, return_weights=True, 
                          weight_gamma=1.0, resample_gamma=1.0, similarity=None)
```

#### **Parameters**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`data`** | `np.ndarray` | *Required* | High-dimensional input array of shape $(N, D)$. |
| **`n_landmarks`** | `int` | *Required* | Total number of landmarks $(K)$ to extract. |
| **`mode`** | `str` | `"minmax"` | Strategy: `"minmax"` (FPS), `"random"`, `"stride"`, `"resample"`, or `"staged"`. |
| **`metric`** | `Metric` | `None` | HD distance metric (defaults to Euclidean). |
| **`input_weights`** | `np.ndarray` | `None` | Bias weights for `"resample"` or `"staged"` modes. |
| **`seed`** | `int` | `12345` | Random seed for stochastic modes. |
| **`first`** | `int` | `-1` | Index of the first landmark (defaults to random). |
| **`unique`** | `bool` | `False` | If True, ensures all landmarks are unique indices. |
| **`return_weights`** | `bool` | `True` | Compute Voronoi/density weights for each landmark. |
| **`weight_gamma`** | `float` | `1.0` | Exponent for density-weight normalization. |
| **`resample_gamma`** | `float` | `1.0` | Exponent for biased resampling in `"resample"` mode. |
| **`similarity`** | `np.ndarray` | `None` | Pre-computed HD distance matrix (for `"minmax"` mode). |

#### **Returns**
| Name | Type | Description |
|:---|:---:|:---|
| **`landmarks`** | `np.ndarray` | High-dimensional coordinates of the selected $(K, D)$ points. |
| **`indices`** | `np.ndarray` | Original indices of the selected points in the input data. |
| **`weights`** | `np.ndarray` | Normalized Voronoi weights (if `return_weights=True`). |

---

#### **B. Classical MDS: `classical_mds`**
Standard linear dimensionality reduction, often used as the initialization for Sketch-map.

```python
from dimredpy.sketchmap import classical_mds

result = classical_mds(data, n_components=2, metric=None, 
                       dist_matrix=None, verbose=False)
```

#### **Parameters**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`data`** | `np.ndarray` | *Required* | Input data of shape $(N, D)$. |
| **`n_components`** | `int` | `2` | Target embedding dimensionality. |
| **`metric`** | `Metric` | `None` | Distance metric for MDS. |
| **`dist_matrix`** | `np.ndarray` | `None` | If provided, computes MDS directly from pre-calculated distances. |

#### **Returns**
| Name | Type | Description |
|:---|:---:|:---|
| **`embedding`** | `np.ndarray` | Low-dimensional coordinates of shape $(N, n\_components)$. |
| **`eigenvalues`** | `np.ndarray` | Eigenvalues of the Gram matrix. |
| **`error`** | `float` | Residual MDS stress. |

---

#### **C. Sketch-map Optimization: `sketch_map`**
The core optimization engine that uses sigmoid-transformed distances.

```python
from dimredpy.sketchmap import sketch_map

result = sketch_map(data, n_components=2, weights=None, init=None, 
                    metric=None, fun_hd=(6.0, 8.0, 8.0), fun_ld=(6.0, 2.0, 8.0), 
                    preopt_steps=100, grid=None, global_steps=0, 
                    imix=0.0, dist_matrix=None, verbose=False)
```

#### **Parameters**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`data`** | `np.ndarray` | *Required* | $(N, D)$ landmark coordinates. |
| **`n_components`** | `int` | `2` | Number of dimensions in the embedding space. |
| **`weights`** | `np.ndarray` | `None` | Per-point statistical weights (from selection). |
| **`init`** | `np.ndarray` | `None` | Starting embedding (e.g., from MDS). |
| **`metric`** | `Metric` | `None` | Distance metric (defaults to Euclidean). |
| **`fun_hd` / `fun_ld`** | `tuple` | *(6,8,8) / (6,2,8)* | Sigmoid parameters $(\sigma, A, B)$. |
| **`preopt_steps`** | `int` | `100` | Maximum Conjugate Gradient iterations. |
| **`grid`** | `tuple` | `None` | Pointwise global spec: `(width, coarse, fine)`. |
| **`global_steps`** | `int` | `0` | Steps of stochastic global search (uncommon). |
| **`imix`** | `float` | `0.0` | Mix ratio between Sketch-map (0.0) and MDS (1.0). |
| **`dist_matrix`** | `np.ndarray` | `None` | Pre-computed HD distance matrix for optimization. |
| **`verbose`** | `bool` | `False` | Print optimization progress and final stress. |

#### **Returns**
| Name | Type | Description |
|:---|:---:|:---|
| **`embedding`** | `np.ndarray` | Optimized coordinates of shape $(N, n\_components)$. |
| **`stress`** | `float` | Final scalar $\chi^2$ stress. |
| **`per_point_errors`** | `np.ndarray` | Contribution of each point to the total stress. |

---

#### **D. Out-of-Sample Projection: `project_out_of_sample`**
Maps new data points into an existing low-dimensional embedding.

```python
from dimredpy.sketchmap import project_out_of_sample

result = project_out_of_sample(samples, landmarks_hd, landmarks_ld, weights=None, 
                               metric=None, fun_hd=(6.0, 8.0, 8.0), 
                               fun_ld=(6.0, 2.0, 8.0), grid=(1.0, 21, 201), 
                               cg_steps=3, gt=0.0, similarity=False, 
                               imix=0.0, use_gpu=False, verbose=False)
```

#### **Parameters**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`samples`** | `np.ndarray` | *Required* | New high-dim points $(M, D)$ to project. |
| **`landmarks_hd`** | `np.ndarray` | *Required* | Coordinates of original landmarks $(K, D)$. |
| **`landmarks_ld`** | `np.ndarray` | *Required* | Embedding of the landmarks $(K, d)$. |
| **`weights`** | `np.ndarray` | `None` | Landmark weights (from selection). |
| **`metric`** | `Metric` | `None` | HD distance metric (defaults to Euclidean). |
| **`fun_hd` / `fun_ld`** | `tuple` | *(6,8,8) / (6,2,8)* | Sigmoid parameters $(\sigma, A, B)$. |
| **`grid`** | `tuple` | `(1,21,201)` | `(width, coarse, fine)` for 2D search. |
| **`cg_steps`** | `int` | `0` | Refinement steps (Adam/CG) after grid search. |
| **`gt`** | `float` | `0.0` | Global threshold (legacy `-gt` flag). |
| **`similarity`** | `bool` | `False` | If True, `samples` are HD distances to landmarks. |
| **`imix`** | `float` | `0.0` | Mix ratio between Sketch-map and MDS stress. |
| **`use_gpu`** | `bool` | `False` | Enables parallel projection using PyTorch CUDA. |
| **`verbose`** | `bool` | `False` | Print progress for large-scale projections. |

#### **Returns**
| Name | Type | Description |
|:---|:---:|:---|
| **`embedding`** | `np.ndarray` | Projected coordinates of shape $(M, n\_components)$. |
| **`error`** | `np.ndarray` | Per-sample projection stress at the optimum. |
| **`nearest_distance`** | `np.ndarray` | HD distance to the nearest landmark for each sample. |

---

### 4. MBAR Free-Energy Surfaces (`dimredpy.mbar`)
High-level workflow for constructing free-energy surfaces (FES) from multi-state simulation data (e.g., Parallel Tempering / Replica Exchange). Wraps `pymbar`.

#### **Main Function: `mbar_free_energy_surface`**
The primary entry point for end-to-end surface construction.

```python
from dimredpy.mbar import mbar_free_energy_surface

result = mbar_free_energy_surface(energies, temperatures, collective_vars, 
                                  target_temperature=None, sample_assignments=None, 
                                  n_bins=50, extent=None, kde=True, 
                                  decorrelate=False, mbar_kwargs=None, **kwargs)
```

#### **Parameters**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`energies`** | `np.ndarray` | *Required* | Flattened array of potential energies for all samples. |
| **`temperatures`** | `np.ndarray` | *Required* | The temperature ladder $(K)$ used in the simulation. |
| **`collective_vars`** | `np.ndarray` | *Required* | $(N, 2)$ array of projection coordinates. |
| **`target_temperature`** | `float` | `None` | Temperature at which to evaluate the FES (defaults to ladder start). |
| **`sample_assignments`** | `np.ndarray` | `None` | Mapping of samples to originating replicas $(0 \dots K-1)$. |
| **`n_bins`** | `int` | `50` | Grid resolution for the output surface. |
| **`extent`** | `tuple` | `None` | Manual grid limits `(x_min, x_max, y_min, y_max)`. |
| **`kde`** | `bool` | `True` | Use Kernel Density Estimation (KDE) for a smooth surface. |
| **`decorrelate`** | `bool` | `False` | Automatically subsamples the time series for independence. |
| **`mbar_kwargs`** | `dict` | `None` | Extra arguments passed directly to `pymbar.MBAR`. |

#### **Returns**
| Name | Type | Description |
|:---|:---:|:---|
| **`probability`** | `np.ndarray` | $(n\_bins, n\_bins)$ normalized probability surface. |
| **`free_energy`** | `np.ndarray` | $(n\_bins, n\_bins)$ free-energy surface in units of $k_B T$. |
| **`bin_centers_x / y`** | `np.ndarray` | Coordinate axes for plotting. |
| **`mbar_weights`** | `np.ndarray` | $(K, N)$ matrix of statistical weights. |
| **`N_eff`** | `np.ndarray` | Effective sample size for each state. |

#### **Modular Sub-functions**
For advanced workflows, you can use the internal components directly:
- **`decorrelate_timeseries(data, energies, method="statistical_inefficiency")`**: Extracts independent samples from a correlated trajectory.
- **`build_u_kn(energies, temperatures)`**: Constructs the reduced potential matrix.
- **`run_mbar(u_kn, N_k, solver="default")`**: The core solver wrapper that returns the weight matrix.
- **`build_probability_surface(cv, weights, state_index)`**: Grids the weights into a 2D surface.

---

---

## 🛠 Shared Utilities (`dimredpy.shared`)

The `shared` module provides the core building blocks for structural analysis, custom metrics, and distance preservation diagnostics.

### **1. Structural Descriptors (`dimredpy.shared.descriptors`)**
Geometric analysis tools for 3D point clouds and trajectories.

| Function | Description | Returns |
|:---|:---|:---|
| **`coordination_histogram`** | Normalized histogram of neighbor counts per point (cutoff based). | `np.ndarray` |
| **`effective_coordination_number`** | **ECN**: A continuous measure of local atomic density. | `float` |
| **`radius_of_gyration`** | Geometric **Rg** (root-mean-square distance to center of mass). | `float` |
| **`hausdorff_chirality_measure`** | Quantifies structural chirality $[0, 1]$ (0 = achiral). | `float` |
| **`radial_distribution_function`** | Computes the trajectory-wide radial distribution **g(r)**. | `(r, g_r)` |
| **`compute_trajectory_descriptors`**| Batch-processes a trajectory for all the above metrics. | `dict` |

---

### **2. Distance Metrics (`dimredpy.shared.metrics`)**
Custom metrics used across all dimensionality reduction modules.

| Metric | Parameters | Description |
|:---|:---|:---|
| **`EuclideanMetric()`** | - | Standard $L_2$ Euclidean distance. |
| **`PBCMetric(period)`** | `period` | Periodic Boundary Conditions (scalar or $(D,)$ array). |
| **`SphericalMetric(period)`** | `period` | Geodesic distances on a hyper-sphere (last dim azimuthal). |
| **`DotMetric()`** | - | Metric defined as $d(a,b) = -\ln(a \cdot b)$. |
| **`get_metric(...)`** | `period`, `sphere_period`, `dot` | Factory function to retrieve metric based on CLI-style flags. |

---

### **3. Transfer Functions (`dimredpy.shared.transfer`)**
Mathematical kernels for Sketch-map. All functions support `.f(x)`, `.df(x)`, and `.fdf(x)`.

| Function | Parameters | Description |
|:---|:---|:---|
| **`XSigmoid`** | `sigma, A, B` | Extended sigmoid: $1 - [1 + (2^{A/B} - 1) (R/\sigma)^A]^{-B/A}$. |
| **`Sigmoid`** | `sigma` | Squared-Lorentzian sigmoid ($A=2, B \to \infty$). |
| **`Identity()`** | - | Identity mapping $F(R) = R$ (standard MDS limit). |
| **`Warp`** | `sigma, A_hd, B_hd, a_ld, b_ld` | Maps HD distances through an LD inverse mapping. |
| **`make_transfer(spec)`** | `spec` (tuple) | Factory mapping tuples (e.g. `(6,8,8)`) to objects. |

---

### **4. Data I/O (`dimredpy.shared.io`)**
Agnostic loaders for simulation and embedding data.

| Component | Description | Format / Context |
|:---|:---|:---|
| **`load_spatial_coordinates`** | Parses 3D trajectories. | Extended `XYZ` |
| **`load_point_set`** | Loads landmarks and weights from disk. | Whitespace-delimited `txt` |
| **`save_grid_surface`** | Saves 2D energy/probability maps. | Supports `gnuplot=True` block format. |

---

### **5. Analysis & Diagnostics (`dimredpy.shared.analysis`)**
Tools for measuring the quality of a dimensionality reduction.

| Function | Description | Returns |
|:---|:---|:---|
| **`distance_histogram`** | 1D (HD) or 2D (HD vs LD) histogram of pairwise distances. | `dict` (bins + counts) |
| **`preservation_score`** | Computes the scalar $\chi^2$ stress for an existing embedding. | `float` (Lower is better) |

---

## 🚀 Advanced Usage: The GPU Advantage

DimRedPy is optimized for high-performance scientific workflows, utilizing hardware acceleration and mathematical transforms to ensure both speed and accuracy.

### **1. GPU Acceleration & Batching**
- **FIt-SNE & UMAP**: Leverages `cuML` (NVIDIA RAPIDS) for up to 100x speedup over CPU implementations.
- **Sketch-map Projection**: Uses **PyTorch CUDA batches** in `project_out_of_sample` to parallelize the grid search and Adam refinement for thousands of samples simultaneously.

### **2. Automated Mahalanobis Whitening**
Since `cuML` does not natively support the Mahalanobis metric, DimRedPy implements a mathematical workaround for GPU execution:
1. It computes the sample covariance matrix $S$.
2. It calculates the whitening transform $S^{-1/2}$.
3. It transforms the data: $X_{white} = (X - \mu) S^{-1/2}$.
Performing standard Euclidean UMAP on the whitened data is **mathematically identical** to performing Mahalanobis UMAP on the original data. This ensures 100% parity between CPU and GPU backends.

---

## ✅ Installation & Testing

```bash
# Development installation
pip install -e .

# Run the comprehensive test suite
python -m pytest dimredpy/tests
```
