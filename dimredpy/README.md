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
from dimredpy.fitsne import fit_sne, fitsne_project

result = fit_sne(data, n_components=2, perplexity=30.0, n_iter=1000, 
                 early_exaggeration=12.0, early_exaggeration_iter=250, 
                 learning_rate="auto", metric="euclidean", 
                 min_num_intervals=50, negative_gradient_method="fft", 
                 seed=42, n_jobs=-1, verbose=False, use_gpu=True, **kwargs)

# Out-of-sample projection
new_embedding = fitsne_project(result, new_data)
```

#### **Parameters (`fit_sne`)**
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
| **`**kwargs`** | - | - | Advanced parameters forwarded to `openTSNE` (e.g., `dof` for t-distribution degrees of freedom, `initialization="pca"`, `perplexity_list` for multi-scale embeddings). |

#### **Returns (`fit_sne`)**
Returns a dictionary containing the embedding and the trained model state for downstream projection.

| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"embedding"`** | `np.ndarray` | Projected coordinates of shape $(N, n\_components)$. |
| **`"model"`** | `object` | Fitted `openTSNE` or `cuML` model object. |

#### **Parameters (`fitsne_project`)**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`model_dict`** | `dict` | *Required* | The results dictionary returned by `fit_sne()`. |
| **`new_data`** | `np.ndarray` | *Required* | Out-of-sample data points of shape $(M, D)$ to project. |
| **`verbose`** | `bool` | `False` | If True, prints projection progress. |
| **`**kwargs`** | - | - | Advanced parameters forwarded to the projection method. |

#### **Returns (`fitsne_project`)**
| Return Value | Type | Description |
|:---|:---:|:---|
| **`embedding`** | `np.ndarray` | Projected coordinates of shape $(M, n\_components)$. |

---

### 2. UMAP (`dimredpy.umap_embed`)
Uniform Manifold Approximation and Projection. Features specialized support for the **Mahalanobis** metric on both CPU and GPU backends.

#### **Function Signature**
```python
from dimredpy.umap_embed import umap_embed, umap_project, umap_inverse_project

result = umap_embed(data, n_components=2, metric="euclidean", n_neighbors=15, 
                    min_dist=0.1, y=None, seed=42, n_jobs=-1, verbose=False, 
                    use_gpu=True, **kwargs)

# Out-of-sample projection
new_embedding = umap_project(result, new_data)

# Inverse projection (CPU only)
reconstructed_data = umap_inverse_project(result, new_embedding)
```

#### **Parameters (`umap_embed`)**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`data`** | `np.ndarray` | *Required* | High-dimensional input array of shape $(N, D)$. |
| **`n_components`** | `int` | `2` | Dimensionality of the target embedding space. |
| **`metric`** | `str` | `"euclidean"` | Distance metric. Supports `"mahalanobis"`, `"cosine"`, etc. |
| **`n_neighbors`** | `int` | `15` | Local neighborhood size. Larger = more global structure. |
| **`min_dist`** | `float` | `0.1` | Minimum distance in embedding space (0.001 to 0.5). |
| **`y`** | `np.ndarray` | `None` | Target labels/values for supervised dimensionality reduction. |
| **`seed`** | `int` | `42` | Random seed for reproducibility. |
| **`n_jobs`** | `int` | `-1` | Number of CPU threads to use (-1 = all cores). |
| **`verbose`** | `bool` | `False` | If True, prints progress and backend selection. |
| **`use_gpu`** | `bool` | `True` | Attempts to use `cuml.UMAP` for massive GPU acceleration. |
| **`**kwargs`** | - | - | Additional arguments passed directly to the UMAP constructor. |

#### **Returns (`umap_embed`)**
Returns a dictionary containing the embedding and the trained model state for downstream projection.

| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"embedding"`** | `np.ndarray` | Projected coordinates of shape $(N, n\_components)$. |
| **`"model"`** | `object` | Fitted `umap-learn` or `cuML` model object. |
| **`"metric"`** | `str` | Distance metric used. |
| **`"mahalanobis_params"`**| `tuple` | Whitening states `(mu, whitener)` if GPU Mahalanobis was used. |

#### **Parameters (`umap_project`)**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`model_dict`** | `dict` | *Required* | The results dictionary returned by `umap_embed()`. |
| **`new_data`** | `np.ndarray` | *Required* | Out-of-sample data points of shape $(M, D)$ to project. |
| **`verbose`** | `bool` | `False` | If True, prints projection progress. |

#### **Returns (`umap_project`)**
| Return Value | Type | Description |
|:---|:---:|:---|
| **`embedding`** | `np.ndarray` | Projected coordinates of shape $(M, n\_components)$. |

#### **Parameters (`umap_inverse_project`)**
> [!WARNING]
> Inverse projection is currently only supported by the CPU backend (`umap-learn`). The underlying UMAP model must have been created with `use_gpu=False`.

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`model_dict`** | `dict` | *Required* | The results dictionary returned by `umap_embed()`. |
| **`new_ld_data`** | `np.ndarray` | *Required* | Low-dimensional data points of shape $(M, n\_components)$. |
| **`verbose`** | `bool` | `False` | If True, prints projection progress. |

#### **Returns (`umap_inverse_project`)**
| Return Value | Type | Description |
|:---|:---:|:---|
| **`embedding`** | `np.ndarray` | Reconstructed high-dimensional coordinates of shape $(M, D)$. |

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
                          weight_gamma=1.0, resample_gamma=1.0, similarity=None,
                          batch_size=None)
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
| **`batch_size`** | `int` | `None` | Optional chunk size for computing Voronoi weights (`return_weights=True`) to prevent Out-Of-Memory (OOM) errors on extremely large datasets. |

#### **Returns**
| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"landmarks"`** | `np.ndarray` | High-dimensional coordinates of the selected $(K, D)$ points. |
| **`"indices"`** | `np.ndarray` | Original indices of the selected points in the input data. |
| **`"weights"`** | `np.ndarray` | Normalized Voronoi weights (if `return_weights=True`). |

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
| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"embedding"`** | `np.ndarray` | Low-dimensional coordinates of shape $(N, n\_components)$. |
| **`"eigenvalues"`** | `np.ndarray` | Eigenvalues of the Gram matrix. |
| **`"error"`** | `float` | Residual MDS stress. |
| **`"per_point_errors"`** | `np.ndarray` | Per-point MDS errors (only if `verbose=True`). |

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

| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"embedding"`** | `np.ndarray` | Optimized coordinates of shape $(N, n\_components)$. |
| **`"stress"`** | `float` | Final scalar $\chi^2$ stress. |
| **`"per_point_errors"`** | `np.ndarray` | Contribution of each point to the total stress (only if `verbose=True`). |

---

#### **D. Out-of-Sample Projection: `project_out_of_sample`**
Maps new data points into an existing low-dimensional embedding.

```python
from dimredpy.sketchmap import project_out_of_sample

result = project_out_of_sample(samples, landmarks_hd, landmarks_ld, weights=None, 
                               metric=None, fun_hd=(6.0, 8.0, 8.0), 
                               fun_ld=(6.0, 2.0, 8.0), grid=(1.0, 21, 201), 
                               cg_steps=0, gt=0.0, similarity=False, 
                               imix=0.0, use_gpu=False, verbose=False)
```

#### **Parameters**
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`samples`** | `np.ndarray` | *Required* | New high-dim points $(M, D)$ to project. *Note: Memory-safe! Automatic VRAM/RAM batching is implemented internally to prevent OOMs on millions of points.* |
| **`landmarks_hd`** | `np.ndarray` | *Required* | Coordinates of original landmarks $(K, D)$. |
| **`landmarks_ld`** | `np.ndarray` | *Required* | Embedding of the landmarks $(K, d)$. |
| **`weights`** | `np.ndarray` | `None` | Landmark weights (from selection). |
| **`metric`** | `Metric` | `None` | HD distance metric (defaults to Euclidean). |
| **`fun_hd` / `fun_ld`** | `tuple` | *(6,8,8) / (6,2,8)* | Sigmoid parameters $(\sigma, A, B)$. |
| **`grid`** | `tuple` | `(1,21,201)` | `(width, coarse, fine)` for initial 2D search. |
| **`cg_steps`** | `int` | `0` | **Crucial:** Refinement steps after the initial grid search. In CPU mode, acts as `maxiter` for Scipy CG. In GPU mode (`use_gpu=True`), any value > 0 triggers exactly **100 steps of continuous Adam optimization** to slide points into their exact geometric basins (matching the old C++ `-cgmin` flag). If `0`, points remain locked to a discrete grid! |
| **`gt`** | `float` | `0.0` | Global threshold (legacy `-gt` flag). |
| **`similarity`** | `bool` | `False` | If True, `samples` are HD distances to landmarks. |
| **`imix`** | `float` | `0.0` | Mix ratio between Sketch-map and MDS stress. |
| **`use_gpu`** | `bool` | `False` | Enables parallel projection using PyTorch CUDA (with automatic batched processing for massive trajectories). |
| **`verbose`** | `bool` | `False` | Print progress for large-scale projections. |

| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"embedding"`** | `np.ndarray` | Projected coordinates of shape $(M, n\_components)$. |
| **`"error"`** | `np.ndarray` | Per-sample projection stress at the optimum. |
| **`"nearest_distance"`** | `np.ndarray` | HD distance to the nearest landmark for each sample. |

---

### 4. MBAR Thermodynamic Analysis (`dimredpy.mbar`)

High-level workflow for constructing free-energy surfaces (FES), extracting free energy differences, and computing observable expectations from multi-state simulation data (e.g., Parallel Tempering / Replica Exchange). Wraps `pymbar` to ensure numerical consistency.

#### **A. End-to-End Workflow: `mbar_free_energy_surface`**
The primary entry point for full surface construction.

```python
from dimredpy.mbar import mbar_free_energy_surface

result = mbar_free_energy_surface(energies, temperatures, collective_vars, 
                                  target_temperature=None, sample_assignments=None, 
                                  n_bins=50, extent=None, kde=True, 
                                  decorrelate=False, mbar_kwargs=None, **kwargs)
```

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`energies`** | `np.ndarray` | *Required* | `(N_total,)` array of potential energies across all replicas. |
| **`temperatures`** | `np.ndarray` | *Required* | `(K,)` array representing the temperature ladder $(K)$. |
| **`collective_vars`** | `np.ndarray` | *Required* | `(N_total, 2)` array of 2D projection coordinates. |
| **`target_temperature`** | `float` | `None` | Temperature at which to evaluate the FES (defaults to ladder start). |
| **`sample_assignments`** | `np.ndarray` | `None` | `(N_total,)` mapping of samples to originating replicas. |
| **`n_bins`** | `int` | `50` | Grid resolution for the output surface. |
| **`extent`** | `tuple` | `None` | Manual grid limits `(x_min, x_max, y_min, y_max)`. |
| **`kde`** | `bool` | `True` | Use Kernel Density Estimation (KDE) for a smooth surface. |
| **`decorrelate`** | `bool` | `False` | Automatically subsamples the time series for independence. |
| **`mbar_kwargs`** | `dict` | `None` | Extra arguments passed directly to `pymbar.MBAR`. |

| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"probability"`** | `np.ndarray` | `(n_bins, n_bins)` normalized probability surface. |
| **`"free_energy"`** | `np.ndarray` | `(n_bins, n_bins)` free-energy surface in units of $k_B T$. |
| **`"bin_centers_x"`** | `np.ndarray` | Coordinate axis along X. |
| **`"bin_centers_y"`** | `np.ndarray` | Coordinate axis along Y. |
| **`"mbar_weights"`** | `np.ndarray` | `(K, N_total)` matrix of statistical weights. |
| **`"f_k"`** | `np.ndarray` | `(K,)` unitless reduced free energies. |
| **`"N_eff"`** | `np.ndarray` | `(K,)` effective sample size for each state. |

---

#### **B. Preprocessing & Utilities**
For advanced workflows, you can use the internal MBAR data preparation components directly.

```python
from dimredpy.mbar import decorrelate_timeseries, build_u_kn, build_probability_surface
```

- **`decorrelate_timeseries(data, energies=None, method="statistical_inefficiency", max_stride=500)`**: Extracts independent samples from a correlated trajectory. Returns a dictionary with `"data"`, `"indices"`, and `"g"` (statistical inefficiency).
- **`build_u_kn(energies, temperatures, sample_assignments=None)`**: Constructs the reduced potential matrix. Returns `(u_kn, N_k)` where `u_kn` is the `(K, N_total)` matrix and `N_k` is sample counts per state.
- **`build_probability_surface(collective_vars, mbar_weights, state_index=0, n_bins=50, extent=None, kde=False, kde_bandwidth=None)`**: Grids the MBAR weights into a 2D surface. Returns a dict containing `"probability"`, `"free_energy"`, and axis bin centers.

---

#### **C. Core MBAR Solver: `run_mbar`**
Solves the MBAR equations to find the unitless free energies $f_k$ and the statistical weights $W_{n,k}$ for all samples.

```python
from dimredpy.mbar import run_mbar

mbar_dict = run_mbar(u_kn, N_k, solver="default")
```

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`u_kn`** | `np.ndarray` | *Required* | `(K, N)` matrix of reduced potentials $u_k(x_n)$. |
| **`N_k`** | `np.ndarray` | *Required* | `(K,)` array of sample counts per state. |
| **`solver`** | `str` | `"default"` | Solver protocol (`"default"`, `"robust"`, `"jax"`). |

| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"f_k"`** | `np.ndarray` | `(K,)` unitless reduced free energies per state. |
| **`"weights"`** | `np.ndarray` | `(K, N)` matrix of statistical weights. |
| **`"mbar"`** | `MBAR` | The raw fitted `pymbar.MBAR` object. |
| **`"N_eff"`** | `np.ndarray` | `(K,)` effective sample size per state. |

---

#### **D. Free Energy Differences: `mbar_free_energy_differences`**
Computes the exact matrix of dimensionless free energy differences ($\Delta f_{ij}$) between all sampled states and their uncertainties.

```python
from dimredpy.mbar import mbar_free_energy_differences

diffs = mbar_free_energy_differences(mbar_dict, compute_uncertainty=True)
```

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`mbar_dict`** | `dict` | *Required* | The results dictionary returned by `run_mbar()`. |
| **`compute_uncertainty`** | `bool` | `True` | Calculate the uncertainties (standard errors). |
| **`uncertainty_method`** | `str` | `None` | Method for uncertainty estimation (e.g., `"svd"`). |

| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"Delta_f"`** | `np.ndarray` | `(K, K)` matrix of dimensionless free energy differences $f_j - f_i$. |
| **`"dDelta_f"`** | `np.ndarray` | `(K, K)` matrix of standard errors in the estimates. |

---

#### **E. Observable Expectations: `mbar_compute_expectations`**
Estimates the expected average value of an observable $A$ across each thermodynamic state.

```python
from dimredpy.mbar import mbar_compute_expectations

exp = mbar_compute_expectations(mbar_dict, A_kn, state_dependent=False, compute_uncertainty=True)
```

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`mbar_dict`** | `dict` | *Required* | The results dictionary returned by `run_mbar()`. |
| **`A_kn`** | `np.ndarray` | *Required* | The observable evaluated for each sample. |
| **`state_dependent`** | `bool` | `False` | Whether $A_{kn}$ has a different functional form in each state. |
| **`compute_uncertainty`** | `bool` | `True` | Calculate the standard errors of the expectations. |

| Dictionary Key | Type | Description |
|:---|:---:|:---|
| **`"mu"`** | `np.ndarray` | `(K,)` expected values of the observable in each state. |
| **`"sigma"`** | `np.ndarray` | `(K,)` standard errors of the expectations. |

---

## 🛠 Shared Utilities (`dimredpy.shared`)

The `shared` module provides the core building blocks for structural analysis, custom metrics, distance preservation diagnostics, and I/O. These components are designed to be used independently or in conjunction with the dimensionality reduction workflows.

### **1. Structural Descriptors (`dimredpy.shared.descriptors`)**
Geometric analysis tools for 3D point clouds and trajectories. These descriptors are highly useful for defining **Collective Variables (CVs)** or analyzing the structural evolution of a system.

```python
from dimredpy.shared.descriptors import (
    coordination_histogram, coordination_histogram_trajectory,
    effective_coordination_number, average_neighbor_distance,
    radius_of_gyration, radial_distribution_function,
    hausdorff_chirality_measure, projection_center,
    compute_trajectory_descriptors
)
```

#### **Functions & API Details**

##### **`coordination_histogram`**
Calculates the coordination number (neighbor count) histogram for a single frame.
- **Parameters**:
  - `positions` (`np.ndarray`): `(N, 3)` coordinates.
  - `cutoff` (`float`): Distance cutoff (defaults to `3.4`).
  - `max_neighbors` (`int`): Maximum neighbor count tracked in the histogram (defaults to `12`).
- **Returns**: `np.ndarray` of shape `(max_neighbors + 1,)` where entry `i` represents the fraction of points having exactly `i` neighbors.

##### **`coordination_histogram_trajectory`**
Computes neighbor count histograms for all frames in a trajectory.
- **Parameters**: Same as `coordination_histogram` but accepts `trajectory` of shape `(T, N, 3)`.
- **Returns**: `np.ndarray` of shape `(T, max_neighbors + 1)`.

##### **`effective_coordination_number`**
Computes the mean continuous Effective Coordination Number (ECN) for a single frame. ECN acts as a soft, continuous version of coordination numbers.
- **Parameters**:
  - `positions` (`np.ndarray`): `(N, 3)` coordinates.
  - `cutoff` (`float`): Cutoff for defining neighbor pairs (defaults to `3.4`).
- **Returns**: `float` representing the average ECN across all points.

##### **`average_neighbor_distance`**
Calculates the mean distance averaged over all pairs within the cutoff.
- **Parameters**: Same as `effective_coordination_number`.
- **Returns**: `float`.

##### **`radius_of_gyration`**
Computes the geometric root-mean-square distance of points to their center of mass.
- **Parameters**:
  - `positions` (`np.ndarray`): `(N, 3)` coordinates.
- **Returns**: `float`.

##### **`radial_distribution_function`**
Computes the radial distribution function $g(r)$ for a trajectory.
- **Parameters**:
  - `trajectory` (`np.ndarray`): `(T, N, 3)` coordinates.
  - `r_max` (`float`): Max distance for RDF computation (defaults to `8.0`).
  - `n_bins` (`int`): Number of histogram bins (defaults to `200`).
  - `cutoff` (`float`, optional): If provided, ignores pairs beyond this distance.
- **Returns**: `(r_centers, g_r)` where both are `np.ndarray` of shape `(n_bins,)`.

##### **`hausdorff_chirality_measure`**
Quantifies structural chirality in range $[0, 1]$ using normalized Hausdorff distance to the PCA reflected mirror image (0 = achiral).
- **Parameters**:
  - `positions` (`np.ndarray`): `(N, 3)` coordinates.
- **Returns**: `float`.

##### **`projection_center`**
Computes the center of gravity of a weighted distribution.
- **Parameters**:
  - `values` (`np.ndarray`): `(M,)` sorted grid values.
  - `weights` (`np.ndarray`): `(M,)` distribution weights.
  - `threshold` (`float`): Upper bound value to filter values (defaults to `0.0`).
- **Returns**: `float` representing the center.

##### **`compute_trajectory_descriptors`**
Batch-processes a trajectory extracting multiple descriptors in a single call.
- **Parameters**: Same as `coordination_histogram_trajectory`.
- **Returns**: `dict` containing `"neighbor_histograms"`, `"ecn"`, `"d_av"`, `"rg"`, and `"hcm"` arrays.

---

### **2. Distance Metrics (`dimredpy.shared.metrics`)**
Custom distance metric objects supporting pairwise computations. In many physical systems, periodic boundary conditions (PBC) or spherical representations require specialized distance logic.

```python
from dimredpy.shared.metrics import (
    EuclideanMetric, PBCMetric, SphericalMetric, DotMetric, get_metric
)

# Instantiate a periodic boundary metric (10.0 Å box)
metric = PBCMetric(period=10.0)

# Compute distances
d = metric.dist(x, y)
dist_matrix = metric.pairwise(dataset)
```

#### **Metric Classes & API Details**
All metric classes inherit from `Metric` and implement:
- `dist(a, b)`: Distance between two vectors.
- `diff(a, b)`: Displacement vector $b - a$ ( PBC-aware where applicable).
- `pairwise(X)`: Pairwise distance matrix of shape `(N, N)` for dataset `X` of shape `(N, D)`.
- `pairwise_vec(X, Y)`: Pairwise distance matrix of shape `(M, K)` between two sets of vectors.

##### **`EuclideanMetric`**
Standard Euclidean distance. Takes no parameters.

##### **`PBCMetric`**
Periodic Boundary Conditions (toroidal distance).
- **Parameters**:
  - `period` (`float` | `np.ndarray`): Period $L_i$ for each dimension.

##### **`SphericalMetric`**
Geodesic (great-circle) distance on a hyper-sphere. The last dimension is treated as azimuthal/periodic.
- **Parameters**:
  - `period` (`float` | `np.ndarray`): Period $L_i$ for each dimension.

##### **`DotMetric`**
Similarity metric: $d(a, b) = -\ln(a \cdot b)$.

##### **`get_metric`** (Factory Function)
Helper to instantiate a metric from CLI style arguments:
- `get_metric(period=0.0, sphere_period=0.0, dot=False)`

---

### **3. Transfer Functions (`dimredpy.shared.transfer`)**
Mathematical kernels used primarily by the Sketch-map algorithm to "squash" distances. By transforming pairwise distances $R \to F(R) \in [0, 1]$, Sketch-map focuses on intermediate-range topology.

```python
from dimredpy.shared.transfer import (
    Identity, Sigmoid, Compress, XSigmoid, Gamma, Warp, make_transfer
)

# Construct using helper function
sigmoid = make_transfer((6.0, 8.0, 8.0))

# Evaluate transfer value and derivative
f_val, df_val = sigmoid.fdf(distances)
```

#### **Transfer Classes & API Details**
All transfer functions inherit from `TransferFunction` and implement:
- `f(x)`: Evaluation of $F(R)$ on NumPy array `x`.
- `df(x)`: Evaluation of the analytical derivative $F'(R)$ on NumPy array `x`.
- `fdf(x)`: Evaluates both simultaneously (saving recomputation).
- `f_torch(x)`: PyTorch equivalent for GPU accelerated graph execution.

##### **`Identity`**
No-op transfer mapping: $F(R) = R$. Takes no parameters.

##### **`Sigmoid`**
Squared-Lorentzian sigmoid: $F(R) = 1 - \frac{1}{1 + (R/\sigma)^2}$.
- **Parameters**: `sigma` (`float`).

##### **`Compress`**
Linear (Lorentzian) compression: $F(R) = 1 - \frac{1}{1 + R/\sigma}$.
- **Parameters**: `sigma` (`float`).

##### **`XSigmoid`**
Extended sigmoid (canonical Sketch-map): $F(R) = 1 - \left[1 + (2^{A/B} - 1) (R/\sigma)^A\right]^{-B/A}$.
- **Parameters**: `sigma` (`float`), `A` (`float`), `B` (`float`).

##### **`Gamma`**
Incomplete gamma function mapping: $F(R) = Q(N/2, (R/\sigma)^2/2)$. Requires `scipy.special`.
- **Parameters**: `sigma` (`float`), `N` (`float`).

##### **`Warp`**
Maps high-dimensional distances through an inverse low-dimensional mapping to match stress functions.
- **Parameters**: `sigma` (`float`), `A` (`float`), `B` (`float`), `a` (`float`), `b` (`float`).

##### **`make_transfer`** (Factory Function)
Builds the transfer function object from spec:
- `None` or `"identity"` $\to$ `Identity()`
- `(sigma,)` or `float` $\to$ `Sigmoid(sigma)`
- `(sigma, N)` $\to$ `Gamma(sigma, N)`
- `(sigma, A, B)` $\to$ `XSigmoid(sigma, A, B)`
- `(sigma, A, B, a, b)` $\to$ `Warp(sigma, A, B, a, b)`


---

### **4. Data I/O (`dimredpy.shared.io`)**
Agnostic, memory-efficient loaders and writers for simulation trajectories, tabular data, and embedding outputs. Transparently supports reading and writing `.gz` compressed files to save disk space.

```python
from dimredpy.shared.io import (
    load_spatial_coordinates, load_tabular_data, save_tabular_data,
    load_point_set, save_point_set, save_grid_surface
)

# Load an XYZ trajectory (automatically handles .gz)
traj = load_spatial_coordinates("trajectory.xyz.gz", format="xyz", label_filter="Au")

# Save a free-energy surface in Gnuplot format
save_grid_surface("fes.dat", surface_dict, gnuplot=True)
```

#### **Functions & API Details**

##### **`load_spatial_coordinates`**
Parses 3D atomic coordinates from a trajectory file (e.g. extended XYZ format). Uses a memory-efficient file iterator.
- **Parameters**:
  - `filepath` (`str` | `Path`): Path to the trajectory file.
  - `format` (`str`): Format of the file (defaults to `"xyz"`).
  - `label_filter` (`str`, optional): If provided, filters coordinates by atom/point label (e.g. `"Au"`). Passed to underlying XYZ parser.
- **Returns**: `np.ndarray` of shape `(T, N, 3)` where `T` is number of frames, `N` is number of coordinates per frame, and `3` is the spatial dimensions.

##### **`load_tabular_data`**
Loads 1D or 2D numerical arrays from plain-text tabular files (e.g., COLVAR or potential energy logs).
- **Parameters**:
  - `filepath` (`str` | `Path`): Path to the text file.
  - `skip_header` (`int`): Number of lines to skip at the beginning of the file (defaults to `0`).
  - `**kwargs`: Additional keyword arguments forwarded to `numpy.loadtxt`.
- **Returns**: `np.ndarray` containing the numerical data.

##### **`save_tabular_data`**
Saves a NumPy array to a text file. Automatically creates parent directories if they do not exist.
- **Parameters**:
  - `filepath` (`str` | `Path`): Destination path.
  - `data` (`np.ndarray`): Array of data to save.
  - `header` (`str`): Header string placed at the top of the file (defaults to `""`).
  - `**kwargs`: Additional keyword arguments forwarded to `numpy.savetxt`.

##### **`load_point_set`**
Loads coordinates (and optional weights) representing high-dimensional and/or low-dimensional point sets (e.g. landmarks).
- **Parameters**:
  - `hd_path` (`str` | `Path`): Path to high-dimensional point set file.
  - `ld_path` (`str` | `Path`, optional): Path to low-dimensional point set file.
  - `has_weights` (`bool`): If `True`, assumes the final column in the high-dimensional file represents the statistical weights (defaults to `False`).
- **Returns**: `dict` containing:
  - `"points_hd"`: `np.ndarray` of high-dimensional points.
  - `"weights"`: `np.ndarray` of weights if `has_weights` is `True`, otherwise `None`.
  - `"points_ld"`: `np.ndarray` of low-dimensional points (present only if `ld_path` is provided).

##### **`save_point_set`**
Saves a point set with optional weights. Automatically creates parent directories.
- **Parameters**:
  - `filepath` (`str` | `Path`): Destination path.
  - `points` (`np.ndarray`): Coordinate matrix.
  - `weights` (`np.ndarray`, optional): Weight array. If provided, is column-stacked as the last column of the output.
  - `**kwargs`: Additional keyword arguments forwarded to `numpy.savetxt`.

##### **`save_grid_surface`**
Saves a 2D probability or free-energy grid surface.
- **Parameters**:
  - `filepath` (`str` | `Path`): Destination path.
  - `surface` (`dict`): Dictionary containing keys `"bin_centers_x"`, `"bin_centers_y"`, `"probability"`, and `"free_energy"`.
  - `gnuplot` (`bool`): If `True`, formats the output with a blank line separating blocks of constant X, compatible with Gnuplot's `splot` command (defaults to `False`).


---

### **5. Analysis & Diagnostics (`dimredpy.shared.analysis`)**
Tools for measuring the quality of a dimensionality reduction mapping by checking how well high-dimensional (HD) pairwise distances are preserved in the low-dimensional (LD) space.

```python
from dimredpy.shared.analysis import distance_histogram, preservation_score, stress_per_pair

# Compute the final Sketch-map chi^2 stress (lower is better)
stress = preservation_score(
    landmarks_hd, landmarks_ld, 
    fun_hd=(6.0, 8.0, 8.0), 
    fun_ld=(6.0, 2.0, 8.0)
)

# Generate a 2D histogram of HD vs LD distances to visualize the mapping
hist_dict = distance_histogram(
    landmarks_hd, landmarks_ld, 
    n_bins=100, 
    fun_hd=(6.0, 8.0, 8.0), 
    fun_ld=(6.0, 2.0, 8.0)
)

# Compute the full pairwise stress matrix to identify problematic points
stress_mat = stress_per_pair(
    landmarks_hd, landmarks_ld, 
    fun_hd=(6.0, 8.0, 8.0), 
    fun_ld=(6.0, 2.0, 8.0)
)
```

#### **Functions & API Details**

##### **`distance_histogram`**
Computes a 1D (HD only) or 2D (HD vs LD) histogram of pairwise distances.
- **Parameters**:
  - `landmarks_hd` (`np.ndarray`): `(N, D)` high-dimensional points.
  - `landmarks_ld` (`np.ndarray`, optional): `(N, d)` low-dimensional points. If `None`, computes a 1D histogram of HD distances.
  - `metric` (`Metric`, optional): Metric used for HD distances (defaults to `EuclideanMetric`).
  - `n_bins` (`int`): Number of histogram bins along each axis (defaults to `100`).
  - `max_d` (`float`, optional): Maximum distance to include. Auto-detected if not specified.
  - `fun_hd` (`spec`, optional): Transfer function spec applied to HD distances.
  - `fun_ld` (`spec`, optional): Transfer function spec applied to LD distances.
  - `weights` (`np.ndarray`, optional): `(N,)` per-point weights. Pair weights are computed as $w_i \cdot w_j$.
- **Returns**: `dict` containing:
  - `"hd_distances"`: `np.ndarray` of upper-triangle HD pairwise distances.
  - `"ld_distances"`: `np.ndarray` of upper-triangle LD pairwise distances (returned if `landmarks_ld` is provided).
  - `"histogram_1d"`: `np.ndarray` of shape `(n_bins,)` containing counts (returned if `landmarks_ld` is `None`).
  - `"histogram_2d"`: `np.ndarray` of shape `(n_bins, n_bins)` containing counts (returned if `landmarks_ld` is provided).
  - `"bin_edges_hd"`: `np.ndarray` of shape `(n_bins + 1,)` defining bin edges for the HD axis.
  - `"bin_edges_ld"`: `np.ndarray` of shape `(n_bins + 1,)` defining bin edges for the LD axis (returned if `landmarks_ld` is provided).

##### **`preservation_score`**
Computes the mean-squared Sketch-map stress $\chi^2$ as a scalar quality score representing overall distance preservation. Lower values indicate better distance preservation.
- **Parameters**:
  - `landmarks_hd` (`np.ndarray`): `(N, D)` high-dimensional points.
  - `landmarks_ld` (`np.ndarray`): `(N, d)` low-dimensional points.
  - `metric` (`Metric`, optional): Metric used for HD distances (defaults to `EuclideanMetric`).
  - `fun_hd` (`spec`, optional): Transfer function spec applied to HD distances.
  - `fun_ld` (`spec`, optional): Transfer function spec applied to LD distances.
  - `weights` (`np.ndarray`, optional): `(N,)` per-point weights.
- **Returns**: `float` representing the $\chi^2$ stress.

##### **`stress_per_pair`**
Computes the full pair-wise stress matrix. Useful for diagnosing exactly which high-dimensional distances are being distorted during dimensionality reduction.
- **Parameters**: Same as `preservation_score`.
- **Returns**: `np.ndarray` of shape `(N, N)`, representing a symmetric matrix where entry $(i, j)$ is the weighted squared difference between transformed high-dimensional and low-dimensional distances.


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
