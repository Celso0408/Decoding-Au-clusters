"""
UMAP: Uniform Manifold Approximation and Projection.

Provides a unified Python interface to UMAP via umap-learn (CPU) or cuML (GPU),
with built-in support for the Mahalanobis metric and automatic covariance
matrix handling.

This module is fully domain-agnostic — it operates on any (N, D) array of
high-dimensional observations. All algorithm parameters are exposed and
documented; no experiment-specific presets are baked in.

Backends
--------
- **umap-learn** (CPU): pip install umap-learn
  Full UMAP implementation with arbitrary distance metrics.
  For Mahalanobis, the sample covariance matrix is passed via metric_kwds.

- **cuML** (GPU): part of NVIDIA RAPIDS
  GPU-accelerated UMAP. For Mahalanobis metric (not natively supported),
  data is pre-whitened so that Euclidean distance on the whitened data
  is mathematically equivalent to Mahalanobis distance on the original.

Parameter guidance
------------------
The key hyperparameters and their effects:

    n_neighbors : int (default 15)
        Number of nearest neighbors for graph construction. Larger values
        incorporate more global context; smaller values emphasize local
        structure.

    min_dist : float (default 0.1)
        Minimum distance between points in the embedding. Smaller values
        produce tighter clusters (down to 0.001); larger values spread
        points more evenly.

    metric : str (default "euclidean")
        Distance metric in HD space. Common choices:
        - "euclidean": standard L2 distance
        - "mahalanobis": covariance-aware distance (good for correlated
          or heterogeneously scaled features)
        - "cosine", "manhattan", "correlation", etc.

Reference
---------
    github.com/lmcinnes/umap
"""

import numpy as np
from typing import Optional

try:
    import cuml
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


def _mahalanobis_whiten(X: np.ndarray) -> np.ndarray:
    """Pre-whiten data for Mahalanobis-equivalent Euclidean distance.

    Transforms X such that Euclidean distance on the result equals
    Mahalanobis distance on the original:

        X_white = (X - mu) @ S^{-1/2}

    where S = cov(X) and S^{-1/2} = V diag(1/sqrt(lambda)) V^T.
    This is used for GPU backends that do not support Mahalanobis natively.
    """
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6  # regularization for numerical stability
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    whitener = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return (X - mu) @ whitener


def umap_embed(
    data: np.ndarray,
    n_components: int = 2,
    metric: str = "euclidean",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> np.ndarray:
    """Run UMAP dimensionality reduction.

    Supports GPU acceleration via cuML or CPU execution via umap-learn.
    All default values follow the standard umap-learn conventions.
    Users should tune parameters for their specific dataset and goals.

    Parameters
    ----------
    data : (N, D) array
        High-dimensional input data.
    n_components : int
        Number of embedding dimensions (default 2).
    metric : str
        Distance metric in the HD space (default "euclidean").
        Use "mahalanobis" for covariance-aware distance.
    n_neighbors : int
        Number of nearest neighbors for graph construction (default 15).
    min_dist : float
        Minimum distance between embedded points (default 0.1).
    seed : int
        Random seed for reproducibility (default 42).
    n_jobs : int
        Number of CPU threads (-1 = all cores).
    verbose : bool
        Print progress information.
    use_gpu : bool
        Attempt GPU acceleration via cuML if available.
    **kwargs
        Additional keyword arguments passed to the UMAP constructor.

    Returns
    -------
    embedding : (N, n_components) array
        Low-dimensional embedding coordinates.
    """
    # --- GPU path via cuML ---
    if use_gpu and HAS_CUML:
        X = np.asarray(data, dtype=np.float32)

        if metric == "mahalanobis":
            if verbose:
                print("   -> [GPU] cuML UMAP with Mahalanobis via pre-whitening")
            X = _mahalanobis_whiten(X).astype(np.float32)
            gpu_metric = "euclidean"
        else:
            if verbose:
                print(f"   -> [GPU] cuML UMAP with metric={metric}")
            gpu_metric = metric

        try:
            from cuml.manifold import UMAP as cumlUMAP
            model = cumlUMAP(
                n_components=n_components,
                metric=gpu_metric,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=seed,
                verbose=verbose,
                **kwargs,
            )
            return model.fit_transform(X)
        except Exception as e:
            if verbose:
                print(f"   -> [GPU] cuML UMAP failed ({e}). Falling back to CPU...")

    if use_gpu and not HAS_CUML and verbose:
        print("   -> [CPU] cuml not found. Falling back to umap-learn.")

    # --- CPU path via umap-learn ---
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for CPU UMAP. Install via: pip install umap-learn"
        )

    # For Mahalanobis with umap-learn, pass the covariance matrix
    umap_kwargs = dict(kwargs)
    if metric == "mahalanobis":
        X = np.asarray(data, dtype=float)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        umap_kwargs["metric_kwds"] = {"V": cov}

    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=verbose,
        **umap_kwargs,
    )

    if verbose:
        print(f"   -> [umap-learn] metric={metric}, n_neighbors={n_neighbors}, "
              f"min_dist={min_dist}")

    return reducer.fit_transform(np.asarray(data, dtype=float))
