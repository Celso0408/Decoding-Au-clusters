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
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple

try:
    import cuml
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


def _mahalanobis_whiten(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-whiten data for Mahalanobis-equivalent Euclidean distance.

    Returns:
        X_white: Whitened data
        mu: Mean of X
        whitener: Whitening matrix
    """
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6  # regularization for numerical stability
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    whitener = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    X_white = (X - mu) @ whitener
    return X_white, mu, whitener


def umap_embed(
    data: np.ndarray,
    n_components: int = 2,
    metric: str = "euclidean",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    y: Optional[np.ndarray] = None,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Run UMAP dimensionality reduction (Core Reduction & Supervised).

    Returns
    -------
    dict with keys:
        - "embedding": (N, n_components) array of coordinates
        - "model": The fitted UMAP model (umap-learn or cuML)
        - "metric": The original metric requested
        - "mahalanobis_params": (mu, whitener) if GPU Mahalanobis was used
    """
    res = {"metric": metric, "mahalanobis_params": None}

    # --- GPU path via cuML ---
    if use_gpu and HAS_CUML:
        X = np.asarray(data, dtype=np.float32)

        if metric == "mahalanobis":
            if verbose:
                print("   -> [GPU] cuML UMAP with Mahalanobis via pre-whitening")
            X, mu, whitener = _mahalanobis_whiten(X)
            X = X.astype(np.float32)
            gpu_metric = "euclidean"
            res["mahalanobis_params"] = (mu, whitener)
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
            emb = model.fit_transform(X, y=y)
            res["embedding"] = emb
            res["model"] = model
            return res
        except Exception as e:
            if verbose:
                print(f"   -> [GPU] cuML UMAP failed ({e}). Falling back to CPU...")

    if use_gpu and not HAS_CUML:
        print("   -> [WARNING] GPU requested (use_gpu=True) but cuML not found. Falling back to CPU (umap-learn).")

    # --- CPU path via umap-learn ---
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for CPU UMAP. Install via: pip install umap-learn"
        )

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
        print(f"   -> [umap-learn] metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist}")

    emb = reducer.fit_transform(np.asarray(data, dtype=float), y=y)
    res["embedding"] = emb
    res["model"] = reducer
    return res


def umap_project(
    model_dict: Dict[str, Any],
    new_data: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Project out-of-sample data using a fitted UMAP model.
    
    Parameters
    ----------
    model_dict : dict
        The result dictionary returned by `umap_embed()`.
    new_data : (M, D) array
        The out-of-sample data to project.
        
    Returns
    -------
    embedding : (M, n_components) array
    """
    model = model_dict["model"]
    metric = model_dict["metric"]
    mah_params = model_dict["mahalanobis_params"]
    
    X = np.asarray(new_data)
    
    # Check if this is a GPU model
    if HAS_CUML and "cuml" in str(type(model)):
        X = X.astype(np.float32)
        if metric == "mahalanobis" and mah_params is not None:
            mu, whitener = mah_params
            X = (X - mu) @ whitener
            X = X.astype(np.float32)
            
    if verbose:
        print(f"   -> Projecting {X.shape[0]} out-of-sample points...")
        
    return model.transform(X)


def umap_inverse_project(
    model_dict: Dict[str, Any],
    new_ld_data: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Project low-dimensional out-of-sample data back into the high-dimensional space.
    
    Parameters
    ----------
    model_dict : dict
        The result dictionary returned by `umap_embed()`.
    new_ld_data : (M, n_components) array
        The low-dimensional data to inversely project.
        
    Returns
    -------
    embedding : (M, D) array
        The reconstructed high-dimensional data.
    """
    model = model_dict["model"]
    metric = model_dict["metric"]
    mah_params = model_dict["mahalanobis_params"]
    
    # cuML UMAP historically does not support inverse_transform, or its support is limited.
    if HAS_CUML and "cuml" in str(type(model)):
        raise NotImplementedError(
            "Inverse projection is not natively supported by the GPU (cuML) backend. "
            "Please rerun `umap_embed` with `use_gpu=False` to use the CPU backend for inverse transformation."
        )
        
    if not hasattr(model, "inverse_transform"):
        raise NotImplementedError(
            "The underlying UMAP model does not support inverse_transform. "
            "Note that umap-learn typically only supports inverse_transform for Euclidean distances."
        )
        
    X_ld = np.asarray(new_ld_data)
    
    if verbose:
        print(f"   -> Inversely projecting {X_ld.shape[0]} low-dimensional points...")
        
    X_hd = model.inverse_transform(X_ld)
    
    # If mahalanobis was used on CPU, umap-learn handles it if the metric was provided appropriately.
    # However, umap-learn's inverse_transform doesn't natively support arbitrary metrics.
    # If the user successfully inverted it, we just return the output. 
    return X_hd
