"""
FIt-SNE: FFT-accelerated interpolation-based t-SNE.

Provides a unified Python interface to t-SNE via openTSNE (CPU) or cuML (GPU),
using the FFT-based negative gradient method for O(N log N) scaling.

This module is fully domain-agnostic — it operates on any (N, D) array of
high-dimensional observations. All algorithm parameters are exposed and
documented; no experiment-specific presets are baked in.

Backends
--------
- **openTSNE** (CPU): pip install openTSNE
  Full FFT negative gradient method with fine-grained control over
  min_num_intervals, early_exaggeration, and learning rate.

- **cuML** (GPU): part of NVIDIA RAPIDS
  GPU-accelerated t-SNE for large datasets (>100k samples).

Parameter guidance
------------------
The key hyperparameters and their effects:

    perplexity : float (default 30)
        Effective number of nearest neighbors. Controls the balance between
        local and global structure. Larger values (100-500) emphasize global
        topology; smaller values (5-30) emphasize local clustering.

    early_exaggeration : float (default 12)
        Multiplicative factor applied to high-dimensional affinities during
        the early phase. Larger values (100-1000) produce tighter, more
        separated clusters but may require more iterations to converge.
        Standard t-SNE default is 12; the original FIt-SNE C++ code
        supports values up to 1000+.

    min_num_intervals : int (default 50)
        FFT interpolation grid resolution for the N-body force approximation.
        Higher values (500-1000) increase force accuracy at the cost of
        memory and compute time. Important for large datasets.

    learning_rate : float or "auto"
        Gradient descent step size. "auto" computes max(200, N / early_exag),
        following the convention in the original FIt-SNE implementation.

Reference
---------
    github.com/KlugerLab/FIt-SNE
"""

import numpy as np
from typing import Optional, Union

try:
    import cuml
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


def _resolve_learning_rate(
    learning_rate: Union[str, float],
    n_samples: int,
    early_exaggeration: float,
) -> float:
    """Resolve 'auto' learning rate following FIt-SNE convention.

    The original FIt-SNE implementation computes:
        lr = max(200, N / early_exag_coeff)
    """
    if isinstance(learning_rate, str) and learning_rate == "auto":
        return float(max(200, n_samples / max(early_exaggeration, 1)))
    return float(learning_rate)


def fit_sne(
    data: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    early_exaggeration: float = 12.0,
    early_exaggeration_iter: int = 250,
    learning_rate: Union[str, float] = "auto",
    metric: str = "euclidean",
    min_num_intervals: int = 50,
    negative_gradient_method: str = "fft",
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> np.ndarray:
    """Run FIt-SNE dimensionality reduction.

    Supports GPU acceleration via cuML or CPU execution via openTSNE.
    All default values follow the standard t-SNE / openTSNE conventions.
    Users should tune parameters for their specific dataset and goals.

    Parameters
    ----------
    data : (N, D) array
        High-dimensional input data.
    n_components : int
        Number of embedding dimensions (default 2).
    perplexity : float
        Effective number of nearest neighbors. Larger values emphasize
        global structure (default 30).
    n_iter : int
        Number of gradient descent iterations (default 1000).
    early_exaggeration : float
        Multiplicative factor for HD affinities in early iterations.
        Larger values produce more separated clusters (default 12).
    early_exaggeration_iter : int
        Number of iterations for the early exaggeration phase (default 250).
    learning_rate : float or "auto"
        Step size. "auto" computes max(200, N / early_exag) (default "auto").
    metric : str
        Distance metric for the HD space (default "euclidean").
    min_num_intervals : int
        FFT interpolation grid resolution for N-body forces (default 50).
    negative_gradient_method : str
        Method for repulsive forces: "fft" (FIt-SNE) or "bh" (Barnes-Hut).
        Default: "fft".
    seed : int
        Random seed for reproducibility (default 42).
    n_jobs : int
        Number of CPU threads (-1 = all cores).
    verbose : bool
        Print progress information.
    use_gpu : bool
        Attempt GPU acceleration via cuML if available.

    Returns
    -------
    embedding : (N, n_components) array
        Low-dimensional embedding coordinates.
    **kwargs
        Additional keyword arguments passed to the underlying openTSNE or cuML TSNE constructor.
    """
    lr_numeric = _resolve_learning_rate(learning_rate, len(data), early_exaggeration)

    # --- GPU path via cuML ---
    if use_gpu and HAS_CUML:
        if verbose:
            print(f"   -> [GPU] Using cuml.TSNE")
            print(f"      perplexity={perplexity}, early_exag={early_exaggeration}, "
                  f"lr={lr_numeric:.1f}, n_iter={n_iter}")
        try:
            from cuml.manifold import TSNE as cumlTSNE
            model = cumlTSNE(
                n_components=n_components,
                perplexity=perplexity,
                early_exaggeration=float(early_exaggeration),
                learning_rate=lr_numeric,
                max_iter=n_iter,
                random_state=seed,
                verbose=verbose,
                **kwargs,
            )
            return model.fit_transform(np.asarray(data, dtype=np.float32))
        except Exception as e:
            if verbose:
                print(f"   -> [GPU] cuML t-SNE failed ({e}). Falling back to CPU...")

    if use_gpu and not HAS_CUML and verbose:
        print("   -> [CPU] cuml not found. Falling back to openTSNE.")

    # --- CPU path via openTSNE ---
    try:
        from openTSNE import TSNE
    except ImportError:
        raise ImportError(
            "openTSNE is required for CPU FIt-SNE. Install via: pip install openTSNE"
        )

    neg_grad_params = {}
    if min_num_intervals is not None:
        neg_grad_params["min_num_intervals"] = min_num_intervals

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        early_exaggeration=early_exaggeration,
        learning_rate=lr_numeric,
        metric=metric,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=verbose,
        negative_gradient_method=negative_gradient_method,
        **neg_grad_params,
        **kwargs,
    )

    if verbose:
        print(f"   -> [openTSNE] perplexity={perplexity}, early_exag={early_exaggeration}, "
              f"lr={lr_numeric:.1f}, min_intervals={min_num_intervals}, "
              f"n_iter={n_iter}, method={negative_gradient_method}")

    return tsne.fit(np.asarray(data, dtype=float))
