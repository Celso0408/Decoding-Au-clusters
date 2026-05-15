"""
Distance-preservation diagnostics.

Mirrors dimdist.cpp:
    distance_histogram  — 1D or 2D histogram of HD vs LD pairwise distances.
    stress_per_pair     — full pair-wise stress matrix.
    preservation_score  — scalar summarising overall distance preservation.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from .metrics import Metric, EuclideanMetric
from .transfer import TransferFunction, make_transfer


def distance_histogram(
    landmarks_hd: np.ndarray,
    landmarks_ld: Optional[np.ndarray] = None,
    metric: Optional[Metric] = None,
    n_bins: int = 100,
    max_d: Optional[float] = None,
    fun_hd=None,
    fun_ld=None,
    weights: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute 1D (HD only) or 2D (HD vs LD) histogram of pairwise distances.

    Mirrors dimdist.cpp.

    Parameters
    ----------
    landmarks_hd : (N, D) high-dim points.
    landmarks_ld : (N, d) low-dim points (optional; if None, only HD histogram).
    metric       : HD distance metric (default Euclidean).
    n_bins       : number of bins in each axis.
    max_d        : maximum distance (auto-detected if None).
    fun_hd       : optional transfer function applied to HD distances.
    fun_ld       : optional transfer function applied to LD distances.
    weights      : (N,) per-point weights; pair weight = w_i * w_j.

    Returns
    -------
    dict with:
        "hd_distances" : 1D array of upper-triangle HD pairwise distances.
        "ld_distances" : 1D array of LD pairwise distances (if ld given).
        "histogram_1d" : (n_bins,) counts if ld is None.
        "histogram_2d" : (n_bins, n_bins) counts if ld is given.
        "bin_edges_hd" : bin edges for HD axis.
        "bin_edges_ld" : bin edges for LD axis (if ld given).
    """
    landmarks_hd = np.asarray(landmarks_hd, dtype=float)
    N = landmarks_hd.shape[0]

    if metric is None:
        metric = EuclideanMetric()

    tf_hd = make_transfer(fun_hd) if fun_hd is not None else None
    tf_ld = make_transfer(fun_ld) if fun_ld is not None else None

    hd_mat = metric.pairwise(landmarks_hd)
    triu   = np.triu_indices(N, k=1)

    hd_dist = hd_mat[triu]
    if tf_hd is not None:
        hd_dist = tf_hd.f(hd_dist)

    # Pair weights
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        pair_w = (w[:, None] * w[None, :])[triu]
    else:
        pair_w = np.ones(len(hd_dist), dtype=float)

    result = {"hd_distances": hd_dist}

    if landmarks_ld is None:
        mx = max_d if max_d is not None else hd_dist.max()
        edges = np.linspace(0, mx, n_bins + 1)
        hist, _ = np.histogram(hd_dist, bins=edges, weights=pair_w)
        result["histogram_1d"] = hist
        result["bin_edges_hd"] = edges
        return result

    # 2D histogram
    landmarks_ld = np.asarray(landmarks_ld, dtype=float)
    if landmarks_ld.shape[0] != N:
        raise ValueError("landmarks_hd and landmarks_ld must have the same number of points.")
    ld_euclid    = EuclideanMetric()
    ld_mat       = ld_euclid.pairwise(landmarks_ld)
    ld_dist      = ld_mat[triu]
    if tf_ld is not None:
        ld_dist = tf_ld.f(ld_dist)

    mx_hd = max_d if max_d is not None else hd_dist.max()
    mx_ld = ld_dist.max()

    edges_hd = np.linspace(0, mx_hd, n_bins + 1)
    edges_ld = np.linspace(0, mx_ld, n_bins + 1)
    hist2d, _, _ = np.histogram2d(hd_dist, ld_dist,
                                  bins=[edges_hd, edges_ld],
                                  weights=pair_w)
    result.update({
        "ld_distances": ld_dist,
        "histogram_2d": hist2d,
        "bin_edges_hd": edges_hd,
        "bin_edges_ld": edges_ld,
    })
    return result


def preservation_score(
    landmarks_hd: np.ndarray,
    landmarks_ld: np.ndarray,
    metric: Optional[Metric] = None,
    fun_hd=None,
    fun_ld=None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the mean-squared Sketch-map stress χ² as a scalar quality score.

    Lower is better.  Equivalent to the final χ² reported by dimred -v.

    Parameters
    ----------
    landmarks_hd : (N, D) high-dim landmarks.
    landmarks_ld : (N, d) embedded landmark coordinates.
    metric       : HD metric (default Euclidean).
    fun_hd       : HD transfer function spec.
    fun_ld       : LD transfer function spec.
    weights      : (N,) per-point weights.

    Returns
    -------
    float : χ² stress.
    """
    if metric is None:
        metric = EuclideanMetric()
    tf_hd = make_transfer(fun_hd)
    tf_ld = make_transfer(fun_ld)

    hd_landmarks_arr = np.asarray(landmarks_hd, dtype=float)
    ld_landmarks_arr = np.asarray(landmarks_ld, dtype=float)
    if hd_landmarks_arr.shape[0] != ld_landmarks_arr.shape[0]:
        raise ValueError("landmarks_hd and landmarks_ld must have the same number of points.")
        
    hd_mat = metric.pairwise(hd_landmarks_arr)
    ld_mat = EuclideanMetric().pairwise(ld_landmarks_arr)

    fhd = tf_hd.f(hd_mat)
    fld = tf_ld.f(ld_mat)

    N = hd_mat.shape[0]
    triu = np.triu_indices(N, k=1)
    diff = (fhd[triu] - fld[triu]) ** 2

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        pw = (w[:, None] * w[None, :])[triu]
        return float((pw * diff).sum() / pw.sum())
    return float(diff.mean())
