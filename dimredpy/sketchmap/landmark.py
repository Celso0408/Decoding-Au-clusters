"""
Landmark selection for Sketch-map.

Implements all five selection modes from the original dimlandmark C++ tool:
    minmax   — Farthest Point Sampling (greedy max-min), default
    random   — uniformly random selection
    stride   — evenly-spaced stride along the dataset
    resample — random selection biased toward undersampled regions
    staged   — two-stage FPS + probability-weighted sub-selection

After selection, Voronoi weights are optionally computed (matching -w flag).
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dimredpy.shared.metrics import Metric, EuclideanMetric


def _voronoi_weights(
    data: np.ndarray,
    landmarks: np.ndarray,
    metric: Metric,
    input_weights: Optional[np.ndarray] = None,
    weight_gamma: float = 1.0,
) -> np.ndarray:
    """
    Assign each data point to its nearest landmark (Voronoi tessellation)
    and return normalised landmark weights.

    Parameters
    ----------
    data          : (N, D) data matrix.
    landmarks     : (K, D) selected landmarks.
    metric        : distance metric.
    input_weights : (N,) per-point weights (default: all 1).
    weight_gamma  : apply weight^gamma before normalising.
    """
    N, K = data.shape[0], landmarks.shape[0]
    if input_weights is None:
        input_weights = np.ones(N, dtype=float)

    dist_matrix = metric.pairwise_vec(data, landmarks)  # (N, K)
    nearest = np.argmin(dist_matrix, axis=1)             # (N,)

    raw = np.zeros(K, dtype=float)
    for j in range(N):
        raw[nearest[j]] += input_weights[j]

    raw = raw ** weight_gamma
    total = raw.sum()
    if total == 0:
        raise RuntimeError("Voronoi weights sum to zero — check data and landmarks.")
    return raw / total


def select_landmarks(
    data: np.ndarray,
    n_landmarks: int,
    mode: str = "minmax",
    metric: Optional[Metric] = None,
    input_weights: Optional[np.ndarray] = None,
    seed: int = 12345,
    first: int = -1,
    unique: bool = False,
    return_weights: bool = True,
    weight_gamma: float = 1.0,
    resample_gamma: float = 1.0,
    similarity: Optional[np.ndarray] = None,
) -> Dict:
    """
    Select landmark points from a dataset.

    Parameters
    ----------
    data          : (N, D) array of high-dimensional data points.
    n_landmarks   : number K of landmarks to select.
    mode          : "minmax" | "random" | "stride" | "resample" | "staged"
    metric        : distance metric (default: Euclidean).
    input_weights : (N,) per-point input weights (used in resample/staged).
    seed          : random seed.
    first         : index of the first landmark (-1 = random).
    unique        : enforce unique selection in random mode.
    return_weights: if True, compute and return Voronoi weights.
    weight_gamma  : exponent applied to Voronoi weights before normalising.
    resample_gamma: gamma parameter for resample / staged modes.
    similarity    : (N,N) pre-computed distance matrix (minmax only).

    Returns
    -------
    dict with keys:
        "landmarks" : (K, D) selected landmark coordinates.
        "indices"   : (K,) integer indices into the original data.
        "weights"   : (K,) normalised weights, or None if return_weights=False.
    """
    data = np.asarray(data, dtype=float)
    N, D = data.shape

    if n_landmarks > N:
        raise ValueError(f"n_landmarks ({n_landmarks}) > N ({N}).")

    if metric is None:
        metric = EuclideanMetric()

    rng = np.random.default_rng(seed)

    if input_weights is None:
        input_weights = np.ones(N, dtype=float)
    input_weights = np.asarray(input_weights, dtype=float)

    # -----------------------------------------------------------------------
    # Pick the first landmark
    # -----------------------------------------------------------------------
    if first < 0:
        first_idx = int(rng.integers(0, N))
    else:
        first_idx = int(first)

    def _compute_dist_to(idx, min_dist):
        """Update min-distance list when a new landmark is added at idx."""
        if similarity is not None:
            new_d = similarity[idx]
        else:
            new_d = metric.pairwise_vec(data, data[idx:idx+1]).ravel()
        return np.minimum(min_dist, new_d)

    # -----------------------------------------------------------------------
    # MODE: stride
    # -----------------------------------------------------------------------
    if mode == "stride":
        stride = N // n_landmarks
        indices = np.array([i * stride for i in range(n_landmarks)], dtype=int)

    # -----------------------------------------------------------------------
    # MODE: random
    # -----------------------------------------------------------------------
    elif mode == "random":
        if unique:
            indices = rng.choice(N, size=n_landmarks, replace=False).astype(int)
        else:
            indices = rng.integers(0, N, size=n_landmarks).astype(int)

    # -----------------------------------------------------------------------
    # MODE: minmax (Farthest Point Sampling)
    # -----------------------------------------------------------------------
    elif mode == "minmax":
        indices = np.empty(n_landmarks, dtype=int)
        indices[0] = first_idx

        if similarity is not None:
            min_dist = similarity[first_idx].copy()
        else:
            min_dist = metric.pairwise_vec(data, data[first_idx:first_idx+1]).ravel()

        for i in range(1, n_landmarks):
            next_idx = int(np.argmax(min_dist))
            indices[i] = next_idx
            min_dist = _compute_dist_to(next_idx, min_dist)

    # -----------------------------------------------------------------------
    # MODE: resample (random biased toward far points)
    # -----------------------------------------------------------------------
    elif mode == "resample":
        indices = np.empty(n_landmarks, dtype=int)
        indices[0] = first_idx

        # accumulate d^{-gamma} distances
        accum = np.zeros(N, dtype=float)
        d_to_last = metric.pairwise_vec(data, data[first_idx:first_idx+1]).ravel()
        inv_d = np.where(d_to_last == 0, 1e200, d_to_last ** (-resample_gamma))
        accum += inv_d

        for i in range(1, n_landmarks):
            weights_prob = input_weights * (accum ** (-resample_gamma))
            weights_prob /= weights_prob.sum()
            chosen = False
            while not chosen:
                next_idx = int(rng.choice(N, p=weights_prob))
                if unique and next_idx in indices[:i]:
                    continue
                chosen = True
            indices[i] = next_idx
            d_new = metric.pairwise_vec(data, data[next_idx:next_idx+1]).ravel()
            inv_d = np.where(d_new == 0, 1e200, d_new ** (-resample_gamma))
            accum += inv_d

    # -----------------------------------------------------------------------
    # MODE: staged (two-stage FPS + probability-weighted sub-selection)
    # -----------------------------------------------------------------------
    elif mode == "staged":
        m = int(np.sqrt(n_landmarks * N))
        # Stage 1: FPS over sqrt(K*N) intermediate points
        mid_indices = np.empty(m, dtype=int)
        mid_indices[0] = first_idx
        min_dist = metric.pairwise_vec(data, data[first_idx:first_idx+1]).ravel()
        for i in range(1, m):
            next_idx = int(np.argmax(min_dist))
            mid_indices[i] = next_idx
            min_dist = _compute_dist_to(next_idx, min_dist)

        # Stage 2: compute Voronoi weights for intermediate points
        mid_pts = data[mid_indices]
        cross_d = metric.pairwise_vec(data, mid_pts)   # (N, m)
        nearest_mid = np.argmin(cross_d, axis=1)
        mid_weights = np.zeros(m, dtype=float)
        for j in range(N):
            mid_weights[nearest_mid[j]] += input_weights[j]
        mid_weights = mid_weights ** resample_gamma
        mid_weights /= mid_weights.sum()

        # Stage 3: sample n_landmarks from intermediate points
        indices = np.empty(n_landmarks, dtype=int)
        vplist = [[] for _ in range(m)]
        for j in range(N):
            vplist[nearest_mid[j]].append(j)

        chosen_set = set()
        for i in range(n_landmarks):
            unique_found = False
            while not unique_found:
                mid_chosen = int(rng.choice(m, p=mid_weights))
                sub_pool = vplist[mid_chosen]
                sub_idx = int(rng.choice(sub_pool))
                if unique and sub_idx in chosen_set:
                    continue
                unique_found = True
            indices[i] = sub_idx
            chosen_set.add(sub_idx)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: minmax, random, stride, resample, staged."
        )

    landmarks = data[indices]

    weights = None
    if return_weights:
        weights = _voronoi_weights(data, landmarks, metric, input_weights, weight_gamma)

    return {"landmarks": landmarks, "indices": indices, "weights": weights}
