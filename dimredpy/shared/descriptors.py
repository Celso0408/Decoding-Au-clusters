"""
Structural descriptors for point cloud and spatial trajectory analysis.

These are common geometric descriptors and analysis tools for 3D point sets:
    coordination_histogram  — multi-dimensional descriptor based on local density
    effective_coordination_number (ECN)
    average_neighbor_distance (d_av)
    radius_of_gyration (Rg)
    radial_distribution_function (RDF / g(r))
    hausdorff_chirality_measure (HCM)
    d_band_center (epsilon_d)  — generic projection-center descriptor

All functions operate on trajectory data stored as numpy arrays.
"""

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Coordination histogram
# ---------------------------------------------------------------------------

def coordination_histogram(
    positions: np.ndarray,
    cutoff: float = 3.4,
    max_neighbors: int = 12,
) -> np.ndarray:
    """
    Compute the coordination (neighbor count) histogram for a single frame.

    For each point, count the number of neighbors within the cutoff distance.
    The histogram bin i contains the fraction of points with exactly i neighbors,
    for i in [0, max_neighbors].

    Parameters
    ----------
    positions     : (N, 3) spatial coordinates.
    cutoff        : distance cutoff for counting neighbors.
    max_neighbors : maximum neighbor count to track in the histogram.

    Returns
    -------
    (max_neighbors+1,) normalised histogram. Entries sum to 1.
    """
    positions = np.asarray(positions, dtype=float)
    N = positions.shape[0]
    # Efficient pairwise distance calculation
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    
    neighbor_counts = (dist < cutoff).sum(axis=1)
    neighbor_counts = np.clip(neighbor_counts, 0, max_neighbors)
    hist, _ = np.histogram(neighbor_counts, bins=np.arange(max_neighbors + 2) - 0.5)
    return hist.astype(float) / N


def coordination_histogram_trajectory(
    trajectory: np.ndarray,
    cutoff: float = 3.4,
    max_neighbors: int = 12,
) -> np.ndarray:
    """
    Compute neighbor count histograms for an entire trajectory.

    Parameters
    ----------
    trajectory    : (T, N, 3) positions for T frames.
    cutoff        : distance cutoff.
    max_neighbors : maximum neighbor count.

    Returns
    -------
    (T, max_neighbors+1) array of histograms.
    """
    T = trajectory.shape[0]
    out = np.zeros((T, max_neighbors + 1), dtype=float)
    for t in range(T):
        out[t] = coordination_histogram(trajectory[t], cutoff, max_neighbors)
    return out


# ---------------------------------------------------------------------------
# Effective Coordination Number (ECN)
# ---------------------------------------------------------------------------

def effective_coordination_number(
    positions: np.ndarray,
    cutoff: float = 3.4,
) -> float:
    """
    Mean effective coordination number for a single frame.

    ECN provides a continuous measure of local density:
        ECN = (1/N) Σ_i Σ_{j≠i} exp(1 - (d_ij / d_av)^6)

    Parameters
    ----------
    positions : (N, 3) spatial coordinates.
    cutoff    : cutoff for defining neighbor pairs.

    Returns
    -------
    float : mean ECN averaged over all points.
    """
    positions = np.asarray(positions, dtype=float)
    N = positions.shape[0]
    diff = positions[:, np.newaxis] - positions[np.newaxis]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)

    # Weighted average neighbor distance per point
    mask = dist < cutoff
    d_av_per_point = np.where(
        mask.sum(axis=1) > 0,
        np.sum(np.where(mask, dist, 0.0), axis=1) / mask.sum(axis=1).clip(min=1),
        0.0,
    )

    # ECN calculation
    ecn_per_point = np.zeros(N, dtype=float)
    for i in range(N):
        if d_av_per_point[i] > 0:
            ecn_per_point[i] = np.exp(1.0 - (dist[i] / d_av_per_point[i]) ** 6).sum()
            ecn_per_point[i] -= 1.0  # remove self-interaction

    return float(ecn_per_point.mean())


def average_neighbor_distance(
    positions: np.ndarray,
    cutoff: float = 3.4,
) -> float:
    """
    Mean distance averaged over all pairs within the cutoff.

    Parameters
    ----------
    positions : (N, 3) spatial coordinates.
    cutoff    : cutoff for distance detection.

    Returns
    -------
    float : mean length.
    """
    positions = np.asarray(positions, dtype=float)
    diff = positions[:, np.newaxis] - positions[np.newaxis]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    pairs = dist[dist < cutoff]
    if len(pairs) == 0:
        return 0.0
    return float(pairs.mean())


# ---------------------------------------------------------------------------
# Radius of Gyration
# ---------------------------------------------------------------------------

def radius_of_gyration(positions: np.ndarray) -> float:
    """
    Radius of gyration Rg for a point cloud.

        Rg = sqrt( (1/N) Σ_i |r_i - r_cm|² )

    Parameters
    ----------
    positions : (N, 3) spatial coordinates.

    Returns
    -------
    float : Rg value.
    """
    positions = np.asarray(positions, dtype=float)
    cm = positions.mean(axis=0)
    return float(np.sqrt(((positions - cm) ** 2).sum(axis=1).mean()))


# ---------------------------------------------------------------------------
# Radial Distribution Function (RDF / g(r))
# ---------------------------------------------------------------------------

def radial_distribution_function(
    trajectory: np.ndarray,
    r_max: float = 8.0,
    n_bins: int = 200,
    cutoff: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the radial distribution function g(r) for a trajectory.

    Parameters
    ----------
    trajectory : (T, N, 3) trajectory of positions.
    r_max      : maximum distance for RDF.
    n_bins     : number of bins.
    cutoff     : optional distance cutoff.

    Returns
    -------
    (r_centers, g_r) : bin centres and g(r) values.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    T, N, _ = trajectory.shape
    edges = np.linspace(0, r_max, n_bins + 1)
    dr    = edges[1] - edges[0]
    hist  = np.zeros(n_bins, dtype=float)

    for t in range(T):
        pos  = trajectory[t]
        diff = pos[:, np.newaxis] - pos[np.newaxis]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        triu = np.triu_indices(N, k=1)
        dists_t = dist[triu]
        if cutoff is not None:
            dists_t = dists_t[dists_t < cutoff]
        h, _ = np.histogram(dists_t, bins=edges)
        hist += h

    # Normalization
    V_sphere = (4.0 / 3.0) * np.pi * r_max ** 3
    rho = N / V_sphere
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    shell_vols = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    norm = T * N * rho * shell_vols
    g_r = hist / np.where(norm > 0, norm, 1.0)
    return r_centers, g_r


# ---------------------------------------------------------------------------
# Hausdorff Chirality Measure (HCM)
# ---------------------------------------------------------------------------

def hausdorff_chirality_measure(positions: np.ndarray) -> float:
    """
    Compute the Hausdorff Chirality Measure (HCM) for a point set.

    Quantifies the degree of chirality by computing the normalized
    Hausdorff distance between a point set and its mirror image.

    Parameters
    ----------
    positions : (N, 3) coordinates.

    Returns
    -------
    float : HCM value in [0, 1]. 0 = achiral, larger = more chiral.
    """
    positions = np.asarray(positions, dtype=float)
    cm = positions.mean(axis=0)
    X  = positions - cm

    # Mirror image: reflect through the xy-plane of the PCA frame
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_pca    = X @ Vt.T
    X_mirror = X_pca.copy()
    X_mirror[:, 2] *= -1

    diff_matrix = np.sqrt(
        ((X_pca[:, np.newaxis] - X_mirror[np.newaxis]) ** 2).sum(axis=2)
    )
    h_fwd = diff_matrix.min(axis=1).max()
    h_bwd = diff_matrix.min(axis=0).max()
    hausdorff = max(h_fwd, h_bwd)

    diameter = np.sqrt(((X_pca[:, np.newaxis] - X_pca[np.newaxis]) ** 2)
                       .sum(axis=2)).max()

    if diameter < 1e-10:
        return 0.0
    return float(hausdorff / diameter)


# ---------------------------------------------------------------------------
# Projection center descriptor (e.g., center of gravity)
# ---------------------------------------------------------------------------

def projection_center(
    values: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Center of gravity of a weighted distribution.

    Parameters
    ----------
    values    : (M,) grid values (sorted ascending).
    weights   : (M,) distribution weights.
    threshold : cutoff for including weights.

    Returns
    -------
    float : distribution center.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = values <= threshold
    if not mask.any():
        return float("nan")
    V_sub   = values[mask]
    W_sub   = weights[mask]
    norm    = np.trapezoid(W_sub, V_sub)
    if abs(norm) < 1e-20:
        return float("nan")
    return float(np.trapezoid(V_sub * W_sub, V_sub) / norm)


# ---------------------------------------------------------------------------
# Batch descriptor extraction
# ---------------------------------------------------------------------------

def compute_trajectory_descriptors(
    trajectory: np.ndarray,
    cutoff: float = 3.4,
    max_neighbors: int = 12,
) -> dict:
    """
    Compute geometric descriptors for every frame in a trajectory.

    Parameters
    ----------
    trajectory    : (T, N, 3) positions.
    cutoff        : cutoff for neighbor detection.
    max_neighbors : max neighbor count for histogram.

    Returns
    -------
    dict with structural analysis results.
    """
    T = trajectory.shape[0]
    coord_hists = np.zeros((T, max_neighbors + 1), dtype=float)
    ecn_arr     = np.zeros(T, dtype=float)
    dav_arr     = np.zeros(T, dtype=float)
    rg_arr      = np.zeros(T, dtype=float)
    hcm_arr     = np.zeros(T, dtype=float)

    for t in range(T):
        pos = trajectory[t]
        coord_hists[t] = coordination_histogram(pos, cutoff, max_neighbors)
        ecn_arr[t]     = effective_coordination_number(pos, cutoff)
        dav_arr[t]     = average_neighbor_distance(pos, cutoff)
        rg_arr[t]      = radius_of_gyration(pos)
        hcm_arr[t]     = hausdorff_chirality_measure(pos)

    return {
        "neighbor_histograms": coord_hists,
        "ecn":  ecn_arr,
        "d_av": dav_arr,
        "rg":   rg_arr,
        "hcm":  hcm_arr,
    }
