"""
Dimensionality reduction: MDS and iterative Sketch-map optimisation.

Implements the full NLDRMDS and NLDRITER logic from libdimred.cpp:
    classical_mds   — Classical MDS via eigendecomposition of the centred
                      squared-distance matrix (Gram matrix).
    spherical_mds   — Spherical MDS (embeds on a sphere, mirrors SMDS mode).
    toroidal_mds    — Toroidal MDS (iterative, mirrors TMDS mode).
    sketch_map      — Iterative Sketch-map optimisation of χ² using
                      conjugate-gradient, with optional pointwise global
                      grid refinement (mirrors NLDRITER + global=True).
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from typing import Optional, Dict, Tuple
from dimredpy.shared.metrics import Metric, EuclideanMetric
from dimredpy.shared.transfer import TransferFunction, Identity, make_transfer


# ---------------------------------------------------------------------------
# Classical MDS
# ---------------------------------------------------------------------------

def classical_mds(
    data: np.ndarray,
    n_components: int = 2,
    metric: Optional[Metric] = None,
    dist_matrix: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict:
    """
    Classical (linear) Multi-Dimensional Scaling.

    Mirrors C++ NLDRMDS with mode=MDS.

    Parameters
    ----------
    data         : (N, D) input points.
    n_components : target low dimensionality d.
    metric       : distance metric (default Euclidean).
    dist_matrix  : (N, N) pre-computed distance matrix; if given, data is
                   only used to determine N.
    verbose      : if True, return per-point errors and eigenvalues.

    Returns
    -------
    dict with:
        "embedding"   : (N, d) low-dimensional coordinates.
        "eigenvalues" : (d,) eigenvalues of the Gram matrix.
        "error"       : scalar stress (fraction of variance explained).
        "per_point_errors" : (N,) per-point errors (only if verbose=True).
    """
    data = np.asarray(data, dtype=float)
    N = data.shape[0]

    if dist_matrix is not None:
        D2 = np.asarray(dist_matrix, dtype=float) ** 2
    else:
        if metric is None:
            metric = EuclideanMetric()
        dist_matrix = metric.pairwise(data)
        D2 = dist_matrix ** 2

    # Double-centring  M = -½ H D² H  where H = I - (1/N) 1 1ᵀ
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    M = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    # Eigendecomposition — take the top n_components eigenpairs
    vals, vecs = eigh(M, subset_by_index=[N - n_components, N - 1])

    # Sort descending
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Clamp negative eigenvalues (numerical noise)
    vals_pos = np.maximum(vals, 0.0)
    embedding = vecs * np.sqrt(vals_pos)

    result = {
        "embedding": embedding,
        "eigenvalues": vals,
        "error": float(vals_pos.sum() / np.maximum(np.trace(M), 1e-300)),
    }

    if verbose:
        # Per-point MDS error: squared difference between original and
        # reconstructed distances
        D_ld = EuclideanMetric().pairwise(embedding)
        errors = ((dist_matrix - D_ld) ** 2).mean(axis=1)
        result["per_point_errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Spherical MDS
# ---------------------------------------------------------------------------

def spherical_mds(
    data: np.ndarray,
    n_components: int = 2,
    metric: Optional[Metric] = None,
    dist_matrix: Optional[np.ndarray] = None,
) -> Dict:
    """
    Spherical MDS: embeds data onto a hyper-sphere by fitting a cosine
    distance matrix.  Mirrors C++ NLDRMDS with mode=SMDS.

    Returns the angular coordinates on [0,1] (i.e. divided by π).
    """
    data = np.asarray(data, dtype=float)
    N = data.shape[0]

    if dist_matrix is not None:
        dist = np.asarray(dist_matrix, dtype=float)
    else:
        if metric is None:
            metric = EuclideanMetric()
        dist = metric.pairwise(data)

    # Radius of the hyper-sphere
    sr = dist.max() / np.pi

    # Cosine Gram matrix  M_ij = cos(d_ij / sr) * sr²
    M = (sr ** 2) * np.cos(dist / sr)

    d = n_components
    vals, vecs = eigh(M, subset_by_index=[N - d - 1, N - 1])
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    vals_pos = np.maximum(vals, 0.0)
    Q = vecs * np.sqrt(vals_pos)     # (N, d+1) Cartesian embedding on sphere

    # Convert to angular coordinates as in C++
    embedding = np.zeros((N, d), dtype=float)
    for i in range(N):
        tx = 0.0
        embedding[i, 0] = np.arctan2(Q[i, d], Q[i, d - 1]) / np.pi
        tx += Q[i, d] ** 2
        for h in range(1, d):
            tx += Q[i, d - h] ** 2
            embedding[i, h] = np.arctan2(np.sqrt(tx), Q[i, d - h - 1]) / np.pi

    return {
        "embedding": embedding,
        "eigenvalues": vals[:d],
        "error": float(vals_pos[:d].sum() / np.maximum(np.trace(M), 1e-300)),
    }


# ---------------------------------------------------------------------------
# Toroidal MDS  (iterative deflation, mirrors TMDS)
# ---------------------------------------------------------------------------

def toroidal_mds(
    data: np.ndarray,
    n_components: int = 2,
    metric: Optional[Metric] = None,
    dist_matrix: Optional[np.ndarray] = None,
) -> Dict:
    """
    Toroidal MDS: iteratively extracts circular components by spherical MDS
    followed by residual-distance deflation.  Mirrors C++ NLDRMDS mode=TMDS.
    """
    data = np.asarray(data, dtype=float)
    N = data.shape[0]

    if dist_matrix is not None:
        dist = np.asarray(dist_matrix, dtype=float).copy()
    else:
        if metric is None:
            metric = EuclideanMetric()
        dist = metric.pairwise(data)

    euclid = EuclideanMetric()
    embedding = np.zeros((N, n_components), dtype=float)
    eigenvalues = []

    for th in range(n_components):
        res = spherical_mds(data, n_components=1, dist_matrix=dist)
        col = res["embedding"][:, 0]
        eigenvalues.append(res["eigenvalues"][0])
        embedding[:, th] = col

        # Deflation: subtract the circular component from distance matrix
        sr = dist.max() / np.pi
        for i in range(N):
            for j in range(i):
                tdij = abs(col[i] - col[j])
                while tdij > 1:
                    tdij -= 2
                tdij = abs(tdij) * np.pi * sr
                residual = dist[i, j] ** 2 - tdij ** 2
                dist[i, j] = dist[j, i] = np.sqrt(max(residual, 0.0))

    return {
        "embedding": embedding,
        "eigenvalues": np.array(eigenvalues),
        "error": 0.0,
    }


# ---------------------------------------------------------------------------
# Sketch-map stress (χ²) and gradient
# ---------------------------------------------------------------------------

def _chi_squared(
    coords_flat: np.ndarray,
    hd_dist: np.ndarray,
    fhd: np.ndarray,
    ld_transfer: TransferFunction,
    weights: Optional[np.ndarray],
    imix: float,
    n: int,
    d: int,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate the Sketch-map stress χ² and its gradient.

    χ² = Σ_{i<j} w_ij * [(F(D_ij) - f(d_ij))² (1-imix) + imix*(D_ij-d_ij)²]
       normalised by Σ w_ij.

    Parameters
    ----------
    coords_flat : (n*d,) flattened low-dim coordinates.
    hd_dist     : (n,n) high-dim raw distances.
    fhd         : (n,n) transformed high-dim distances F(D_ij).
    ld_transfer : low-dim transfer function f.
    weights     : (n,n) pairwise weight matrix, or None (uniform).
    imix        : mixing ratio (0 = pure Sketch-map, 1 = pure MDS).
    n, d        : number of points and low dimensionality.

    Returns
    -------
    (chi, gradient_flat)
    """
    coords = coords_flat.reshape(n, d)

    # Low-dim pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n,n,d)
    ld_dist2 = (diff ** 2).sum(axis=2)                           # (n,n)
    ld_dist = np.sqrt(np.maximum(ld_dist2, 1e-200))

    fld, dfld = ld_transfer.fdf(ld_dist)

    # Stress components per pair
    delta_f = fhd - fld                               # (n,n)
    delta_d = hd_dist - ld_dist

    if weights is not None:
        W = weights
    else:
        W = np.ones((n, n), dtype=float)
        np.fill_diagonal(W, 0.0)

    # Upper triangle only
    triu = np.triu_indices(n, k=1)
    w_ij = W[triu]
    tw = w_ij.sum()
    if tw == 0:
        tw = 1.0

    # Chi squared
    stress_f = delta_f[triu] ** 2
    stress_d = delta_d[triu] ** 2
    chi = (w_ij * ((1.0 - imix) * stress_f + imix * stress_d)).sum() / tw

    # Gradient
    safe_ld = np.where(ld_dist < 1e-100, 1e-100, ld_dist)
    # factor per pair (i,j): g_ij * (r_i - r_j)
    g_ij_full = (
        (delta_f * dfld * (1.0 - imix) + imix * delta_d) / safe_ld
    ) * W

    # Only upper triangle contributes, symmetrised with sign
    grad_coords = np.zeros((n, d), dtype=float)
    for h in range(d):
        contrib = g_ij_full * diff[:, :, h]        # (n,n)
        # sum over j for each i: gradient_i = Σ_j g_ij*(x_i-x_j)
        grad_coords[:, h] = contrib.sum(axis=1) - contrib.sum(axis=0)

    grad_flat = (-2.0 / tw) * grad_coords.ravel()
    return chi, grad_flat


# ---------------------------------------------------------------------------
# Sketch-map  (iterative optimisation, mirrors NLDRITER)
# ---------------------------------------------------------------------------

def sketch_map(
    data: np.ndarray,
    n_components: int = 2,
    weights: Optional[np.ndarray] = None,
    init: Optional[np.ndarray] = None,
    metric: Optional[Metric] = None,
    fun_hd=(6.0, 8.0, 8.0),
    fun_ld=(6.0, 2.0, 8.0),
    preopt_steps: int = 100,
    grid: Optional[Tuple] = None,
    global_steps: int = 0,
    imix: float = 0.0,
    dist_matrix: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict:
    """
    Full Sketch-map embedding.  Mirrors dimred.cpp workflow:
        1. Compute HD distance matrix.
        2. Initialise from classical MDS (or provided init).
        3. Optimise χ² with conjugate gradient (preopt_steps).
        4. Optionally run pointwise global grid optimisation.

    Parameters
    ----------
    data          : (N, D) landmark points.
    n_components  : output dimensionality d (usually 2).
    weights       : (N,) per-landmark weights (Voronoi weights from selection).
    init          : (N, d) initial low-dim positions (None -> classical MDS).
    metric        : distance metric (default: Euclidean).
    fun_hd        : HD transfer function spec, e.g. (6.0, 8.0, 8.0).
    fun_ld        : LD transfer function spec.
    preopt_steps  : max CG iterations for global optimisation.
    grid          : (width, coarse_pts, fine_pts) for pointwise global search.
                   If None, no pointwise global refinement is done.
    global_steps  : CG steps run after each pointwise global move.
    imix          : mix ratio between Sketch-map (0) and MDS (1) stress.
    dist_matrix   : (N,N) pre-computed distance matrix.
    verbose       : if True, return per-point errors.

    Returns
    -------
    dict with:
        "embedding"        : (N, d) low-dimensional coordinates.
        "stress"           : final χ² value.
        "per_point_errors" : (N,) per-point contributions (if verbose).
    """
    data = np.asarray(data, dtype=float)
    N, D = data.shape
    d = n_components

    if metric is None:
        metric = EuclideanMetric()

    # --- HD distance matrix ---
    if dist_matrix is not None:
        hd_dist = np.asarray(dist_matrix, dtype=float)
    else:
        hd_dist = metric.pairwise(data)

    # --- Transfer functions ---
    tf_hd = make_transfer(fun_hd)
    tf_ld = make_transfer(fun_ld)

    # --- Transformed HD distances ---
    fhd = tf_hd.f(hd_dist)
    np.fill_diagonal(fhd, 0.0)

    # --- Weight matrix W_ij = w_i * w_j ---
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        W = np.outer(w, w)
    else:
        W = np.ones((N, N), dtype=float)
    np.fill_diagonal(W, 0.0)

    # --- Initialisation ---
    if init is not None:
        coords = np.asarray(init, dtype=float).copy()
    else:
        mds_res = classical_mds(data, n_components=d, dist_matrix=hd_dist)
        coords = mds_res["embedding"].copy()

    coords_flat = coords.ravel()

    # --- Pre-optimisation: CG on all points jointly ---
    if preopt_steps > 0:
        def obj(x):
            v, g = _chi_squared(x, hd_dist, fhd, tf_ld, W, imix, N, d)
            return v, g

        res = minimize(
            obj,
            coords_flat,
            jac=True,
            method="CG",
            options={"maxiter": preopt_steps, "gtol": 1e-6},
        )
        coords_flat = res.x
        coords = coords_flat.reshape(N, d)

    # --- Pointwise global grid refinement (mirrors NLDRITER global=True) ---
    if grid is not None and d == 2:
        gwidth, g1, g2 = float(grid[0]), int(grid[1]), int(grid[2])
        coords = _global_grid_refinement(
            coords, hd_dist, fhd, tf_ld, W, imix, N, d,
            gwidth, g1, g2, global_steps
        )

    # --- Final stress ---
    chi, _ = _chi_squared(coords.ravel(), hd_dist, fhd, tf_ld, W, imix, N, d)

    result = {"embedding": coords, "stress": float(chi)}

    if verbose:
        # Per-point errors
        ld_euclid = EuclideanMetric()
        ld_dist = ld_euclid.pairwise(coords)
        fld = tf_ld.f(ld_dist)
        delta_f = (fhd - fld) ** 2
        delta_d = (hd_dist - ld_dist) ** 2
        per_point = (((1.0 - imix) * delta_f + imix * delta_d) * W).sum(axis=1)
        w_sums = W.sum(axis=1)
        result["per_point_errors"] = np.where(w_sums > 0, per_point / w_sums, 0.0)

    return result


# ---------------------------------------------------------------------------
# Pointwise global grid refinement
# ---------------------------------------------------------------------------

def _global_grid_refinement(
    coords, hd_dist, fhd, tf_ld, W, imix, N, d,
    gwidth, g1, g2, cg_steps
):
    """
    For each point ip, scan a 2D grid and move to the global minimum,
    then refine with CG.  Mirrors NLDRITER global mode in C++.
    Requires d == 2.
    """
    from scipy.interpolate import RectBivariateSpline

    coords = coords.copy()
    chi1_weights = W.sum(axis=1)         # row sums used as chi1 weights
    chi1_weights = np.where(chi1_weights > 0, chi1_weights, 1.0)

    gx = np.linspace(-gwidth, gwidth, g1)
    gy = np.linspace(-gwidth, gwidth, g1)

    for ip in range(N):
        # Compute stress at each coarse grid point for landmark ip
        grid_vals = np.zeros((g1, g1), dtype=float)
        grid_dx   = np.zeros((g1, g1), dtype=float)
        grid_dy   = np.zeros((g1, g1), dtype=float)

        for i, xi in enumerate(gx):
            for j, yj in enumerate(gy):
                v, g = _chi1(
                    np.array([xi, yj]), ip, coords, hd_dist, fhd, tf_ld, imix, W
                )
                grid_vals[i, j] = v
                grid_dx[i, j]   = g[0]
                grid_dy[i, j]   = g[1]

        # Bicubic interpolation on fine grid
        interp = RectBivariateSpline(gx, gy, grid_vals, kx=3, ky=3)
        gx_f = np.linspace(-gwidth, gwidth, g2)
        gy_f = np.linspace(-gwidth, gwidth, g2)
        fine_vals = interp(gx_f, gy_f)

        init_v, _ = _chi1(coords[ip], ip, coords, hd_dist, fhd, tf_ld, imix, W)
        best_i, best_j = np.unravel_index(np.argmin(fine_vals), fine_vals.shape)
        min_v = fine_vals[best_i, best_j]

        if min_v >= init_v:
            continue   # interpolant found no improvement — skip

        # Verify on true function (guard against interpolation artefacts)
        cand = np.array([gx_f[best_i], gy_f[best_j]])
        true_v, _ = _chi1(cand, ip, coords, hd_dist, fhd, tf_ld, imix, W)
        if true_v >= init_v:
            continue

        coords[ip] = cand

        # CG refinement after global move
        if cg_steps > 0:
            def obj_ip(x):
                return _chi1(x, ip, coords, hd_dist, fhd, tf_ld, imix, W)
            res = minimize(obj_ip, coords[ip], jac=True, method="CG",
                           options={"maxiter": cg_steps, "gtol": 1e-8})
            coords[ip] = res.x

    return coords


def _chi1(x_ip, ip, coords, hd_dist, fhd, tf_ld, imix, W):
    """
    Single-point stress for landmark ip at trial position x_ip.
    Used during pointwise global optimisation.
    """
    N = coords.shape[0]
    diff = coords - x_ip                  # (N, d)
    ld = np.sqrt((diff ** 2).sum(axis=1)) # (N,)
    ld[ip] = 0.0

    fld, dfld = tf_ld.fdf(ld)
    safe_ld = np.where(ld < 1e-100, 1e-100, ld)

    w = W[ip].copy()
    w[ip] = 0.0
    tw = w.sum()
    if tw == 0:
        tw = 1.0

    delta_f = fhd[ip] - fld
    delta_d = hd_dist[ip] - ld

    val = (w * ((1.0 - imix) * delta_f**2 + imix * delta_d**2)).sum() / tw

    g_factor = ((delta_f * dfld * (1.0 - imix) + imix * delta_d) / safe_ld) * w
    g_factor[ip] = 0.0
    grad = (-2.0 / tw) * (g_factor[:, np.newaxis] * diff).sum(axis=0)

    return val, grad
