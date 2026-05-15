"""
Out-of-sample projection into an existing Sketch-map embedding.

Mirrors the C++ dimproj tool and NLDRProjection::project():
    - Evaluates the per-point stress χ₁²(x) on a coarse 2D grid.
    - Builds a bicubic interpolant on the grid.
    - Scans a fine grid for the global minimum.
    - Optionally refines with conjugate-gradient minimisation.
    - Supports a simple path-based (exponential-averaging) fallback.

Only d=2 is supported for the grid/bicubic path (same limitation as C++).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline
from typing import Optional, Dict, Tuple
from dimredpy.shared.metrics import Metric, EuclideanMetric
from dimredpy.shared.transfer import TransferFunction, make_transfer

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Core per-point stress function χ₁²(x_new | landmarks)
# ---------------------------------------------------------------------------

def _single_point_stress(
    x_new: np.ndarray,
    hd_dists_to_landmarks: np.ndarray,
    fhd_to_landmarks: np.ndarray,
    landmarks_ld: np.ndarray,
    tf_ld: TransferFunction,
    weights: np.ndarray,
    imix: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Compute stress and gradient for a single new point at low-dim position x_new.

    χ₁²(x) = (1/Σw_i) Σ_i w_i * [ (F(D_i) - f(|x - p_i|))² (1-imix)
                                    + imix * (D_i - |x - p_i|)² ]

    Parameters
    ----------
    x_new                  : (d,) candidate low-dim position.
    hd_dists_to_landmarks  : (K,) HD distances from the new point to each landmark.
    fhd_to_landmarks       : (K,) F(HD distances).
    landmarks_ld           : (K, d) low-dim landmark positions.
    tf_ld                  : low-dim transfer function f.
    weights                : (K,) landmark weights.
    imix                   : stress mixing ratio.

    Returns
    -------
    (stress_scalar, gradient_d)
    """
    diff   = landmarks_ld - x_new             # (K, d)
    ld     = np.sqrt((diff ** 2).sum(axis=1)) # (K,)
    safe   = np.where(ld < 1e-100, 1e-100, ld)

    fld, dfld = tf_ld.fdf(ld)

    delta_f = fhd_to_landmarks - fld
    delta_d = hd_dists_to_landmarks - ld

    tw = weights.sum()
    if tw == 0:
        tw = 1.0

    val = (weights * ((1.0 - imix) * delta_f**2 + imix * delta_d**2)).sum() / tw

    g_factor = ((delta_f * dfld * (1.0 - imix) + imix * delta_d) / safe) * weights
    # gradient w.r.t. x_new:  +2 * Σ_i g_i * (x_new - p_i) / tw
    grad = (2.0 / tw) * (g_factor[:, np.newaxis] * (-diff)).sum(axis=0)

    return val, grad


# ---------------------------------------------------------------------------
# Main projection function
# ---------------------------------------------------------------------------

def project_out_of_sample(
    samples: np.ndarray,
    landmarks_hd: np.ndarray,
    landmarks_ld: np.ndarray,
    weights: Optional[np.ndarray] = None,
    metric: Optional[Metric] = None,
    fun_hd=(6.0, 8.0, 8.0),
    fun_ld=(6.0, 2.0, 8.0),
    grid: Tuple = (1.0, 21, 201),
    cg_steps: int = 0,
    gt: float = 0.0,
    similarity: bool = False,
    imix: float = 0.0,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Project new high-dimensional samples into the existing landmark embedding.

    Mirrors dimproj.cpp / NLDRProjection::project().

    Parameters
    ----------
    samples      : (M, D) new data points to project.
    landmarks_hd : (K, D) high-dim landmark coordinates.
    landmarks_ld : (K, d) low-dim landmark embedding (from sketch_map).
    weights      : (K,) landmark weights. Default: uniform.
    metric       : HD distance metric (default: Euclidean).
    fun_hd       : HD transfer function spec.
    fun_ld       : LD transfer function spec.
    grid         : (width, coarse_pts, fine_pts).
                   The 2D search box spans ±width.
    cg_steps     : conjugate-gradient refinement steps after grid search.
    gt           : if > 0, use exponential-temperature averaging over the fine
                   grid instead of strict minimum  (mirrors -gt flag).
    similarity   : if True, rows of `samples` are pre-computed HD distances
                   to each landmark (not raw coordinates).
    imix         : mix ratio (Sketch-map vs MDS stress).

    Returns
    -------
    dict with:
        "embedding"         : (M, d) projected low-dim coordinates.
        "error"             : (M,) per-point projection stress at the optimum.
        "nearest_distance"  : (M,) HD distance to the nearest landmark.
    """
    samples      = np.asarray(samples, dtype=float)
    landmarks_hd = np.asarray(landmarks_hd, dtype=float)
    landmarks_ld = np.asarray(landmarks_ld, dtype=float)
    M = samples.shape[0]
    K, d_lm = landmarks_ld.shape

    # --- Batched CPU/GPU Branch ---
    if use_gpu and HAS_TORCH and torch.cuda.is_available():
        if verbose:
            print(f"   -> [GPU] Using {torch.cuda.get_device_name()} for Sketch-map projection...")
        try:
            return _project_out_of_sample_gpu(
                samples, landmarks_hd, landmarks_ld, weights, metric,
                fun_hd, fun_ld, grid, similarity, imix, verbose, cg_steps
            )
        except Exception as e:
            if verbose:
                print(f"   -> [GPU] PyTorch projection failed ({e}). Falling back to Batched CPU...")

    # --- High-Performance Batched CPU ---
    if verbose:
        print(f"   -> [CPU] Using Vectorized Batch Projection (Speed-Optimized)...")
    
    return _cpu_project_all_batched(
        samples, landmarks_hd, landmarks_ld, weights, metric,
        fun_hd, fun_ld, grid, similarity, imix, verbose, cg_steps
    )

def _cpu_project_all_batched(
    samples, landmarks_hd, landmarks_ld, weights, metric,
    fun_hd, fun_ld, grid, similarity, imix, verbose, cg_steps
) -> Dict:
    """
    Vectorized CPU projection using NumPy matrix operations.
    Replaces the slow one-by-one Python loop.
    """
    from scipy.optimize import minimize
    
    M = samples.shape[0]
    K, d = landmarks_ld.shape
    gwidth, g1, g2 = grid
    
    if weights is None:
        weights = np.ones(K, dtype=float)
    tw = weights.sum()
    if tw == 0: tw = 1.0

    tf_hd = make_transfer(fun_hd)
    tf_ld = make_transfer(fun_ld)
    
    # 1. Setup Grid and Pre-calculate f(ld)
    # We use the fine grid resolution (g2) directly for maximum accuracy
    # g2xG2 scan is much faster in NumPy batches than Bicubic + Fine scan in Python loops
    res_grid = g2
    gx = np.linspace(-gwidth, gwidth, res_grid)
    gy = np.linspace(-gwidth, gwidth, res_grid)
    grid_pts = np.stack(np.meshgrid(gx, gy, indexing='ij'), axis=-1).reshape(-1, 2)
    G = grid_pts.shape[0]
    
    if verbose:
        print(f"      Grid: {res_grid}x{res_grid} ({G} points)")

    # Pairwise distances from grid points to landmarks: (G, K)
    from scipy.spatial.distance import cdist
    dist_grid_lm = cdist(grid_pts, landmarks_ld)
    f_ld_grid = tf_ld.f(dist_grid_lm) # (G, K)
    
    # 2. Batch Processing
    # Smaller batch size for high-res grid to keep memory safe (~400MB)
    batch_size = 500
    all_pos = np.zeros((M, d))
    all_err = np.zeros(M)
    all_near = np.zeros(M)
    
    for i in range(0, M, batch_size):
        if verbose and i % 20000 == 0:
            print(f"      Batch {i}/{M}...")
            
        b_idx = slice(i, min(i + batch_size, M))
        b_samples = samples[b_idx]
        B = b_samples.shape[0]
        
        # HD distances: (B, K)
        if similarity:
            hd_dists = b_samples
        else:
            hd_dists = cdist(b_samples, landmarks_hd)
            
        all_near[b_idx] = hd_dists.min(axis=1)
        
        # F_hd: (B, K)
        f_hd = tf_hd.f(hd_dists)
        
        # Stress matrix: (B, G)
        # Expansion: Σ w * (F-f)^2 = Σ w*F^2 + Σ w*f^2 - 2 * Σ w*F*f
        # term1: (B, 1)
        term1 = (f_hd**2 @ weights)[:, np.newaxis]
        # term2: (1, G)
        term2 = (f_ld_grid**2 @ weights)[np.newaxis, :]
        # term3: (B, G)
        term3 = (f_hd * weights) @ f_ld_grid.T
        
        stress_mat = (term1 + term2 - 2.0 * term3) / tw
        
        # Find best grid points
        best_g_idx = np.argmin(stress_mat, axis=1)
        batch_pos = grid_pts[best_g_idx]
        
        # 3. CG Refinement (only if requested)
        if cg_steps > 0:
            # We refine each point in the batch.
            # While this is a loop, it's only for a few steps and after a very good start.
            for j in range(B):
                def obj(x):
                    return _single_point_stress(
                        x, hd_dists[j], f_hd[j], landmarks_ld, tf_ld, weights, imix
                    )
                res = minimize(obj, batch_pos[j], jac=True, method="CG",
                               options={"maxiter": cg_steps, "gtol": 1e-7})
                batch_pos[j] = res.x
                all_err[i+j] = res.fun
        else:
            all_err[b_idx] = np.min(stress_mat, axis=1)
            
        all_pos[b_idx] = batch_pos

    return {
        "embedding": all_pos,
        "error": all_err,
        "nearest_distance": all_near,
    }

def _project_out_of_sample_gpu(
    samples, landmarks_hd, landmarks_ld, weights, metric,
    fun_hd, fun_ld, grid, similarity, imix, verbose, cg_steps
) -> Dict:
    import torch
    device = torch.device("cuda")
    
    M = samples.shape[0]
    L = landmarks_hd.shape[0]
    
    # 1. Setup sigmoids on GPU
    sigma_hd, A, B = fun_hd
    sigma_ld, a, b = fun_ld
    pre_A = 2.0**(A/B) - 1.0
    pre_a = 2.0**(a/b) - 1.0
    
    # 2. Setup Grid on GPU
    gwidth, g1, g2 = grid
    coarse_vec = torch.linspace(-gwidth, gwidth, g1, device=device)
    grid_coarse = torch.stack(torch.meshgrid(coarse_vec, coarse_vec, indexing='ij'), dim=-1).reshape(-1, 2)
    
    # Pre-embed landmarks on GPU
    lm_ld_t = torch.tensor(landmarks_ld, dtype=torch.float32, device=device)
    w_t = torch.tensor(weights if weights is not None else np.ones(L), dtype=torch.float32, device=device)
    
    # f_ld for every grid point to every landmark
    # dist_grid_lm: (G, L)
    dist_grid_lm = torch.cdist(grid_coarse, lm_ld_t)
    f_ld_grid = 1.0 - (1.0 + pre_a * (dist_grid_lm / sigma_ld)**a)**(-b/a) # (G, L)
    
    # 3. Batch Processing
    batch_size = 5000
    all_pos = []
    all_err = []
    all_near = []
    
    lm_hd_t = torch.tensor(landmarks_hd, dtype=torch.float32, device=device)
    
    for i in range(0, M, batch_size):
        if verbose and i % 50000 == 0:
            print(f"      Batch {i}/{M}...")
            
        batch_end = min(i + batch_size, M)
        batch_samples = torch.tensor(samples[i:batch_end], dtype=torch.float32, device=device)
        
        # HD distances: (B, L)
        if similarity:
            hd_dists = batch_samples
        else:
            hd_dists = torch.cdist(batch_samples, lm_hd_t)
            
        all_near.append(torch.min(hd_dists, dim=1)[0].cpu().numpy())
        
        # F_hd: (B, L)
        f_hd = 1.0 - (1.0 + pre_A * (hd_dists / sigma_hd)**A)**(-B/A)
        
        # Stress: (B, G) = sum_j w_j * (F_hd_ij - f_ld_gj)^2
        # Expansion: sum w*F^2 + sum w*f^2 - 2 * sum w*F*f
        term1 = torch.matmul(f_hd**2, w_t.unsqueeze(1)) # (B, 1)
        term2 = torch.matmul(w_t.unsqueeze(0), (f_ld_grid**2).T) # (1, G)
        term3 = torch.matmul(f_hd * w_t, f_ld_grid.T) # (B, G)
        
        stress = term1 + term2 - 2.0 * term3
        best_idx = torch.argmin(stress, dim=1)
        best_pos = grid_coarse[best_idx].clone() # (B, 2)
        
        # 4. Continuous Refinement (Pro-Grade Adam on GPU)
        if cg_steps > 0:
            # Shake them off the grid to break symmetry
            best_pos += torch.randn_like(best_pos) * (gwidth / g2)
            best_pos.requires_grad_(True)
            
            # Adam parameters
            m = torch.zeros_like(best_pos)
            v = torch.zeros_like(best_pos)
            lr = 0.02
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            
            # 100 steps for deep convergence
            for t in range(1, 101):
                dist_lp = torch.cdist(best_pos, lm_ld_t)
                f_ld_p = 1.0 - (1.0 + pre_a * (dist_lp / sigma_ld)**a)**(-b/a)
                b_term2 = torch.matmul(f_ld_p**2, w_t.unsqueeze(1))
                b_term3 = (f_hd * w_t * f_ld_p).sum(dim=1, keepdim=True)
                batch_stress = (term1 + b_term2 - 2.0 * b_term3).sum()
                
                batch_stress.backward()
                
                with torch.no_grad():
                    grad = best_pos.grad
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    best_pos -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                    best_pos.grad.zero_()
            
            with torch.no_grad():
                dist_lp = torch.cdist(best_pos, lm_ld_t)
                f_ld_p = 1.0 - (1.0 + pre_a * (dist_lp / sigma_ld)**a)**(-b/a)
                b_term2 = torch.matmul(f_ld_p**2, w_t.unsqueeze(1))
                b_term3 = (f_hd * w_t * f_ld_p).sum(dim=1, keepdim=True)
                final_stress = (term1 + b_term2 - 2.0 * b_term3) / w_t.sum()
        else:
            final_stress = torch.min(stress, dim=1)[0].unsqueeze(1) / w_t.sum()

        all_pos.append(best_pos.detach().cpu().numpy())
        all_err.append(final_stress.detach().cpu().numpy().ravel())

    return {
        "embedding": np.concatenate(all_pos, axis=0),
        "error": np.concatenate(all_err, axis=0),
        "nearest_distance": np.concatenate(all_near, axis=0),
    }


# ---------------------------------------------------------------------------
# Grid-based projection (2D only)
# ---------------------------------------------------------------------------

def _grid_project(hd_dists, fhd, landmarks_ld, tf_ld, weights, imix,
                  gwidth, g1, g2, cg_steps, gt):
    """Run coarse grid -> bicubic interpolation -> fine grid scan -> CG."""

    gx = np.linspace(-gwidth, gwidth, g1)
    gy = np.linspace(-gwidth, gwidth, g1)
    grid_vals = np.zeros((g1, g1), dtype=float)
    grid_dx   = np.zeros((g1, g1), dtype=float)
    grid_dy   = np.zeros((g1, g1), dtype=float)

    for i, xi in enumerate(gx):
        for j, yj in enumerate(gy):
            v, g = _single_point_stress(
                np.array([xi, yj]), hd_dists, fhd, landmarks_ld, tf_ld, weights, imix
            )
            grid_vals[i, j] = v
            grid_dx[i, j]   = g[0]
            grid_dy[i, j]   = g[1]

    # Bicubic interpolation
    interp = RectBivariateSpline(gx, gy, grid_vals, kx=3, ky=3)

    gx_f = np.linspace(-gwidth, gwidth, g2)
    gy_f = np.linspace(-gwidth, gwidth, g2)
    fine_vals = interp(gx_f, gy_f)

    if gt > 0.0:
        # Exponential-temperature averaging (mirrors -gt flag)
        reff = fine_vals.min()
        echi = np.exp(-(fine_vals - reff) / gt)
        tt   = echi.sum()
        XX, YY = np.meshgrid(gx_f, gy_f, indexing='ij')
        tx = (XX * echi).sum() / tt
        ty = (YY * echi).sum() / tt
        tr = np.sqrt(XX**2 + YY**2)
        tr = (tr * echi).sum() / tt
        # Angle-weighted radial position (mirrors C++)
        angle = np.arctan2(ty, tx)
        best_pos = np.array([tr * np.cos(angle), tr * np.sin(angle)])
    else:
        best_idx = np.unravel_index(np.argmin(fine_vals), fine_vals.shape)
        best_pos = np.array([gx_f[best_idx[0]], gy_f[best_idx[1]]])

    # CG refinement
    if cg_steps > 0:
        def obj(x):
            return _single_point_stress(x, hd_dists, fhd, landmarks_ld, tf_ld, weights, imix)
        res = minimize(obj, best_pos, jac=True, method="CG",
                       options={"maxiter": cg_steps, "gtol": 1e-9})
        best_pos = res.x
        err = res.fun
    else:
        err, _ = _single_point_stress(best_pos, hd_dists, fhd, landmarks_ld, tf_ld, weights, imix)

    return best_pos, err


# ---------------------------------------------------------------------------
# Simple path-based projection (mirrors -path flag)
# ---------------------------------------------------------------------------

def path_project(
    samples: np.ndarray,
    landmarks_hd: np.ndarray,
    landmarks_ld: np.ndarray,
    lam: float,
    metric: Optional[Metric] = None,
) -> np.ndarray:
    """
    Path-like projection: exponential-distance-weighted average of LD landmarks.

        p_new = Σ_i exp(-D_i/λ) * p_i  /  Σ_i exp(-D_i/λ)

    Mirrors dimproj.cpp -path option.

    Parameters
    ----------
    samples      : (M, D) new data points.
    landmarks_hd : (K, D) HD landmark coordinates.
    landmarks_ld : (K, d) LD landmark coordinates.
    lam          : exponential decay length λ.
    metric       : HD distance metric (default Euclidean).

    Returns
    -------
    (M, d) projected coordinates.
    """
    if metric is None:
        metric = EuclideanMetric()
    samples      = np.asarray(samples, dtype=float)
    landmarks_hd = np.asarray(landmarks_hd, dtype=float)
    landmarks_ld = np.asarray(landmarks_ld, dtype=float)

    hd_dists = metric.pairwise_vec(samples, landmarks_hd)  # (M, K)
    w = np.exp(-hd_dists / lam)                            # (M, K)
    tw = w.sum(axis=1, keepdims=True)                      # (M, 1)
    return (w[:, :, np.newaxis] * landmarks_ld[np.newaxis]).sum(axis=1) / tw


# ---------------------------------------------------------------------------
# GPU-accelerated projection (PyTorch)
# ---------------------------------------------------------------------------

