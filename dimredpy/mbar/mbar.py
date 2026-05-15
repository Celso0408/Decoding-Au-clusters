"""
MBAR (Multistate Bennett Acceptance Ratio) free-energy surface estimation.

Implements the MBAR workflow commonly used for multi-state free energy calculation:
    1. Solve the self-consistent MBAR equations to obtain reduced free energies f_k.
    2. Compute MBAR statistical weights w_{n,k} for all samples across all states.
    3. Project onto a 2D collective-variable grid to build a probability surface.
    4. Convert to a free-energy surface  F(s) = -kB T ln p(s).

This module wraps pymbar for the core MBAR solver and adds the 2D gridding and
free-energy surface construction that are specific to this workflow.

Reference implementation: https://github.com/choderalab/pymbar

References
----------
Shirts & Chodera, J. Chem. Phys. 129, 124105 (2008).
"""

import numpy as np
from typing import Optional, Dict, Tuple


# Scaling constant (e.g., Boltzmann constant kB)
_kB_CONST = 8.617333262e-5


# ---------------------------------------------------------------------------
# Timeseries decorrelation
# ---------------------------------------------------------------------------

def decorrelate_timeseries(
    data: np.ndarray,
    energies: Optional[np.ndarray] = None,
    method: str = "statistical_inefficiency",
    max_stride: int = 500,
) -> Dict:
    """
    Subsample correlated time series to extract statistically independent samples.

    This is a crucial preprocessing step before MBAR analysis: pymbar assumes
    all input samples are uncorrelated. Applying this function to each replica's
    time series before constructing u_kn ensures this assumption holds.

    Parameters
    ----------
    data     : (N, D) trajectory data (e.g. feature vectors or descriptors).
    energies : (N,) potential energies; if provided, uses energy-based
               autocorrelation for estimating the statistical inefficiency g.
               If None, uses a simple stride-based estimate from data variance.
    method   : "statistical_inefficiency" to use pymbar.timeseries, or
               "stride" for a simple fixed-stride subsampling.
    max_stride : upper bound on the stride for safety.

    Returns
    -------
    dict with:
        "data"    : (N_eff, D) decorrelated data.
        "indices" : (N_eff,) original indices of retained samples.
        "g"       : statistical inefficiency estimate.
    """
    N = len(data)

    if method == "statistical_inefficiency" and energies is not None:
        try:
            from pymbar import timeseries as ts
            g = ts.statistical_inefficiency(energies)
            g = min(g, max_stride)
            indices = ts.subsample_correlated_data(energies, g=g)
            return {
                "data": data[indices],
                "indices": np.asarray(indices),
                "g": g,
            }
        except ImportError:
            pass  # fall through to stride method

    # Simple stride-based fallback
    if energies is not None:
        # Estimate autocorrelation time from energy variance ratio
        # g ≈ 1 + 2 * τ_int
        block_sizes = [1, 2, 5, 10, 20, 50, 100]
        var_ratios = []
        base_var = np.var(energies)
        if base_var > 0:
            for bs in block_sizes:
                n_blocks = N // bs
                if n_blocks < 10:
                    break
                block_means = energies[:n_blocks * bs].reshape(n_blocks, bs).mean(axis=1)
                var_ratios.append(np.var(block_means) * bs / base_var)
            g = max(var_ratios[-1] if var_ratios else 1.0, 1.0)
        else:
            g = 1.0
    else:
        g = 20.0  # conservative default stride

    g = min(g, max_stride)
    stride = max(1, int(np.round(g)))
    indices = np.arange(0, N, stride)
    return {
        "data": data[indices],
        "indices": indices,
        "g": g,
    }


# ---------------------------------------------------------------------------
# MBAR wrapper
# ---------------------------------------------------------------------------

def run_mbar(
    u_kn: np.ndarray,
    N_k: np.ndarray,
    solver: str = "default",
    **kwargs,
) -> Dict:
    """
    Run the MBAR estimator on a multi-state dataset.

    Requires pymbar to be installed (pip install pymbar).

    Uses pymbar's native W_nk weight matrix (log-space stable) instead of
    manual weight computation, ensuring numerical consistency with the
    reference implementation.

    Parameters
    ----------
    u_kn : (K, N_total) reduced potential matrix.
        u_kn[k, n] = β_k * U(x_n) where x_n is sample n from ANY state.
        N_total = sum(N_k).
    N_k  : (K,) number of samples from each thermodynamic state k.
    solver : MBAR solver protocol: "default", "robust", or "jax".
    **kwargs : Additional arguments passed directly to `pymbar.MBAR`.

    Returns
    -------
    dict with:
        "f_k"     : (K,) reduced free energies per state.
        "weights" : (K, N_total) MBAR weights w_{k,n}.
        "mbar"    : the raw pymbar.MBAR object.
        "N_eff"   : (K,) effective sample sizes per state.
    """
    try:
        from pymbar import MBAR
    except ImportError:
        raise ImportError(
            "pymbar is required for MBAR analysis.  Install via: pip install pymbar"
        )

    solver_kw = {}
    if solver != "default":
        solver_kw["solver_protocol"] = solver

    mbar = MBAR(u_kn, N_k, **solver_kw, **kwargs)
    f_k  = mbar.f_k

    # Use pymbar's native W_nk (N, K) weight matrix (log-space stable)
    # W_nk[n, k] is the weight of sample n for state k
    W_nk = mbar.W_nk  # (N, K)  -- uses mbar.Log_W_nk internally
    weights = W_nk.T  # (K, N)  -- transpose to match our convention

    # Effective sample sizes (Kish formula)
    N_eff = mbar.compute_effective_sample_number()

    return {
        "f_k": f_k,
        "weights": weights,
        "mbar": mbar,
        "N_eff": N_eff,
    }


# ---------------------------------------------------------------------------
# 2D probability surface
# ---------------------------------------------------------------------------

def build_probability_surface(
    collective_vars: np.ndarray,
    mbar_weights: np.ndarray,
    state_index: int = 0,
    n_bins: int = 50,
    extent: Optional[Tuple] = None,
    kde: bool = False,
    kde_bandwidth: Optional[float] = None,
) -> Dict:
    """
    Build a 2D probability surface from MBAR weights.

    Parameters
    ----------
    collective_vars : (N_total, 2) 2D projection coordinates for each sample.
    mbar_weights    : (K, N_total) MBAR weight matrix from run_mbar().
    state_index     : which thermodynamic state k to evaluate (usually the
                      temperature of interest, e.g. 300 K replica index).
    n_bins          : number of bins along each axis.
    extent          : (x_min, x_max, y_min, y_max) manual bin limits.
                     If None, auto-detected from data.
    kde             : if True, use kernel density estimation instead of binning.
    kde_bandwidth   : KDE bandwidth (None = Scott's rule).

    Returns
    -------
    dict with:
        "probability"  : (n_bins, n_bins) normalised probability surface p(s).
        "free_energy"  : (n_bins, n_bins) free-energy surface F(s) = -ln(p).
        "bin_centers_x": (n_bins,) bin centres along x-axis.
        "bin_centers_y": (n_bins,) bin centres along y-axis.
        "extent"       : (x_min, x_max, y_min, y_max).
    """
    cv  = np.asarray(collective_vars, dtype=float)
    w_k = np.asarray(mbar_weights[state_index], dtype=float)   # (N,)
    w_k = w_k / w_k.sum()   # normalise

    if extent is None:
        x_min, x_max = cv[:, 0].min(), cv[:, 0].max()
        y_min, y_max = cv[:, 1].min(), cv[:, 1].max()
    else:
        x_min, x_max, y_min, y_max = extent

    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    if kde:
        from scipy.stats import gaussian_kde
        bw = kde_bandwidth or "scott"
        kde_obj = gaussian_kde(cv.T, bw_method=bw, weights=w_k)
        XX, YY = np.meshgrid(x_centers, y_centers, indexing='ij')
        prob = kde_obj(np.vstack([XX.ravel(), YY.ravel()])).reshape(n_bins, n_bins)
    else:
        prob, _, _ = np.histogram2d(
            cv[:, 0], cv[:, 1],
            bins=[x_edges, y_edges],
            weights=w_k,
        )

    prob = prob / prob.sum()

    # Free-energy surface F = -kB T ln p  (in units of kBT)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(prob > 0, np.log(prob), -np.inf)
        free_energy = -log_p
        free_energy -= free_energy[np.isfinite(free_energy)].min()

    return {
        "probability":   prob,
        "free_energy":   free_energy,
        "bin_centers_x": x_centers,
        "bin_centers_y": y_centers,
        "extent":        (x_min, x_max, y_min, y_max),
    }


# ---------------------------------------------------------------------------
# Build reduced potential matrix from Replica Exchange data
# ---------------------------------------------------------------------------

def build_u_kn(
    energies: np.ndarray,
    temperatures: np.ndarray,
    sample_assignments: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the reduced potential matrix u_kn for Parallel Tempering simulations.

    u_kn[k, n] = β_k * E(x_n)
               = E(x_n) / (kB * T_k)

    Biasing potentials V_k(x) are assumed to be zero for NVT simulations.

    Parameters
    ----------
    energies           : (N_total,) potential energies for every sample,
                         concatenated across all replicas in temperature order.
    temperatures       : (K,) temperatures for each replica.
    sample_assignments : (N_total,) integer array mapping each sample to its
                         originating replica index.  If None, assumes equal
                         numbers of samples from each replica.

    Returns
    -------
    (u_kn, N_k):
        u_kn : (K, N_total) reduced potential matrix.
        N_k  : (K,) number of samples per state.
    """
    energies     = np.asarray(energies, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    K  = len(temperatures)
    N  = len(energies)
    beta_k = 1.0 / (_kB_CONST * temperatures)

    u_kn = beta_k[:, np.newaxis] * energies[np.newaxis, :]   # (K, N)

    if sample_assignments is not None:
        N_k = np.bincount(sample_assignments, minlength=K).astype(int)
    else:
        n_each = N // K
        N_k = np.array([n_each] * K)
        N_k[-1] += N - n_each * K

    return u_kn, N_k


# ---------------------------------------------------------------------------
# Convenience: full MBAR workflow from multi-state data
# ---------------------------------------------------------------------------

def mbar_free_energy_surface(
    energies: np.ndarray,
    temperatures: np.ndarray,
    collective_vars: np.ndarray,
    target_temperature: Optional[float] = None,
    sample_assignments: Optional[np.ndarray] = None,
    n_bins: int = 50,
    extent: Optional[Tuple] = None,
    kde: bool = True,
    decorrelate: bool = False,
    mbar_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Dict:
    """
    End-to-end MBAR probability/free-energy surface workflow.

    Parameters
    ----------
    energies           : (N_total,) potential energies.
    temperatures       : (K,) temperature ladder.
    collective_vars    : (N_total, 2) low-dim projection for all samples.
    target_temperature : temperature at which to evaluate the surface
                         (default: first temperature in ladder).
    sample_assignments : (N_total,) replica indices. None = equal distribution.
    n_bins             : grid resolution.
    extent             : manual bin limits (x_min, x_max, y_min, y_max).
    kde                : use kernel density estimation (recommended).
    decorrelate        : if True, subsample each replica's timeseries
                         using pymbar.timeseries.statistical_inefficiency
                         before MBAR analysis.

    mbar_kwargs        : dictionary of arguments to pass directly to `run_mbar` / `pymbar.MBAR`.
    **kwargs           : additional arguments passed to `build_probability_surface`.

    Returns
    -------
    dict with "probability", "free_energy", "bin_centers_x", "bin_centers_y",
    "extent", "mbar_weights", "f_k", "N_eff".
    """
    energies     = np.asarray(energies, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    K = len(temperatures)
    N = len(energies)

    # --- Optional decorrelation ---
    if decorrelate:
        if sample_assignments is None:
            n_each = N // K
            sample_assignments = np.repeat(np.arange(K), n_each)
            if len(sample_assignments) < N:
                sample_assignments = np.concatenate([
                    sample_assignments,
                    np.full(N - len(sample_assignments), K - 1)
                ])

        keep_indices = []
        for k in range(K):
            mask = sample_assignments == k
            idx_k = np.where(mask)[0]
            dec = decorrelate_timeseries(
                collective_vars[idx_k],
                energies=energies[idx_k],
            )
            keep_indices.append(idx_k[dec["indices"]])

        keep = np.sort(np.concatenate(keep_indices))
        energies = energies[keep]
        collective_vars = collective_vars[keep]
        sample_assignments = sample_assignments[keep]

    if mbar_kwargs is None:
        mbar_kwargs = {}
        
    u_kn, N_k = build_u_kn(energies, temperatures, sample_assignments)
    mbar_out  = run_mbar(u_kn, N_k, **mbar_kwargs)

    if target_temperature is None:
        state_idx = 0
    else:
        state_idx = int(np.argmin(np.abs(temperatures - target_temperature)))

    surf = build_probability_surface(
        collective_vars, mbar_out["weights"],
        state_index=state_idx,
        n_bins=n_bins,
        extent=extent,
        kde=kde,
        **kwargs,
    )
    surf["mbar_weights"] = mbar_out["weights"]
    surf["f_k"]          = mbar_out["f_k"]
    surf["N_eff"]        = mbar_out["N_eff"]
    surf["state_index"]  = state_idx
    surf["temperature"]  = float(temperatures[state_idx])
    return surf
