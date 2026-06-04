"""
Downstream analysis functions for MBAR.

Extracts free energy differences and observable expectations directly from a fitted
MBAR object (provided via the `mbar_dict` returned by `run_mbar`).
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any

def mbar_free_energy_differences(
    mbar_dict: Dict[str, Any],
    compute_uncertainty: bool = True,
    uncertainty_method: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute dimensionless free energy differences between all states.

    Parameters
    ----------
    mbar_dict : dict
        The result dictionary returned by `run_mbar()`. Must contain the "mbar" key.
    compute_uncertainty : bool
        If True, calculates the uncertainties (standard errors) of the free energy differences.
    uncertainty_method : str, optional
        Method for uncertainty estimation passed to pymbar (e.g., "svd").

    Returns
    -------
    dict with keys:
        - "Deltaf_ij" : (K, K) array of dimensionless free energy differences f_j - f_i.
        - "dDeltaf_ij": (K, K) array of standard errors in the estimates (if requested).
        - "Theta"     : (K, K) covariance matrix (if computed).
    """
    if "mbar" not in mbar_dict:
        raise ValueError("The provided dictionary must contain the fitted 'mbar' object.")

    mbar_obj = mbar_dict["mbar"]
    
    kwargs = {"compute_uncertainty": compute_uncertainty}
    if uncertainty_method is not None:
        kwargs["uncertainty_method"] = uncertainty_method
        
    results = mbar_obj.compute_free_energy_differences(**kwargs)
    
    # Depending on pymbar version, results is either a dict (pymbar 4) or a tuple (pymbar 3)
    if isinstance(results, dict):
        return results
    elif isinstance(results, tuple):
        # Fallback for pymbar <= 3
        out = {"Delta_f": results[0]}
        if compute_uncertainty and len(results) > 1:
            out["dDelta_f"] = results[1]
        if len(results) > 2:
            out["Theta"] = results[2]
        return out
    else:
        raise TypeError(f"Unexpected return type from compute_free_energy_differences: {type(results)}")


def mbar_compute_expectations(
    mbar_dict: Dict[str, Any],
    A_kn: np.ndarray,
    state_dependent: bool = False,
    compute_uncertainty: bool = True,
    uncertainty_method: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute the expectations of an observable for all thermodynamic states.

    Parameters
    ----------
    mbar_dict : dict
        The result dictionary returned by `run_mbar()`. Must contain the "mbar" key.
    A_kn : (N_total,) or (K, N_total) array
        The observable evaluated for each sample. If state_dependent is False, 
        A_kn is shape (N_total,). If True, A_kn is shape (K, N_total) where the 
        observable depends on the thermodynamic state.
    state_dependent : bool
        Whether the observable A_kn is state-dependent (shape K, N_total).
    compute_uncertainty : bool
        If True, calculates the uncertainties (standard errors) of the expectations.
    uncertainty_method : str, optional
        Method for uncertainty estimation passed to pymbar.

    Returns
    -------
    dict with keys:
        - "mu"    : (K,) array of expected values of the observable in each state.
        - "sigma" : (K,) array of standard errors of the expectations (if requested).
        - "Theta" : (K, K) covariance matrix of the log weights (if computed).
    """
    if "mbar" not in mbar_dict:
        raise ValueError("The provided dictionary must contain the fitted 'mbar' object.")

    mbar_obj = mbar_dict["mbar"]
    
    kwargs = {
        "state_dependent": state_dependent,
        "compute_uncertainty": compute_uncertainty
    }
    if uncertainty_method is not None:
        kwargs["uncertainty_method"] = uncertainty_method
        
    results = mbar_obj.compute_expectations(A_kn, **kwargs)
    
    if isinstance(results, dict):
        return results
    elif isinstance(results, tuple):
        # Fallback for pymbar <= 3
        out = {"mu": results[0]}
        if compute_uncertainty and len(results) > 1:
            out["sigma"] = results[1]
        if len(results) > 2:
            out["Theta"] = results[2]
        return out
    else:
        raise TypeError(f"Unexpected return type from compute_expectations: {type(results)}")
