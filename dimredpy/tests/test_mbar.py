import numpy as np
import pytest

from dimredpy.mbar import (
    run_mbar,
    mbar_free_energy_differences,
    mbar_compute_expectations,
    build_u_kn,
    decorrelate_timeseries,
    build_probability_surface,
    mbar_free_energy_surface
)

def test_build_u_kn_with_assignments():
    """Test build_u_kn with explicit sample assignments."""
    energies = np.array([10.0, 12.0, 15.0, 9.0, 11.0])
    temperatures = np.array([300.0, 310.0])
    # 3 samples from state 0, 2 samples from state 1
    assignments = np.array([0, 0, 0, 1, 1])
    
    u_kn, N_k = build_u_kn(energies, temperatures, sample_assignments=assignments)
    assert u_kn.shape == (2, 5)
    assert np.array_equal(N_k, [3, 2])

def test_build_u_kn_without_assignments():
    """Test build_u_kn without explicit sample assignments (equal split)."""
    energies = np.array([10.0, 12.0, 15.0, 9.0])
    temperatures = np.array([300.0, 310.0])
    
    u_kn, N_k = build_u_kn(energies, temperatures)
    assert u_kn.shape == (2, 4)
    assert np.array_equal(N_k, [2, 2])

def test_decorrelate_timeseries_stride():
    """Test time series decorrelation using fallback stride method."""
    np.random.seed(42)
    data = np.random.randn(100, 2)
    energies = np.random.randn(100)
    
    res = decorrelate_timeseries(data, energies, method="stride", max_stride=10)
    assert "data" in res
    assert "indices" in res
    assert "g" in res
    assert len(res["data"]) <= 100

def test_decorrelate_timeseries_statistical_inefficiency():
    """Test time series decorrelation using pymbar's statistical_inefficiency."""
    try:
        import pymbar
    except ImportError:
        pytest.skip("pymbar not installed")
        
    np.random.seed(42)
    data = np.random.randn(100, 2)
    energies = np.random.randn(100)
    
    res = decorrelate_timeseries(data, energies, method="statistical_inefficiency")
    assert "data" in res
    assert "indices" in res
    assert "g" in res

def test_mbar_basic():
    """Test basic MBAR execution with toy data."""
    try:
        import pymbar
    except ImportError:
        pytest.skip("pymbar not installed")
        
    K = 3
    N_each = 50
    N_total = K * N_each
    
    np.random.seed(42)
    energies = np.random.randn(N_total)
    temperatures = np.array([300.0, 310.0, 320.0])
    
    u_kn, N_k = build_u_kn(energies, temperatures)
    res = run_mbar(u_kn, N_k)
    
    assert "f_k" in res
    assert "weights" in res
    assert "mbar" in res
    assert res["f_k"].shape == (K,)
    assert res["weights"].shape == (K, N_total)

def test_mbar_differences_comprehensive():
    """Test free energy differences with and without uncertainty."""
    try:
        import pymbar
    except ImportError:
        pytest.skip("pymbar not installed")
        
    K = 2
    N_each = 50
    N_total = K * N_each
    np.random.seed(42)
    energies = np.random.randn(N_total)
    temperatures = np.array([300.0, 310.0])
    
    u_kn, N_k = build_u_kn(energies, temperatures)
    res = run_mbar(u_kn, N_k)
    
    # With uncertainty
    diffs = mbar_free_energy_differences(res, compute_uncertainty=True)
    assert "Delta_f" in diffs or "Deltaf_ij" in diffs
    assert "dDelta_f" in diffs or "dDeltaf_ij" in diffs
    
    # Without uncertainty
    diffs_no_unc = mbar_free_energy_differences(res, compute_uncertainty=False)
    assert "dDelta_f" not in diffs_no_unc and "dDeltaf_ij" not in diffs_no_unc

def test_mbar_expectations_comprehensive():
    """Test observable expectations with state dependence and uncertainty options."""
    try:
        import pymbar
    except ImportError:
        pytest.skip("pymbar not installed")
        
    K = 2
    N_each = 50
    N_total = K * N_each
    np.random.seed(42)
    energies = np.random.randn(N_total)
    temperatures = np.array([300.0, 310.0])
    
    u_kn, N_k = build_u_kn(energies, temperatures)
    res = run_mbar(u_kn, N_k)
    
    # State independent, with uncertainty
    A_kn = energies
    exp = mbar_compute_expectations(res, A_kn, state_dependent=False, compute_uncertainty=True)
    assert "mu" in exp
    assert "sigma" in exp
    
    # State dependent, without uncertainty
    A_kn_state = np.vstack([energies, energies * 1.1]) # shape (2, N_total)
    exp_state = mbar_compute_expectations(res, A_kn_state, state_dependent=True, compute_uncertainty=False)
    assert "mu" in exp_state
    assert "sigma" not in exp_state

def test_build_probability_surface():
    """Test probability surface gridding with and without KDE."""
    cv = np.random.rand(100, 2)
    weights = np.ones((1, 100)) / 100.0
    
    # Without KDE
    res_hist = build_probability_surface(cv, weights, state_index=0, n_bins=10, kde=False)
    assert "probability" in res_hist
    assert "free_energy" in res_hist
    assert res_hist["probability"].shape == (10, 10)
    
    # With KDE
    res_kde = build_probability_surface(cv, weights, state_index=0, n_bins=10, kde=True)
    assert "probability" in res_kde
    assert "free_energy" in res_kde
    assert res_kde["probability"].shape == (10, 10)

def test_mbar_free_energy_surface_full():
    """Test the full end-to-end MBAR free energy surface workflow with decorrelation."""
    try:
        import pymbar
    except ImportError:
        pytest.skip("pymbar not installed")
        
    K = 2
    N_each = 50
    N_total = K * N_each
    
    np.random.seed(42)
    energies = np.random.randn(N_total)
    temperatures = np.array([300.0, 310.0])
    cv = np.random.rand(N_total, 2)
    
    res = mbar_free_energy_surface(
        energies=energies,
        temperatures=temperatures,
        collective_vars=cv,
        target_temperature=310.0, # Target the second temperature
        n_bins=10,
        kde=False,
        decorrelate=True
    )
    
    assert "probability" in res
    assert "free_energy" in res
    assert "f_k" in res
    assert "mbar_weights" in res
    assert res["probability"].shape == (10, 10)
    assert res["temperature"] == 310.0

def test_mbar_analysis_exceptions():
    """Test that downstream analysis functions correctly raise errors for invalid input."""
    invalid_dict = {"f_k": np.array([1, 2])} # Missing "mbar" key
    
    with pytest.raises(ValueError, match="must contain the fitted 'mbar' object"):
        mbar_free_energy_differences(invalid_dict)
        
    with pytest.raises(ValueError, match="must contain the fitted 'mbar' object"):
        mbar_compute_expectations(invalid_dict, np.array([1, 2, 3]))
