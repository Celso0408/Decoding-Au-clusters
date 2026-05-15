import numpy as np
import pytest
from dimredpy.mbar.mbar import (
    build_u_kn, 
    run_mbar,
    build_probability_surface,
    mbar_free_energy_surface,
    decorrelate_timeseries
)

def test_build_u_kn():
    energies = np.random.rand(100)
    temps = np.array([300, 310, 320])
    u_kn, n_k = build_u_kn(energies, temps)
    assert u_kn.shape == (3, 100)
    assert n_k.tolist() == [33, 33, 34]
    
    # Test specific assignments
    assignments = np.random.randint(0, 3, 100)
    u_kn, n_k = build_u_kn(energies, temps, sample_assignments=assignments)
    assert u_kn.shape == (3, 100)
    assert np.sum(n_k) == 100

def test_run_mbar():
    try:
        import pymbar
    except ImportError:
        pytest.skip("pymbar not installed")
        
    u_kn = np.random.rand(2, 50)
    n_k = np.array([25, 25])
    
    # Test default
    res = run_mbar(u_kn, n_k)
    assert "f_k" in res
    assert res["weights"].shape == (2, 50)
    
    # Test solver switch and kwargs
    res2 = run_mbar(u_kn, n_k, solver="robust", maximum_iterations=100)
    assert "f_k" in res2

def test_build_probability_surface():
    cvs = np.random.rand(50, 2)
    weights = np.random.rand(2, 50)
    
    # Basic
    res = build_probability_surface(cvs, weights, state_index=0, n_bins=10, kde=False)
    assert res["probability"].shape == (10, 10)
    
    # KDE
    res_kde = build_probability_surface(cvs, weights, state_index=0, n_bins=10, kde=True, kde_bandwidth=0.1)
    assert res_kde["probability"].shape == (10, 10)
    
    # Extent
    res_ext = build_probability_surface(cvs, weights, state_index=0, n_bins=10, kde=False, extent=(0, 1, 0, 1))
    assert res_ext["probability"].shape == (10, 10)

def test_mbar_free_energy_surface():
    try:
        import pymbar
    except ImportError:
        pytest.skip("pymbar not installed")
        
    energies = np.random.rand(60)
    temps = np.array([300, 400])
    cvs = np.random.rand(60, 2)
    
    # Target temp and kwargs pass-through
    res = mbar_free_energy_surface(
        energies, temps, cvs, 
        target_temperature=350.0, 
        n_bins=5, 
        kde=False,
        mbar_kwargs={"solver_protocol": "robust"}
    )
    assert res["probability"].shape == (5, 5)

def test_decorrelate_timeseries():
    try:
        import pymbar
    except ImportError:
        pytest.skip("pymbar not installed")
        
    cvs = np.random.rand(100, 2)
    energies = np.random.rand(100)
    
    # Both CVs and energies
    res = decorrelate_timeseries(cvs, energies)
    assert "g" in res
    assert "indices" in res
    
    # Only CVs
    res2 = decorrelate_timeseries(cvs)
    assert "g" in res2

def test_mbar_decorrelate_integration():
    """Test full mbar pipeline with decorrelation flag."""
    try:
        import pymbar
    except ImportError:
        pytest.skip("pymbar not installed")
        
    energies = np.random.rand(100)
    temps = np.array([300, 400])
    cvs = np.random.rand(100, 2)
    
    res = mbar_free_energy_surface(
        energies, temps, cvs,
        decorrelate=True,
        n_bins=5,
        kde=False
    )
    assert res["probability"].shape == (5, 5)
