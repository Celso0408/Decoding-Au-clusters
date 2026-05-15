import numpy as np
import pytest
from dimredpy.shared import (
    coordination_histogram, effective_coordination_number,
    radius_of_gyration, hausdorff_chirality_measure,
    radial_distribution_function, projection_center,
    compute_trajectory_descriptors
)

def test_coordination_histogram():
    pos = np.array([[0,0,0], [1,0,0], [2,0,0]], dtype=float)
    # Generic cutoff
    ch = coordination_histogram(pos, cutoff=1.1, max_neighbors=2)
    assert np.allclose(ch, [0, 2/3, 1/3])
    
    # Very small cutoff (everyone has 0 neighbors)
    ch_zero = coordination_histogram(pos, cutoff=0.1, max_neighbors=2)
    assert np.allclose(ch_zero, [1.0, 0, 0])

def test_effective_coordination_number():
    pos = np.array([[0,0,0], [1,0,0], [2,0,0]], dtype=float)
    
    # Generic
    ecn = effective_coordination_number(pos, cutoff=1.5)
    assert ecn > 0

def test_radius_of_gyration():
    pos = np.array([[-1,0,0], [1,0,0]], dtype=float)
    
    # No masses (COM is 0,0,0)
    rg = radius_of_gyration(pos)
    assert np.isclose(rg, 1.0) # sqrt((1^2 + (-1)^2)/2) = 1.0

def test_hausdorff_chirality_measure():
    # Symmetric object (e.g. square) should have near 0 HCM
    pos_sym = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], dtype=float)
    hcm_sym = hausdorff_chirality_measure(pos_sym)
    assert np.isclose(hcm_sym, 0.0, atol=1e-5)
    
    # Asymmetric random object should have HCM > 0
    pos_asym = np.random.rand(5, 3)
    hcm_asym = hausdorff_chirality_measure(pos_asym)
    assert hcm_asym >= 0.0

def test_radial_distribution_function():
    traj = np.array([
        [[0,0,0], [1,0,0], [2,0,0]],
        [[0,0,0], [1.1,0,0], [2.1,0,0]]
    ])
    
    r, g_r = radial_distribution_function(traj, n_bins=10, r_max=3.0)
    assert len(r) == 10
    assert len(g_r) == 10
    assert not np.any(np.isnan(g_r))

def test_projection_center():
    values = np.linspace(-10, 5, 100)
    weights = np.exp(-(values + 3.0)**2 / 1.0)
    
    pc = projection_center(values, weights)
    assert np.allclose(pc, -3.0, atol=0.1)

def test_compute_trajectory_descriptors_exhaustive():
    traj = np.random.rand(5, 10, 3) # 5 frames, 10 atoms
    
    res = compute_trajectory_descriptors(
        traj, 
        cutoff=0.5, 
        max_neighbors=5
    )
    
    assert res["neighbor_histograms"].shape == (5, 6) # max_neighbors+1
    assert res["ecn"].shape == (5,)
    assert res["rg"].shape == (5,)
    assert res["hcm"].shape == (5,)
