import numpy as np
import pytest
from dimredpy.shared import distance_histogram, preservation_score
from dimredpy.shared import PBCMetric

def test_distance_histogram_1d():
    """Test 1D distance histogram computation."""
    X = np.array([[0,0], [1,0], [0,1]])
    res = distance_histogram(X, n_bins=5, max_d=2.0)
    assert "histogram_1d" in res
    assert res["histogram_1d"].shape == (5,)
    assert "hd_distances" in res
    assert len(res["hd_distances"]) == 3

def test_distance_histogram_2d():
    """Test 2D distance histogram computation with HD and LD."""
    X = np.array([[0,0], [1,0], [0,1]])
    Y = np.array([[0,0], [0.1,0], [0,0.1]])
    
    res = distance_histogram(X, Y, n_bins=5)
    assert "histogram_2d" in res
    assert res["histogram_2d"].shape == (5, 5)

def test_distance_histogram_weighted():
    """Test histogram with weights."""
    X = np.array([[0,0], [1,0], [0,1]])
    Y = np.array([[0,0], [0.1,0], [0,0.1]])
    w = np.array([1.0, 2.0, 3.0])
    
    res = distance_histogram(X, Y, weights=w, n_bins=5)
    assert "histogram_2d" in res
    # Pair weights: (1,2)->2, (1,3)->3, (2,3)->6. Total sum = 11
    assert np.isclose(res["histogram_2d"].sum(), 11.0)


def test_preservation_score_basic():
    """Test basic preservation score."""
    X = np.array([[0,0], [1,0], [0,1]])
    Y = np.array([[0,0], [0.1,0], [0,0.1]])
    
    score = preservation_score(X, Y, fun_hd=None, fun_ld=None)
    assert score > 0

def test_preservation_score_functions():
    """Test preservation score with transfer functions."""
    X = np.random.rand(10, 3)
    Y = np.random.rand(10, 2)
    
    score_tf = preservation_score(X, Y, fun_hd=(2.0, 1.0, 1.0), fun_ld=(2.0, 1.0, 1.0))
    assert score_tf > 0

def test_preservation_score_custom_metric():
    """Test custom metric object."""
    X = np.array([[0,0], [0.8,0]])
    Y = np.array([[0,0], [0.8,0]])
    m = PBCMetric(period=1.0) # distance in X becomes 0.2
    
    score = preservation_score(X, Y, metric=m)
    assert np.allclose(score, (0.2 - 0.8)**2)

def test_diagnostics_shape_errors():
    """Test that mismatched shapes raise errors."""
    X = np.array([[0,0], [1,0]])
    Y = np.array([[0,0], [1,0], [0,1]]) # Different length
    
    with pytest.raises(ValueError):
        distance_histogram(X, Y)
        
    with pytest.raises(ValueError):
        preservation_score(X, Y)
