import numpy as np
import pytest
from dimredpy.sketchmap import select_landmarks

def test_select_landmarks_minmax():
    """Test minmax (Sketch-map original) landmark selection."""
    data = np.random.rand(100, 5)
    
    res = select_landmarks(data, n_landmarks=10, mode="minmax")
    
    assert "landmarks" in res
    assert "indices" in res
    assert "weights" in res
    assert res["landmarks"].shape == (10, 5)
    assert len(res["indices"]) == 10
    assert len(res["weights"]) == 10

def test_select_landmarks_metrics():
    """Test other distance metrics for landmark selection."""
    data = np.random.rand(100, 5)
    from dimredpy.shared.metrics import PBCMetric
    
    res = select_landmarks(data, n_landmarks=10, mode="minmax", metric=PBCMetric([10]))
    assert res["landmarks"].shape == (10, 5)

def test_select_landmarks_edge_cases():
    """Test boundary constraints and invalid inputs."""
    data = np.random.rand(10, 5)
    
    # Requesting more landmarks than data points should raise ValueError
    with pytest.raises(ValueError):
        select_landmarks(data, n_landmarks=20, mode="minmax")
    
    # Invalid mode should raise ValueError
    with pytest.raises(ValueError):
        select_landmarks(data, n_landmarks=5, mode="invalid_mode")

def test_select_landmarks_seed():
    """Test reproducibility using seed."""
    data = np.random.rand(50, 5)
    
    res1 = select_landmarks(data, n_landmarks=10, mode="minmax", seed=42)
    res2 = select_landmarks(data, n_landmarks=10, mode="minmax", seed=42)
    
    np.testing.assert_array_equal(res1["indices"], res2["indices"])
