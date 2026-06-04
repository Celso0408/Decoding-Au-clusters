import numpy as np
import pytest
from dimredpy.sketchmap import select_landmarks
from dimredpy.sketchmap.landmark import _voronoi_weights
from dimredpy.shared.metrics import EuclideanMetric, PBCMetric

def test_select_landmarks_minmax():
    """Test minmax (FPS) landmark selection."""
    data = np.random.rand(100, 5)
    res = select_landmarks(data, n_landmarks=10, mode="minmax")
    assert "landmarks" in res
    assert "indices" in res
    assert "weights" in res
    assert res["landmarks"].shape == (10, 5)

def test_select_landmarks_minmax_similarity():
    """Test minmax with precomputed similarity matrix."""
    data = np.random.rand(100, 5)
    metric = EuclideanMetric()
    similarity = metric.pairwise(data) # fixed argument count
    
    res = select_landmarks(data, n_landmarks=10, mode="minmax", similarity=similarity)
    assert res["landmarks"].shape == (10, 5)

def test_select_landmarks_stride():
    """Test stride mode."""
    data = np.random.rand(100, 5)
    res = select_landmarks(data, n_landmarks=10, mode="stride")
    assert res["landmarks"].shape == (10, 5)
    # Stride of 100/10 = 10 -> indices should be 0, 10, 20...
    np.testing.assert_array_equal(res["indices"], np.arange(0, 100, 10))

def test_select_landmarks_random():
    """Test random modes with and without uniqueness."""
    data = np.random.rand(100, 5)
    
    res_unique = select_landmarks(data, n_landmarks=10, mode="random", unique=True, seed=42)
    assert len(np.unique(res_unique["indices"])) == 10
    
    res_non_unique = select_landmarks(data, n_landmarks=10, mode="random", unique=False, seed=42)
    assert len(res_non_unique["indices"]) == 10

def test_select_landmarks_resample_and_staged():
    """Test resample and staged advanced algorithms."""
    data = np.random.rand(100, 5)
    weights = np.ones(100)
    weights[0] = 100.0 # Heavy weight
    
    res_resample = select_landmarks(data, n_landmarks=10, mode="resample", input_weights=weights)
    assert res_resample["landmarks"].shape == (10, 5)
    
    res_staged = select_landmarks(data, n_landmarks=10, mode="staged", input_weights=weights)
    assert res_staged["landmarks"].shape == (10, 5)

def test_voronoi_weights_edge_cases():
    """Test the voronoi weight calculator explicitly."""
    data = np.array([[0,0], [1,0], [0,1], [10,10]])
    landmarks = np.array([[0.1, 0.1], [9.9, 9.9]])
    
    metric = EuclideanMetric()
    w = _voronoi_weights(data, landmarks, metric)
    
    assert len(w) == 2
    assert np.isclose(w[0], 0.75) # 3/4
    assert np.isclose(w[1], 0.25) # 1/4

def test_select_landmarks_metrics():
    """Test other distance metrics for landmark selection."""
    data = np.random.rand(100, 5)
    res = select_landmarks(data, n_landmarks=10, mode="minmax", metric=PBCMetric([10]))
    assert res["landmarks"].shape == (10, 5)

def test_select_landmarks_edge_cases():
    """Test boundary constraints and invalid inputs."""
    data = np.random.rand(10, 5)
    
    with pytest.raises(ValueError):
        select_landmarks(data, n_landmarks=20, mode="minmax")
    
    with pytest.raises(ValueError):
        select_landmarks(data, n_landmarks=5, mode="invalid_mode")

def test_select_landmarks_seed():
    """Test reproducibility using seed."""
    data = np.random.rand(50, 5)
    res1 = select_landmarks(data, n_landmarks=10, mode="minmax", seed=42)
    res2 = select_landmarks(data, n_landmarks=10, mode="minmax", seed=42)
    np.testing.assert_array_equal(res1["indices"], res2["indices"])
