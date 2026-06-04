import numpy as np
import pytest
from dimredpy.sketchmap import sketch_map, classical_mds
from dimredpy.shared.metrics import PBCMetric

def test_classical_mds_basic():
    """Test classical MDS basic execution."""
    data = np.random.rand(20, 5)
    emb = classical_mds(data, n_components=2)
    assert emb["embedding"].shape == (20, 2)
    assert not np.any(np.isnan(emb["embedding"]))
    assert "error" in emb # instead of stress

def test_classical_mds_metrics():
    """Test classical MDS with alternative metrics."""
    data = np.random.rand(20, 5)
    emb_sq = classical_mds(data, n_components=2, metric=PBCMetric([10]))
    assert emb_sq["embedding"].shape == (20, 2)

def test_classical_mds_invalid_dims():
    """Test classical MDS dimensions validation."""
    data = np.random.rand(5, 5) # Only 5 samples
    with pytest.raises(ValueError, match="Requested eigenvalue indices"):
        classical_mds(data, n_components=10)

def test_sketch_map_basic():
    """Test basic Sketch-map reduction."""
    data = np.random.rand(30, 5)
    res = sketch_map(data, n_components=2, preopt_steps=5, global_steps=5)
    assert "embedding" in res
    assert res["embedding"].shape == (30, 2)
    assert "stress" in res

def test_sketch_map_weights():
    """Test Sketch-map with explicit sample weights."""
    data = np.random.rand(30, 5)
    weights = np.random.rand(30)
    res = sketch_map(data, weights=weights, preopt_steps=5, global_steps=5)
    assert res["embedding"].shape == (30, 2)

def test_sketch_map_functions():
    """Test custom high/low dim sigmoids."""
    data = np.random.rand(30, 5)
    fun_hd = (8.0, 10.0, 10.0)
    fun_ld = (8.0, 2.0, 10.0)
    
    res = sketch_map(data, fun_hd=fun_hd, fun_ld=fun_ld, preopt_steps=5, global_steps=5)
    assert res["embedding"].shape == (30, 2)

def test_sketch_map_imix_pointwise_global():
    """Test optimization with intermediate mix fractions."""
    data = np.random.rand(30, 5)
    res = sketch_map(data, imix=0.5, grid=(1.0, 21, 201), preopt_steps=5, global_steps=5)
    assert res["embedding"].shape == (30, 2)

def test_sketch_map_different_metrics():
    """Test metric injection in Sketch-map."""
    data = np.random.rand(30, 5)
    res = sketch_map(data, metric=PBCMetric([10]), preopt_steps=5, global_steps=5)
    assert res["embedding"].shape == (30, 2)
