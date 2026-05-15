import numpy as np
import pytest
from dimredpy.sketchmap import sketch_map, classical_mds

def test_classical_mds_basic():
    data = np.random.rand(20, 5)
    # Default metric euclidean
    emb = classical_mds(data, n_components=2)
    assert emb["embedding"].shape == (20, 2)
    assert not np.any(np.isnan(emb["embedding"]))

def test_classical_mds_metrics():
    from dimredpy.shared.metrics import PBCMetric
    data = np.random.rand(20, 5)
    # Other metrics
    emb_sq = classical_mds(data, n_components=2, metric=PBCMetric([10]))
    assert emb_sq["embedding"].shape == (20, 2)

def test_sketch_map_basic():
    data = np.random.rand(30, 5)
    res = sketch_map(data)
    assert "embedding" in res
    assert res["embedding"].shape == (30, 2)
    assert res["stress"] >= 0

def test_sketch_map_weights():
    data = np.random.rand(30, 5)
    weights = np.random.rand(30)
    res = sketch_map(data, weights=weights)
    assert res["embedding"].shape == (30, 2)

def test_sketch_map_functions():
    data = np.random.rand(30, 5)
    fun_hd = (8.0, 10.0, 10.0)
    fun_ld = (8.0, 2.0, 10.0)
    
    res = sketch_map(data, fun_hd=fun_hd, fun_ld=fun_ld, preopt_steps=5)
    assert res["embedding"].shape == (30, 2)

def test_sketch_map_imix_pointwise_global():
    data = np.random.rand(30, 5)
    res = sketch_map(data, imix=0.5, grid=(1.0, 21, 201), preopt_steps=5)
    assert res["embedding"].shape == (30, 2)

def test_sketch_map_different_metrics():
    from dimredpy.shared.metrics import PBCMetric
    data = np.random.rand(30, 5)
    res = sketch_map(data, metric=PBCMetric([10]), preopt_steps=5)
    assert res["embedding"].shape == (30, 2)
