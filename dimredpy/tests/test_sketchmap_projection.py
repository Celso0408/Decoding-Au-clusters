import numpy as np
import pytest
from dimredpy.sketchmap import project_out_of_sample

def test_project_points_basic():
    """Test basic projection mapping using Euclidean distance."""
    data = np.random.rand(50, 5) # 50 frames, 5 CVs
    landmarks = np.random.rand(10, 5) # 10 landmarks, 5 CVs
    landmark_embeddings = np.random.rand(10, 2) # 10 landmarks mapped to 2D
    
    emb = project_out_of_sample(
        data, 
        landmarks, 
        landmark_embeddings
    )
    
    assert emb["embedding"].shape == (50, 2)
    assert not np.any(np.isnan(emb["embedding"]))

def test_project_points_metrics():
    """Test using non-Euclidean metrics like sqeuclidean."""
    from dimredpy.shared.metrics import EuclideanMetric
    data = np.random.rand(50, 5)
    landmarks = np.random.rand(10, 5)
    landmark_embeddings = np.random.rand(10, 2)
    
    emb = project_out_of_sample(
        data, 
        landmarks, 
        landmark_embeddings, 
        metric=EuclideanMetric()
    )
    
    assert emb["embedding"].shape == (50, 2)

def test_project_points_with_grid():
    """Test interpolation projection onto a precomputed bounding grid."""
    data = np.random.rand(50, 5)
    landmarks = np.random.rand(10, 5)
    landmark_embeddings = np.random.rand(10, 2)
    
    emb = project_out_of_sample(
        data,
        landmarks,
        landmark_embeddings,
        grid=(1.0, 21, 201)
    )
    
    assert emb["embedding"].shape == (50, 2)

def test_project_points_functions():
    """Test providing specific fun_hd and fun_ld for Sketch-map projection."""
    data = np.random.rand(50, 5)
    landmarks = np.random.rand(10, 5)
    landmark_embeddings = np.random.rand(10, 2)
    
    fun_hd = (8.0, 10.0, 10.0)
    fun_ld = (8.0, 2.0, 10.0)
    
    emb = project_out_of_sample(
        data,
        landmarks,
        landmark_embeddings,
        fun_hd=fun_hd,
        fun_ld=fun_ld
    )
    
    assert emb["embedding"].shape == (50, 2)

def test_projection_equivalence():
    """
    Verify that the new vectorized batched CPU projection is mathematically
    consistent with the original grid-based search logic.
    """
    from dimredpy.sketchmap.projection import _grid_project
    from dimredpy.shared.transfer import make_transfer
    
    # 1. Setup Synthetic Test Data
    np.random.seed(123)
    K = 15  # Landmarks
    M = 10  # Samples to project
    D = 13  # Dimensions
    
    landmarks_hd = np.random.rand(K, D)
    landmarks_ld = np.random.rand(K, 2)
    samples = np.random.rand(M, D)
    weights = np.random.rand(K) # Use non-uniform weights to be thorough
    
    fun_hd = (1.2, 10.5, 1.0)
    fun_ld = (1.2, 1.0, 1.0)
    # Use the same resolution for both to ensure bit-perfect match
    grid = (2.0, 21, 21) 
    
    # 2. Reference Implementation (Loop + Original Logic)
    tf_hd = make_transfer(fun_hd)
    tf_ld = make_transfer(fun_ld)
    
    expected_pos = []
    expected_err = []
    for m in range(M):
        hd_d = np.linalg.norm(samples[m] - landmarks_hd, axis=1)
        fhd = tf_hd.f(hd_d)
        # _grid_project is the original paper logic
        pos, err = _grid_project(
            hd_d, fhd, landmarks_ld, tf_ld, weights, imix=0.0,
            gwidth=grid[0], g1=grid[1], g2=grid[2], cg_steps=10, gt=0.0
        )
        expected_pos.append(pos)
        expected_err.append(err)
    
    expected_pos = np.array(expected_pos)
    expected_err = np.array(expected_err)
    
    # 3. New Batched Implementation
    res = project_out_of_sample(
        samples, landmarks_hd, landmarks_ld, 
        weights=weights, fun_hd=fun_hd, fun_ld=fun_ld, 
        grid=grid, cg_steps=10, use_gpu=False, verbose=False
    )
    
    batched_pos = res["embedding"]
    batched_err = res["error"]
    
    # 4. Assert Equivalence
    pos_diff = np.linalg.norm(expected_pos - batched_pos, axis=1)
    err_diff = np.abs(expected_err - batched_err)
    
    assert pos_diff.max() < 1e-10, f"Positions do not match! Diff: {pos_diff.max()}"
    assert err_diff.max() < 1e-10, f"Errors do not match! Diff: {err_diff.max()}"
