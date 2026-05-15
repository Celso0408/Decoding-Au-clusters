import numpy as np
import pytest
from dimredpy.umap_embed import umap_embed

def test_umap_embed_basic():
    """Test basic UMAP execution with default parameters."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(50, 10)
    emb = umap_embed(data, n_neighbors=5, n_epochs=10, min_dist=0.1)
    assert emb.shape == (50, 2)
    assert not np.any(np.isnan(emb))

def test_umap_embed_mahalanobis():
    """Test Mahalanobis metric which triggers covariance matrix logic."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    # Provide enough data to build a non-singular covariance matrix
    data = np.random.rand(100, 5)
    emb = umap_embed(data, n_neighbors=5, n_epochs=10, metric="mahalanobis")
    assert emb.shape == (100, 2)
    assert not np.any(np.isnan(emb))

def test_umap_embed_advanced_kwargs():
    """Test passing arbitrary kwargs to umap-learn."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(50, 5)
    # Passed to UMAP init: densmap=True, dens_lambda=2.0
    emb = umap_embed(data, n_neighbors=5, n_epochs=10, densmap=True, dens_lambda=2.0)
    assert emb.shape == (50, 2)

def test_umap_embed_different_metrics():
    """Test other metrics like cosine."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(50, 5)
    emb = umap_embed(data, n_neighbors=5, n_epochs=10, metric="cosine")
    assert emb.shape == (50, 2)

def test_umap_gpu_fallback():
    """Test GPU fallback behavior."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(50, 5)
    try:
        import cuml
        has_cuml = True
    except ImportError:
        has_cuml = False
        
    if not has_cuml:
        # Should fallback to umap-learn cleanly without raising an error
        emb = umap_embed(data, n_neighbors=5, n_epochs=10, use_gpu=True, verbose=True)
        assert emb.shape == (50, 2)
