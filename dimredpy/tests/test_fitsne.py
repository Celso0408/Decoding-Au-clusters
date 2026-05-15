import numpy as np
import pytest
from dimredpy.fitsne import fit_sne
import sys

def test_fit_sne_basic():
    """Test basic FIt-SNE execution with default parameters."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    emb = fit_sne(data, perplexity=5, n_iter=10)
    assert emb.shape == (50, 2)
    assert not np.any(np.isnan(emb))

def test_fit_sne_auto_learning_rate():
    """Test the automatic learning rate calculation logic."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    # If learning_rate="auto", it shouldn't crash
    emb = fit_sne(data, perplexity=5, learning_rate="auto", early_exaggeration=12.0, n_iter=10)
    assert emb.shape == (50, 2)

def test_fit_sne_explicit_learning_rate():
    """Test providing a specific float learning rate."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    emb = fit_sne(data, perplexity=5, learning_rate=50.0, n_iter=10)
    assert emb.shape == (50, 2)

def test_fit_sne_different_metrics():
    """Test that metric parameter is correctly handled."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    # Test cosine metric
    emb_cosine = fit_sne(data, perplexity=5, metric="cosine", n_iter=10)
    assert emb_cosine.shape == (50, 2)

def test_fit_sne_advanced_kwargs():
    """Test passing arbitrary kwargs to openTSNE via the wrapper."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    # Passed to TSNE init: initialization="pca", dof=0.5
    emb = fit_sne(data, perplexity=5, n_iter=10, initialization="pca", dof=0.5)
    assert emb.shape == (50, 2)

def test_fit_sne_negative_gradient_methods():
    """Test switching negative gradient methods."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    emb_bh = fit_sne(data, perplexity=5, n_iter=10, negative_gradient_method="bh")
    assert emb_bh.shape == (50, 2)
    
    emb_fft = fit_sne(data, perplexity=5, n_iter=10, negative_gradient_method="fft", min_num_intervals=10)
    assert emb_fft.shape == (50, 2)

def test_fit_sne_gpu_fallback():
    """Test that if use_gpu=True but cuML is missing, it falls back to CPU (if implemented)."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    # cuML is likely not installed in this test env, so it should fallback or raise
    try:
        import cuml
        has_cuml = True
    except ImportError:
        has_cuml = False
        
    if not has_cuml:
        # Currently, if use_gpu=True but cuml missing, our wrapper prints a message and falls back to CPU.
        # Ensure it runs cleanly.
        emb = fit_sne(data, perplexity=5, n_iter=10, use_gpu=True, verbose=True)
        assert emb.shape == (50, 2)
