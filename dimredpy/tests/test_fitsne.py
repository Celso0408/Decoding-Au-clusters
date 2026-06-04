import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from dimredpy.fitsne import fit_sne, fitsne_project
from dimredpy.fitsne.fitsne import _resolve_learning_rate

def test_resolve_learning_rate():
    """Test the FIt-SNE auto learning rate calculation."""
    # N / early_exag > 200
    lr1 = _resolve_learning_rate("auto", n_samples=10000, early_exaggeration=10.0)
    assert lr1 == 1000.0
    
    # N / early_exag < 200 -> bounds to 200
    lr2 = _resolve_learning_rate("auto", n_samples=1000, early_exaggeration=12.0)
    assert lr2 == 200.0
    
    # Explicit float
    lr3 = _resolve_learning_rate(500.0, n_samples=1000, early_exaggeration=12.0)
    assert lr3 == 500.0

def test_fit_sne_cpu_basic():
    """Test basic FIt-SNE execution via openTSNE (CPU)."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    res = fit_sne(data, perplexity=5, n_iter=10, use_gpu=False, min_num_intervals=None)
    emb = res["embedding"]
    assert emb.shape == (50, 2)
    assert not np.any(np.isnan(emb))
    assert "model" in res

def test_fit_sne_different_metrics():
    """Test distance metrics and method routing."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    res_cosine = fit_sne(data, perplexity=5, metric="cosine", n_iter=10, use_gpu=False)
    assert res_cosine["embedding"].shape == (50, 2)
    
    res_bh = fit_sne(data, perplexity=5, negative_gradient_method="bh", n_iter=10, use_gpu=False)
    assert res_bh["embedding"].shape == (50, 2)

def test_fit_sne_advanced_kwargs():
    """Test arbitrary kwargs are passed down properly to openTSNE."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    res = fit_sne(data, perplexity=5, n_iter=10, initialization="pca", dof=0.5, use_gpu=False)
    assert res["embedding"].shape == (50, 2)

@patch("dimredpy.fitsne.fitsne.HAS_CUML", True)
def test_fit_sne_gpu_cuml_success():
    """Test the cuML GPU execution path via mocking."""
    data = np.random.rand(50, 10)
    
    mock_cuml_tsne = MagicMock()
    mock_cuml_tsne_instance = MagicMock()
    mock_cuml_tsne_instance.fit_transform.return_value = np.zeros((50, 2))
    mock_cuml_tsne.return_value = mock_cuml_tsne_instance
    
    with patch.dict('sys.modules', {'cuml': MagicMock(), 'cuml.manifold': MagicMock(TSNE=mock_cuml_tsne)}):
        res = fit_sne(data, perplexity=5, n_iter=10, use_gpu=True, verbose=True)
        
    assert res["embedding"].shape == (50, 2)
    assert res["model"] == mock_cuml_tsne_instance
    mock_cuml_tsne_instance.fit_transform.assert_called_once()

@patch("dimredpy.fitsne.fitsne.HAS_CUML", True)
def test_fit_sne_gpu_cuml_failure_fallback():
    """Test fallback to CPU if cuML throws an exception during execution."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    
    mock_cuml_tsne = MagicMock()
    mock_cuml_tsne_instance = MagicMock()
    mock_cuml_tsne_instance.fit_transform.side_effect = RuntimeError("GPU Out of Memory")
    mock_cuml_tsne.return_value = mock_cuml_tsne_instance
    
    with patch.dict('sys.modules', {'cuml': MagicMock(), 'cuml.manifold': MagicMock(TSNE=mock_cuml_tsne)}):
        # Should catch the error, print a fallback message, and run via openTSNE
        res = fit_sne(data, perplexity=5, n_iter=10, use_gpu=True, verbose=True)
        
    assert res["embedding"].shape == (50, 2)
    # The returned model should be from openTSNE (an array/TSNEEmbedding), not the mocked cuML
    assert res["model"] is not mock_cuml_tsne_instance

def test_fit_sne_project_cpu():
    """Test out-of-sample projection using CPU model."""
    try:
        import openTSNE
    except ImportError:
        pytest.skip("openTSNE not installed")
        
    data = np.random.rand(50, 10)
    res = fit_sne(data, perplexity=5, n_iter=10, use_gpu=False)
    
    new_data = np.random.rand(10, 10)
    proj = fitsne_project(res, new_data)
    assert proj.shape == (10, 2)

@patch("dimredpy.fitsne.fitsne.HAS_CUML", True)
def test_fit_sne_project_gpu_mock():
    """Test out-of-sample projection handling for GPU models."""
    new_data = np.random.rand(10, 10)
    
    # 1. cuML model that DOES support transform
    mock_model_supported = MagicMock()
    # Mock its type to include 'cuml' so the check passes
    type(mock_model_supported).__name__ = "cuml.manifold.TSNE"
    mock_model_supported.__class__.__module__ = "cuml.manifold"
    mock_model_supported.transform.return_value = np.zeros((10, 2))
    
    res_dict_1 = {"model": mock_model_supported}
    
    # Needs some tricky patching to bypass the type string check, 
    # instead we just patch the __class__ str check
    with patch("dimredpy.fitsne.fitsne.type", return_value=MagicMock(__str__=lambda x: "cuml")):
        proj = fitsne_project(res_dict_1, new_data, verbose=True)
        assert proj.shape == (10, 2)
        mock_model_supported.transform.assert_called_once()

    # 2. cuML model that DOES NOT support transform
    mock_model_unsupported = MagicMock()
    del mock_model_unsupported.transform  # Remove the attribute
    res_dict_2 = {"model": mock_model_unsupported}
    
    with patch("dimredpy.fitsne.fitsne.type", return_value=MagicMock(__str__=lambda x: "cuml")):
        with pytest.raises(NotImplementedError, match="does not support out-of-sample projection"):
            fitsne_project(res_dict_2, new_data, verbose=True)

def test_fit_sne_open_tsne_missing():
    """Test exception when openTSNE is not installed."""
    data = np.random.rand(50, 10)
    
    # Hide openTSNE completely
    with patch.dict('sys.modules', {'openTSNE': None}):
        with pytest.raises(ImportError, match="openTSNE is required"):
            fit_sne(data, use_gpu=False)
