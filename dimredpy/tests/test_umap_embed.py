import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from dimredpy.umap_embed import umap_embed, umap_project, umap_inverse_project
from dimredpy.umap_embed.umap_embed import _mahalanobis_whiten

def test_mahalanobis_whiten():
    """Test the Mahalanobis pre-whitening logic."""
    np.random.seed(42)
    # 100 samples, 3 features
    data = np.random.randn(100, 3)
    # Scale one feature heavily
    data[:, 0] *= 10.0
    
    X_white, mu, whitener = _mahalanobis_whiten(data)
    
    assert X_white.shape == (100, 3)
    assert mu.shape == (3,)
    assert whitener.shape == (3, 3)
    
    # After whitening, the covariance should be close to identity
    cov_white = np.cov(X_white, rowvar=False)
    assert np.allclose(cov_white, np.eye(3), atol=0.1)

def test_umap_embed_cpu_basic():
    """Test basic UMAP execution on CPU."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(50, 10)
    res = umap_embed(data, n_neighbors=5, n_epochs=10, min_dist=0.1, use_gpu=False)
    emb = res["embedding"]
    assert emb.shape == (50, 2)
    assert not np.any(np.isnan(emb))
    assert "model" in res
    assert res["metric"] == "euclidean"

def test_umap_embed_cpu_mahalanobis():
    """Test CPU UMAP with Mahalanobis metric."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(100, 5)
    res = umap_embed(data, n_neighbors=5, n_epochs=10, metric="mahalanobis", use_gpu=False)
    emb = res["embedding"]
    assert emb.shape == (100, 2)
    assert res["metric"] == "mahalanobis"

def test_umap_supervised():
    """Test supervised UMAP reduction."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(50, 5)
    y = np.random.randint(0, 3, size=50)
    res = umap_embed(data, y=y, n_neighbors=5, n_epochs=10, use_gpu=False)
    emb = res["embedding"]
    assert emb.shape == (50, 2)

@patch("dimredpy.umap_embed.umap_embed.HAS_CUML", True)
def test_umap_gpu_cuml_success():
    """Test GPU UMAP via cuML mocking."""
    data = np.random.rand(50, 5)
    
    mock_cuml_umap = MagicMock()
    mock_cuml_umap_instance = MagicMock()
    mock_cuml_umap_instance.fit_transform.return_value = np.zeros((50, 2))
    mock_cuml_umap.return_value = mock_cuml_umap_instance
    
    with patch.dict('sys.modules', {'cuml': MagicMock(), 'cuml.manifold': MagicMock(UMAP=mock_cuml_umap)}):
        res = umap_embed(data, n_neighbors=5, n_epochs=10, use_gpu=True, verbose=True)
        
    assert res["embedding"].shape == (50, 2)
    assert res["model"] == mock_cuml_umap_instance
    mock_cuml_umap_instance.fit_transform.assert_called_once()

@patch("dimredpy.umap_embed.umap_embed.HAS_CUML", True)
def test_umap_gpu_cuml_mahalanobis():
    """Test GPU UMAP Mahalanobis fallback (pre-whitening)."""
    data = np.random.rand(50, 5)
    
    mock_cuml_umap = MagicMock()
    mock_cuml_umap_instance = MagicMock()
    mock_cuml_umap_instance.fit_transform.return_value = np.zeros((50, 2))
    mock_cuml_umap.return_value = mock_cuml_umap_instance
    
    with patch.dict('sys.modules', {'cuml': MagicMock(), 'cuml.manifold': MagicMock(UMAP=mock_cuml_umap)}):
        res = umap_embed(data, n_neighbors=5, n_epochs=10, metric="mahalanobis", use_gpu=True, verbose=True)
        
    assert res["embedding"].shape == (50, 2)
    assert res["metric"] == "mahalanobis"
    assert res["mahalanobis_params"] is not None
    mu, whitener = res["mahalanobis_params"]
    assert mu.shape == (5,)
    assert whitener.shape == (5, 5)

@patch("dimredpy.umap_embed.umap_embed.HAS_CUML", True)
def test_umap_gpu_failure_fallback():
    """Test fallback to CPU if cuML throws an exception."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(50, 5)
    
    mock_cuml_umap = MagicMock()
    mock_cuml_umap_instance = MagicMock()
    mock_cuml_umap_instance.fit_transform.side_effect = RuntimeError("GPU crash")
    mock_cuml_umap.return_value = mock_cuml_umap_instance
    
    with patch.dict('sys.modules', {'cuml': MagicMock(), 'cuml.manifold': MagicMock(UMAP=mock_cuml_umap)}):
        res = umap_embed(data, n_neighbors=5, n_epochs=10, use_gpu=True, verbose=True)
        
    assert res["embedding"].shape == (50, 2)
    assert res["model"] is not mock_cuml_umap_instance # Falls back to umap-learn

def test_umap_project_cpu():
    """Test out-of-sample projection on CPU."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(50, 5)
    res = umap_embed(data, n_neighbors=5, n_epochs=10, use_gpu=False)
    
    new_data = np.random.rand(10, 5)
    proj = umap_project(res, new_data)
    assert proj.shape == (10, 2)

@patch("dimredpy.umap_embed.umap_embed.HAS_CUML", True)
def test_umap_project_gpu_mock():
    """Test out-of-sample projection on GPU."""
    new_data = np.random.rand(10, 5)
    
    mock_model = MagicMock()
    mock_model.transform.return_value = np.zeros((10, 2))
    
    # Test Euclidean
    res_dict = {
        "model": mock_model,
        "metric": "euclidean",
        "mahalanobis_params": None
    }
    
    with patch("dimredpy.umap_embed.umap_embed.type", return_value=MagicMock(__str__=lambda x: "cuml")):
        proj = umap_project(res_dict, new_data, verbose=True)
        assert proj.shape == (10, 2)
        mock_model.transform.assert_called_once()
        
    # Test Mahalanobis projection branch
    mock_model.transform.reset_mock()
    mu = np.zeros(5)
    whitener = np.eye(5)
    res_dict_mah = {
        "model": mock_model,
        "metric": "mahalanobis",
        "mahalanobis_params": (mu, whitener)
    }
    with patch("dimredpy.umap_embed.umap_embed.type", return_value=MagicMock(__str__=lambda x: "cuml")):
        proj_mah = umap_project(res_dict_mah, new_data, verbose=True)
        assert proj_mah.shape == (10, 2)
        mock_model.transform.assert_called_once()

def test_umap_inverse_project_cpu():
    """Test inverse projection on CPU."""
    try:
        import umap
    except ImportError:
        pytest.skip("umap-learn not installed")
        
    data = np.random.rand(50, 10)
    res = umap_embed(data, n_epochs=10, use_gpu=False)
    
    new_data = np.random.rand(10, 10)
    proj = umap_project(res, new_data)
    inv_proj = umap_inverse_project(res, proj)
    
    assert inv_proj.shape == (10, 10)

@patch("dimredpy.umap_embed.umap_embed.HAS_CUML", True)
def test_umap_inverse_project_gpu_error():
    """Test that inverse project raises NotImplementedError for GPU models."""
    mock_model = MagicMock()
    res_dict = {
        "model": mock_model,
        "metric": "euclidean",
        "mahalanobis_params": None
    }
    
    with patch("dimredpy.umap_embed.umap_embed.type", return_value=MagicMock(__str__=lambda x: "cuml")):
        with pytest.raises(NotImplementedError, match="not natively supported by the GPU"):
            umap_inverse_project(res_dict, np.zeros((10, 2)))

def test_umap_inverse_project_unsupported_model():
    """Test inverse project on model without inverse_transform."""
    mock_model = MagicMock()
    del mock_model.inverse_transform # Ensure it doesn't have it
    res_dict = {
        "model": mock_model,
        "metric": "euclidean",
        "mahalanobis_params": None
    }
    
    with pytest.raises(NotImplementedError, match="does not support inverse_transform"):
        umap_inverse_project(res_dict, np.zeros((10, 2)))
