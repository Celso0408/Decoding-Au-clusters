import numpy as np
import pytest
from dimredpy.shared import (
    EuclideanMetric, PBCMetric, SphericalMetric, DotMetric, get_metric,
    TransferFunction, Sigmoid, XSigmoid, Identity, Warp, Compress, Gamma, make_transfer
)

# =============================================================================
# METRICS
# =============================================================================

def test_euclidean_metric():
    """Test standard Euclidean metric properties and pairwise."""
    m = EuclideanMetric()
    a = np.array([0, 0])
    b = np.array([3, 4])
    assert m.dist(a, b) == 5.0
    
    # Check pairwise symmetry and zeros
    X = np.random.rand(10, 3)
    D = m.pairwise(X)
    assert D.shape == (10, 10)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    
    # Check pairwise_vec
    Y = np.random.rand(5, 3)
    D2 = m.pairwise_vec(X, Y)
    assert D2.shape == (10, 5)
    for i in range(10):
        for j in range(5):
            assert np.isclose(D2[i, j], m.dist(X[i], Y[j]))

def test_pbc_metric_scalar_period():
    """Test Periodic Boundary Conditions with a scalar period."""
    m = PBCMetric(period=10.0)
    
    # Wrapping around the boundary
    assert m.dist([1], [9]) == 2.0  
    assert np.allclose(m.diff([1], [9]), [-2.0])
    
    # Distance should be symmetric
    assert m.dist([9], [1]) == 2.0
    assert np.allclose(m.diff([9], [1]), [2.0])
    
    # Check pairwise consistency
    X = np.array([[1], [5], [9]])
    D = m.pairwise(X)
    assert np.allclose(D, D.T)
    assert D[0, 2] == 2.0
    
    # Check pairwise_vec consistency
    Y = np.array([[0], [2]])
    D_vec = m.pairwise_vec(X, Y)
    assert D_vec.shape == (3, 2)
    assert D_vec[0, 0] == 1.0
    assert D_vec[2, 0] == 1.0 # 9 and 0 wrapped is 1

def test_pbc_metric_array_period():
    """Test PBC with different periods per dimension."""
    m = PBCMetric(period=[10.0, 5.0])
    
    # Dim 0: 1 and 9 wrapped by 10 is 2.
    # Dim 1: 1 and 4 wrapped by 5 is 2.
    # dist = sqrt(2^2 + 2^2) = sqrt(8)
    assert np.isclose(m.dist([1, 1], [9, 4]), np.sqrt(8))
    
    X = np.array([[1, 1], [9, 4]])
    D = m.pairwise(X)
    assert np.isclose(D[0, 1], np.sqrt(8))

def test_spherical_metric():
    """Test geodesic distance on a hypersphere."""
    m = SphericalMetric(period=[2*np.pi])
    
    # On a unit circle, distance between 0 and pi/2 is pi/2
    assert np.allclose(m.dist([0], [np.pi/2]), np.pi/2)
    
    # Diametrically opposite points
    assert np.allclose(m.dist([0], [np.pi]), np.pi)
    
    # Test diff (azimuthal wrapping)
    d = m.diff([0.1], [2*np.pi - 0.1])
    assert np.allclose(d, [-0.2])
    
    # Check pairwise
    X = np.array([[0], [np.pi/2], [np.pi]])
    D = m.pairwise(X)
    assert np.allclose(D[0, 1], np.pi/2)
    assert np.allclose(D[0, 2], np.pi)
    
    # Check pairwise_vec
    Y = np.array([[3*np.pi/2]])
    D_vec = m.pairwise_vec(X, Y)
    assert np.allclose(D_vec[0, 0], np.pi/2)

def test_dot_metric():
    """Test dot product distance -log(a.b)"""
    m = DotMetric()
    a = np.array([1, 0])
    b = np.array([0, 1])
    
    # dot is 0, dist is -log(1e-300) approx 690.77
    assert m.dist(a, b) > 600
    
    c = np.array([1, 0])
    assert m.dist(a, c) == 0.0 # -log(1) = 0
    
    X = np.array([[1, 0], [0, 1]])
    D = m.pairwise(X)
    assert np.allclose(np.diag(D), 0.0)
    assert D[0, 1] > 600

def test_get_metric_factory():
    """Test the metric factory."""
    assert isinstance(get_metric(0, 0, False), EuclideanMetric)
    assert isinstance(get_metric(10.0), PBCMetric)
    assert isinstance(get_metric(0, 2.0), SphericalMetric)
    assert isinstance(get_metric(dot=True), DotMetric)
    
    with pytest.raises(ValueError, match="incompatible"):
        get_metric(period=1.0, dot=True)

# =============================================================================
# TRANSFER FUNCTIONS
# =============================================================================

def test_identity_transfer():
    tf = Identity()
    assert tf.f(5.0) == 5.0
    assert tf.df(5.0) == 1.0
    f, df = tf.fdf(5.0)
    assert f == 5.0 and df == 1.0

def test_sigmoid_transfer():
    tf = Sigmoid(sigma=2.0)
    # f(x) = 1 - 1 / (1 + (x/sigma)^2)
    assert np.allclose(tf.f(2.0), 0.5)
    
    # Check gradient numerically over array
    x = np.linspace(0.1, 5.0, 10)
    h = 1e-6
    df_num = (tf.f(x + h) - tf.f(x - h)) / (2 * h)
    assert np.allclose(tf.df(x), df_num, atol=1e-5)

def test_xsigmoid_transfer():
    tf = XSigmoid(sigma=2.0, A=4.0, B=6.0)
    x = np.linspace(0.1, 5.0, 10)
    
    f, df = tf.fdf(x)
    h = 1e-6
    df_num = (tf.f(x + h) - tf.f(x - h)) / (2 * h)
    assert np.allclose(df, df_num, atol=1e-5)

def test_transfer_f_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    
    # Test Identity
    tf_id = Identity()
    t_val = torch.tensor([5.0])
    assert torch.allclose(tf_id.f_torch(t_val), torch.tensor([5.0]))
    
    # Test XSigmoid
    tf_x = XSigmoid(sigma=2.0, A=4.0, B=6.0)
    t_x = torch.linspace(0.1, 5.0, 10)
    t_out = tf_x.f_torch(t_x)
    n_out = tf_x.f(t_x.numpy())
    assert torch.allclose(t_out, torch.from_numpy(n_out), atol=1e-5)


def test_compress_transfer():
    tf = Compress(sigma=2.0)
    assert np.allclose(tf.f(2.0), 0.5)
    
    x = np.linspace(0.1, 5.0, 10)
    h = 1e-6
    df_num = (tf.f(x + h) - tf.f(x - h)) / (2 * h)
    assert np.allclose(tf.df(x), df_num, atol=1e-5)

def test_gamma_transfer():
    tf = Gamma(sigma=2.0, N=2.0)
    x = np.linspace(0.1, 5.0, 10)
    
    f, df = tf.fdf(x)
    h = 1e-6
    df_num = (tf.f(x + h) - tf.f(x - h)) / (2 * h)
    assert np.allclose(df, df_num, atol=1e-5)

def test_warp_transfer():
    tf = Warp(sigma=2.0, A=4.0, B=6.0, a=2.0, b=100.0)
    x = np.linspace(0.1, 5.0, 10)
    
    f, df = tf.fdf(x)
    h = 1e-6
    df_num = (tf.f(x + h) - tf.f(x - h)) / (2 * h)
    assert np.allclose(df, df_num, atol=1e-5)

def test_make_transfer_factory():
    assert isinstance(make_transfer(None), Identity)
    assert isinstance(make_transfer((2.0,)), Sigmoid)
    assert isinstance(make_transfer((2.0, 4.0, 6.0)), XSigmoid)
    assert isinstance(make_transfer((2.0, 2.0)), Gamma)
    assert isinstance(make_transfer((2.0, 4.0, 6.0, 2.0, 8.0)), Warp)
    
    with pytest.raises(ValueError):
        make_transfer((1, 2, 3, 4)) # Invalid length
