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
    m = EuclideanMetric()
    a = np.array([0, 0])
    b = np.array([3, 4])
    assert m.dist(a, b) == 5.0
    
    X = np.array([[0,0], [1,0], [0,1]])
    D = m.pairwise(X)
    assert D.shape == (3, 3)
    assert np.allclose(D[0,1], 1.0)
    assert np.allclose(D[1,2], np.sqrt(2))

def test_pbc_metric():
    # Scalar period
    m1 = PBCMetric(period=10.0)
    assert m1.dist([1], [9]) == 2.0  # wraps around
    assert np.allclose(m1.diff([1], [9]), [-2.0])
    
    # Array period
    m2 = PBCMetric(period=[10.0, 5.0])
    assert m2.dist([1, 1], [9, 4]) == np.sqrt(2**2 + 2**2) 
    
    X = np.array([[1, 1], [9, 4]])
    D = m2.pairwise(X)
    assert np.allclose(D[0, 1], np.sqrt(8))

def test_spherical_metric():
    m = SphericalMetric(period=[2*np.pi])
    # On a unit circle, distance between 0 and pi/2 is pi/2
    assert np.allclose(m.dist([0], [np.pi/2]), np.pi/2)
    
    # Test diff (azimuthal wrapping)
    d = m.diff([0.1], [2*np.pi - 0.1])
    assert np.allclose(d, [-0.2])

def test_dot_metric():
    m = DotMetric()
    a = np.array([1, 0])
    b = np.array([0, 1])
    # dot is 0, dist is -log(1e-300) approx 690
    assert m.dist(a, b) > 600
    
    c = np.array([1, 0])
    assert m.dist(a, c) == 0.0

def test_get_metric():
    assert isinstance(get_metric(0, 0, False), EuclideanMetric)
    assert isinstance(get_metric(10.0), PBCMetric)
    assert isinstance(get_metric(0, 2.0), SphericalMetric)
    assert isinstance(get_metric(dot=True), DotMetric)
    
    with pytest.raises(ValueError):
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
    # x=2, sigma=2 -> f(2) = 1 - 1 / (1+1) = 0.5
    assert np.allclose(tf.f(2.0), 0.5)
    
    # Check gradient numerically
    x = 3.0
    h = 1e-6
    df_num = (tf.f(x + h) - tf.f(x - h)) / (2 * h)
    assert np.allclose(tf.df(x), df_num, atol=1e-5)

def test_xsigmoid_transfer():
    # Extended sigmoid: (sigma, A, B)
    tf = XSigmoid(sigma=2.0, A=4.0, B=6.0)
    val = tf.f(2.0)
    assert val > 0
    
    f, df = tf.fdf(2.0)
    h = 1e-6
    df_num = (tf.f(2.0 + h) - tf.f(2.0 - h)) / (2 * h)
    assert np.allclose(df, df_num, atol=1e-5)

def test_compress_transfer():
    tf = Compress(sigma=2.0)
    # f(x) = 1 - 1 / (1 + x/sigma)
    assert np.allclose(tf.f(2.0), 0.5)
    assert tf.df(0.0) == 0.5 # (1/2) * [1/1]^2

def test_gamma_transfer():
    tf = Gamma(sigma=2.0, N=2.0)
    val = tf.f(1.0)
    assert val > 0

def test_warp_transfer():
    # Warp(sigma, A, B, a, b)
    tf = Warp(sigma=2.0, A=4.0, B=6.0, a=2.0, b=100.0)
    val = tf.f(2.0)
    assert val > 0

def test_make_transfer():
    assert isinstance(make_transfer(None), Identity)
    assert isinstance(make_transfer((2.0,)), Sigmoid)
    assert isinstance(make_transfer((2.0, 4.0, 6.0)), XSigmoid)
    assert isinstance(make_transfer((2.0, 2.0)), Gamma)
    assert isinstance(make_transfer((2.0, 4.0, 6.0, 2.0, 8.0)), Warp)
