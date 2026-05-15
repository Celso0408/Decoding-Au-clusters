"""
Distance metrics for Sketch-map.

Mirrors the four C++ metric classes:
    NLDRMetricEuclid   -> EuclideanMetric
    NLDRMetricPBC      -> PBCMetric      (toroidal periodic boundary conditions)
    NLDRMetricSphere   -> SphericalMetric (geodesic distance on hyper-sphere)
    NLDRMetricDot      -> DotMetric       (-log(a·b))
"""

import numpy as np
from scipy.spatial.distance import cdist


class Metric:
    """Abstract base for all distance metrics."""

    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        raise NotImplementedError

    def diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(b, dtype=float) - np.asarray(a, dtype=float)

    def pairwise(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pairwise_vec(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class EuclideanMetric(Metric):
    """Standard Euclidean distance."""

    def dist(self, a, b):
        d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        return float(np.sqrt(np.dot(d, d)))

    def pairwise(self, X):
        return cdist(np.asarray(X, dtype=float), np.asarray(X, dtype=float))

    def pairwise_vec(self, X, Y):
        return cdist(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))


class PBCMetric(Metric):
    """
    Toroidal (periodic boundary conditions) distance.

    Parameters
    ----------
    period : float or array-like of shape (D,)
        Period L_i for each dimension.  Scalar applies same period to all dims.
    """

    def __init__(self, period):
        self.period = np.asarray(period, dtype=float)

    def _wrap(self, delta):
        L = self.period
        return delta - L * np.round(delta / L)

    def diff(self, a, b):
        return self._wrap(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))

    def dist(self, a, b):
        d = self.diff(a, b)
        return float(np.sqrt(np.dot(d, d)))

    def pairwise(self, X):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            delta = self._wrap(X - X[i])
            D[i] = np.sqrt((delta ** 2).sum(axis=1))
        return D

    def pairwise_vec(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        M, K = X.shape[0], Y.shape[0]
        D = np.zeros((M, K), dtype=float)
        for i in range(M):
            delta = self._wrap(Y - X[i])
            D[i] = np.sqrt((delta ** 2).sum(axis=1))
        return D


class SphericalMetric(Metric):
    """
    Geodesic (great-circle) distance on a hyper-sphere.

    The last dimension is treated as periodic (azimuthal).
    Mirrors C++ NLDRMetricSphere.

    Parameters
    ----------
    period : float or array-like of shape (D,)
        Period L_i for each dimension.
    """

    def __init__(self, period):
        self.period = np.asarray(period, dtype=float)

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """Convert angular coordinates to unit-sphere Cartesian embedding."""
        L = self.period
        angles = x * (2.0 * np.pi / L)     # (..., D)
        n = angles.shape[-1]
        result = np.zeros(angles.shape[:-1] + (n + 1,), dtype=float)
        carry = np.ones(angles.shape[:-1], dtype=float)
        for i in range(n):
            result[..., i] = carry * np.cos(angles[..., i])
            carry = carry * np.sin(angles[..., i])
        result[..., n] = carry
        return result

    def dist(self, a, b):
        ea = self._embed(np.asarray(a, dtype=float))
        eb = self._embed(np.asarray(b, dtype=float))
        return float(np.arccos(np.clip(np.dot(ea, eb), -1.0, 1.0)))

    def diff(self, a, b):
        d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        L = self.period
        d[-1] = d[-1] - L[-1] * np.round(d[-1] / L[-1])
        return d

    def pairwise(self, X):
        X = np.asarray(X, dtype=float)
        E = self._embed(X)
        return np.arccos(np.clip(E @ E.T, -1.0, 1.0))

    def pairwise_vec(self, X, Y):
        EX = self._embed(np.asarray(X, dtype=float))
        EY = self._embed(np.asarray(Y, dtype=float))
        return np.arccos(np.clip(EX @ EY.T, -1.0, 1.0))


class DotMetric(Metric):
    """Dot-product distance: d(a,b) = -log(a · b)."""

    def dist(self, a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return float(-np.log(max(np.dot(a, b), 1e-300)))

    def pairwise(self, X):
        X = np.asarray(X, dtype=float)
        dots = np.clip(X @ X.T, 1e-300, None)
        return -np.log(dots)

    def pairwise_vec(self, X, Y):
        dots = np.clip(np.asarray(X) @ np.asarray(Y).T, 1e-300, None)
        return -np.log(dots)


def get_metric(period=0.0, sphere_period=0.0, dot=False) -> Metric:
    """
    Return the appropriate Metric given CLI-style parameters.

    Parameters
    ----------
    period        : float  — toroidal period (0 = Euclidean).
    sphere_period : float  — spherical period (0 = not spherical).
    dot           : bool   — use dot-product metric.
    """
    if dot:
        if period != 0.0 or sphere_period != 0.0:
            raise ValueError("Dot-product metric is incompatible with periodic options.")
        return DotMetric()
    if sphere_period != 0.0:
        return SphericalMetric(sphere_period)
    if period != 0.0:
        return PBCMetric(period)
    return EuclideanMetric()
