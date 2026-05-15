"""
Transfer functions for Sketch-map.

Each function maps pairwise distances R -> F(R) in [0, 1].

The canonical Sketch-map sigmoid is the extended sigmoid (XSigmoid):

    F_σ,A,B(R) = 1 - [1 + (2^(A/B) - 1) * (R/σ)^A]^(-B/A)

which reduces to the ordinary squared-Lorentzian (Sigmoid) when A=2, B->∞,
and to the identity (Identity) when σ -> ∞.

All classes share the same interface:
    f(x)  : apply the function element-wise.
    df(x) : apply the derivative.
    fdf(x): returns (f, df) together (saves recomputation).
"""

import numpy as np
try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TransferFunction:
    """Abstract base for all transfer functions."""

    def f(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def df(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fdf(self, x: np.ndarray):
        return self.f(x), self.df(x)

    def f_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        if torch is None:
            raise ImportError("torch is required for f_torch.")
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.f(x)


# ---------------------------------------------------------------------------
# Identity  F(R) = R
# ---------------------------------------------------------------------------

class Identity(TransferFunction):
    """Identity: F(R) = R.  Used when no nonlinear filter is needed."""

    def f(self, x):
        return np.asarray(x, dtype=float)

    def df(self, x):
        return np.ones_like(x, dtype=float)

    def fdf(self, x):
        x = np.asarray(x, dtype=float)
        return x.copy(), np.ones_like(x)

    def f_torch(self, x):
        return x.clone()


# ---------------------------------------------------------------------------
# Sigmoid  F(R) = 1 - 1/(1+(R/σ)²)
# ---------------------------------------------------------------------------

class Sigmoid(TransferFunction):
    """
    Squared-Lorentzian sigmoid.
        F(R) = 1 - 1 / (1 + (R/σ)²)
    Derivative:
        F'(R) = 2R/σ² * [1/(1+(R/σ)²)]²
    """

    def __init__(self, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = float(sigma)
        self._inv_s  = 1.0 / self.sigma
        self._inv_s2 = self._inv_s ** 2

    def f(self, x):
        x = np.asarray(x, dtype=float)
        sx = x * self._inv_s
        return 1.0 - 1.0 / (1.0 + sx * sx)

    def df(self, x):
        x = np.asarray(x, dtype=float)
        sx = x * self._inv_s
        t  = 1.0 / (1.0 + sx * sx)
        return x * (t * t) * 2.0 * self._inv_s2

    def fdf(self, x):
        x = np.asarray(x, dtype=float)
        sx = x * self._inv_s
        t  = 1.0 / (1.0 + sx * sx)
        return 1.0 - t, x * (t * t) * 2.0 * self._inv_s2


# ---------------------------------------------------------------------------
# Compress  F(R) = 1 - 1/(1+R/σ)
# ---------------------------------------------------------------------------

class Compress(TransferFunction):
    """
    Linear (Lorentzian) compression.
        F(R) = 1 - 1 / (1 + R/σ)
    Derivative:
        F'(R) = (1/σ) * [1/(1+R/σ)]²
    """

    def __init__(self, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = float(sigma)
        self._inv_s = 1.0 / self.sigma

    def f(self, x):
        x = np.asarray(x, dtype=float)
        sx = x * self._inv_s
        return 1.0 - 1.0 / (1.0 + sx)

    def df(self, x):
        x = np.asarray(x, dtype=float)
        sx = x * self._inv_s
        t  = 1.0 / (1.0 + sx)
        return (t * t) * self._inv_s

    def fdf(self, x):
        x = np.asarray(x, dtype=float)
        sx = x * self._inv_s
        t  = 1.0 / (1.0 + sx)
        return 1.0 - t, (t * t) * self._inv_s


# ---------------------------------------------------------------------------
# XSigmoid  (the primary Sketch-map transfer function)
# F(R) = 1 - [1 + (2^(A/B)-1)*(R/σ)^A]^(-B/A)
# ---------------------------------------------------------------------------

class XSigmoid(TransferFunction):
    """
    Extended sigmoid — the main Sketch-map transfer function.

    Parameters
    ----------
    sigma : float
        Length-scale parameter σ.  Controls which range of distances are
        "squashed" by the sigmoid (distances near σ are most affected).
    A : float
        High-dimensional exponent (called A or a in the paper).
    B : float
        Low-dimensional exponent (called B or b in the paper).

    Formula
    -------
    F_σ,A,B(R) = 1 - [1 + (2^(A/B) - 1) * (R/σ)^A]^(-B/A)

    The derivative is
    F'(R) = (B/σ) * (2^(A/B)-1) * (R/σ)^(A-1)
               * [1 + (2^(A/B)-1)*(R/σ)^A]^(-B/A - 1)
    which is equivalent to the C++ implementation using pre-computed pars.
    """

    def __init__(self, sigma: float, A: float, B: float):
        if sigma <= 0 or A <= 0 or B <= 0:
            raise ValueError("sigma, A, B must all be positive.")
        self.sigma = float(sigma)
        self.A = float(A)
        self.B = float(B)
        # Pre-computed internal constants (matching C++ layout)
        self._p0 = 1.0 / sigma                     # 1/σ
        self._p1 = 2.0 ** (A / B) - 1.0            # 2^(A/B)-1
        self._p2 = A                                # a
        self._p3 = B                                # b
        self._p4 = -B / A                           # -b/a  (outer exponent)
        # gradient constant: B * (2^(A/B)-1) / σ
        self._gfac = B * self._p1 / sigma

    def f(self, x):
        x  = np.asarray(x, dtype=float)
        sx = x * self._p0                           # R/σ
        return 1.0 - (1.0 + self._p1 * sx ** self._p2) ** self._p4

    def df(self, x):
        x  = np.asarray(x, dtype=float)
        sx = x * self._p0
        inner = 1.0 + self._p1 * sx ** self._p2
        # chain-rule: (B/A) * p1 * (R/σ)^(A-1) * (1/σ) * inner^(-B/A - 1)
        # rewritten as: B * p1 / σ * (R/σ)^(A-1) / A * inner^(-B/A - 1)
        # but simplest: derivative w.r.t. x
        # d/dx [1 - inner^p4] = -p4 * inner^(p4-1) * p1 * A * (R/σ)^(A-1) * p0
        return (-self._p4) * self._p1 * self._p2 * self._p0 * (
            sx ** (self._p2 - 1.0)) * (inner ** (self._p4 - 1.0))

    def fdf(self, x):
        x  = np.asarray(x, dtype=float)
        sx = x * self._p0
        t  = self._p1 * sx ** self._p2
        inner = 1.0 + t
        f_val = 1.0 - inner ** self._p4
        # Guard for x=0 (sx^(A-1) would be 0 if A>1, but nan if A<1)
        with np.errstate(invalid='ignore', divide='ignore'):
            # Corrected: p0 (1/sigma) is already implicitly handled by t/x if we consider t = p1*(x/sigma)^A
            df_val = (-self._p4) * self._p2 * (t / np.where(x == 0, 1.0, x)) * inner ** (self._p4 - 1.0)
            df_val = np.where(x == 0, 0.0, df_val)
        return f_val, df_val

    def f_torch(self, x):
        # sx = x / sigma
        # f = 1 - (1 + p1 * sx**p2)**p4
        sx = x * self._p0
        return 1.0 - torch.pow(1.0 + self._p1 * torch.pow(sx, self._p2), self._p4)


# ---------------------------------------------------------------------------
# Gamma  F(R) = Q(N/2, (R/σ)²/2)  (upper incomplete gamma)
# ---------------------------------------------------------------------------

class Gamma(TransferFunction):
    """
    Gamma-function sigmoid.
        F(R) = Q(N/2, (R/σ)²/2)
    where Q is the regularised upper incomplete gamma function.

    Requires scipy.special.
    """

    def __init__(self, sigma: float, N: float):
        from scipy.special import gammaincc, gamma as g_fn
        if sigma <= 0 or N <= 0:
            raise ValueError("sigma and N must be positive.")
        self.sigma = float(sigma)
        self.N = float(N)
        self._inv_s_sq2 = 1.0 / (sigma * np.sqrt(2.0))
        self._half_N = N * 0.5
        self._norm = 2.0 / g_fn(self._half_N)
        self._gammaincc = gammaincc

    def f(self, x):
        x  = np.asarray(x, dtype=float)
        sx = x * self._inv_s_sq2
        return self._gammaincc(self._half_N, sx * sx)

    def df(self, x):
        x  = np.asarray(x, dtype=float)
        sx = x * self._inv_s_sq2
        return -self._norm * (sx ** (self.N - 1.0)) * np.exp(-sx * sx) * self._inv_s_sq2

    def fdf(self, x):
        return self.f(x), self.df(x)


# ---------------------------------------------------------------------------
# Warp  F_warp(R) = g( F_HD(R) )  where g is the inverse of F_LD
# Used with -warp flag in the original CLI
# ---------------------------------------------------------------------------

class Warp(TransferFunction):
    """
    Warp transfer function: F_warp(R) = F_LD^{-1}( F_HD(R) ).

    This maps high-dim distances through the HD sigmoid and then through
    the inverse of the LD sigmoid, so that minimising the identity stress
        χ² = Σ wij (F_warp(Dij) - dij)²
    is equivalent to the full Sketch-map objective.

    Parameters are (σ, A_HD, B_HD, a_LD, b_LD) as in the C++ -warp mode.
    """

    def __init__(self, sigma: float, A: float, B: float, a: float, b: float):
        self._hd = XSigmoid(sigma, A, B)
        # Pre-compute inverse-LD constants (matching C++ pars[5..9])
        self._sigma_ld = sigma
        self._p5 = sigma                            # σ_LD
        self._p6 = 2.0 ** (a / b) - 1.0            # 2^(a/b)-1
        self._p7 = 1.0 / a                          # 1/a
        self._p8 = b                                # b
        self._p9 = -a / b                           # -a/b

    def _g(self, y):
        """Inverse of the LD XSigmoid evaluated at y = F_HD(R)."""
        sx = (1.0 - y) ** self._p9               # (1-y)^(-a/b)
        sx = (sx - 1.0) / self._p6
        return self._p5 * sx ** self._p7

    def _dg(self, y):
        """Derivative of g with respect to y."""
        g_val = self._g(y)
        # d/dy g(y)  by chain rule
        sx = (1.0 - y) ** self._p9
        denom = (sx - 1.0) * (y - 1.0) * self._p8
        return g_val / denom

    def f(self, x):
        x = np.asarray(x, dtype=float)
        return self._g(self._hd.f(x))

    def df(self, x):
        x = np.asarray(x, dtype=float)
        fx, dfx = self._hd.fdf(x)
        return self._dg(fx) * dfx

    def fdf(self, x):
        x = np.asarray(x, dtype=float)
        fx, dfx = self._hd.fdf(x)
        return self._g(fx), self._dg(fx) * dfx


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_transfer(spec) -> TransferFunction:
    """
    Construct a TransferFunction from a short specification.

    Parameters
    ----------
    spec : None | "identity" | tuple
        - None or "identity"       -> Identity()
        - (sigma,)                 -> Sigmoid(sigma)
        - (sigma, A, B)            -> XSigmoid(sigma, A, B)
        - (sigma, N)  [2-tuple]    -> Gamma(sigma, N)

    Examples
    --------
    >>> make_transfer((6.0, 8.0, 8.0))
    XSigmoid(sigma=6.0, A=8.0, B=8.0)
    """
    if spec is None or spec == "identity":
        return Identity()
    if isinstance(spec, (int, float)):
        return Sigmoid(float(spec))
    spec = tuple(spec)
    if len(spec) == 1:
        return Sigmoid(spec[0])
    if len(spec) == 2:
        return Gamma(spec[0], spec[1])
    if len(spec) == 3:
        return XSigmoid(spec[0], spec[1], spec[2])
    if len(spec) == 5:
        return Warp(*spec)
    raise ValueError(f"Cannot interpret transfer-function spec {spec!r}.")
