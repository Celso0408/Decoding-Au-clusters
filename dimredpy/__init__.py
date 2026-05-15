"""
DimRedPy -- Dimensionality Reduction Framework for Python
==========================================================
A unified, pure-Python / PyTorch / scikit-learn framework for nonlinear
dimensionality reduction, free-energy surface construction, and structural
analysis of high-dimensional datasets.

Sub-frameworks
--------------
Sketch-map (sketchmap)
    Nonlinear dimensionality reduction preserving local and global structure
    via optimised sigmoid-based transfer functions. Pure-Python reimplementation
    of the lab-cosmo/sketchmap C++ codebase.

FIt-SNE (fitsne)
    FFT-accelerated interpolation-based t-SNE via openTSNE / cuML.
    O(N log N) scaling for large datasets.

UMAP (umap_embed)
    Uniform Manifold Approximation and Projection via umap-learn / cuML.
    Supports arbitrary distance metrics including Mahalanobis.

MBAR (mbar)
    Multistate Bennett Acceptance Ratio for free-energy surface estimation
    from replica-exchange or expanded-ensemble simulations. Wraps pymbar.

Shared modules
--------------
shared
    Consolidated shared utilities including metrics, transfer functions,
    distance-preservation diagnostics, IO loaders, and structural descriptors.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dimredpy")
except PackageNotFoundError:
    __version__ = "0.1.0+dev"
