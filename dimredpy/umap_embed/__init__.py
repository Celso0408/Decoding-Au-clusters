"""
DimRedPy UMAP Sub-Framework
============================
Uniform Manifold Approximation and Projection for dimensionality reduction.

Uses umap-learn (CPU) or cuML (GPU) as the compute backend.

Reference:
    McInnes et al., "UMAP: Uniform Manifold Approximation and Projection
    for Dimension Reduction", JOSS 3(29), 861 (2018).
"""

from .umap_embed import umap_embed

__all__ = ["umap_embed"]
