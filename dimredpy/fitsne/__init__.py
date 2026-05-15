"""
DimRedPy FIt-SNE Sub-Framework
==============================
FFT-accelerated interpolation-based t-SNE for large-scale
dimensionality reduction.

Uses openTSNE (CPU) or cuML (GPU) as the compute backend.

Reference:
    Linderman et al., "Fast interpolation-based t-SNE for improved
    visualization of single-cell RNA-seq data", Nature Methods 16,
    243-245 (2019).
"""

from .fitsne import fit_sne

__all__ = ["fit_sne"]
