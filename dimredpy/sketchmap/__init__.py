"""
Sketch-map dimensionality reduction pipeline.

Includes:
    landmark   — Landmark selection from datasets.
    reduction  — Classical MDS and Sketch-map SMACOF optimization.
    projection — Out-of-sample embedding of new points.
"""

from .reduction import sketch_map, classical_mds
from .projection import project_out_of_sample
from .landmark import select_landmarks

__all__ = [
    "sketch_map",
    "classical_mds",
    "project_out_of_sample",
    "select_landmarks"
]
