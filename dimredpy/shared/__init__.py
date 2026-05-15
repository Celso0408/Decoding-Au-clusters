"""
Shared utilities for the DimRedPy framework.

Includes:
    metrics     — Distance metrics (Euclidean, PBC, Spherical, Dot).
    transfer    — Transfer functions (Sigmoid, XSigmoid, etc.).
    io          — Data loaders and savers.
    analysis    — Distance-preservation diagnostics.
    descriptors — Structural and geometric descriptors.
"""

from .metrics import Metric, EuclideanMetric, PBCMetric, SphericalMetric, DotMetric, get_metric
from .transfer import TransferFunction, Sigmoid, XSigmoid, Identity, Warp, Compress, Gamma, make_transfer
from .io import (
    load_spatial_coordinates, load_tabular_data, save_tabular_data,
    load_point_set, save_point_set, save_grid_surface
)
from .analysis import distance_histogram, preservation_score
from .descriptors import (
    coordination_histogram,
    coordination_histogram_trajectory,
    effective_coordination_number,
    average_neighbor_distance,
    radius_of_gyration,
    radial_distribution_function,
    hausdorff_chirality_measure,
    projection_center,
    compute_trajectory_descriptors
)

__all__ = [
    "Metric", "EuclideanMetric", "PBCMetric", "SphericalMetric", "DotMetric", "get_metric",
    "TransferFunction", "Sigmoid", "XSigmoid", "Identity", "Warp", "Compress", "Gamma", "make_transfer",
    "load_spatial_coordinates", "load_tabular_data", "save_tabular_data",
    "load_point_set", "save_point_set", "save_grid_surface",
    "distance_histogram", "preservation_score",
    "coordination_histogram", "coordination_histogram_trajectory",
    "effective_coordination_number", "average_neighbor_distance",
    "radius_of_gyration", "radial_distribution_function",
    "hausdorff_chirality_measure", "projection_center",
    "compute_trajectory_descriptors"
]
