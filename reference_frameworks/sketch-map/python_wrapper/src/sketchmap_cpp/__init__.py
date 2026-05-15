"""Python bindings for the sketch-map C++ core."""

from ._core import (
    __version__,
    mds,
    pairwise_distances,
    project,
    select_landmarks,
    sketch_map,
)

__all__ = [
    "__version__",
    "mds",
    "pairwise_distances",
    "project",
    "select_landmarks",
    "sketch_map",
]

