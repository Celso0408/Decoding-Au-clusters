"""
Data loaders for spatial coordinates and simulation results.

This module provides agnostic parsers for common spatial data formats (XYZ, etc.)
and generic trajectory data.
"""

import numpy as np
import os
from typing import Optional, List, Dict, Tuple


def load_spatial_coordinates(
    filepath: str,
    format: str = "xyz",
    **kwargs,
) -> np.ndarray:
    """
    Load 3D spatial coordinates from a file.

    Parameters
    ----------
    filepath : path to the data file.
    format   : "xyz" or other supported coordinate formats.

    Returns
    -------
    (T, N, 3) float array of coordinates.
    """
    if format.lower() == "xyz":
        return _load_xyz(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_xyz(
    filepath: str,
    label_filter: Optional[str] = None,
) -> np.ndarray:
    """
    Parse an extended XYZ file and return spatial positions.
    Uses a memory-efficient file iterator instead of readlines().

    Parameters
    ----------
    filepath     : path to .xyz file.
    label_filter : if given, only points with this label are kept.

    Returns
    -------
    (T, N, 3) float array.
    """
    pos_list = []
    with open(filepath, "r") as f:
        while True:
            try:
                line = next(f).strip()
                if not line:
                    continue
                n_points = int(line)
                next(f)  # Skip comment line
                
                frame_pos = []
                for _ in range(n_points):
                    parts = next(f).split()
                    if not parts:
                        continue
                    label = parts[0]
                    if label_filter is None or label == label_filter:
                        frame_pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
                # Always append the frame to keep frame count T consistent
                pos_list.append(frame_pos)
            except StopIteration:
                break
            except (ValueError, IndexError):
                # Skip malformed frames
                continue
            
    # Try to make a consistent 3D array if possible
    return np.asarray(pos_list, dtype=float)


def load_tabular_data(
    filepath: str,
    skip_header: int = 0,
    **kwargs,
) -> np.ndarray:
    """Load data from a text file (e.g., COLVAR or energies)."""
    return np.loadtxt(filepath, skiprows=skip_header, **kwargs)


def save_tabular_data(
    filepath: str,
    data: np.ndarray,
    header: str = "",
    **kwargs,
) -> None:
    """Save data to a text file."""
    np.savetxt(filepath, data, header=header, **kwargs)


def load_point_set(
    hd_path: str,
    ld_path: Optional[str] = None,
    has_weights: bool = False,
) -> Dict:
    """
    Load high-dimensional (and optionally low-dimensional) point sets.

    Commonly used for loading landmarks and their embeddings.
    """
    hd_data = np.loadtxt(hd_path)
    out = {}
    
    if has_weights:
        out["points_hd"] = hd_data[:, :-1]
        out["weights"]   = hd_data[:, -1]
    else:
        out["points_hd"] = hd_data
        out["weights"]   = None
        
    if ld_path is not None:
        out["points_ld"] = np.loadtxt(ld_path)
        
    return out


def save_point_set(
    filepath: str,
    points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> None:
    """Save a point set with optional weights."""
    if weights is not None:
        data = np.column_stack([points, weights])
    else:
        data = points
    np.savetxt(filepath, data, **kwargs)


def save_grid_surface(
    filepath: str,
    surface: Dict,
    gnuplot: bool = False,
) -> None:
    """
    Save a 2D grid surface (e.g., probability or energy) to a file.
    Optimized to use vectorized NumPy operations.
    """
    x = np.asarray(surface["bin_centers_x"])
    y = np.asarray(surface["bin_centers_y"])
    prob = np.asarray(surface["probability"])
    fe   = np.asarray(surface["free_energy"])
    
    # Create a 2D meshgrid of X and Y coordinates
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    if gnuplot:
        # Gnuplot requires a blank line between blocks of different X
        with open(filepath, "w") as f:
            f.write("# x  y  probability  value\n")
            for i in range(len(x)):
                # Vectorize the inner loop by stacking arrays
                block = np.column_stack((
                    np.full(len(y), x[i]),
                    y,
                    prob[i, :],
                    fe[i, :]
                ))
                np.savetxt(f, block, fmt="%12.6f %12.6f %12.6e %12.6f")
                f.write("\n")
    else:
        # Fully vectorized dump if no blank lines needed
        flat_data = np.column_stack((X.ravel(), Y.ravel(), prob.ravel(), fe.ravel()))
        np.savetxt(
            filepath, 
            flat_data, 
            fmt="%12.6f %12.6f %12.6e %12.6f", 
            header="x  y  probability  value", 
            comments="#"
        )
