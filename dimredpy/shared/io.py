"""
Data loaders for spatial coordinates and simulation results.

This module provides agnostic parsers for common spatial data formats (XYZ, etc.)
and generic trajectory data.
"""

import numpy as np
import os
import gzip
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union


def _open_file(filepath: Union[str, Path], mode: str = "rt"):
    """Transparently open regular or gzipped files."""
    filepath = Path(filepath)
    if filepath.suffix == '.gz':
        return gzip.open(filepath, mode)
    return open(filepath, mode)


def load_spatial_coordinates(
    filepath: Union[str, Path],
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
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Coordinate file not found: {filepath}")

    if format.lower() == "xyz":
        return _load_xyz(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_xyz(
    filepath: Union[str, Path],
    label_filter: Optional[str] = None,
) -> np.ndarray:
    """
    Parse an extended XYZ file and return spatial positions.
    Uses a memory-efficient file iterator instead of readlines().
    Transparently supports .gz files.

    Parameters
    ----------
    filepath     : path to .xyz file.
    label_filter : if given, only points with this label are kept.

    Returns
    -------
    (T, N, 3) float array.
    """
    pos_list = []
    
    with _open_file(filepath, "rt") as f:
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
            except (ValueError, IndexError) as e:
                # Malformed frame
                raise ValueError(f"Malformed XYZ frame encountered in {filepath}: {e}")
            
    if not pos_list:
        raise ValueError(f"No valid coordinate data found in {filepath}")
        
    return np.asarray(pos_list, dtype=float)


def load_tabular_data(
    filepath: Union[str, Path],
    skip_header: int = 0,
    **kwargs,
) -> np.ndarray:
    """Load data from a text file (e.g., COLVAR or energies)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
        
    try:
        return np.loadtxt(filepath, skiprows=skip_header, **kwargs)
    except Exception as e:
        raise IOError(f"Failed to load tabular data from {filepath}: {e}")


def save_tabular_data(
    filepath: Union[str, Path],
    data: np.ndarray,
    header: str = "",
    **kwargs,
) -> None:
    """Save data to a text file."""
    filepath = Path(filepath)
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        np.savetxt(filepath, data, header=header, **kwargs)
    except Exception as e:
        raise IOError(f"Failed to save tabular data to {filepath}: {e}")


def load_point_set(
    hd_path: Union[str, Path],
    ld_path: Optional[Union[str, Path]] = None,
    has_weights: bool = False,
) -> Dict:
    """
    Load high-dimensional (and optionally low-dimensional) point sets.
    """
    hd_path = Path(hd_path)
    if not hd_path.exists():
        raise FileNotFoundError(f"High-dimensional point set file not found: {hd_path}")
        
    try:
        hd_data = np.loadtxt(hd_path)
    except Exception as e:
        raise IOError(f"Failed to load high-dimensional points from {hd_path}: {e}")
        
    out = {}
    if has_weights:
        out["points_hd"] = hd_data[:, :-1]
        out["weights"]   = hd_data[:, -1]
    else:
        out["points_hd"] = hd_data
        out["weights"]   = None
        
    if ld_path is not None:
        ld_path = Path(ld_path)
        if not ld_path.exists():
            raise FileNotFoundError(f"Low-dimensional point set file not found: {ld_path}")
        try:
            out["points_ld"] = np.loadtxt(ld_path)
        except Exception as e:
            raise IOError(f"Failed to load low-dimensional points from {ld_path}: {e}")
        
    return out


def save_point_set(
    filepath: Union[str, Path],
    points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> None:
    """Save a point set with optional weights."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if weights is not None:
        data = np.column_stack([points, weights])
    else:
        data = points
        
    try:
        np.savetxt(filepath, data, **kwargs)
    except Exception as e:
        raise IOError(f"Failed to save point set to {filepath}: {e}")


def save_grid_surface(
    filepath: Union[str, Path],
    surface: Dict,
    gnuplot: bool = False,
) -> None:
    """
    Save a 2D grid surface (e.g., probability or energy) to a file.
    Optimized to use vectorized NumPy operations.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    x = np.asarray(surface["bin_centers_x"])
    y = np.asarray(surface["bin_centers_y"])
    prob = np.asarray(surface["probability"])
    fe   = np.asarray(surface["free_energy"])
    
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    try:
        if gnuplot:
            with _open_file(filepath, "wt") as f:
                f.write("# x  y  probability  value\n")
                for i in range(len(x)):
                    block = np.column_stack((
                        np.full(len(y), x[i]),
                        y,
                        prob[i, :],
                        fe[i, :]
                    ))
                    np.savetxt(f, block, fmt="%12.6f %12.6f %12.6e %12.6f")
                    f.write("\n")
        else:
            flat_data = np.column_stack((X.ravel(), Y.ravel(), prob.ravel(), fe.ravel()))
            np.savetxt(
                filepath, 
                flat_data, 
                fmt="%12.6f %12.6f %12.6e %12.6f", 
                header="x  y  probability  value", 
                comments="#"
            )
    except Exception as e:
        raise IOError(f"Failed to save grid surface to {filepath}: {e}")
