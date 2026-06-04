import numpy as np
import os
import gzip
import pytest
from pathlib import Path

from dimredpy.shared.io import (
    load_spatial_coordinates, load_tabular_data, save_tabular_data,
    load_point_set, save_point_set, save_grid_surface
)

def test_spatial_io_xyz_basic(tmp_path):
    """Test reading basic xyz format."""
    path = tmp_path / "test.xyz"
    content = "2\ncomment\nX 0.0 0.0 0.0\nX 1.0 1.0 1.0\n2\ncomment2\nX 2.0 2.0 2.0\nX 3.0 3.0 3.0\n"
    path.write_text(content)
    
    pos = load_spatial_coordinates(path, format="xyz")
    assert pos.shape == (2, 2, 3)

def test_spatial_io_xyz_gzipped(tmp_path):
    """Test transparently reading gzipped xyz file."""
    path = tmp_path / "test.xyz.gz"
    content = "2\ncomment\nX 0.0 0.0 0.0\nX 1.0 1.0 1.0\n2\ncomment2\nX 2.0 2.0 2.0\nX 3.0 3.0 3.0\n"
    with gzip.open(path, "wt") as f:
        f.write(content)
        
    pos = load_spatial_coordinates(path, format="xyz")
    assert pos.shape == (2, 2, 3)

def test_spatial_io_xyz_filtered(tmp_path):
    """Test filtering by label in xyz format."""
    path = tmp_path / "test.xyz"
    content = "3\ncomment\nX 0.0 0.0 0.0\nY 1.0 1.0 1.0\nX 2.0 2.0 2.0\n"
    path.write_text(content)
    
    pos = load_spatial_coordinates(path, format="xyz", label_filter="X")
    assert pos.shape == (1, 2, 3)

def test_spatial_io_xyz_malformed(tmp_path):
    """Test error on malformed frame."""
    path = tmp_path / "test.xyz"
    content = "2\ncomment\nX 0.0 0.0 0.0\n# missing atom\n"
    path.write_text(content)
    with pytest.raises(ValueError, match="Malformed XYZ"):
        load_spatial_coordinates(path, format="xyz")

def test_spatial_io_xyz_empty(tmp_path):
    """Test error on empty file."""
    path = tmp_path / "test.xyz"
    path.write_text("")
    with pytest.raises(ValueError, match="No valid coordinate data"):
        load_spatial_coordinates(path, format="xyz")

def test_spatial_io_missing_file(tmp_path):
    """Test FileNotFoundError when loading missing coordinates."""
    path = tmp_path / "missing.xyz"
    with pytest.raises(FileNotFoundError):
        load_spatial_coordinates(path, format="xyz")

def test_spatial_io_unsupported(tmp_path):
    """Test unsupported format error handling."""
    path = tmp_path / "test.pdb"
    path.write_text("dummy")
    with pytest.raises(ValueError):
        load_spatial_coordinates(path, format="pdb")

def test_tabular_io_basic(tmp_path):
    """Test basic saving and loading of tabular data."""
    cpath = tmp_path / "data.dat"
    raw_data = np.random.rand(10, 3)
    save_tabular_data(cpath, raw_data, header="c1 c2 c3")
    
    loaded = load_tabular_data(cpath)
    assert loaded.shape == (10, 3)
    assert np.allclose(loaded, raw_data)

def test_tabular_io_skip_rows(tmp_path):
    """Test skipping header rows when loading."""
    cpath = tmp_path / "data.dat"
    raw_data = np.random.rand(10, 3)
    save_tabular_data(cpath, raw_data, header="c1 c2 c3")
    
    loaded = load_tabular_data(cpath, skip_header=1)
    assert loaded.shape == (10, 3)

def test_tabular_io_missing_file(tmp_path):
    """Test FileNotFoundError for missing tabular data."""
    path = tmp_path / "missing.dat"
    with pytest.raises(FileNotFoundError):
        load_tabular_data(path)

def test_tabular_io_invalid_load(tmp_path):
    """Test IOError when loading invalid tabular data."""
    path = tmp_path / "invalid.dat"
    path.write_text("a b c\nd e f\n")
    with pytest.raises(IOError):
        load_tabular_data(path)

def test_point_set_io_weighted(tmp_path):
    """Test saving and loading point sets with weights."""
    hd_path = tmp_path / "hd.dat"
    ld_path = tmp_path / "ld.dat"
    hd = np.random.rand(5, 10)
    ld = np.random.rand(5, 2)
    w = np.random.rand(5)
    
    save_point_set(hd_path, hd, weights=w)
    save_point_set(ld_path, ld)
    
    res = load_point_set(hd_path, ld_path, has_weights=True)
    assert np.allclose(res["points_hd"], hd)
    assert np.allclose(res["points_ld"], ld)
    assert np.allclose(res["weights"], w)

def test_point_set_io_unweighted(tmp_path):
    """Test point sets without weights."""
    hd_path = tmp_path / "hd.dat"
    ld_path = tmp_path / "ld.dat"
    hd = np.random.rand(5, 10)
    ld = np.random.rand(5, 2)
    
    save_point_set(hd_path, hd)
    save_point_set(ld_path, ld)
    
    res = load_point_set(hd_path, ld_path, has_weights=False)
    assert np.allclose(res["points_hd"], hd)
    assert np.allclose(res["points_ld"], ld)
    assert res.get("weights") is None

def test_point_set_missing_files(tmp_path):
    """Test missing file handling in point sets."""
    hd_path = tmp_path / "missing_hd.dat"
    ld_path = tmp_path / "missing_ld.dat"
    
    with pytest.raises(FileNotFoundError):
        load_point_set(hd_path)
        
    hd = np.random.rand(5, 10)
    save_point_set(hd_path, hd)
    
    with pytest.raises(FileNotFoundError):
        load_point_set(hd_path, ld_path)

def test_save_grid_surface(tmp_path):
    """Test saving grid surfaces with gnuplot formatting."""
    path = tmp_path / "surface.dat"
    surf = {
        "bin_centers_x": np.linspace(0, 1, 5),
        "bin_centers_y": np.linspace(0, 1, 5),
        "probability": np.ones((5, 5)),
        "free_energy": np.zeros((5, 5))
    }
    
    # Save with gnuplot flag
    save_grid_surface(path, surf, gnuplot=True)
    assert path.exists()
    
    content = path.read_text()
    assert "# x  y  probability  value" in content
    assert "\n\n" in content # Check for gnuplot block separators

    # Save without gnuplot flag
    path_nogp = tmp_path / "surface_nogp.dat"
    save_grid_surface(path_nogp, surf, gnuplot=False)
    
    content_nogp = path_nogp.read_text()
    assert "\n\n" not in content_nogp # Should just be pure tabular
