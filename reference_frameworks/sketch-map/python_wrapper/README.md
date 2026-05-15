# sketchmap-cpp Python Wrapper

This folder contains a `pybind11` package that exposes the existing sketch-map
C++ core to Python. The goal is to keep the expensive numerical work in C++
while making the workflow usable from Python scripts, notebooks, and downstream
tools.

The package builds a native extension named `sketchmap_cpp._core` from the
repository's existing `libs/` and `tools/libdimred.cpp` sources. It does not
replace the original command-line tools.

## What Is Exposed

Python function | C++ behavior
--- | ---
`select_landmarks(...)` | Fast landmark selection using the C++ distance metrics. Supports `minmax`, `random`, and `stride`.
`mds(...)` | Classical MDS via `NLDRMDS`.
`sketch_map(...)` | Iterative sketch-map/MDS optimization via `NLDRITER`.
`project(...)` | Out-of-sample projection via `NLDRProjection::project`.
`pairwise_distances(...)` | Pairwise distances using the C++ metric implementations.

## Requirements

System requirements:

- Python 3.9+
- A C++ compiler such as `g++`
- LAPACK development libraries
- BLAS development libraries

Python build/runtime dependencies are declared in `pyproject.toml`:

- `setuptools`
- `wheel`
- `pybind11`
- `numpy`

On Debian/Ubuntu-like systems the native libraries are usually:

```sh
sudo apt-get install build-essential liblapack-dev libblas-dev
```

If you use Conda or Micromamba, install equivalent packages from your channel:

```sh
micromamba install compilers lapack blas pybind11 numpy
```

## Install

Run from the repository root:

```sh
python -m pip install ./python_wrapper
```

For editable development:

```sh
python -m pip install -e ./python_wrapper
```

The build uses the C++ files from the parent repository:

```text
../libs/*.cpp
../tools/libdimred.cpp
```

So keep `python_wrapper/` inside this repository when building.

Advanced build knobs:

```sh
SKETCHMAP_EXTRA_COMPILE_ARGS="-DUSE_BOOST" \
SKETCHMAP_EXTRA_LINK_ARGS="-lboost_math_c99" \
python -m pip install -e ./python_wrapper
```

Use this only when you know the extra flags match your local libraries. Gamma
transfer functions require `USE_BOOST`; the default build supports identity and
extended-sigmoid transfer functions.

## Basic Usage

```python
import numpy as np
import sketchmap_cpp as smap

# One sample per row, one collective variable per column.
data = np.loadtxt("examples/protein/colvar.wt.30cv.4")[:, :30]

land = smap.select_landmarks(
    data,
    n_landmarks=200,
    mode="minmax",
    period=2 * np.pi,
    return_weights=True,
)

mds = smap.mds(
    land["landmarks"],
    lowdim=2,
    period=2 * np.pi,
    center=True,
)

fit = smap.sketch_map(
    land["landmarks"],
    lowdim=2,
    weights=land["weights"],
    init=mds["embedding"],
    period=2 * np.pi,
    fun_hd=(6, 8, 8),
    fun_ld=(6, 2, 8),
    preopt_steps=100,
    grid=(60, 51, 501),
    global_steps=3,
    center=True,
)

projected = smap.project(
    data,
    landmarks_hd=land["landmarks"],
    landmarks_ld=fit["embedding"],
    weights=land["weights"],
    period=2 * np.pi,
    fun_hd=(6, 8, 8),
    fun_ld=(6, 2, 8),
    grid=(60, 51, 501),
    cg_steps=3,
)

np.savetxt("projection.smap", projected["embedding"])
```

## API Notes

### `select_landmarks`

```python
select_landmarks(
    points,
    n_landmarks,
    mode="minmax",
    input_weights=None,
    period=0.0,
    sphere_period=0.0,
    dot=False,
    seed=12345,
    first=-1,
    unique=False,
    return_weights=True,
    weight_gamma=1.0,
)
```

Returns a dictionary:

- `landmarks`: selected high-dimensional points.
- `indices`: selected source row indices.
- `weights`: normalized Voronoi landmark weights when `return_weights=True`.

### `mds`

```python
mds(points, lowdim=2, period=0.0, sphere_period=0.0, dot=False)
```

Returns:

- `embedding`
- `eigenvalues`
- `error`
- optionally `per_point_errors` when `verbose=True`

### `sketch_map`

```python
sketch_map(
    points,
    lowdim=2,
    weights=None,
    init=None,
    period=0.0,
    fun_hd=(6, 8, 8),
    fun_ld=(6, 2, 8),
    preopt_steps=100,
    grid=None,
    global_steps=0,
    imix=0.0,
    minimizer="conjgrad",
)
```

`fun_hd` and `fun_ld` accept:

- `None` or `"identity"`
- `(sigma, a, b)` for the extended sigmoid transfer function
- `(sigma, n)` for gamma mode, only if the extension is compiled with Boost

`grid=(width, coarse_points, fine_points)` enables the original pointwise
global optimizer. The grid path is only implemented for `lowdim=2`, matching the
legacy C++ implementation.

### `project`

```python
project(
    samples,
    landmarks_hd,
    landmarks_ld,
    weights=None,
    period=0.0,
    fun_hd=(6, 8, 8),
    fun_ld=(6, 2, 8),
    grid=(1.0, 21, 201),
    cg_steps=0,
)
```

Returns:

- `embedding`: projected low-dimensional coordinates.
- `error`: projection stress at the selected coordinate.
- `nearest_distance`: distance to the nearest landmark.

The original bicubic out-of-sample projector is 2D-only, so
`landmarks_ld.shape[1]` must be `2`.

## Matching the Original CLI Example

The command in `examples/protein/README`:

```sh
awk '{for(i=1;i<=30;i++){ printf "%12.6f",$i} ; print ""}' colvar.wt.30cv.4 | \
dimproj -D 30 -d 2 -P lm4.30cv.w01.1 -p lm4.30cv.w01.1.proj \
  -pi 6.283185 -w -grid 60,51,501 -fun-hd 6,8,8 -fun-ld 6,2,8 -cgmin 3
```

is equivalent to:

```python
import numpy as np
import sketchmap_cpp as smap

data = np.loadtxt("colvar.wt.30cv.4")[:, :30]
landmarks_hd = np.loadtxt("lm4.30cv.w01.1")[:, :30]
weights = np.loadtxt("lm4.30cv.w01.1")[:, 30]
landmarks_ld = np.loadtxt("lm4.30cv.w01.1.proj")[:, :2]

result = smap.project(
    data,
    landmarks_hd,
    landmarks_ld,
    weights=weights,
    period=6.283185,
    grid=(60, 51, 501),
    fun_hd=(6, 8, 8),
    fun_ld=(6, 2, 8),
    cg_steps=3,
)

np.savetxt("colvar.wt.30cv.4.smap", result["embedding"])
```

## Performance Expectations

The wrapper keeps pairwise distance loops, MDS, interpolation, and optimization
inside compiled C++. Python mainly passes NumPy arrays across the boundary and
receives NumPy arrays back.

Expected performance should be close to the original command-line tools. There
is some conversion overhead when arrays are copied into the legacy `FMatrix`
containers, but that overhead is usually small compared with `O(N^2)` pairwise
distance work and iterative optimization.

## Limitations

- The legacy C++ `ERROR(...)` macro exits the process for deep internal errors.
  The wrapper validates common bad inputs first, but invalid states inside the
  old core can still terminate Python.
- `select_landmarks` currently exposes the common `minmax`, `random`, and
  `stride` modes. The CLI-only `resample` and `staged` modes are not wrapped
  yet.
- The out-of-sample projector and grid optimizer are 2D-only.
- The wrapper builds from the parent repository sources. It is not a standalone
  vendored source distribution yet.

## Troubleshooting

`pybind11` missing:

```sh
python -m pip install pybind11
python -m pip install -e ./python_wrapper
```

LAPACK or BLAS link errors:

```sh
sudo apt-get install liblapack-dev libblas-dev
```

or pass custom link flags:

```sh
SKETCHMAP_EXTRA_LINK_ARGS="-L/path/to/libs -llapack -lblas" \
python -m pip install -e ./python_wrapper
```

Compiler too old:

Use a compiler with C++11 support or newer.

