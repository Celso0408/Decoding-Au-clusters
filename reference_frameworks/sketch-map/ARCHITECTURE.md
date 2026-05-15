# Sketch-map Code Instructions and Architecture

This repository contains command-line and VMD/Tcl tools for multidimensional
scaling (MDS), sketch-map nonlinear dimensionality reduction, landmark
selection, out-of-sample projection, and distance-preservation diagnostics.

The project is split into three main layers:

- `libs/`: reusable C++ numerical utilities, matrix containers, interpolation,
  optimization, random numbers, parsing, and LAPACK/BLAS wrappers.
- `tools/`: the sketch-map dimensionality-reduction algorithms and the command
  line programs built on top of them.
- `gismo/`: Tcl/Tk and VMD integration for the GISMO graphical interface.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `README` | Very short build overview. |
| `Makefile` | Top-level build entry point. Builds `libs` first, then `tools`. |
| `make.in` | Local compiler/linker configuration used by sub-Makefiles. |
| `make.in.example` | Older example configuration. |
| `libs/` | General numerical toolbox compiled into `libs/libtoolbox.a`. |
| `tools/` | CLI programs and shared sketch-map implementation. |
| `bin/` | Intended location for generated executables. Ignored by Git. |
| `gismo/` | VMD/Tcl GUI, plotting, PLUMED driver, and sketch-map wrappers. |
| `utils/sketch-map.sh` | Interactive shell workflow around `dimred`. |
| `examples/protein/` | Example protein data and a documented `dimproj` command. |

## Build Instructions

The intended build flow is:

```sh
cp make.in.example make.in   # only if make.in is absent
# edit make.in for compiler, include, LAPACK, and linker settings
make
```

The top-level `make` runs:

1. `make -C libs all`, producing `libs/libtoolbox.a`.
2. `make -C tools all`, producing `bin/dimdist`, `bin/dimlandmark`,
   `bin/dimproj`, and `bin/dimred`.

Useful Make targets:

```sh
make              # build libraries and tools
make clean        # remove .o and .d files from libs/ and tools/
make distclean    # also remove bin/* and libs/libtoolbox.a
```

Dependencies:

- A C++ compiler. The current `make.in` uses `g++`.
- LAPACK, linked as `-llapack`.
- BLAS symbols for `dgemm`/`zgemm` may be needed by the matrix multiplication
  specialization in `libs/libfmblas.cpp`, depending on how LAPACK is packaged.
- Optional `USE_BOOST` enables the gamma transfer function in
  `NLDRFunction`; without it, gamma mode intentionally exits with an error.
- Optional `ARPACK` enables an iterative sparse eigen-solver path in
  `NLDRLLE`.

Build note from this checkout: a modern-GCC compatibility issue in
`libs/tbdefs.hpp` was fixed by adding a `std::valarray` overload for
`toolbox::mpiostream`. After that patch, `make` builds `libs/libtoolbox.a` and
the four binaries in `bin/`.

The repository also contains a Python wrapper package in `python_wrapper/`.
It builds a `pybind11` extension against the same C++ core so Python code can
call the compiled implementation directly.

## Command-line Tools

### `dimlandmark`

Selects representative high-dimensional landmark points from data on standard
input.

Common usage:

```sh
bin/dimlandmark -D 30 -n 200 -mode minmax -i -w < data.dat > landmarks.dat
```

Important options:

- `-D`: input dimensionality.
- `-n`: number of landmarks to select.
- `-mode`: `stride`, `random`, `minmax`, `resample`, or `staged`.
- `-pi`: toroidal periodicity for all dimensions.
- `-spi`: spherical periodicity.
- `-dot`: use dot-product distance, implemented as `-log(a . b)`.
- `-wi`: input rows include a point weight.
- `-w`: output landmark weights from Voronoi assignment.
- `-wgamma`: transform landmark weights by a power.
- `-i`: print selected source indices.
- `-unique`: avoid duplicate random selections where applicable.
- `-similarity`: treat input as a similarity/distance matrix. This is only
  rudimentarily supported for `minmax`.

Input rows are:

```text
x1 x2 ... xD [weight-if--wi]
```

Output starts with comments, then one row per landmark:

```text
[index-if--i] x1 x2 ... xD [source-weight-if--i-and--wi] [landmark-weight-if--w]
```

### `dimred`

Computes a low-dimensional embedding. With no iterative options it performs
classical MDS. With transfer functions and optimization options it performs the
sketch-map objective.

Common MDS usage:

```sh
bin/dimred -D 30 -d 2 -pi 6.283185 -w < landmarks.dat > landmarks.mds
```

Common sketch-map usage:

```sh
bin/dimred \
  -D 30 -d 2 -pi 6.283185 -w \
  -fun-hd 6,8,8 -fun-ld 6,2,8 \
  -preopt 100 -grid 60,51,501 -gopt 3 \
  -init landmarks.mds \
  < landmarks.dat > landmarks.smap
```

Important options:

- `-D`, `-d`: input and output dimensionality. In practice several projection
  paths assume `-d 2`.
- `-w`: input rows include weights.
- `-pi`, `-spi`, `-dot`: metric selection.
- `-similarity`: input is an `N x N` distance/similarity matrix rather than raw
  coordinates.
- `-init`: low-dimensional coordinates used as the optimizer starting point.
- `-randomize`: add Gaussian noise to `-init` coordinates.
- `-center`: center output coordinates around zero.
- `-fun-hd`, `-fun-ld`: transfer functions. `identity`, `sigma,a,b`, or
  `sigma,n` are supported by the parser.
- `-preopt`: local optimizer steps before global refinement.
- `-grid`: global grid as `width,coarse_points,fine_points`.
- `-gopt`: conjugate-gradient steps after global pointwise moves.
- `-imix`: mix original distance stress with transformed-distance stress.
- `-imode`: optimizer mode: `conjgrad`, `simplex`, `anneal`, `paratemp`, or
  `nested`.
- `-plumed`: emit PLUMED-compatible landmark records.
- `-v`, `-vv`: include report lines and optionally per-point errors.

Output is normally one low-dimensional point per input row. With `-vv`, an
extra per-point error column is appended.

### `dimproj`

Projects new high-dimensional samples into an existing landmark embedding.

Example from `examples/protein/README`:

```sh
awk '{for(i=1;i<=30;i++){ printf "%12.6f",$i} ; print ""}' colvar.wt.30cv.4 \
  | dimproj -D 30 -d 2 \
      -P lm4.30cv.w01.1 \
      -p lm4.30cv.w01.1.proj \
      -pi 6.283185 -w \
      -grid 60,51,501 \
      -fun-hd 6,8,8 -fun-ld 6,2,8 \
      -cgmin 3 \
  | awk '{print $1,$2}' > colvar.wt.30cv.4.smap
```

Important options:

- `-P`: high-dimensional landmark file.
- `-p`: low-dimensional landmark projection file.
- `-D`, `-d`, `-pi`, `-spi`, `-dot`, `-w`: same meaning as above.
- `-grid`: global projection grid `width,coarse_points,fine_points`.
- `-cgmin`: conjugate-gradient refinement steps after the grid search.
- `-gt`: use exponential averaging over grid values instead of strict minimum.
- `-path`: use path-like exponential averaging instead of sketch-map
  minimization.
- `-print`: write interpolation surfaces named `interpolant.N`.
- `-similarity`: input rows are distances from the new point to landmarks.

For each input point, normal output is:

```text
y1 y2 projection_error nearest_landmark_distance
```

### `dimdist`

Computes 1D or 2D histograms of high-dimensional and low-dimensional pairwise
distances. It is mainly a diagnostic for how well the projection preserves the
selected distance scale.

Examples:

```sh
bin/dimdist -D 30 -d 2 -P landmarks.dat -p landmarks.smap -pi 6.283185 \
  -nbin 100 100 -gnuplot > distance_histogram.dat

bin/dimdist -D 30 -P landmarks.dat -pi 6.283185 -nbin 200 > hd_distances.dat
```

Important options:

- `-P`, `-p`: high- and low-dimensional point files.
- `-D`, `-d`: dimensions. If `-d 0`, only high-dimensional distances are binned.
- `-nbin`, `-maxd`, `-wbin`: histogram resolution, maximum distance, and
  smoothing window.
- `-gnuplot`: output blank-line-separated blocks suitable for gnuplot.
- `-lowmem`: recompute distances instead of storing all pair distances.
- `-osim`: print the high-dimensional pairwise distance matrix and exit.

## Core C++ Architecture

All C++ code lives in the `toolbox` namespace, except the low-level Fortran
symbols wrapped in `tblapack` and `tbblas`.

### Numerical Toolbox (`libs/`)

`libs/` is a standalone toolbox used by `tools/`:

- `tbdefs.hpp` / `libtb.cpp`: shared constants, error macros, string conversion,
  CSV parsing, timers, iteration stopping criteria, and MPI-aware streams.
- `matrix-full.hpp`: row-major dense `FMatrix<T>` with slices for rows,
  columns, and diagonals.
- `matrix-crs.hpp`: compressed-row sparse matrix implementation.
- `matrix-coord.hpp`: coordinate-list matrix representation.
- `tensor-full.hpp`: dense tensor container used by bicubic interpolation.
- `matrix-conv.hpp`, `matrix-io.hpp`, `ioparser.hpp`: matrix conversion and
  generic structured I/O helpers.
- `linalg.hpp` / `liblinalg.cpp`: LAPACK wrappers for linear solves,
  eigen-solvers, SVD, matrix inverse, pseudoinverse, matrix functions, and
  Cholesky.
- `matrix-full-blas.hpp` / `libfmblas.cpp`: BLAS-backed `mult` specialization
  for dense real and complex matrix multiplication.
- `interpol.hpp` / `libinterpol.cpp`: 1D spline and 2D bicubic interpolation.
- `minsearch.hpp` / `libminsearch.cpp`: minimizers using a function-object
  interface. Includes conjugate gradient, simplex, simulated annealing,
  parallel tempering, nested sampling, and helper constructors.
- `rndgen.hpp` / `librndgen.cpp`: standard and Mersenne Twister uniform random
  generators, Gaussian generators, and correlated Gaussian support.
- `tools-histogram.hpp`, `tools-autocorr.hpp`: analysis utilities.
- `clparser.hpp` / `libclparser.cpp`: simple command-line option parser.

### Sketch-map Layer (`tools/dimreduce.hpp`, `tools/libdimred.cpp`)

The shared dimensionality-reduction code is in `tools/libdimred.cpp` with its
public API in `tools/dimreduce.hpp`.

Main types:

- `NLDRFunction`: nonlinear transfer function used to transform distances.
  Supports identity, compress, sigmoid, extended sigmoid, gamma, and warp. It
  can build a spline interpolation table lazily for speed.
- `NLDRMetric`: metric interface. Implementations are Euclidean, toroidal PBC,
  spherical geodesic, and dot-product distance.
- `NLDRNeighborList`: all-pairs neighbor construction with max-neighbor,
  cutoff, and symmetrization controls.
- `NLDRProjection`: stores high-dimensional landmarks `P`, low-dimensional
  points `p`, weights, projection options, neighbor-derived local maps, and the
  out-of-sample objective.
- `NLDRMDS`: classical, spherical, or toroidal MDS initialization.
- `NLDRLLE`: locally linear embedding variants. This code exists in the library
  but is not exposed by the current CLI programs.
- `NLDRITERChi`: objective function adapter for the minimizers.
- `NLDRITER`: iterative sketch-map/MDS optimization over all projected points.

The principal sketch-map objective compares transformed high-dimensional and
low-dimensional distances:

```text
chi = mean_ij weight_ij * (f_hd(D_ij) - f_ld(d_ij))^2
```

When `-imix` is nonzero, the objective also mixes in a direct metric-MDS stress
term based on `(D_ij - d_ij)^2`.

## Runtime Data Flow

The typical command-line workflow is:

1. Prepare a whitespace-separated collective-variable matrix with one sample per
   row.
2. Select landmarks with `dimlandmark`, optionally including Voronoi weights.
3. Embed landmarks with `dimred`, usually starting from MDS and then applying
   sketch-map transfer functions.
4. Project the full dataset with `dimproj` using the high-dimensional landmarks
   and their low-dimensional embedding.
5. Inspect quality with `dimdist`.

The `examples/protein/` directory already contains:

- `colvar.wt.30cv.4`: 30-dimensional collective variables.
- `lm4.30cv.w01.1`: high-dimensional landmarks.
- `lm4.30cv.w01.1.proj`: low-dimensional landmark projection.
- `colvar.wt.30cv.4.smap`: projected full dataset.
- `bhp.pdb`: topology for visualization.

## GUI and VMD Architecture (`gismo/`)

The Tcl layer wraps the same command-line tools for interactive use in VMD.

- `plumed_gui.tcl`: registers the GISMO extension in VMD, creates the main
  window, connects menus, traces variables, and drives plot updates.
- `gtplot.tcl`: generic Tk canvas plotting, axes, zooming, lasso selection,
  point/line/pixel drawing, and color maps.
- `colvarTools.tcl`: stores and plots collective-variable trajectories and frame
  selections.
- `cvlist.tcl`: central data registry for CV columns, selected CVs, landmarks,
  dimensionality-reduction results, and out-of-sample embedding. It invokes
  `dimlandmark`, `dimred`, and `dimproj` through `$env(smapdir)/bin/...`.
- `sketchmap.tcl`: sketch-map-specific Tcl wrapper. It validates GUI settings,
  creates temporary files, runs MDS/sketch-map/out-of-sample commands, and reads
  resulting coordinates back into Tcl lists.
- `driver_interface.tcl`: writes PLUMED input files for torsions, DRMSD, RDF,
  and ADF, then runs `$env(plumedir)/utilities/driver/driver`.
- `fesTools.tcl`: reads/plots free-energy surfaces and can run PLUMED
  `sum_hills`.
- `progress_bar.tcl`: small Tk progress bar helper.

Expected GUI environment variables:

- `smapdir`: repository/install prefix containing `bin/dimred`,
  `bin/dimlandmark`, and `bin/dimproj`.
- `plumedir`: PLUMED installation used by the driver and `sum_hills`.
- `VMDDIR`: VMD installation path used by `plumed_gui.tcl` to source scripts.

The GUI writes temporary working files under a per-process directory such as
`/tmp/plumedvis.<pid>` or VMD's temporary directory. Intermediate files include
`CV_DATA`, `MDS_DATA`, `IMDS_DATA`, `SMAP_IN`, `SMAP_OUT`, `LANDMARK_FILE`, and
`PROJECTION_FILE`.

## Implementation Details Worth Knowing

- Most algorithms compute all pairwise distances, so memory and runtime scale
  roughly as `O(N^2)` for many operations.
- `dimproj` and global sketch-map refinement are effectively 2D-only because the
  grid and bicubic interpolation code require `d == 2`.
- The command-line parser ignores unknown options unless the program explicitly
  checks them, so typos may silently do nothing.
- `ERROR(...)` prints to `stderr` and exits unless compiled in debug mode.
- Transfer-function parameters parsed as `sigma,a,b` use extended sigmoid mode;
  parameters parsed as `sigma,n` use gamma mode and require Boost support.
- The dot-product metric computes `-log(dot)`, so input vectors must produce
  positive dot products.
- The Tcl GUI assumes VMD molecule/frame APIs are available and is not intended
  to run as standalone Tcl.
- `utils/sketch-map.sh` is an interactive helper around `dimred`, but it should
  be reviewed before production use; the `if` structure around the similarity
  prompt is fragile.

## Adding or Changing Code

When adding a new command-line option:

1. Add a variable and `clp.getoption(...)` call in the relevant `tools/*.cpp`
   file.
2. Add text to that file's `banner()` usage output.
3. If it controls shared behavior, add a field to the appropriate options class
   in `tools/dimreduce.hpp`.
4. Implement the behavior in `tools/libdimred.cpp`.
5. If the GUI should expose it, update `gismo/sketchmap.tcl` or
   `gismo/cvlist.tcl` and make sure the generated command includes the flag.

When adding a new metric:

1. Derive from `NLDRMetric`.
2. Implement `pdist(...)`; implement `pdiff(...)` if local vector differences
   are meaningful for projection or neighbor logic.
3. Instantiate it in `dimred.cpp`, `dimproj.cpp`, `dimdist.cpp`, and
   `dimlandmark.cpp` as needed.
4. Document incompatible combinations such as dot product with periodic metrics.

When adding a new optimizer:

1. Add a value to `NLDRIterMin`.
2. Add options to `NLDRITEROptions` if needed.
3. Add parser support in `dimred.cpp` for `-imode`.
4. Add a `case` in `NLDRITER`.

When changing matrix or linear algebra code:

1. Check row-major `FMatrix` layout versus Fortran column-major LAPACK/BLAS
   expectations.
2. Follow existing transpose-before-call and transpose-after-call patterns.
3. Keep dense and sparse matrix operations separate; the LLE path relies on
   `CrsMatrix`, while MDS and sketch-map mostly use dense `FMatrix`.
