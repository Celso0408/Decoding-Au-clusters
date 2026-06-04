from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


class BuildExt(build_ext):
    c_opts = {
        "unix": ["-O0", "-std=c++11", "-D_hypot=hypot"],
        "mingw32": ["-O0", "-std=c++11", "-D_hypot=hypot", "--param", "ggc-min-expand=0", "--param", "ggc-min-heapsize=65536"],
    }
    l_opts = {
        "unix": [],
        "mingw32": [],
    }

    def build_extensions(self):
        # Enable parallel compilation (compile multiple .cpp files at once)
        self.parallel = 4
        ct = self.compiler.compiler_type
        opts = list(self.c_opts.get(ct, []))
        link_opts = list(self.l_opts.get(ct, []))

        extra_compile = os.environ.get("SKETCHMAP_EXTRA_COMPILE_ARGS")
        if extra_compile:
            opts.extend(extra_compile.split())

        extra_link = os.environ.get("SKETCHMAP_EXTRA_LINK_ARGS")
        if extra_link:
            link_opts.extend(extra_link.split())

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        super().build_extensions()


def pybind11_include_dirs() -> list[str]:
    import pybind11

    return [pybind11.get_include(), pybind11.get_include(user=True)]


sources = [
    "src/sketchmap_cpp/_bindings.cpp",
    "../tools/libdimred.cpp",
    "../libs/libtb.cpp",
    "../libs/librndgen.cpp",
    "../libs/libminsearch.cpp",
    "../libs/liblinalg.cpp",
    "../libs/libioparser.cpp",
    "../libs/libinterpol.cpp",
    "../libs/libfmblas.cpp",
    "../libs/libclparser.cpp",
]

import sys
import glob

extra_link_args = []
libraries = []  # Avoid hanging linker; will be resolved below

if sys.platform == "win32":
    # Robustly find OpenBLAS DLL via installed packages (works in pip isolated envs too)
    openblas_dll = None

    # Strategy 1: find via scipy package location
    try:
        import scipy
        scipy_site = Path(scipy.__file__).parent.parent
        candidates = (
            list(scipy_site.glob("scipy.libs/*openblas*.dll")) +
            list(scipy_site.glob("numpy.libs/*openblas*.dll"))
        )
        if candidates:
            openblas_dll = candidates[0]
    except ImportError:
        pass

    # Strategy 2: find via numpy package location
    if openblas_dll is None:
        try:
            import numpy
            numpy_site = Path(numpy.__file__).parent.parent
            candidates = (
                list(numpy_site.glob("scipy.libs/*openblas*.dll")) +
                list(numpy_site.glob("numpy.libs/*openblas*.dll")) +
                list(numpy_site.glob("numpy/.libs/*openblas*.dll"))
            )
            if candidates:
                openblas_dll = candidates[0]
        except ImportError:
            pass

    # Strategy 3: fall back to the venv-relative path (legacy)
    if openblas_dll is None:
        # Walk up from HERE to find a .venv directory
        search = HERE
        for _ in range(6):
            candidate_venv = search / ".venv" / "Lib" / "site-packages"
            hits = (
                list(candidate_venv.glob("scipy.libs/*openblas*.dll")) +
                list(candidate_venv.glob("numpy.libs/*openblas*.dll"))
            )
            if hits:
                openblas_dll = hits[0]
                break
            search = search.parent

    if openblas_dll:
        libraries = []
        extra_link_args = [str(openblas_dll)]
        print(f"Windows build: Linking against OpenBLAS DLL found at {openblas_dll}")
    else:
        # Last resort — system LAPACK/BLAS (may hang if not installed)
        libraries = ["lapack", "blas"]
        print("WARNING: OpenBLAS DLL not found. Falling back to system lapack/blas.")

else:
    # Linux build
    libraries = ["lapack", "blas"]

extra_objects = []
if sys.platform == "win32" and extra_link_args:
    extra_objects = extra_link_args
    extra_link_args = []

extension = Extension(
    "sketchmap_cpp._core",
    sources=sources,
    include_dirs=[
        str(HERE / "mingw_compat") if sys.platform == "win32" else "",
        *pybind11_include_dirs(),
        str(REPO_ROOT / "libs"),
        str(REPO_ROOT / "tools"),
    ] if sys.platform == "win32" else [
        *pybind11_include_dirs(),
        str(REPO_ROOT / "libs"),
        str(REPO_ROOT / "tools"),
    ],
    libraries=libraries,
    extra_link_args=extra_link_args,
    extra_objects=extra_objects,
    language="c++",
)


setup(
    ext_modules=[extension],
    package_dir={"": "src"},
    packages=["sketchmap_cpp"],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
