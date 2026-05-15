from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


class BuildExt(build_ext):
    c_opts = {
        "unix": ["-O3", "-std=c++11"],
    }
    l_opts = {
        "unix": [],
    }

    def build_extensions(self):
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

extension = Extension(
    "sketchmap_cpp._core",
    sources=sources,
    include_dirs=[
        *pybind11_include_dirs(),
        str(REPO_ROOT / "libs"),
        str(REPO_ROOT / "tools"),
    ],
    libraries=["lapack", "blas"],
    language="c++",
)


setup(
    ext_modules=[extension],
    package_dir={"": "src"},
    packages=["sketchmap_cpp"],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
