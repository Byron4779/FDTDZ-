#!/usr/bin/env python

import codecs
import os
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
  with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
    return f.read()


# This custom class for building the extensions uses CMake to compile. You
# don't have to use CMake for this task, but I found it to be the easiest when
# compiling ops with GPU support since setuptools doesn't have great CUDA
# support.
class CMakeBuildExt(build_ext):

  def build_extensions(self):
    # First: configure CMake build
    import sys

    import pybind11

    # FindPython resolves development headers/libraries for this interpreter.
    self.cmake_config = "Debug" if self.debug else "Release"
    install_dir = os.path.abspath(
        os.path.dirname(self.get_ext_fullpath("dummy")))
    os.makedirs(install_dir, exist_ok=True)
    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
        "-DPython_EXECUTABLE={}".format(sys.executable),
        "-DCMAKE_BUILD_TYPE={}".format(self.cmake_config),
        "-DCMAKE_PREFIX_PATH={}".format(pybind11.get_cmake_dir()),
    ]
    architectures = os.environ.get("FDTDZ_CUDA_ARCHITECTURES")
    if architectures:
      cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=" + architectures)

    os.makedirs(self.build_temp, exist_ok=True)
    subprocess.check_call(["cmake", HERE] + cmake_args, cwd=self.build_temp)

    # Build all the extensions
    super().build_extensions()

    # Finally run install
    subprocess.check_call(
        ["cmake", "--install", ".", "--config", self.cmake_config],
        cwd=self.build_temp,
    )

  def build_extension(self, ext):
    target_name = ext.name.split(".")[-1]
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", target_name,
         "--config", self.cmake_config],
        cwd=self.build_temp,
    )


# Opt-in pure-Python builds support the NumPy/JAX ADE solvers without nvcc.
# CUDA builds remain the default; a failed CUDA build never silently falls back.
build_cuda = os.environ.get("FDTDZ_BUILD_CUDA", "1")
if build_cuda not in {"0", "1"}:
  raise ValueError("FDTDZ_BUILD_CUDA must be 0 or 1")
extensions = [
    Extension(
        "fdtdz_jax.gpu_ops",
        [
            "cuda/jax_ops.cc",
            "cuda/kernel_jax.cc.cu",
        ],
    )
] if build_cuda == "1" else []

setup(
    name="fdtdz",
    author="Jesse Lu",
    author_email="jesselu@spinsphotonics.com",
    url="https://github.com/spinsphotonics/fdtdz",
    license="MIT",
    description=("Fast, scalable, and free photonic simulation"),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src", include=["fdtdz_jax", "fdtdz_jax.*"]),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"fdtdz_jax.ptx": ["*.ptx"]},
    exclude_package_data={"": ["*.pyd", "*.so", "*.pyc"]},
    zip_safe=False,
    # Public FFI registration (jax.extend.ffi on 0.4, jax.ffi on 0.5/0.6).
    # Keep the range bounded by the CPU/GPU version matrix we validate.
    install_requires=["jax>=0.4.32,<0.7", "jaxlib>=0.4.32,<0.7"],
    python_requires=">=3.10",
    use_scm_version=True,
    extras_require={
        "test": ["pytest>=7", "matplotlib>=3.5", "h5py>=3.8,<4"],
        "plot": ["matplotlib>=3.5"],
        "hdf5": ["h5py>=3.8,<4"],
        "fit": ["scipy>=1.10,<2"],
    },
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
)
