"""Check release archive contents without extracting or importing the package."""

import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def check(directory, pure_python=False):
  sdists = list(directory.glob("*.tar.gz"))
  wheels = list(directory.glob("*.whl"))
  if len(sdists) != 1 or len(wheels) != 1:
    raise RuntimeError("Provide a directory containing exactly one sdist and wheel")
  with tarfile.open(sdists[0]) as archive:
    source_names = {
        str(PurePosixPath(name).relative_to(PurePosixPath(name).parts[0]))
        for name in archive.getnames()}
  required = {
      "CMakeLists.txt", "cuda/CMakeLists.txt", "cuda/jax_ops.cc",
      "cuda/kernel_jax.cc.cu", "cuda/kernel_jax.h", "cuda/shapedefs.h",
      "src/fdtdz_jax/simulation_3d.py", "src/fdtdz_jax/fdtdz_jax_version.py",
      "src/fdtdz_jax/result_io_3d.py", "src/fdtdz_jax/plotting_3d.py",
      "src/fdtdz_jax/geometry_3d.py",
      "src/fdtdz_jax/bloch_3d.py",
      "examples/oblique_film_3d.py", "examples/profile_jax_gpu_3d.py",
      "docs/oblique_incidence_3d.md", "docs/jax_gpu_profiling.md",
      "examples/validate_jax_gpu_3d.py",
      "tools/run_wsl_gpu.py",
      "tools/validate_gpu_environment.py", "docs/gpu_validation.md",
      "complex-permittivity-metasurface-2d.ipynb",
      "complex-permittivity-metasurface-3d.ipynb",
  }
  if missing := required - source_names:
    raise RuntimeError(f"Source distribution is missing: {sorted(missing)}")
  with zipfile.ZipFile(wheels[0]) as archive:
    wheel_names = set(archive.namelist())
  # All runtime Python modules and PTX assets in the sdist must reach the wheel.
  expected_runtime = {
      name.removeprefix("src/") for name in source_names
      if name.startswith("src/fdtdz_jax/") and name.endswith((".py", ".ptx"))}
  if missing := expected_runtime - wheel_names:
    raise RuntimeError(f"Wheel is missing: {sorted(missing)}")
  if not any(name.endswith(".ptx") for name in wheel_names):
    raise RuntimeError("Wheel contains no PTX kernels")
  for names in (source_names, wheel_names):
    if any("__pycache__" in name or name.endswith(".pyc") for name in names):
      raise RuntimeError("Distribution includes Python cache files")
  if any(name.endswith((".pyd", ".so")) for name in source_names):
    raise RuntimeError("Source distribution includes locally compiled extensions")
  if pure_python and any(name.endswith((".pyd", ".so")) for name in wheel_names):
    raise RuntimeError("Pure-Python wheel contains a stale native extension")
  print(f"Archive checks passed: {sdists[0].name}, {wheels[0].name}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directory", type=Path)
  parser.add_argument("--pure-python", action="store_true")
  args = parser.parse_args()
  check(args.directory, args.pure_python)
