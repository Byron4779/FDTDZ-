"""Layer stack, concave polygon and binary bitmap in one 3D unit cell.

Run: python examples/metasurface_3d_geometry.py --backend numpy
Optional: --output result.npz (existing files are protected).
Coordinates use normalized units; these illustrative materials are not fits.
"""

import argparse
from pathlib import Path
import sys

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
  sys.path.insert(0, str(SOURCE))

from fdtdz_jax import BitmapZ3D, BoundarySpec3D, GridSpec3D, LayerStack3D  # noqa: E402
from fdtdz_jax import Lossless, PlaneWaveSource3D, PolygonZ3D  # noqa: E402
from fdtdz_jax import ScatteringMonitor3D, Simulation3D  # noqa: E402


def build_simulation(backend="numpy"):
  grid = GridSpec3D((8, 8, 96), (0.5, 0.5, 0.5), 0.2)
  simulation = Simulation3D(grid, BoundarySpec3D(12), backend=backend)
  simulation.add_material("substrate", Lossless(2.25))
  simulation.add_material("pattern", Lossless(4.0))
  simulation.add_geometry(LayerStack3D(23.0, (
      (0.5, "substrate"), (0.5, "pattern"))))
  simulation.add_geometry(PolygonZ3D(
      ((-0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5),
       (0.5, 1.5), (-0.5, 1.5)), 24.0, 25.0, "pattern"))
  # bitmap[x_pixel, y_pixel]; conventional image arrays need a transpose.
  simulation.add_geometry(BitmapZ3D(
      [[True, False], [False, True]], origin_xy=(2.0, 2.0),
      pixel_size=(0.5, 0.5), z_min=25.0, z_max=26.0, material="pattern"))
  times = np.arange(600) * grid.dt
  pulse = np.exp(-0.5 * ((times - 12.0) / 4.0)**2) * np.sin(
      2.0 * np.pi * 0.12 * (times - 12.0))
  simulation.set_source(PlaneWaveSource3D(pulse, 20))
  simulation.set_monitors(ScatteringMonitor3D(32, 72))
  return simulation


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = build_simulation(args.backend).run()
  for index, name in enumerate(result.material_names):
    print(f"{name}: {np.count_nonzero(result.material_ids == index)} Yee material nodes")
  spectrum = result.spectrum
  bins = np.flatnonzero(spectrum.valid)
  if not bins.size:
    raise RuntimeError("No valid spectrum bins for this source.")
  peak = bins[np.argmax(np.sum(np.abs(spectrum.reference_incident_field[bins])**2, axis=1))]
  print(f"backend: {result.backend}; peak frequency: {spectrum.frequencies[peak]:.6g}")
  print(f"R/T/A: {spectrum.reflection[peak]:.6g} / "
        f"{spectrum.transmission[peak]:.6g} / {spectrum.absorption[peak]:.6g}")
  if args.output is not None:
    result.to_npz(args.output)
    print(f"Exported: {args.output}")


if __name__ == "__main__":
  main()
