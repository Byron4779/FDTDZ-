"""Minimal high-level 3D complex-permittivity metasurface simulation.

Run from any directory, for example:

  python examples/metasurface_3d_high_level.py
  python examples/metasurface_3d_high_level.py --backend numpy
  python examples/metasurface_3d_high_level.py --backend numpy --output result.npz
"""

import argparse
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
  sys.path.insert(0, str(SOURCE))

from fdtdz_jax import BoundarySpec3D  # noqa: E402
from fdtdz_jax import CylinderZ3D  # noqa: E402
from fdtdz_jax import DrudePole  # noqa: E402
from fdtdz_jax import GridSpec3D  # noqa: E402
from fdtdz_jax import Layer3D  # noqa: E402
from fdtdz_jax import LorentzPole  # noqa: E402
from fdtdz_jax import Lossless  # noqa: E402
from fdtdz_jax import Multipole  # noqa: E402
from fdtdz_jax import PlaneWaveSource3D  # noqa: E402
from fdtdz_jax import ScatteringMonitor3D  # noqa: E402
from fdtdz_jax import Simulation3D  # noqa: E402


def gaussian_pulse(num_steps, dt, frequency=0.06):
  """Finite broadband carrier pulse in normalized solver units."""
  time = np.arange(num_steps) * dt
  center = 18.0
  width = 6.0
  return np.exp(-0.5 * ((time - center) / width)**2) * np.sin(
      2.0 * np.pi * frequency * (time - center))


def build_simulation(backend="jax"):
  grid = GridSpec3D(shape=(8, 8, 96), spacing=(1.0, 1.0, 1.0), dt=0.45)
  simulation = Simulation3D(
      grid, BoundarySpec3D(cpml_width=12),
      background=Lossless(1.0), backend=backend, dtype=np.float32)

  # Illustrative normalized material, not a fit to a particular real metal.
  metal = Multipole(1.0, (
      DrudePole(plasma_frequency=0.55, collision_frequency=0.025),
      LorentzPole(delta_eps=0.8, omega_0=0.42, damping=0.025),
  ))
  simulation.add_material("substrate", Lossless(2.25))
  simulation.add_material("metal", metal)

  # Later geometry overrides earlier geometry in overlapping Yee nodes.
  simulation.add_geometry(Layer3D(47.0, 50.0, "substrate"))
  simulation.add_geometry(CylinderZ3D(
      center_xy=(4.0, 4.0), radius=2.4,
      z_center=48.5, height=3.0, material="metal"))

  simulation.set_source(PlaneWaveSource3D(
      gaussian_pulse(240, grid.dt), z_index=20,
      polarization_xy=(1.0, 0.0), direction=1))
  simulation.set_monitors(ScatteringMonitor3D(
      reflection_z=32, transmission_z=72))
  return simulation


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--backend", choices=("jax", "numpy"), default="jax")
  parser.add_argument("--output", type=Path,
                      help="Export to .npz, .h5 or .hdf5; existing files are protected")
  args = parser.parse_args()
  if args.output is not None and args.output.suffix.lower() not in (
      ".npz", ".h5", ".hdf5"):
    parser.error("--output must end with .npz, .h5 or .hdf5")

  result = build_simulation(args.backend).run(window="hann")
  spectrum = result.diffraction
  incident = np.sqrt(np.sum(
      np.abs(spectrum.reference_incident_field)**2, axis=1))
  valid = np.flatnonzero(spectrum.valid)
  if valid.size == 0:
    raise RuntimeError("No frequency bin passed the incident-amplitude threshold.")
  peak = valid[np.argmax(incident[valid])]

  print(f"backend: {result.backend}")
  print(f"device: {result.device}")
  print(f"peak normalized frequency: {spectrum.frequencies[peak]:.6f}")
  print(f"R/T/A: {spectrum.reflection[peak]:.6f} / "
        f"{spectrum.transmission[peak]:.6f} / "
        f"{spectrum.absorption[peak]:.6f}")
  print(f"max |Ex(final)|: {np.max(np.abs(result.final_field('Ex'))):.6e}")
  if args.output is not None:
    if args.output.suffix.lower() == ".npz":
      result.to_npz(args.output)
    else:
      result.to_hdf5(args.output)
    print(f"Exported result: {args.output}")


if __name__ == "__main__":
  main()
