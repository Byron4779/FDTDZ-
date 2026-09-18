"""Independent temporal and CPML sensitivity studies on a dielectric film.

Run: python examples/convergence_3d.py --backend numpy
Reported differences are against each study's baseline; they are not an
absolute error estimate. Frequencies and coordinates use normalized units.
"""

import argparse
from pathlib import Path
import sys

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
  sys.path.insert(0, str(SOURCE))

from fdtdz_jax import BoundarySpec3D, GridSpec3D, Layer3D, Lossless  # noqa: E402
from fdtdz_jax import PlaneWaveSource3D, ScatteringMonitor3D, Simulation3D  # noqa: E402


def pulse(times):
  return np.exp(-0.5 * ((times - 15.0) / 5.0)**2) * np.sin(
      2.0 * np.pi * 0.06 * (times - 15.0))


def build_simulation(backend):
  grid = GridSpec3D((2, 2, 64), (1.0, 1.0, 1.0), 0.4)
  simulation = Simulation3D(grid, BoundarySpec3D(6), backend=backend)
  simulation.add_material("film", Lossless(2.25))
  simulation.add_geometry(Layer3D(29.0, 32.0, "film"))
  simulation.set_source(PlaneWaveSource3D(pulse(np.arange(300) * grid.dt), 12))
  simulation.set_monitors(ScatteringMonitor3D(20, 46))
  return simulation


def print_study(study):
  print(f"{study.parameter}: {study.values}; baseline={study.baseline_value}")
  for name, report in study.comparisons.comparisons.items():
    print(f"  {name}: max |delta R/T/A| = "
          f"{report.maximum_reflection_difference:.6g} / "
          f"{report.maximum_transmission_difference:.6g} / "
          f"{report.maximum_absorption_difference:.6g}")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
  args = parser.parse_args()
  simulation = build_simulation(args.backend)
  temporal = simulation.run_temporal_study(
      (1, 2, 4), waveform_factory=pulse,
      frequency_min=0.04, frequency_max=0.09)
  cpml = simulation.run_cpml_study(
      (6, 8, 10), frequency_min=0.04, frequency_max=0.09)
  print_study(temporal)
  print_study(cpml)
  finer = temporal.results[2].compare_spectrum(
      temporal.results[4], frequency_min=0.04, frequency_max=0.09)
  print(f"dt/2 versus dt/4: max |delta R| = {finer.maximum_reflection_difference:.6g}")


if __name__ == "__main__":
  main()
