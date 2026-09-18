"""Import optical data, fit passive ADE strengths and simulate a thin film.

Run: python examples/fit_material_3d.py --backend numpy
Optional: --input measured.csv --representation nk --skiprows 1
Input is wavelength_nm,n,k (or wavelength_nm,epsilon_real,epsilon_imag).
The default uses synthetic data to make the example reproducible. Pole
positions/widths below are illustrative: adapt them to measured data.
"""

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
  sys.path.insert(0, str(SOURCE))

from fdtdz_jax import BoundarySpec3D, GridSpec3D, Layer3D  # noqa: E402
from fdtdz_jax import DrudePole, LorentzPole, Multipole, PhysicalScale  # noqa: E402
from fdtdz_jax import PlaneWaveSource3D, ScatteringMonitor3D, Simulation3D  # noqa: E402
from fdtdz_jax import fit_passive_material, material_accuracy_report  # noqa: E402
from fdtdz_jax import read_material_table  # noqa: E402


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
  parser.add_argument("--input", type=Path, help="CSV with wavelength in nm")
  parser.add_argument("--representation", choices=("nk", "epsilon"), default="nk")
  parser.add_argument("--skiprows", type=int, default=1)
  parser.add_argument("--imaginary-sign", type=int, choices=(-1, 1), default=1)
  args = parser.parse_args()
  scale = PhysicalScale(10e-9)
  # Fit strengths only; these resonance/collision frequencies stay fixed.
  candidates = (DrudePole(1, .025), LorentzPole(1, .42, .025))
  if args.input is None:
    known = Multipole(1.5, (DrudePole(.55, .025), LorentzPole(.8, .42, .025)))
    omega = np.linspace(.15, .8, 100)
    index = np.sqrt(known.relative_permittivity(omega))
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "synthetic.csv"
      np.savetxt(path, np.column_stack((2*np.pi*scale.length_unit_m/omega*1e9,
                                       index.real, index.imag)), delimiter=",",
                 header="wavelength_nm,n,k", comments="")
      table = read_material_table(path, delimiter=",", skiprows=1,
                                  axis="wavelength", unit="nm", scale=scale)
    print("data: synthetic (not experimental material parameters)")
  else:
    table = read_material_table(
        args.input, delimiter=",", skiprows=args.skiprows,
        representation=args.representation, imaginary_sign=args.imaginary_sign,
        axis="wavelength", unit="nm", scale=scale)
    print(f"data: {args.input}")
  fit = fit_passive_material(table, candidates)
  print(f"fitted material: {fit.material}")
  print(f"fit RMS epsilon error: {fit.rms_error:.6e}")
  print(f"fit maximum relative error: {fit.maximum_relative_error:.6e}")
  # The data fit and time-discretization errors are separate diagnostics.
  grid = GridSpec3D((2, 2, 48), (1., 1., 1.), .3)
  report = material_accuracy_report(fit.material, grid.dt, table.angular_frequencies)
  print(f"ADE maximum relative error: {report.maximum_relative_error:.6e}")
  simulation = Simulation3D(grid, BoundarySpec3D(6), backend=args.backend)
  simulation.add_material("fitted-film", fit.material)
  simulation.add_geometry(Layer3D(23, 26, "fitted-film"))
  time = np.arange(200)*grid.dt
  pulse = np.exp(-.5*((time-15)/5)**2)*np.sin(2*np.pi*.06*(time-15))
  simulation.set_source(PlaneWaveSource3D(pulse, 10))
  simulation.set_monitors(ScatteringMonitor3D(16, 34))
  result = simulation.run(window="hann")
  print(f"simulation backend/device: {result.backend} / {result.device}")
  print(f"max |Ex(final)|: {np.max(np.abs(result.final_field('Ex'))):.6e}")


if __name__ == "__main__":
  main()
