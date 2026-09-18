"""Oblique TE/TM film simulation with x/y Bloch phases and a Fresnel comparison.

python examples/oblique_film_3d.py --backend numpy --polarization te
python examples/oblique_film_3d.py --backend jax --polarization tm --azimuth 30
Use --spacing 0.125 and temporal/CPML studies to assess numerical error.
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


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
  parser.add_argument("--polarization", choices=("te", "tm"), default="te")
  parser.add_argument("--angle", type=float, default=30, help="Polar angle in degrees at f=0.06")
  parser.add_argument("--azimuth", type=float, default=0, help="Azimuth in degrees")
  parser.add_argument("--spacing", type=float, choices=(.5, .25, .125), default=.25)
  parser.add_argument("--output", type=Path, help="Optional .npz analysis archive")
  args = parser.parse_args()
  if not np.isfinite(args.angle) or not 0 <= args.angle < 90:
    parser.error("angle must lie in [0, 90) degrees")
  if not np.isfinite(args.azimuth):
    parser.error("azimuth must be finite")
  f, h = .06, args.spacing
  theta, phi = np.deg2rad((args.angle, args.azimuth))
  kt = 2*np.pi*f*np.sin(theta)
  kxy = tuple(kt*np.array([np.cos(phi), np.sin(phi)]))
  polarization = ((-np.sin(phi), np.cos(phi)) if args.polarization == "te"
                  else (np.cos(phi), np.sin(phi)))
  grid = GridSpec3D((2, 2, round(80/h)), (h, h, h), .4*h)
  simulation = Simulation3D(grid, BoundarySpec3D(round(8/h), bloch_wavevector=kxy),
      backend=args.backend, dtype=np.float64 if args.backend == "numpy" else np.float32)
  simulation.add_material("film", Lossless(2.25))
  simulation.add_geometry(Layer3D(40, 44, "film"))
  time = np.arange(round(140/grid.dt))*grid.dt
  waveform = np.exp(-.5*((time-30)/8)**2)*np.sin(2*np.pi*f*(time-30))
  simulation.set_source(PlaneWaveSource3D(waveform, round(16/h),
      polarization_xy=polarization, reference_frequency=f))
  simulation.set_monitors(ScatteringMonitor3D(round(25/h), round(60/h)))
  result = simulation.run()
  spectrum = result.spectrum
  valid = np.flatnonzero(spectrum.valid)
  if not valid.size:
    raise RuntimeError("No propagating frequency passed the incident threshold.")
  index = valid[np.argmin(abs(spectrum.frequencies[valid]-f))]
  omega = 2*np.pi*spectrum.frequencies[index]
  kz1, kz2 = np.sqrt(omega**2-kt**2), np.sqrt(2.25*omega**2-kt**2)
  y1, y2 = ((kz1, kz2) if args.polarization == "te" else (1/kz1, 2.25/kz2))
  r = (y1-y2)/(y1+y2)
  phase = np.exp(2j*kz2*4)
  fresnel = abs(r*(1-phase)/(1-r*r*phase))**2
  print(f"backend/device: {result.backend} / {result.device}")
  print(f"fixed Bloch kx,ky (radians/length): {kxy}")
  print(f"reference frequency/angle: {f:.6f} / {args.angle:.3f} degrees")
  print(f"reported frequency/angle: {spectrum.frequencies[index]:.6f} / "
        f"{np.rad2deg(np.arcsin(kt/omega)):.3f} degrees")
  print(f"R/T/A: {spectrum.reflection[index]:.6f} / "
        f"{spectrum.transmission[index]:.6f} / {spectrum.absorption[index]:.6f}")
  print(f"Fresnel film R: {fresnel:.6f}; absolute R error: "
        f"{abs(spectrum.reflection[index]-fresnel):.6e}")
  print("This is a finite-grid convergence example; A includes numerical error.")
  if args.output is not None:
    result.to_npz(args.output)


if __name__ == "__main__":
  main()
