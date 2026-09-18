"""Validate and benchmark the 3D dispersive JAX backend on the active device."""

import argparse
from pathlib import Path
import sys
import time


_SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_DIRECTORY) not in sys.path:
  sys.path.insert(0, str(_SOURCE_DIRECTORY))

import numpy as np
import jax

from fdtdz_jax import DrudePole
from fdtdz_jax import LorentzPole
from fdtdz_jax import Lossless
from fdtdz_jax import Multipole
from fdtdz_jax import huygens_plane_source
from fdtdz_jax import initialize_yee_ade_3d
from fdtdz_jax import jax_material_grid_3d
from fdtdz_jax import jax_runtime_report
from fdtdz_jax import jax_yee_ade_3d_state
from fdtdz_jax import prepare_material_grid_3d
from fdtdz_jax import prepare_z_cpml
from fdtdz_jax import simulate_jax_yee_ade_3d_plane_probes
from fdtdz_jax import simulate_yee_ade_3d_plane_probes


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--allow-cpu", action="store_true",
      help="allow CPU JAX for a functional comparison")
  parser.add_argument("--nx", type=int, default=12)
  parser.add_argument("--ny", type=int, default=12)
  parser.add_argument("--nz", type=int, default=80)
  parser.add_argument("--steps", type=int, default=160)
  parser.add_argument("--bloch-wavevector", type=float, nargs=2, default=(0., 0.),
                      metavar=("KX", "KY"), help="Normalized radians/length")
  args = parser.parse_args()

  report = jax_runtime_report()
  print("JAX runtime:", report)
  if not report["has_gpu"] and not args.allow_cpu:
    raise SystemExit(
        "GPU validation stopped: this Python environment does not expose a "
        "JAX GPU. Use the GPU environment or pass --allow-cpu for a functional "
        "comparison.")
  if args.nx < 4 or args.ny < 4 or args.nz < 64 or args.steps < 50:
    raise SystemExit("Use nx/ny >= 4, nz >= 64, and steps >= 50.")

  shape = (args.nx, args.ny, args.nz)
  dt = 0.4
  cpml_width = max(10, args.nz // 10)
  source_z = cpml_width + 6
  reflection_z = source_z + 10
  transmission_z = args.nz - cpml_width - 10
  if reflection_z >= args.nz // 2 or transmission_z <= args.nz // 2:
    raise SystemExit("nz is too small for the source, structure and probes.")
  materials = [
      Lossless(1.0),
      Multipole(1.0, (
          DrudePole(0.65, 0.035), LorentzPole(0.6, 0.5, 0.03))),
  ]
  material_ids = np.zeros(shape, dtype=np.uint8)
  material_ids[
      args.nx // 4:3 * args.nx // 4,
      args.ny // 4:3 * args.ny // 4,
      args.nz // 2 - 2:args.nz // 2 + 2] = 1
  cpml = prepare_z_cpml(
      shape, cpml_width, dt, 1.0, dtype=np.float32)
  grid = prepare_material_grid_3d(
      materials, material_ids, dt, 1.0, 1.0, 1.0,
      dtype=np.float32, z_cpml=cpml, bloch_wavevector=args.bloch_wavevector)
  time_axis = np.arange(args.steps, dtype=np.float32) * dt
  waveform = (np.exp(-0.5 * ((time_axis - 16.0) / 5.0)**2) *
              np.sin(2.0 * np.pi * 0.065 * (time_axis - 16.0))).astype(
                  np.float32)
  source = huygens_plane_source(
      waveform, shape, source_z, polarization_xy=(1.0, 0.0),
      direction=1, dtype=np.float32, expand=True,
      bloch_wavevector=args.bloch_wavevector, reference_frequency=.065)

  numpy_start = time.perf_counter()
  numpy_result = simulate_yee_ade_3d_plane_probes(
      materials, material_ids, 1.0, 1.0, 1.0, dt, args.steps,
      (reflection_z, transmission_z),
      electric_current=source.electric_current,
      magnetic_current=source.magnetic_current,
      dtype=np.float32, z_cpml=cpml, bloch_wavevector=args.bloch_wavevector)
  numpy_seconds = time.perf_counter() - numpy_start

  jax_grid = jax_material_grid_3d(grid)
  jax_initial = jax_yee_ade_3d_state(initialize_yee_ade_3d(grid), grid)
  call = lambda: simulate_jax_yee_ade_3d_plane_probes(
      jax_grid, jax_initial, source.electric_waveform,
      source.magnetic_waveform, source.transverse_profile,
      np.asarray(source.polarization_xy, dtype=np.float32),
      source.electric_z_index, source.magnetic_z_index,
      reflection_z, transmission_z)
  compile_start = time.perf_counter()
  jax.block_until_ready(call())
  compile_seconds = time.perf_counter() - compile_start
  execute_start = time.perf_counter()
  jax_state, jax_probes = call()
  jax.block_until_ready((jax_state, jax_probes))
  execute_seconds = time.perf_counter() - execute_start

  comparisons = {
      "final E": (np.asarray(jax_state.electric),
                  numpy_result.final_state.electric),
      "final D": (np.asarray(jax_state.displacement),
                  numpy_result.final_state.displacement),
      "reflection plane": (np.asarray(jax_probes[0]),
                           numpy_result.electric_tangential[:, 0]),
      "transmission plane": (np.asarray(jax_probes[1]),
                             numpy_result.electric_tangential[:, 1]),
  }
  for name, (actual, expected) in comparisons.items():
    error = float(np.max(np.abs(actual - expected)))
    scale = max(float(np.max(np.abs(expected))), 1e-12)
    print(f"{name}: max_abs={error:.6e}, relative_to_peak={error / scale:.6e}")
    try:
      np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=5e-5)
    except AssertionError as failure:
      raise SystemExit(f"Numerical comparison failed for {name}: {failure}") from failure

  updates = args.nx * args.ny * args.nz * args.steps
  print(f"NumPy reference: {numpy_seconds:.6f} s")
  print(f"JAX compile + first run: {compile_seconds:.6f} s")
  print(f"JAX cached run: {execute_seconds:.6f} s")
  print(f"JAX throughput: {updates / execute_seconds / 1e6:.3f} Mcell-updates/s")
  print("3D GPU/JAX validation passed." if report["has_gpu"] else
        "3D JAX CPU functional validation passed.")


if __name__ == "__main__":
  main()
