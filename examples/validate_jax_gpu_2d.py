"""Validate and benchmark the 2D TMz JAX backend on the active GPU.

Run directly from any directory.  By default the script fails when JAX is not
using a GPU; pass ``--allow-cpu`` only for functional development checks.
"""

import argparse
from pathlib import Path
import sys
import time


_SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_DIRECTORY) not in sys.path:
  sys.path.insert(0, str(_SOURCE_DIRECTORY))

import numpy as np

from fdtdz_jax import DrudePole
from fdtdz_jax import LorentzPole
from fdtdz_jax import Lossless
from fdtdz_jax import Multipole
from fdtdz_jax import huygens_current_waveforms
from fdtdz_jax import huygens_line_source
from fdtdz_jax import initialize_yee_ade_2d
from fdtdz_jax import jax_material_grid_2d
from fdtdz_jax import jax_runtime_report
from fdtdz_jax import jax_yee_ade_2d_state
from fdtdz_jax import prepare_material_grid_2d
from fdtdz_jax import prepare_y_cpml
from fdtdz_jax import sample_line
from fdtdz_jax import simulate_jax_yee_ade_2d_probes
from fdtdz_jax import simulate_yee_ade_2d


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--allow-cpu", action="store_true",
      help="allow a CPU JAX backend for functional validation")
  parser.add_argument("--nx", type=int, default=48)
  parser.add_argument("--ny", type=int, default=160)
  parser.add_argument("--steps", type=int, default=300)
  args = parser.parse_args()

  report = jax_runtime_report()
  print("JAX runtime:", report)
  if not report["has_gpu"] and not args.allow_cpu:
    raise SystemExit(
        "GPU validation stopped: JAX does not expose a GPU in this Python "
        "environment. Check that this command uses the same virtual "
        "environment in which jax.devices() reports a GPU, or rerun with "
        "--allow-cpu for a functional CPU-only check.")

  if args.nx < 8 or args.ny < 80 or args.steps < 50:
    raise SystemExit("Use nx >= 8, ny >= 80, and steps >= 50.")
  shape = (args.nx, args.ny)
  dt = 0.4
  cpml_width = max(12, args.ny // 10)
  source_y = cpml_width + 8
  reflection_y = source_y + 15
  transmission_y = args.ny - cpml_width - 18
  structure_start = args.ny // 2 - 4
  structure_stop = args.ny // 2 + 4
  materials = [
      Lossless(1.0),
      Multipole(1.0, (
          DrudePole(0.65, 0.035),
          LorentzPole(0.6, 0.5, 0.03),
      )),
  ]
  ids = np.zeros(shape, dtype=np.uint8)
  ids[args.nx // 4:3 * args.nx // 4,
      structure_start:structure_stop] = 1
  cpml = prepare_y_cpml(
      shape, cpml_width, dt, 1.0, dtype=np.float32)
  grid = prepare_material_grid_2d(
      materials, ids, dt, 1.0, 1.0, dtype=np.float32, y_cpml=cpml)

  time_axis = np.arange(args.steps, dtype=np.float32) * dt
  waveform = (np.exp(-0.5 * ((time_axis - 25.0) / 8.0)**2) *
              np.sin(2.0 * np.pi * 0.065 * (time_axis - 25.0))).astype(
                  np.float32)
  expanded_source = huygens_line_source(
      waveform, shape, source_y, direction=1, dtype=np.float32)
  compact_electric, compact_magnetic = huygens_current_waveforms(
      waveform, direction=1, dtype=np.float32)

  numpy_start = time.perf_counter()
  numpy_result = simulate_yee_ade_2d(
      materials, ids, 1.0, 1.0, dt, args.steps,
      current_density_z=expanded_source.current_density_z,
      magnetic_current_x=expanded_source.magnetic_current_x,
      dtype=np.float32, y_cpml=cpml)
  numpy_seconds = time.perf_counter() - numpy_start

  jax_grid = jax_material_grid_2d(grid)
  jax_initial = jax_yee_ade_2d_state(initialize_yee_ade_2d(grid), grid)
  x_profile = np.ones(args.nx, dtype=np.float32)
  compile_start = time.perf_counter()
  compiled_state, compiled_probes = simulate_jax_yee_ade_2d_probes(
      jax_grid, jax_initial, compact_electric, compact_magnetic, x_profile,
      source_y, reflection_y, transmission_y)
  compiled_state.electric_z.block_until_ready()
  compile_seconds = time.perf_counter() - compile_start
  execute_start = time.perf_counter()
  jax_state, jax_probes = simulate_jax_yee_ade_2d_probes(
      jax_grid, jax_initial, compact_electric, compact_magnetic, x_profile,
      source_y, reflection_y, transmission_y)
  jax_state.electric_z.block_until_ready()
  execute_seconds = time.perf_counter() - execute_start

  numpy_reflection = sample_line(
      numpy_result.electric_z, reflection_y, average_x=True)
  numpy_transmission = sample_line(
      numpy_result.electric_z, transmission_y, average_x=True)
  comparisons = {
      "final Ez": (np.asarray(jax_state.electric_z),
                   numpy_result.final_state.electric_z),
      "reflection probe": (np.asarray(jax_probes[0]), numpy_reflection),
      "transmission probe": (np.asarray(jax_probes[1]), numpy_transmission),
  }
  maximum_error = 0.0
  for name, (actual, expected) in comparisons.items():
    error = float(np.max(np.abs(actual - expected)))
    scale = max(float(np.max(np.abs(expected))), 1e-12)
    relative = error / scale
    maximum_error = max(maximum_error, error)
    print(f"{name}: max_abs={error:.6e}, relative_to_peak={relative:.6e}")
  if maximum_error > 2e-5:
    raise SystemExit(
        f"Numerical comparison failed: maximum absolute error {maximum_error}.")

  updates = args.nx * args.ny * args.steps
  print(f"NumPy reference: {numpy_seconds:.6f} s")
  print(f"JAX compile + first run: {compile_seconds:.6f} s")
  print(f"JAX cached run: {execute_seconds:.6f} s")
  print(f"JAX throughput: {updates / execute_seconds / 1e6:.3f} Mcell-updates/s")
  print("GPU/JAX validation passed." if report["has_gpu"] else
        "JAX CPU functional validation passed.")


if __name__ == "__main__":
  main()
