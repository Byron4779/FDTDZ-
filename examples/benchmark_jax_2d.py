"""Warm up and time the JAX TMz backend on the active device.

The repository's local ``src`` directory is added automatically, so this file
also works before an editable package installation.
"""

from pathlib import Path
import sys
import time


_SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_DIRECTORY) not in sys.path:
  sys.path.insert(0, str(_SOURCE_DIRECTORY))

import numpy as np

from fdtdz_jax import DrudePole
from fdtdz_jax import Lossless
from fdtdz_jax import Multipole
from fdtdz_jax import huygens_current_waveforms
from fdtdz_jax import initialize_yee_ade_2d
from fdtdz_jax import jax_material_grid_2d
from fdtdz_jax import jax_runtime_report
from fdtdz_jax import jax_yee_ade_2d_state
from fdtdz_jax import prepare_material_grid_2d
from fdtdz_jax import prepare_y_cpml
from fdtdz_jax import simulate_jax_yee_ade_2d_line_probes


shape = (128, 256)
num_steps = 500
dt = 0.45
ids = np.zeros(shape, dtype=np.uint8)
ids[48:80, 120:136] = 1
materials = [
    Lossless(1.0),
    Multipole(1.0, (DrudePole(0.55, 0.025),)),
]
cpml = prepare_y_cpml(shape, 32, dt, 1.0, dtype=np.float32)
grid = prepare_material_grid_2d(
    materials, ids, dt, 1.0, 1.0, dtype=np.float32, y_cpml=cpml)
state = jax_yee_ade_2d_state(initialize_yee_ade_2d(grid), grid)
jax_grid = jax_material_grid_2d(grid)
time_axis = np.arange(num_steps, dtype=np.float32) * dt
waveform = (np.exp(-0.5 * ((time_axis - 25.0) / 8.0)**2) *
            np.sin(2.0 * np.pi * 0.06 * (time_axis - 25.0))).astype(np.float32)
electric_waveform, magnetic_waveform = huygens_current_waveforms(
    waveform, direction=1, dtype=np.float32)
x_profile = np.ones(shape[0], dtype=np.float32)
source_y = 60
reflection_probe_y = 90
transmission_probe_y = 190

print("Runtime:", jax_runtime_report())
print("Recorded probe memory:",
      f"{2 * num_steps * shape[0] * np.dtype(np.float32).itemsize / 2**20:.3f} MiB")
print("Equivalent four-field history memory:",
      f"{4 * num_steps * shape[0] * shape[1] * np.dtype(np.float32).itemsize / 2**20:.3f} MiB")
compiled = simulate_jax_yee_ade_2d_line_probes(
    jax_grid, state, electric_waveform, magnetic_waveform, x_profile,
    source_y, reflection_probe_y, transmission_probe_y)
compiled[0].electric_z.block_until_ready()
start = time.perf_counter()
result = simulate_jax_yee_ade_2d_line_probes(
    jax_grid, state, electric_waveform, magnetic_waveform, x_profile,
    source_y, reflection_probe_y, transmission_probe_y)
result[0].electric_z.block_until_ready()
elapsed = time.perf_counter() - start
cell_updates = shape[0] * shape[1] * num_steps
print(f"Elapsed: {elapsed:.6f} s")
print(f"Throughput: {cell_updates / elapsed / 1e6:.3f} Mcell-updates/s")
