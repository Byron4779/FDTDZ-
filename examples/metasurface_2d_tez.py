"""Small non-notebook TEz dispersive metasurface workflow."""

from pathlib import Path
import sys


_SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_DIRECTORY) not in sys.path:
  sys.path.insert(0, str(_SOURCE_DIRECTORY))

import numpy as np

from fdtdz_jax import DrudePole
from fdtdz_jax import LorentzPole
from fdtdz_jax import Lossless
from fdtdz_jax import Multipole
from fdtdz_jax import gaussian_sine_pulse
from fdtdz_jax import jax_runtime_report
from fdtdz_jax import rectangle_mask
from fdtdz_jax import run_tez_scattering_2d


shape = (48, 180)
dt = 0.45
num_steps = 450
materials = [
    Lossless(1.0),
    Multipole(1.0, (
        DrudePole(plasma_frequency=0.55, collision_frequency=0.025),
        LorentzPole(delta_eps=0.8, omega_0=0.42, damping=0.025),
    )),
]
reference_ids = np.zeros(shape, dtype=np.uint8)
device_ids = reference_ids.copy()
device_ids[rectangle_mask(shape, 12, 36, 88, 94)] = 1
waveform = gaussian_sine_pulse(
    num_steps, dt, center_time=35.0, width=12.0, frequency=0.06,
    dtype=np.float32)

print("JAX runtime:", jax_runtime_report())
result = run_tez_scattering_2d(
    materials, reference_ids, device_ids, 1.0, 1.0, dt, waveform,
    source_y=45, reflection_probe_y=70, transmission_probe_y=115,
    cpml_width=24, backend="jax")
valid = result.spectrum.valid & (result.spectrum.frequencies > 0.0)
peak = np.argmax(np.where(
    valid, np.abs(np.fft.rfft(result.reference_transmission)), 0.0))
print("Execution device:", result.device)
print("Frequency:", result.spectrum.frequencies[peak])
print("R, T, A:", result.spectrum.reflection[peak],
      result.spectrum.transmission[peak], result.spectrum.absorption[peak])
print("Total diffraction R, T, A:", result.diffraction.reflection[peak],
      result.diffraction.transmission[peak], result.diffraction.absorption[peak])
