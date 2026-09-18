"""Convert eV Drude-Lorentz fit parameters into normalized solver units."""

from pathlib import Path
import sys


_SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_DIRECTORY) not in sys.path:
  sys.path.insert(0, str(_SOURCE_DIRECTORY))

import numpy as np

from fdtdz_jax import Multipole
from fdtdz_jax import PhysicalScale
from fdtdz_jax import cells_per_wavelength
from fdtdz_jax import material_accuracy_report


# One grid length unit represents 10 nm.  If dx=1 below, the physical grid
# spacing is therefore 10 nm and one normalized time unit is L0/c0.
scale = PhysicalScale(length_unit_m=10e-9)

# Example fit parameters only: replace them with values from the chosen metal
# fit and its stated convention.  Lorentz damping here is delta; if a source
# quotes a full gamma in P'' + gamma P', pass gamma/2 to lorentz_pole_from_ev.
material = Multipole(1.0, (
    scale.drude_pole_from_ev(
        plasma_energy_ev=8.9, collision_energy_ev=0.07),
    scale.lorentz_pole_from_ev(
        delta_eps=0.8, resonance_energy_ev=3.2,
        damping_delta_energy_ev=0.25),
))

wavelengths_m = np.linspace(450e-9, 900e-9, 200)
frequencies_hz = 299_792_458.0 / wavelengths_m
normalized_omega = 2.0 * np.pi * scale.frequency(frequencies_hz)
dt = 0.45
accuracy = material_accuracy_report(material, dt, normalized_omega)

print("Length unit (m):", scale.length_unit_m)
print("Time unit (s):", scale.time_unit_s)
print("Normalized wavelength band:",
      scale.wavelength(wavelengths_m[0]), "to",
      scale.wavelength(wavelengths_m[-1]))
print("Cells per shortest vacuum wavelength at dx=1:",
      cells_per_wavelength(scale.wavelength(wavelengths_m[0]), dx=1.0))
print("Maximum omega*dt:", accuracy.maximum_omega_dt)
print("Below temporal Nyquist:", accuracy.below_nyquist)
print("Maximum ADE relative permittivity error:",
      accuracy.maximum_relative_error)
