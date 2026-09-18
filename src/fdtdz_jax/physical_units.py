"""Conversions between SI/material-fit units and normalized FDTD units."""

from dataclasses import dataclass
import math

import numpy as np

from .materials import Debye
from .materials import DebyePole
from .materials import Drude
from .materials import DrudePole
from .materials import Lorentz
from .materials import LorentzPole


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
ELECTRON_VOLT_J = 1.602_176_634e-19
REDUCED_PLANCK_J_S = 1.054_571_817e-34


def _positive(name, value):
  value = float(value)
  if not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"{name} must be finite and positive.")
  return value


@dataclass(frozen=True)
class PhysicalScale:
  """Normalized-unit scale defined by one physical length unit.

  If one normalized spatial unit represents ``length_unit_m`` metres, one
  normalized time unit represents ``length_unit_m / c0`` seconds.
  """

  length_unit_m: float

  def __post_init__(self):
    object.__setattr__(self, "length_unit_m", _positive(
        "length_unit_m", self.length_unit_m))

  @property
  def time_unit_s(self):
    return self.length_unit_m / SPEED_OF_LIGHT_M_PER_S

  def length(self, metres):
    return np.asarray(metres) / self.length_unit_m

  def metres(self, normalized_length):
    return np.asarray(normalized_length) * self.length_unit_m

  def time(self, seconds):
    return np.asarray(seconds) / self.time_unit_s

  def seconds(self, normalized_time):
    return np.asarray(normalized_time) * self.time_unit_s

  def frequency(self, hertz):
    """Convert ordinary frequency f in Hz to normalized cycles/time."""
    return np.asarray(hertz) * self.time_unit_s

  def hertz(self, normalized_frequency):
    return np.asarray(normalized_frequency) / self.time_unit_s

  def angular_frequency(self, radians_per_second):
    return np.asarray(radians_per_second) * self.time_unit_s

  def radians_per_second(self, normalized_angular_frequency):
    return np.asarray(normalized_angular_frequency) / self.time_unit_s

  def angular_frequency_from_ev(self, energy_ev):
    """Convert a quoted hbar*omega energy in eV to normalized omega."""
    omega_si = np.asarray(energy_ev) * ELECTRON_VOLT_J / REDUCED_PLANCK_J_S
    return self.angular_frequency(omega_si)

  def wavelength(self, wavelength_m):
    return self.length(wavelength_m)

  def drude(self, eps_inf, plasma_angular_frequency_si,
            collision_angular_frequency_si):
    return Drude(
        eps_inf,
        float(self.angular_frequency(plasma_angular_frequency_si)),
        float(self.angular_frequency(collision_angular_frequency_si)))

  def drude_from_ev(self, eps_inf, plasma_energy_ev, collision_energy_ev):
    return Drude(
        eps_inf, float(self.angular_frequency_from_ev(plasma_energy_ev)),
        float(self.angular_frequency_from_ev(collision_energy_ev)))

  def lorentz(self, eps_inf, eps_static, resonance_angular_frequency_si,
              damping_delta_si):
    """Create Lorentz material; damping_delta is half the usual gamma width."""
    return Lorentz(
        eps_inf, eps_static,
        float(self.angular_frequency(resonance_angular_frequency_si)),
        float(self.angular_frequency(damping_delta_si)))

  def debye(self, eps_inf, eps_static, relaxation_time_s):
    return Debye(
        eps_inf, eps_static, float(self.time(relaxation_time_s)))

  def drude_pole_from_ev(self, plasma_energy_ev, collision_energy_ev):
    return DrudePole(
        float(self.angular_frequency_from_ev(plasma_energy_ev)),
        float(self.angular_frequency_from_ev(collision_energy_ev)))

  def lorentz_pole_from_ev(self, delta_eps, resonance_energy_ev,
                           damping_delta_energy_ev):
    return LorentzPole(
        delta_eps, float(self.angular_frequency_from_ev(resonance_energy_ev)),
        float(self.angular_frequency_from_ev(damping_delta_energy_ev)))

  def debye_pole(self, delta_eps, relaxation_time_s):
    return DebyePole(delta_eps, float(self.time(relaxation_time_s)))


__all__ = [
    "ELECTRON_VOLT_J", "PhysicalScale", "REDUCED_PLANCK_J_S",
    "SPEED_OF_LIGHT_M_PER_S",
]
