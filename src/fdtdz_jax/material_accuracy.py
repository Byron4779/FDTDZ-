"""Frequency-band accuracy checks for time-discrete ADE materials."""

from dataclasses import dataclass

import numpy as np

from .ade_reference import discrete_relative_permittivity
from .materials import Multipole
from .multipole_reference import discrete_multipole_permittivity


@dataclass(frozen=True)
class MaterialAccuracyReport:
  angular_frequencies: np.ndarray
  analytic_permittivity: np.ndarray
  discrete_permittivity: np.ndarray
  absolute_error: np.ndarray
  relative_error: np.ndarray
  maximum_absolute_error: float
  maximum_relative_error: float
  maximum_omega_dt: float
  below_nyquist: bool


def material_accuracy_report(material, dt, angular_frequencies,
                             relative_floor=1e-12):
  """Compare continuous material response with its exact discrete recurrence."""
  frequencies = np.asarray(angular_frequencies, dtype=float)
  if frequencies.ndim != 1 or frequencies.size == 0:
    raise ValueError("angular_frequencies must be a non-empty 1D array.")
  if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
    raise ValueError("angular_frequencies must be finite and non-negative.")
  dt = float(dt)
  relative_floor = float(relative_floor)
  if not np.isfinite(dt) or dt <= 0.0:
    raise ValueError("dt must be finite and positive.")
  if not np.isfinite(relative_floor) or relative_floor <= 0.0:
    raise ValueError("relative_floor must be finite and positive.")
  analytic = material.relative_permittivity(frequencies)
  if isinstance(material, Multipole):
    discrete = discrete_multipole_permittivity(material, dt, frequencies)
  else:
    discrete = discrete_relative_permittivity(material, dt, frequencies)
  absolute = np.abs(discrete - analytic)
  relative = absolute / np.maximum(np.abs(analytic), relative_floor)
  maximum_omega_dt = float(np.max(frequencies) * dt)
  return MaterialAccuracyReport(
      angular_frequencies=frequencies,
      analytic_permittivity=analytic,
      discrete_permittivity=discrete,
      absolute_error=absolute,
      relative_error=relative,
      maximum_absolute_error=float(np.max(absolute)),
      maximum_relative_error=float(np.max(relative)),
      maximum_omega_dt=maximum_omega_dt,
      below_nyquist=maximum_omega_dt < np.pi)


__all__ = ["MaterialAccuracyReport", "material_accuracy_report"]
