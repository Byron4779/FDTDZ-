"""Convergence comparisons for 2D scattering spectra."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SpectrumConvergenceReport:
  frequencies: np.ndarray
  valid: np.ndarray
  reflection_difference: np.ndarray
  transmission_difference: np.ndarray
  absorption_difference: np.ndarray
  maximum_reflection_difference: float
  maximum_transmission_difference: float
  maximum_absorption_difference: float
  rms_reflection_difference: float
  rms_transmission_difference: float
  rms_absorption_difference: float

  def converged(self, absolute_tolerance):
    """Return true when every maximum R/T/A difference is within tolerance."""
    tolerance = float(absolute_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
      raise ValueError("absolute_tolerance must be finite and non-negative.")
    return max(
        self.maximum_reflection_difference,
        self.maximum_transmission_difference,
        self.maximum_absorption_difference) <= tolerance


def compare_spectra(baseline, candidate, frequency_min=None,
                    frequency_max=None):
  """Compare two R/T/A spectrum objects on the baseline frequency grid.

  The candidate is linearly interpolated when its frequency grid differs.
  Both objects must expose ``frequencies``, ``reflection``, ``transmission``,
  ``absorption``, and ``valid`` attributes, as do both scattering spectrum
  classes in this package.
  """
  required = ("frequencies", "reflection", "transmission", "absorption",
              "valid")
  for name, spectrum in (("baseline", baseline), ("candidate", candidate)):
    if any(not hasattr(spectrum, attribute) for attribute in required):
      raise TypeError(f"{name} does not provide the required spectrum fields.")
  base_frequency = np.asarray(baseline.frequencies, dtype=float)
  candidate_frequency = np.asarray(candidate.frequencies, dtype=float)
  if (base_frequency.ndim != 1 or candidate_frequency.ndim != 1 or
      base_frequency.size < 2 or candidate_frequency.size < 2 or
      np.any(np.diff(base_frequency) <= 0.0) or
      np.any(np.diff(candidate_frequency) <= 0.0)):
    raise ValueError("spectrum frequency grids must be strictly increasing 1D arrays.")
  valid = np.asarray(baseline.valid, dtype=bool).copy()
  candidate_valid = np.asarray(candidate.valid, dtype=bool)
  candidate_valid_interpolated = np.interp(
      base_frequency, candidate_frequency, candidate_valid.astype(float),
      left=0.0, right=0.0) == 1.0
  valid &= candidate_valid_interpolated
  if frequency_min is not None:
    frequency_min = float(frequency_min)
    valid &= base_frequency >= frequency_min
  if frequency_max is not None:
    frequency_max = float(frequency_max)
    valid &= base_frequency <= frequency_max
  if not np.any(valid):
    raise ValueError("the spectra have no jointly valid samples in the requested band.")

  differences = []
  for field in ("reflection", "transmission", "absorption"):
    base_values = np.asarray(getattr(baseline, field), dtype=float)
    candidate_values = np.asarray(getattr(candidate, field), dtype=float)
    interpolated = np.interp(
        base_frequency, candidate_frequency, candidate_values,
        left=np.nan, right=np.nan)
    differences.append(interpolated - base_values)
  maxima = [float(np.max(np.abs(value[valid]))) for value in differences]
  rms = [float(np.sqrt(np.mean(value[valid]**2))) for value in differences]
  return SpectrumConvergenceReport(
      frequencies=base_frequency, valid=valid,
      reflection_difference=differences[0],
      transmission_difference=differences[1],
      absorption_difference=differences[2],
      maximum_reflection_difference=maxima[0],
      maximum_transmission_difference=maxima[1],
      maximum_absorption_difference=maxima[2],
      rms_reflection_difference=rms[0],
      rms_transmission_difference=rms[1],
      rms_absorption_difference=rms[2])


def cells_per_wavelength(wavelength, dx, refractive_index=1.0):
  """Return grid cells per material wavelength for a real lossless index."""
  wavelength, dx, refractive_index = (
      float(value) for value in (wavelength, dx, refractive_index))
  if not all(math.isfinite(value) and value > 0.0 for value in
             (wavelength, dx, refractive_index)):
    raise ValueError("wavelength, dx, and refractive_index must be positive.")
  return wavelength / (refractive_index * dx)


__all__ = [
    "SpectrumConvergenceReport", "cells_per_wavelength", "compare_spectra",
]
