from types import SimpleNamespace

import numpy as np
import pytest

from fdtdz_jax.convergence_2d import cells_per_wavelength
from fdtdz_jax.convergence_2d import compare_spectra


def _spectrum(frequencies, offset=0.0, valid=None):
  frequencies = np.asarray(frequencies, dtype=float)
  if valid is None:
    valid = np.ones(frequencies.shape, dtype=bool)
  reflection = 0.2 + 0.1 * frequencies + offset
  transmission = 0.7 - 0.05 * frequencies - offset
  return SimpleNamespace(
      frequencies=frequencies, reflection=reflection,
      transmission=transmission, absorption=1.0-reflection-transmission,
      valid=np.asarray(valid))


def test_identical_spectra_are_converged():
  spectrum = _spectrum(np.linspace(0.0, 1.0, 11))
  report = compare_spectra(spectrum, spectrum, 0.2, 0.8)
  assert report.maximum_reflection_difference == pytest.approx(0.0)
  assert report.maximum_transmission_difference == pytest.approx(0.0)
  assert report.converged(1e-12)


def test_comparison_interpolates_candidate_and_reports_known_offset():
  baseline = _spectrum(np.linspace(0.0, 1.0, 11))
  candidate = _spectrum(np.linspace(0.0, 1.0, 21), offset=0.02)
  report = compare_spectra(baseline, candidate, 0.1, 0.9)
  assert report.maximum_reflection_difference == pytest.approx(0.02)
  assert report.maximum_transmission_difference == pytest.approx(0.02)
  assert report.maximum_absorption_difference == pytest.approx(0.0, abs=1e-14)
  assert not report.converged(0.01)
  assert report.converged(0.021)


def test_valid_masks_and_requested_band_are_combined():
  frequencies = np.linspace(0.0, 1.0, 6)
  baseline = _spectrum(frequencies, valid=[0, 1, 1, 1, 1, 0])
  candidate = _spectrum(frequencies, valid=[1, 1, 0, 1, 1, 1])
  report = compare_spectra(baseline, candidate, 0.2, 0.8)
  np.testing.assert_array_equal(
      report.valid, [False, True, False, True, True, False])


def test_cells_per_wavelength_includes_material_index():
  assert cells_per_wavelength(1.55, 0.01, 2.0) == pytest.approx(77.5)
