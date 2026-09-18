import numpy as np
import pytest

from fdtdz_jax.diffraction_2d import diffraction_spectrum


def _traveling_mode(num_steps, nx, dt, frequency, order, amplitude=1.0):
  time = np.arange(num_steps)[:, None] * dt
  x = np.arange(nx)[None, :]
  return amplitude * np.cos(
      2.0 * np.pi * frequency * time - 2.0 * np.pi * order * x / nx)


def test_empty_periodic_cell_has_unit_zero_order_transmission():
  incident = _traveling_mode(256, 16, 0.25, 0.125, 0)
  spectrum = diffraction_spectrum(
      incident, incident, incident, incident, 0.25, 1.0, window=None)
  index = np.argmin(np.abs(spectrum.frequencies - 0.125))
  zero = np.where(spectrum.orders == 0)[0][0]
  assert spectrum.valid[index]
  assert spectrum.reflection[index] == pytest.approx(0.0)
  assert spectrum.transmission[index] == pytest.approx(1.0)
  assert spectrum.transmission_orders[index, zero] == pytest.approx(1.0)
  assert spectrum.absorption[index] == pytest.approx(0.0)


@pytest.mark.parametrize("polarization", ["tmz", "tez"])
def test_single_propagating_diffraction_order_uses_polarization_weight(
    polarization):
  num_steps, nx, dt, frequency = 256, 16, 0.25, 0.25
  incident = _traveling_mode(num_steps, nx, dt, frequency, 0)
  diffracted = _traveling_mode(
      num_steps, nx, dt, frequency, order=1, amplitude=0.4)
  spectrum = diffraction_spectrum(
      incident, incident + diffracted, incident, 0.5 * incident,
      dt, dx=1.0, polarization=polarization, window=None)
  frequency_index = np.argmin(np.abs(spectrum.frequencies - frequency))
  populated = np.argmax(
      spectrum.reflection_orders[frequency_index])
  k = 2.0 * np.pi * frequency
  kx = abs(spectrum.transverse_wavevectors[populated])
  ky = np.sqrt(k * k - kx * kx)
  weight = ky / k if polarization == "tmz" else k / ky
  assert spectrum.reflection_orders[frequency_index, populated] == pytest.approx(
      weight * 0.4**2)
  assert spectrum.transmission[frequency_index] == pytest.approx(0.25)


def test_evanescent_order_has_zero_far_field_power():
  num_steps, nx, dt, frequency = 256, 16, 0.25, 0.0625
  incident = _traveling_mode(num_steps, nx, dt, frequency, 0)
  high_order = _traveling_mode(num_steps, nx, dt, frequency, 3, amplitude=0.7)
  spectrum = diffraction_spectrum(
      incident, incident + high_order, incident, incident,
      dt, dx=1.0, window=None)
  index = np.argmin(np.abs(spectrum.frequencies - frequency))
  assert spectrum.reflection[index] == pytest.approx(0.0, abs=1e-14)
  assert spectrum.transmission[index] == pytest.approx(1.0)
  assert spectrum.absorption[index] == pytest.approx(0.0, abs=1e-14)
