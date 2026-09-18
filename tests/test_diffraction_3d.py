import math

import numpy as np
import pytest

from fdtdz_jax.diffraction_3d import diffraction_spectrum_3d


def _uniform_plane(waveform, nx, ny, component=0, amplitude=1.0):
  plane = np.zeros((waveform.size, 2, nx, ny))
  plane[:, component] = amplitude * waveform[:, None, None]
  return plane


def test_vector_zero_order_spectrum_recovers_known_power_ratios():
  num_steps = 128
  dt = 0.25
  frequency = 0.125
  time = np.arange(num_steps) * dt
  waveform = np.sin(2.0 * np.pi * frequency * time)
  reference_reflection = _uniform_plane(waveform, 4, 6)
  reference_transmission = _uniform_plane(waveform, 4, 6, amplitude=0.8)
  device_reflection = reference_reflection + _uniform_plane(
      waveform, 4, 6, amplitude=0.5)
  device_transmission = _uniform_plane(waveform, 4, 6, amplitude=0.4)
  spectrum = diffraction_spectrum_3d(
      reference_reflection, device_reflection,
      reference_transmission, device_transmission,
      dt, 1.0, 1.0, window=None)
  index = np.argmin(np.abs(spectrum.frequencies - frequency))

  assert spectrum.valid[index]
  assert spectrum.reflection[index] == pytest.approx(0.25)
  assert spectrum.transmission[index] == pytest.approx(0.25)
  assert spectrum.absorption[index] == pytest.approx(0.5)
  assert spectrum.reflection_orders[index, 0, 0] == pytest.approx(0.25)
  assert np.count_nonzero(spectrum.reflection_orders[index]) == 1


@pytest.mark.parametrize(
    "component,power_weight",
    [(0, 1.0 / math.sqrt(0.75)), (1, math.sqrt(0.75))])
def test_vector_power_uses_tm_and_te_weights_for_nonzero_orders(
    component, power_weight):
  num_steps = 128
  nx, ny = 8, 4
  dt = 0.25
  frequency = 0.25
  amplitude = 0.4
  time = np.arange(num_steps) * dt
  waveform = np.sin(2.0 * np.pi * frequency * time)
  reference = _uniform_plane(waveform, nx, ny)
  device_reflection = reference.copy()
  x_profile = np.cos(2.0 * np.pi * np.arange(nx) / nx)
  device_reflection[:, component] += (
      amplitude * waveform[:, None, None] * x_profile[None, :, None])
  spectrum = diffraction_spectrum_3d(
      reference, device_reflection, reference, reference,
      dt, 1.0, 1.0, window=None)
  index = np.argmin(np.abs(spectrum.frequencies - frequency))
  expected_each_order = power_weight * (amplitude / 2.0)**2

  assert spectrum.propagating[index, 1, 0]
  assert spectrum.propagating[index, -1, 0]
  assert spectrum.reflection_orders[index, 1, 0] == pytest.approx(
      expected_each_order)
  assert spectrum.reflection_orders[index, -1, 0] == pytest.approx(
      expected_each_order)
  assert spectrum.reflection[index] == pytest.approx(
      2.0 * expected_each_order)
  assert spectrum.transmission[index] == pytest.approx(1.0)


def test_orders_outside_light_cone_are_excluded_from_power():
  num_steps = 128
  nx, ny = 8, 4
  dt = 0.25
  frequency = 0.0625
  time = np.arange(num_steps) * dt
  waveform = np.sin(2.0 * np.pi * frequency * time)
  reference = _uniform_plane(waveform, nx, ny)
  device = reference.copy()
  device[:, 1] += (
      waveform[:, None, None] *
      np.cos(2.0 * np.pi * np.arange(nx) / nx)[None, :, None])
  spectrum = diffraction_spectrum_3d(
      reference, device, reference, reference, dt, 1.0, 1.0)
  index = np.argmin(np.abs(spectrum.frequencies - frequency))

  assert not spectrum.propagating[index, 1, 0]
  assert spectrum.reflection_orders[index, 1, 0] == 0.0
  assert spectrum.reflection[index] == pytest.approx(0.0, abs=1e-14)


def test_diffraction_spectrum_validates_vector_plane_shape():
  valid = np.zeros((8, 2, 3, 4))
  invalid = np.zeros((8, 3, 4))
  with pytest.raises(ValueError, match="time, 2, nx, ny"):
    diffraction_spectrum_3d(invalid, valid, valid, valid, 0.1, 1.0, 1.0)
