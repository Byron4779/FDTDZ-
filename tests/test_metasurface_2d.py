import numpy as np
import pytest

from fdtdz_jax.metasurface_2d import LineProbe2D
from fdtdz_jax.metasurface_2d import circle_mask
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse
from fdtdz_jax.metasurface_2d import graded_y_sponge
from fdtdz_jax.metasurface_2d import huygens_line_source
from fdtdz_jax.metasurface_2d import huygens_current_waveforms
from fdtdz_jax.metasurface_2d import line_current_source
from fdtdz_jax.metasurface_2d import paint_material
from fdtdz_jax.metasurface_2d import rectangle_mask
from fdtdz_jax.metasurface_2d import sample_line
from fdtdz_jax.metasurface_2d import scattering_spectrum


def test_rectangle_mask_is_half_open_and_paint_is_non_mutating():
  mask = rectangle_mask((8, 10), 2, 6, 3, 8)
  assert mask.sum() == 20
  ids = np.zeros((8, 10), dtype=np.uint8)
  painted = paint_material(ids, mask, 2)
  assert not np.any(ids)
  np.testing.assert_array_equal(painted == 2, mask)


def test_periodic_circle_wraps_across_x_boundary():
  mask = circle_mask((10, 12), center=(0.0, 5.0), radius=1.1)
  assert mask[0, 5]
  assert mask[1, 5]
  assert mask[9, 5]
  assert not mask[8, 5]
  nonperiodic = circle_mask(
      (10, 12), center=(0.0, 5.0), radius=1.1, periodic_x=False)
  assert not nonperiodic[9, 5]


def test_gaussian_sine_pulse_has_requested_shape_and_dtype():
  pulse = gaussian_sine_pulse(
      101, 0.01, center_time=0.5, width=0.1, frequency=2.0,
      amplitude=3.0, phase=np.pi / 2, dtype=np.float32)
  assert pulse.shape == (101,)
  assert pulse.dtype == np.float32
  assert pulse[50] == pytest.approx(3.0)
  assert abs(pulse[0]) < 2e-5


def test_line_source_supports_uniform_and_shaped_profiles():
  waveform = np.asarray([1.0, -2.0])
  uniform = line_current_source(waveform, (4, 6), y_index=2)
  np.testing.assert_allclose(
      uniform[:, :, 2], np.repeat(waveform[:, None], 4, axis=1))
  assert np.count_nonzero(uniform[:, :, [0, 1, 3, 4, 5]]) == 0

  profile = np.asarray([1.0, 0.5, 0.0, -1.0])
  shaped = line_current_source(waveform, (4, 6), 3, profile)
  np.testing.assert_allclose(shaped[:, :, 3], waveform[:, None] * profile)


def test_huygens_source_builds_staggered_paired_lines():
  source = huygens_line_source(
      np.asarray([2.0, 4.0, 6.0]), (3, 8), 4,
      direction=1, wave_impedance=2.0)
  np.testing.assert_allclose(source.current_density_z[:, :, 4],
                             [[2, 2, 2], [4, 4, 4], [6, 6, 6]])
  np.testing.assert_allclose(source.magnetic_current_x[:, :, 4],
                             [[2, 2, 2], [6, 6, 6], [10, 10, 10]])
  assert np.count_nonzero(source.current_density_z[:, :, :4]) == 0


def test_compact_huygens_waveforms_match_expanded_source():
  waveform = np.asarray([2.0, 4.0, 6.0])
  electric, magnetic = huygens_current_waveforms(
      waveform, direction=-1, wave_impedance=2.0)
  source = huygens_line_source(
      waveform, (3, 8), 4, direction=-1, wave_impedance=2.0)
  np.testing.assert_allclose(source.current_density_z[:, 0, 4], electric)
  np.testing.assert_allclose(source.magnetic_current_x[:, 0, 4], magnetic)


def test_line_probe_extracts_or_averages_x_samples():
  fields = np.arange(3 * 4 * 5).reshape(3, 4, 5)
  np.testing.assert_array_equal(sample_line(fields, 2), fields[:, :, 2])
  probe = LineProbe2D(2, average_x=True, name="reflection")
  np.testing.assert_allclose(probe.sample(fields), fields[:, :, 2].mean(axis=1))


def test_graded_y_sponge_is_symmetric_and_x_uniform():
  sponge = graded_y_sponge((4, 20), width=5, maximum_conductivity=1.5, order=3)
  sigma = sponge.electric_conductivity
  assert sigma.shape == (4, 20)
  np.testing.assert_array_equal(sigma, sponge.magnetic_conductivity)
  np.testing.assert_array_equal(sigma, np.repeat(sigma[:1], 4, axis=0))
  np.testing.assert_allclose(sigma[:, :5], sigma[:, -1:-6:-1])
  assert np.all(sigma[:, 5:15] == 0.0)
  assert sigma[0, 0] == pytest.approx(1.5)


def test_scattering_spectrum_recovers_known_field_ratios():
  num_steps = 128
  dt = 0.25
  time = np.arange(num_steps) * dt
  incident = np.sin(2.0 * np.pi * 0.125 * time)
  transmitted_reference = 0.8 * incident
  device_reflection = incident + 0.5 * incident
  device_transmission = 0.4 * incident
  spectrum = scattering_spectrum(
      incident, device_reflection, transmitted_reference,
      device_transmission, dt, window=None)
  index = np.argmin(np.abs(spectrum.frequencies - 0.125))
  assert spectrum.valid[index]
  assert spectrum.reflection[index] == pytest.approx(0.25)
  assert spectrum.transmission[index] == pytest.approx(0.25)
  assert spectrum.absorption[index] == pytest.approx(0.5)


def test_identical_reference_and_device_have_unit_transmission():
  rng = np.random.default_rng(4)
  reflection_probe = rng.normal(size=64)
  transmission_probe = rng.normal(size=64)
  spectrum = scattering_spectrum(
      reflection_probe, reflection_probe, transmission_probe,
      transmission_probe, dt=0.1)
  np.testing.assert_allclose(spectrum.reflection[spectrum.valid], 0.0)
  np.testing.assert_allclose(spectrum.transmission[spectrum.valid], 1.0)
  np.testing.assert_allclose(spectrum.absorption[spectrum.valid], 0.0, atol=1e-14)


@pytest.mark.parametrize("function,args", [
    (rectangle_mask, ((4, 5), 2, 2, 0, 1)),
    (circle_mask, ((4, 5), (0, 0), -1)),
    (line_current_source, ([1], (4, 5), 5)),
    (sample_line, (np.zeros((2, 4, 5)), 5)),
])
def test_invalid_geometry_and_line_arguments_are_rejected(function, args):
  with pytest.raises((TypeError, ValueError)):
    function(*args)
