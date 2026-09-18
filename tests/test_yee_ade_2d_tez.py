import numpy as np

from fdtdz_jax.ade_reference import simulate_ade
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.multipole_reference import simulate_multipole_ade
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d
from fdtdz_jax.yee_ade_2d import prepare_y_cpml
from fdtdz_jax.yee_ade_2d_tez import YeeADE2DTEzState
from fdtdz_jax.yee_ade_2d_tez import initialize_yee_ade_2d_tez
from fdtdz_jax.yee_ade_2d_tez import simulate_yee_ade_2d_tez
from fdtdz_jax.yee_ade_2d_tez import yee_ade_2d_tez_step
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse
from fdtdz_jax.metasurface_2d import huygens_tez_line_source


def test_zero_mixed_tez_state_stays_zero():
  ids = np.arange(24).reshape(4, 6) % 2
  result = simulate_yee_ade_2d_tez(
      [Lossless(1.0), Debye(1.0, 2.0, 0.4)], ids,
      1.0, 1.0, 0.2, 10)
  assert not np.any(result.electric_x)
  assert not np.any(result.electric_y)
  assert not np.any(result.magnetic_z)


def test_lossless_tez_step_matches_explicit_equations():
  shape = (3, 4)
  grid = prepare_material_grid_2d(
      [Lossless(2.0)], np.zeros(shape, int), 0.1, 0.5, 0.25)
  hz = np.arange(12, dtype=float).reshape(shape) / 9.0
  state = initialize_yee_ade_2d_tez(grid)
  state = YeeADE2DTEzState(**{
      **state.__dict__, "magnetic_z": hz,
      "displacement_x": np.ones(shape),
      "displacement_y": -np.ones(shape)})
  actual = yee_ade_2d_tez_step(grid, state)
  expected_dx = 1.0 + 0.4 * (hz - np.roll(hz, 1, axis=1))
  expected_dy = -1.0 - 0.2 * (hz - np.roll(hz, 1, axis=0))
  expected_ex = expected_dx / 2.0
  expected_ey = expected_dy / 2.0
  expected_hz = hz + 0.4 * (
      np.roll(expected_ex, -1, axis=1) - expected_ex)
  expected_hz -= 0.2 * (np.roll(expected_ey, -1, axis=0) - expected_ey)
  np.testing.assert_allclose(actual.electric_x, expected_ex)
  np.testing.assert_allclose(actual.electric_y, expected_ey)
  np.testing.assert_allclose(actual.magnetic_z, expected_hz)


def test_uniform_tez_source_matches_scalar_direct_ade():
  shape = (3, 4)
  material = Debye(1.5, 3.0, 0.5)
  num_steps = 300
  dt = 0.01
  displacement = np.sin(np.arange(num_steps) * dt)
  previous = np.concatenate(([0.0], displacement[:-1]))
  source = -(displacement - previous) / dt
  result = simulate_yee_ade_2d_tez(
      [material], np.zeros(shape, int), 1.0, 1.0, dt, num_steps,
      current_density_x=source)
  np.testing.assert_allclose(
      result.electric_x[:, 0, 0], simulate_ade(material, dt, displacement))
  assert not np.any(result.electric_y)


def test_uniform_tez_source_matches_scalar_multipole_ade():
  shape = (3, 4)
  material = Multipole(1.2, (
      LorentzPole(1.0, 3.0, 0.2), DrudePole(2.0, 0.3)))
  num_steps = 300
  dt = 0.01
  displacement = np.sin(np.arange(num_steps) * dt)
  previous = np.concatenate(([0.0], displacement[:-1]))
  source = -(displacement - previous) / dt
  result = simulate_yee_ade_2d_tez(
      [material], np.zeros(shape, int), 1.0, 1.0, dt, num_steps,
      current_density_y=source)
  np.testing.assert_allclose(
      result.electric_y[:, 0, 0],
      simulate_multipole_ade(material, dt, displacement))
  assert result.final_state.polarization_y_current.shape == shape + (2,)


def test_tez_periodic_wave_remains_finite():
  shape = (24, 20)
  grid = prepare_material_grid_2d(
      [Lossless(1.0)], np.zeros(shape, int), 0.4, 1.0, 1.0)
  state = initialize_yee_ade_2d_tez(grid)
  ex = np.sin(2 * np.pi * np.arange(shape[1])[None, :] / shape[1])
  ex = np.repeat(ex, shape[0], axis=0)
  state = YeeADE2DTEzState(**{
      **state.__dict__, "electric_x": ex,
      "displacement_x": ex.copy()})
  for _ in range(1000):
    state = yee_ade_2d_tez_step(grid, state)
  assert np.all(np.isfinite(state.electric_x))
  assert np.max(np.abs(state.electric_x)) < 1.2


def test_tez_y_cpml_suppresses_late_echo():
  shape = (3, 180)
  num_steps = 520
  dt = 0.5
  time = np.arange(num_steps) * dt
  waveform = np.exp(-0.5 * ((time - 25.0) / 7.0)**2) * np.sin(
      2.0 * np.pi * 0.05 * (time - 25.0))
  source = np.zeros((num_steps,) + shape)
  source[:, :, 90] = waveform[:, None]
  ids = np.zeros(shape, dtype=int)
  periodic = simulate_yee_ade_2d_tez(
      [Lossless(1.0)], ids, 1.0, 1.0, dt, num_steps,
      current_density_x=source)
  cpml = prepare_y_cpml(shape, 24, dt, 1.0)
  absorbing = simulate_yee_ade_2d_tez(
      [Lossless(1.0)], ids, 1.0, 1.0, dt, num_steps,
      current_density_x=source, y_cpml=cpml)
  periodic_late = np.max(np.abs(periodic.electric_x[390:, :, 90]))
  absorbing_late = np.max(np.abs(absorbing.electric_x[390:, :, 90]))
  assert absorbing_late < 1e-3 * periodic_late


def test_tez_huygens_source_propagates_in_requested_direction():
  shape = (3, 180)
  num_steps = 260
  dt = 0.5
  waveform = gaussian_sine_pulse(
      num_steps, dt, center_time=22.0, width=6.0, frequency=0.05)
  source = huygens_tez_line_source(waveform, shape, 90, direction=1)
  assert source.magnetic_y_index == 89
  cpml = prepare_y_cpml(shape, 24, dt, 1.0)
  result = simulate_yee_ade_2d_tez(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 1.0, 1.0, dt,
      num_steps, current_density_x=source.current_density_x,
      magnetic_current_z=source.magnetic_current_z, y_cpml=cpml)
  positive_y_energy = np.sum(result.electric_x[:, :, 110]**2)
  negative_y_energy = np.sum(result.electric_x[:, :, 70]**2)
  assert positive_y_energy > 100.0 * negative_y_energy


def test_tez_huygens_negative_direction_uses_mirrored_staggering():
  shape = (3, 180)
  num_steps = 260
  dt = 0.5
  waveform = gaussian_sine_pulse(
      num_steps, dt, center_time=22.0, width=6.0, frequency=0.05)
  source = huygens_tez_line_source(waveform, shape, 90, direction=-1)
  assert source.magnetic_y_index == 90
  cpml = prepare_y_cpml(shape, 24, dt, 1.0)
  result = simulate_yee_ade_2d_tez(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 1.0, 1.0, dt,
      num_steps, current_density_x=source.current_density_x,
      magnetic_current_z=source.magnetic_current_z, y_cpml=cpml)
  positive_y_energy = np.sum(result.electric_x[:, :, 110]**2)
  negative_y_energy = np.sum(result.electric_x[:, :, 70]**2)
  assert negative_y_energy > 100.0 * positive_y_energy
