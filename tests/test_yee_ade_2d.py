import numpy as np
import pytest

from fdtdz_jax.ade_reference import simulate_ade
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import Lorentz
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.materials import DebyePole
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.multipole_reference import simulate_multipole_ade
from fdtdz_jax.yee_ade_2d import YeeADE2DState
from fdtdz_jax.yee_ade_2d import initialize_yee_ade_2d
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d
from fdtdz_jax.yee_ade_2d import prepare_y_cpml
from fdtdz_jax.yee_ade_2d import simulate_yee_ade_2d
from fdtdz_jax.yee_ade_2d import yee_ade_2d_step
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse
from fdtdz_jax.metasurface_2d import huygens_line_source


def test_material_grid_maps_two_dimensional_geometry():
  materials = [Lossless(1.0), Debye(2.0, 4.0, 0.5)]
  material_ids = np.asarray([[0, 0, 1], [1, 0, 1]])
  grid = prepare_material_grid_2d(
      materials, material_ids, dt=0.1, dx=0.5, dy=0.25)

  assert grid.shape == (2, 3)
  assert grid.coefficients.shape == (2, 3, 5)
  np.testing.assert_array_equal(grid.material_ids, material_ids)
  np.testing.assert_array_equal(
      grid.coefficients[0, 0],
      materials[0].ade_coefficients(0.1).as_array(np.float64))
  np.testing.assert_array_equal(
      grid.coefficients[0, 2],
      materials[1].ade_coefficients(0.1).as_array(np.float64))


def test_zero_state_remains_zero_for_mixed_materials():
  materials = [
      Lossless(1.0), Debye(2.0, 4.0, 0.5),
      Lorentz(1.0, 2.25, 4.0, 0.28),
  ]
  material_ids = np.arange(24).reshape(4, 6) % 3
  result = simulate_yee_ade_2d(
      materials, material_ids, dx=1.0, dy=1.0, dt=0.25, num_steps=6)

  assert not np.any(result.electric_z)
  assert not np.any(result.magnetic_x)
  assert not np.any(result.magnetic_y)
  assert not np.any(result.displacement_z)


def test_one_lossless_step_matches_explicit_tmz_difference_equations():
  shape = (3, 4)
  grid = prepare_material_grid_2d(
      [Lossless(2.0)], np.zeros(shape, dtype=int), dt=0.1, dx=0.5, dy=0.25)
  electric = np.arange(12, dtype=float).reshape(shape) / 7.0
  magnetic_x = np.arange(12, dtype=float).reshape(shape) / 11.0
  magnetic_y = -np.arange(12, dtype=float).reshape(shape) / 13.0
  displacement = 2.0 * electric
  state = YeeADE2DState(
      electric_z=electric,
      magnetic_x=magnetic_x,
      magnetic_y=magnetic_y,
      displacement_z=displacement,
      displacement_z_previous=displacement.copy(),
      electric_z_previous=electric.copy(),
  )

  actual = yee_ade_2d_step(grid, state)
  expected_hx = magnetic_x - 0.4 * (
      np.roll(electric, -1, axis=1) - electric)
  expected_hy = magnetic_y + 0.2 * (
      np.roll(electric, -1, axis=0) - electric)
  expected_d = displacement + 0.2 * (
      expected_hy - np.roll(expected_hy, 1, axis=0))
  expected_d -= 0.4 * (
      expected_hx - np.roll(expected_hx, 1, axis=1))

  np.testing.assert_allclose(actual.magnetic_x, expected_hx)
  np.testing.assert_allclose(actual.magnetic_y, expected_hy)
  np.testing.assert_allclose(actual.displacement_z, expected_d)
  np.testing.assert_allclose(actual.electric_z, expected_d / 2.0)


@pytest.mark.parametrize("material", [
    Lossless(2.0),
    Debye(2.0, 4.0, 0.5),
    Lorentz(1.0, 2.25, 4.0, 0.28),
])
def test_uniform_source_reduces_homogeneous_2d_grid_to_scalar_ade(material):
  shape = (3, 4)
  num_steps = 300
  dt = 0.01
  displacement = np.sin(1.2 * dt * np.arange(num_steps))
  previous = np.concatenate(([0.0], displacement[:-1]))
  current = -(displacement - previous) / dt
  result = simulate_yee_ade_2d(
      [material], np.zeros(shape, dtype=int), dx=1.0, dy=1.0, dt=dt,
      num_steps=num_steps, current_density_z=current)
  expected = simulate_ade(material, dt, displacement)

  np.testing.assert_allclose(result.displacement_z[:, 0, 0], displacement)
  np.testing.assert_allclose(result.electric_z[:, 0, 0], expected)
  assert not np.any(result.magnetic_x)
  assert not np.any(result.magnetic_y)


def test_uniform_source_reduces_2d_multipole_grid_to_scalar_reference():
  material = Multipole(1.2, (
      DebyePole(0.7, 0.4), LorentzPole(1.1, 4.0, 0.2),
      DrudePole(2.5, 0.3)))
  shape = (3, 4)
  num_steps = 500
  dt = 0.01
  displacement = np.sin(1.2 * dt * np.arange(num_steps))
  previous = np.concatenate(([0.0], displacement[:-1]))
  current = -(displacement - previous) / dt
  result = simulate_yee_ade_2d(
      [material], np.zeros(shape, dtype=int), 1.0, 1.0, dt, num_steps,
      current_density_z=current)
  expected = simulate_multipole_ade(material, dt, displacement)
  np.testing.assert_allclose(result.electric_z[:, 0, 0], expected)
  assert result.final_state.polarization_current.shape == shape + (3,)
  np.testing.assert_allclose(
      material.eps_inf * result.final_state.electric_z +
      np.sum(result.final_state.polarization_current, axis=-1),
      result.final_state.displacement_z)


def test_mixed_direct_and_multipole_material_grid_remains_finite():
  materials = [Lossless(1.0), Multipole(1.0, (
      LorentzPole(1.5, 3.0, 0.2), DrudePole(2.0, 0.4)))]
  shape = (12, 20)
  ids = np.zeros(shape, dtype=int)
  ids[4:8, 8:12] = 1
  source = np.zeros((200,) + shape)
  source[0, :, 4] = 0.1
  result = simulate_yee_ade_2d(
      materials, ids, 1.0, 1.0, 0.3, 200,
      current_density_z=source, record_interval=20)
  assert np.all(np.isfinite(result.electric_z))
  assert np.all(result.final_state.polarization_current[ids == 0] == 0.0)


def test_point_and_line_current_sources_are_supported():
  shape = (5, 6)
  point_source = np.zeros((1,) + shape)
  point_source[0, 2, 3] = 4.0
  point = simulate_yee_ade_2d(
      [Lossless(2.0)], np.zeros(shape, dtype=int), 1.0, 1.0, 0.1, 1,
      current_density_z=point_source)
  assert point.displacement_z[0, 2, 3] == pytest.approx(-0.4)
  assert np.count_nonzero(point.displacement_z) == 1

  line_source = np.zeros(shape)
  line_source[:, 2] = 3.0
  line = simulate_yee_ade_2d(
      [Lossless(2.0)], np.zeros(shape, dtype=int), 1.0, 1.0, 0.1, 1,
      current_density_z=line_source)
  np.testing.assert_allclose(line.displacement_z[0, :, 2], -0.3)
  assert np.count_nonzero(line.displacement_z) == shape[0]


def test_magnetic_current_source_drives_expected_local_curl():
  shape = (5, 6)
  magnetic_source = np.zeros(shape)
  magnetic_source[:, 2] = 3.0
  result = simulate_yee_ade_2d(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 1.0, 1.0, 0.1, 1,
      magnetic_current_x=magnetic_source)
  np.testing.assert_allclose(result.magnetic_x[0, :, 2], -0.3)
  assert np.count_nonzero(result.magnetic_x) == shape[0]
  np.testing.assert_allclose(result.electric_z[0, :, 2], 0.03)
  np.testing.assert_allclose(result.electric_z[0, :, 3], -0.03)
  assert np.count_nonzero(result.electric_z) == 2 * shape[0]
  assert not np.any(result.magnetic_y)


def test_two_dimensional_cfl_uses_high_frequency_permittivity():
  material = Debye(eps_inf=0.25, eps_static=2.0, tau=0.5)
  maximum_dt = 0.5 / np.sqrt(2.0)
  prepare_material_grid_2d(
      [material], np.zeros((2, 2), dtype=int), maximum_dt, 1.0, 1.0)
  with pytest.raises(ValueError, match="CFL condition"):
    prepare_material_grid_2d(
        [material], np.zeros((2, 2), dtype=int), maximum_dt * 1.01, 1.0, 1.0)


def test_conductivity_cn_update_matches_uniform_lossless_solution():
  shape = (3, 4)
  sigma_e = np.full(shape, 0.7)
  sigma_m = np.full(shape, 0.4)
  grid = prepare_material_grid_2d(
      [Lossless(2.0)], np.zeros(shape, dtype=int), 0.1, 1.0, 1.0,
      electric_conductivity=sigma_e, magnetic_conductivity=sigma_m)
  state = YeeADE2DState(
      electric_z=np.ones(shape), magnetic_x=np.full(shape, 2.0),
      magnetic_y=np.full(shape, -3.0), displacement_z=np.full(shape, 2.0),
      displacement_z_previous=np.full(shape, 2.0),
      electric_z_previous=np.ones(shape))
  actual = yee_ade_2d_step(grid, state)
  expected_e_decay = (2.0 - 0.5 * 0.1 * 0.7) / (
      2.0 + 0.5 * 0.1 * 0.7)
  expected_h_decay = (1.0 - 0.5 * 0.1 * 0.4) / (
      1.0 + 0.5 * 0.1 * 0.4)
  np.testing.assert_allclose(actual.electric_z, expected_e_decay)
  np.testing.assert_allclose(actual.displacement_z, 2.0 * expected_e_decay)
  np.testing.assert_allclose(actual.magnetic_x, 2.0 * expected_h_decay)
  np.testing.assert_allclose(actual.magnetic_y, -3.0 * expected_h_decay)


def test_conductivity_rejects_overlap_with_dispersive_material():
  materials = [Lossless(1.0), Debye(1.0, 2.0, 0.5)]
  ids = np.zeros((4, 6), dtype=int)
  ids[2, 3] = 1
  sigma = np.zeros_like(ids, dtype=float)
  sigma[2, 3] = 0.1
  with pytest.raises(ValueError, match="restricted to Lossless"):
    prepare_material_grid_2d(
        materials, ids, 0.1, 1.0, 1.0, electric_conductivity=sigma)


def test_y_cpml_coefficients_are_identity_in_interior_and_symmetric():
  cpml = prepare_y_cpml((5, 30), 6, dt=0.2, dy=1.0)
  assert cpml.inverse_kappa_e.shape == (30,)
  np.testing.assert_allclose(cpml.inverse_kappa_e[6:24], 1.0)
  np.testing.assert_allclose(cpml.b_e[6:24], 0.0)
  np.testing.assert_allclose(cpml.c_e[6:24], 0.0)
  np.testing.assert_allclose(cpml.inverse_kappa_e, cpml.inverse_kappa_e[::-1])
  assert np.all(cpml.c_e[:6] <= 0.0)


def test_y_cpml_zero_state_stays_zero_and_memory_is_allocated():
  shape = (4, 30)
  cpml = prepare_y_cpml(shape, 6, 0.2, 1.0)
  result = simulate_yee_ade_2d(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 1.0, 1.0, 0.2, 5,
      y_cpml=cpml)
  assert not np.any(result.electric_z)
  assert result.final_state.cpml_hx_y.shape == shape
  assert result.final_state.cpml_dz_y.shape == shape
  assert not np.any(result.final_state.cpml_hx_y)


def test_y_cpml_rejects_dispersive_material_inside_layer():
  shape = (4, 30)
  ids = np.zeros(shape, dtype=int)
  ids[:, 2] = 1
  cpml = prepare_y_cpml(shape, 6, 0.2, 1.0)
  with pytest.raises(ValueError, match="restricted to Lossless"):
    prepare_material_grid_2d(
        [Lossless(1.0), Lorentz(1.0, 2.0, 3.0, 0.2)], ids,
        0.2, 1.0, 1.0, y_cpml=cpml)


def test_y_cpml_suppresses_late_periodic_echo():
  shape = (3, 180)
  num_steps = 520
  dt = 0.5
  time = np.arange(num_steps) * dt
  waveform = np.exp(-0.5 * ((time - 25.0) / 7.0)**2) * np.sin(
      2.0 * np.pi * 0.05 * (time - 25.0))
  source = np.zeros((num_steps,) + shape)
  source[:, :, 90] = waveform[:, None]
  ids = np.zeros(shape, dtype=int)
  periodic = simulate_yee_ade_2d(
      [Lossless(1.0)], ids, 1.0, 1.0, dt, num_steps,
      current_density_z=source)
  cpml = prepare_y_cpml(shape, 24, dt, 1.0, target_reflection=1e-8)
  absorbing = simulate_yee_ade_2d(
      [Lossless(1.0)], ids, 1.0, 1.0, dt, num_steps,
      current_density_z=source, y_cpml=cpml)
  periodic_late = np.max(np.abs(periodic.electric_z[390:, :, 90]))
  absorbing_late = np.max(np.abs(absorbing.electric_z[390:, :, 90]))
  assert absorbing_late < 1e-3 * periodic_late


def test_huygens_source_preferentially_propagates_in_requested_direction():
  shape = (3, 180)
  num_steps = 260
  dt = 0.5
  waveform = gaussian_sine_pulse(
      num_steps, dt, center_time=22.0, width=6.0, frequency=0.05)
  source = huygens_line_source(waveform, shape, 90, direction=1)
  cpml = prepare_y_cpml(shape, 24, dt, 1.0)
  result = simulate_yee_ade_2d(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 1.0, 1.0, dt,
      num_steps, current_density_z=source.current_density_z,
      magnetic_current_x=source.magnetic_current_x, y_cpml=cpml)
  positive_y_energy = np.sum(result.electric_z[:, :, 110]**2)
  negative_y_energy = np.sum(result.electric_z[:, :, 70]**2)
  assert positive_y_energy > 100.0 * negative_y_energy


def test_float32_fields_and_record_shapes_are_preserved():
  shape = (4, 7)
  result = simulate_yee_ade_2d(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 1.0, 1.0, 0.25, 5,
      dtype=np.float32, record_interval=2)
  assert result.electric_z.shape == (3,) + shape
  assert result.electric_z.dtype == np.float32
  assert result.magnetic_x.dtype == np.float32
  assert result.magnetic_y.dtype == np.float32
  assert result.steps.tolist() == [2, 4, 5]


def test_simulation_resume_matches_single_run():
  shape = (5, 6)
  material_ids = np.zeros(shape, dtype=int)
  material_ids[2:4, 2:5] = 1
  materials = [Lossless(1.0), Lorentz(1.0, 2.25, 4.0, 0.28)]
  source = np.zeros((10,) + shape)
  source[0, 1, 3] = 1.0
  full = simulate_yee_ade_2d(
      materials, material_ids, 1.0, 1.0, 0.1, 10,
      current_density_z=source, record_interval=20)
  first = simulate_yee_ade_2d(
      materials, material_ids, 1.0, 1.0, 0.1, 4,
      current_density_z=source[:4], record_interval=20)
  second = simulate_yee_ade_2d(
      materials, material_ids, 1.0, 1.0, 0.1, 6,
      current_density_z=source[4:], initial_state=first.final_state,
      record_interval=20)

  assert second.steps.tolist() == [10]
  for field in ("electric_z", "magnetic_x", "magnetic_y", "displacement_z",
                "displacement_z_previous", "electric_z_previous"):
    np.testing.assert_allclose(getattr(second.final_state, field),
                               getattr(full.final_state, field))


def test_lossless_periodic_tmz_wave_remains_finite_below_cfl():
  shape = (32, 28)
  grid = prepare_material_grid_2d(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 0.4, 1.0, 1.0)
  x = np.arange(shape[0])[:, None]
  y = np.arange(shape[1])[None, :]
  electric = np.sin(2.0 * np.pi * x / shape[0]) * np.cos(
      2.0 * np.pi * y / shape[1])
  state = YeeADE2DState(
      electric_z=electric,
      magnetic_x=np.zeros(shape),
      magnetic_y=np.zeros(shape),
      displacement_z=electric.copy(),
      displacement_z_previous=electric.copy(),
      electric_z_previous=electric.copy())
  for _ in range(1000):
    state = yee_ade_2d_step(grid, state)

  assert np.all(np.isfinite(state.electric_z))
  assert np.max(np.abs(state.electric_z)) < 1.1


def test_public_api_exports_two_dimensional_solver():
  import fdtdz_jax

  assert fdtdz_jax.prepare_material_grid_2d is prepare_material_grid_2d
  assert fdtdz_jax.simulate_yee_ade_2d is simulate_yee_ade_2d
  assert fdtdz_jax.yee_ade_2d_step is yee_ade_2d_step
