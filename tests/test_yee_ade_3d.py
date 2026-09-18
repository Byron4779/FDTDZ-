import numpy as np
import pytest

from fdtdz_jax.ade_reference import simulate_ade
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import Lorentz
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.multipole_reference import simulate_multipole_ade
from fdtdz_jax.metasurface_3d import huygens_plane_source
from fdtdz_jax.yee_ade_2d import YeeADE2DState
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d
from fdtdz_jax.yee_ade_2d import yee_ade_2d_step
from fdtdz_jax.yee_ade_3d import YeeADE3DState
from fdtdz_jax.yee_ade_3d import initialize_yee_ade_3d
from fdtdz_jax.yee_ade_3d import prepare_material_grid_3d
from fdtdz_jax.yee_ade_3d import prepare_z_cpml
from fdtdz_jax.yee_ade_3d import simulate_yee_ade_3d
from fdtdz_jax.yee_ade_3d import simulate_yee_ade_3d_compact_plane_probes
from fdtdz_jax.yee_ade_3d import simulate_yee_ade_3d_plane_probes
from fdtdz_jax.yee_ade_3d import yee_ade_3d_step


def test_material_grid_accepts_cell_and_component_material_maps():
  materials = [Lossless(1.0), Debye(2.0, 4.0, 0.5)]
  cell_ids = np.zeros((2, 3, 4), dtype=int)
  cell_ids[1, 1:, 2:] = 1
  cell_grid = prepare_material_grid_3d(
      materials, cell_ids, dt=0.1, dx=0.5, dy=0.5, dz=0.5)

  assert cell_grid.shape == (2, 3, 4)
  assert cell_grid.vector_shape == (3, 2, 3, 4)
  assert cell_grid.coefficients.shape == (3, 2, 3, 4, 5)
  for component in range(3):
    np.testing.assert_array_equal(cell_grid.material_ids[component], cell_ids)

  component_ids = np.zeros((3, 2, 3, 4), dtype=int)
  component_ids[1] = 1
  component_grid = prepare_material_grid_3d(
      materials, component_ids, dt=0.1, dx=0.5, dy=0.5, dz=0.5)
  np.testing.assert_array_equal(
      component_grid.coefficients[0, 0, 0, 0],
      materials[0].ade_coefficients(0.1).as_array(np.float64))
  np.testing.assert_array_equal(
      component_grid.coefficients[1, 0, 0, 0],
      materials[1].ade_coefficients(0.1).as_array(np.float64))


def test_three_dimensional_cfl_limit_is_enforced():
  ids = np.zeros((2, 2, 2), dtype=int)
  prepare_material_grid_3d([Lossless(1.0)], ids, 0.5, 1.0, 1.0, 1.0)
  with pytest.raises(ValueError, match="three-dimensional CFL"):
    prepare_material_grid_3d([Lossless(1.0)], ids, 0.6, 1.0, 1.0, 1.0)


def test_z_cpml_coefficients_and_material_boundary_validation():
  shape = (3, 4, 16)
  cpml = prepare_z_cpml(shape, width=4, dt=0.25, dz=1.0)
  assert cpml.inverse_kappa_e.shape == (shape[2],)
  assert cpml.inverse_kappa_h.shape == (shape[2],)
  assert np.all(cpml.inverse_kappa_e > 0.0)
  assert np.all(cpml.inverse_kappa_e <= 1.0)
  np.testing.assert_allclose(cpml.inverse_kappa_e[4:-4], 1.0)

  materials = [Lossless(1.0), Debye(2.0, 4.0, 0.5)]
  valid_ids = np.zeros(shape, dtype=int)
  valid_ids[:, :, 7:9] = 1
  grid = prepare_material_grid_3d(
      materials, valid_ids, 0.25, 1.0, 1.0, 1.0, z_cpml=cpml)
  assert grid.z_cpml is cpml
  invalid_ids = valid_ids.copy()
  invalid_ids[:, :, 1] = 1
  with pytest.raises(ValueError, match="Lossless boundary"):
    prepare_material_grid_3d(
        materials, invalid_ids, 0.25, 1.0, 1.0, 1.0, z_cpml=cpml)


def test_zero_state_with_z_cpml_remains_exactly_zero():
  shape = (3, 4, 18)
  cpml = prepare_z_cpml(shape, width=5, dt=0.25, dz=1.0)
  result = simulate_yee_ade_3d(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 1.0, 1.0, 1.0,
      0.25, 8, z_cpml=cpml)
  assert not np.any(result.electric)
  assert not np.any(result.magnetic)
  state = result.final_state
  assert not np.any(state.cpml_hx_ey_z)
  assert not np.any(state.cpml_hy_ex_z)
  assert not np.any(state.cpml_dx_hy_z)
  assert not np.any(state.cpml_dy_hx_z)


def test_z_cpml_absorbs_a_normally_propagating_pulse():
  shape = (2, 2, 80)
  num_steps = 300
  dt = 0.4
  time = np.arange(num_steps) * dt
  waveform = np.exp(-0.5 * ((time - 16.0) / 4.0)**2) * np.sin(
      2.0 * np.pi * 0.08 * (time - 16.0))
  source = np.zeros((num_steps, 3) + shape)
  source[:, 0, :, :, shape[2] // 2] = waveform[:, None, None]
  material_ids = np.zeros(shape, dtype=int)
  periodic = simulate_yee_ade_3d(
      [Lossless(1.0)], material_ids, 1.0, 1.0, 1.0, dt, num_steps,
      electric_current=source, record_interval=num_steps)
  cpml = prepare_z_cpml(shape, width=12, dt=dt, dz=1.0)
  absorbed = simulate_yee_ade_3d(
      [Lossless(1.0)], material_ids, 1.0, 1.0, 1.0, dt, num_steps,
      electric_current=source, record_interval=num_steps, z_cpml=cpml)

  def final_field_energy(result):
    return (np.sum(result.final_state.electric**2) +
            np.sum(result.final_state.magnetic**2))
  assert final_field_energy(absorbed) < 1e-3 * final_field_energy(periodic)


def test_zero_state_remains_zero_for_mixed_three_dimensional_materials():
  materials = [
      Lossless(1.0), Debye(2.0, 4.0, 0.5),
      Lorentz(1.0, 2.25, 4.0, 0.28),
  ]
  material_ids = np.arange(60).reshape(3, 4, 5) % len(materials)
  result = simulate_yee_ade_3d(
      materials, material_ids, 1.0, 1.0, 1.0, 0.25, 6)

  assert not np.any(result.electric)
  assert not np.any(result.magnetic)
  assert not np.any(result.displacement)
  assert result.final_state.step == 6


def test_one_lossless_step_matches_explicit_full_vector_curl_equations():
  shape = (3, 4, 5)
  grid = prepare_material_grid_3d(
      [Lossless(2.0)], np.zeros(shape, dtype=int),
      dt=0.1, dx=0.5, dy=0.4, dz=0.25)
  values = np.arange(3 * np.prod(shape), dtype=float).reshape((3,) + shape)
  electric = np.sin(values / 17.0)
  magnetic = np.cos(values / 23.0)
  displacement = 2.0 * electric
  state = YeeADE3DState(
      electric=electric, magnetic=magnetic, displacement=displacement,
      displacement_previous=displacement.copy(),
      electric_previous=electric.copy())

  actual = yee_ade_3d_step(grid, state)
  ex, ey, ez = electric
  hx, hy, hz = magnetic
  forward = lambda field, axis, spacing: (
      np.roll(field, -1, axis=axis) - field) / spacing
  backward = lambda field, axis, spacing: (
      field - np.roll(field, 1, axis=axis)) / spacing
  expected_hx = hx - 0.1 * (
      forward(ez, 1, 0.4) - forward(ey, 2, 0.25))
  expected_hy = hy - 0.1 * (
      forward(ex, 2, 0.25) - forward(ez, 0, 0.5))
  expected_hz = hz - 0.1 * (
      forward(ey, 0, 0.5) - forward(ex, 1, 0.4))
  expected_d = displacement.copy()
  expected_d[0] += 0.1 * (
      backward(expected_hz, 1, 0.4) - backward(expected_hy, 2, 0.25))
  expected_d[1] += 0.1 * (
      backward(expected_hx, 2, 0.25) - backward(expected_hz, 0, 0.5))
  expected_d[2] += 0.1 * (
      backward(expected_hy, 0, 0.5) - backward(expected_hx, 1, 0.4))

  np.testing.assert_allclose(
      actual.magnetic, np.stack((expected_hx, expected_hy, expected_hz)))
  np.testing.assert_allclose(actual.displacement, expected_d)
  np.testing.assert_allclose(actual.electric, expected_d / 2.0)


@pytest.mark.parametrize("material", [
    Lossless(2.0),
    Debye(2.0, 4.0, 0.5),
    Lorentz(1.0, 2.25, 4.0, 0.28),
])
def test_uniform_vector_source_reduces_three_dimensional_grid_to_scalar_ade(
    material):
  shape = (2, 2, 2)
  num_steps = 240
  dt = 0.01
  displacement = np.sin(1.2 * dt * np.arange(num_steps))
  previous = np.concatenate(([0.0], displacement[:-1]))
  current = np.zeros((num_steps, 3))
  current[:, 2] = -(displacement - previous) / dt
  result = simulate_yee_ade_3d(
      [material], np.zeros(shape, dtype=int), 1.0, 1.0, 1.0, dt,
      num_steps, electric_current=current)
  expected = simulate_ade(material, dt, displacement)

  np.testing.assert_allclose(
      result.displacement[:, 2, 0, 0, 0], displacement)
  np.testing.assert_allclose(result.electric[:, 2, 0, 0, 0], expected)
  assert not np.any(result.magnetic)
  assert not np.any(result.electric[:, :2])


def test_uniform_source_reduces_three_dimensional_multipole_to_reference():
  material = Multipole(1.2, (
      LorentzPole(1.1, 4.0, 0.2), DrudePole(2.5, 0.3)))
  shape = (2, 2, 2)
  num_steps = 300
  dt = 0.01
  displacement = np.sin(1.2 * dt * np.arange(num_steps))
  previous = np.concatenate(([0.0], displacement[:-1]))
  current = np.zeros((num_steps, 3))
  current[:, 0] = -(displacement - previous) / dt
  result = simulate_yee_ade_3d(
      [material], np.zeros(shape, dtype=int), 1.0, 1.0, 1.0, dt,
      num_steps, electric_current=current)
  expected = simulate_multipole_ade(material, dt, displacement)

  np.testing.assert_allclose(result.electric[:, 0, 0, 0, 0], expected)
  state = result.final_state
  np.testing.assert_allclose(
      material.eps_inf * state.electric +
      np.sum(state.polarization_current, axis=-1), state.displacement)


def test_z_invariant_tmz_three_dimensional_update_matches_two_dimensional_step():
  nx, ny, nz = 5, 6, 3
  dt, dx, dy, dz = 0.1, 0.8, 0.9, 1.1
  material = Lossless(2.0)
  grid_2d = prepare_material_grid_2d(
      [material], np.zeros((nx, ny), dtype=int), dt, dx, dy)
  grid_3d = prepare_material_grid_3d(
      [material], np.zeros((nx, ny, nz), dtype=int), dt, dx, dy, dz)
  x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
  electric_z = np.sin(2.0 * np.pi * x / nx) + 0.3 * np.cos(
      2.0 * np.pi * y / ny)
  magnetic_x = 0.2 * np.sin(2.0 * np.pi * y / ny)
  magnetic_y = 0.1 * np.cos(2.0 * np.pi * x / nx)
  displacement_z = 2.0 * electric_z
  state_2d = YeeADE2DState(
      electric_z=electric_z, magnetic_x=magnetic_x,
      magnetic_y=magnetic_y, displacement_z=displacement_z,
      displacement_z_previous=displacement_z.copy(),
      electric_z_previous=electric_z.copy())
  electric_3d = np.zeros(grid_3d.vector_shape)
  magnetic_3d = np.zeros(grid_3d.vector_shape)
  displacement_3d = np.zeros(grid_3d.vector_shape)
  electric_3d[2] = np.repeat(electric_z[..., None], nz, axis=2)
  magnetic_3d[0] = np.repeat(magnetic_x[..., None], nz, axis=2)
  magnetic_3d[1] = np.repeat(magnetic_y[..., None], nz, axis=2)
  displacement_3d[2] = np.repeat(displacement_z[..., None], nz, axis=2)
  state_3d = YeeADE3DState(
      electric=electric_3d, magnetic=magnetic_3d,
      displacement=displacement_3d,
      displacement_previous=displacement_3d.copy(),
      electric_previous=electric_3d.copy())

  for _ in range(12):
    state_2d = yee_ade_2d_step(grid_2d, state_2d)
    state_3d = yee_ade_3d_step(grid_3d, state_3d)

  np.testing.assert_allclose(
      state_3d.electric[2],
      np.repeat(state_2d.electric_z[..., None], nz, axis=2))
  np.testing.assert_allclose(
      state_3d.magnetic[0],
      np.repeat(state_2d.magnetic_x[..., None], nz, axis=2))
  np.testing.assert_allclose(
      state_3d.magnetic[1],
      np.repeat(state_2d.magnetic_y[..., None], nz, axis=2))
  np.testing.assert_allclose(
      state_3d.displacement[2],
      np.repeat(state_2d.displacement_z[..., None], nz, axis=2))
  assert not np.any(state_3d.electric[:2])
  assert not np.any(state_3d.magnetic[2])


def test_plane_probe_history_matches_full_field_recording():
  shape = (3, 4, 12)
  num_steps = 18
  source = np.zeros((num_steps, 3) + shape)
  source[0, 0, 1, 2, 5] = 0.2
  material_ids = np.zeros(shape, dtype=int)
  common = dict(
      materials=[Lossless(1.0)], material_ids=material_ids,
      dx=1.0, dy=1.0, dz=1.0, dt=0.25, num_steps=num_steps,
      electric_current=source)
  full = simulate_yee_ade_3d(**common)
  compact = simulate_yee_ade_3d_plane_probes(
      probe_z_indices=(3, 8), **common)
  expected = np.stack(
      [full.electric[:, :2, :, :, index] for index in (3, 8)], axis=1)

  np.testing.assert_allclose(compact.electric_tangential, expected)
  assert compact.electric_tangential.shape == (num_steps, 2, 2, 3, 4)
  assert compact.steps.tolist() == list(range(1, num_steps + 1))
  np.testing.assert_allclose(
      compact.final_state.electric, full.final_state.electric)
  assert compact.electric_tangential.nbytes < full.electric.nbytes


def test_compact_numpy_huygens_source_matches_expanded_source_planes():
  shape = (3, 4, 24)
  num_steps = 20
  waveform = np.linspace(-0.02, 0.03, num_steps)
  source = huygens_plane_source(
      waveform, shape, 7, polarization_xy=(0.6, 0.8),
      direction=1, expand=True)
  common = dict(
      materials=[Lossless(1.0)], material_ids=np.zeros(shape, dtype=int),
      dx=1.0, dy=1.0, dz=1.0, dt=0.2)
  expanded = simulate_yee_ade_3d_plane_probes(
      num_steps=num_steps, probe_z_indices=(11, 17),
      electric_current=source.electric_current,
      magnetic_current=source.magnetic_current, **common)
  full = simulate_yee_ade_3d(
      num_steps=num_steps, electric_current=source.electric_current,
      magnetic_current=source.magnetic_current, **common)
  frequencies = np.asarray((0.07, 0.13))
  weights = np.linspace(0.2, 1.0, num_steps)
  compact = simulate_yee_ade_3d_compact_plane_probes(
      electric_waveform=source.electric_waveform,
      magnetic_waveform=source.magnetic_waveform,
      transverse_profile=source.transverse_profile,
      polarization_xy=source.polarization_xy,
      electric_source_z=source.electric_z_index,
      magnetic_source_z=source.magnetic_z_index,
      reflection_probe_z=11, transmission_probe_z=17,
      frequencies=frequencies, frequency_window=weights, **common)

  np.testing.assert_allclose(
      compact.electric_tangential, expanded.electric_tangential)
  np.testing.assert_allclose(
      compact.final_state.electric, expanded.final_state.electric)
  electric_times = (np.arange(num_steps) + 1.0) * common["dt"]
  magnetic_times = (np.arange(num_steps) + 0.5) * common["dt"]
  electric_phases = (
      common["dt"] * weights[None, :] *
      np.exp(2j * np.pi * frequencies[:, None] * electric_times[None, :]))
  magnetic_phases = (
      common["dt"] * weights[None, :] *
      np.exp(2j * np.pi * frequencies[:, None] * magnetic_times[None, :]))
  np.testing.assert_allclose(
      compact.frequency_fields.electric,
      np.einsum("ft,tcxyz->fcxyz", electric_phases, full.electric))
  np.testing.assert_allclose(
      compact.frequency_fields.magnetic,
      np.einsum("ft,tcxyz->fcxyz", magnetic_phases, full.magnetic))
  np.testing.assert_allclose(
      compact.frequency_fields.displacement,
      np.einsum("ft,tcxyz->fcxyz", electric_phases, full.displacement))
