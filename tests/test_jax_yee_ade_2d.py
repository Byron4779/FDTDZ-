import numpy as np

from fdtdz_jax.jax_yee_ade_2d import jax_material_grid_2d
from fdtdz_jax.jax_yee_ade_2d import advance_jax_yee_ade_2d
from fdtdz_jax.jax_yee_ade_2d import jax_yee_ade_2d_state
from fdtdz_jax.jax_yee_ade_2d import jax_yee_ade_2d_step
from fdtdz_jax.jax_yee_ade_2d import simulate_jax_yee_ade_2d
from fdtdz_jax.jax_yee_ade_2d import simulate_jax_yee_ade_2d_probes
from fdtdz_jax.jax_yee_ade_2d import simulate_jax_yee_ade_2d_line_probes
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.yee_ade_2d import initialize_yee_ade_2d
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d
from fdtdz_jax.yee_ade_2d import prepare_y_cpml
from fdtdz_jax.yee_ade_2d import yee_ade_2d_step


def _assert_state_close(actual, expected, rtol=2e-6, atol=2e-6):
  for name in (
      "electric_z", "magnetic_x", "magnetic_y", "displacement_z",
      "displacement_z_previous", "electric_z_previous", "cpml_hx_y",
      "cpml_dz_y", "polarization_current", "polarization_previous"):
    np.testing.assert_allclose(
        np.asarray(getattr(actual, name)), getattr(expected, name),
        rtol=rtol, atol=atol)
  assert int(actual.step) == expected.step


def test_jax_step_matches_numpy_for_mixed_direct_materials():
  shape = (7, 9)
  ids = np.arange(np.prod(shape)).reshape(shape) % 2
  grid = prepare_material_grid_2d(
      [Lossless(1.0), Debye(1.5, 3.0, 0.4)], ids,
      0.2, 1.0, 1.0, dtype=np.float32)
  numpy_state = initialize_yee_ade_2d(grid)
  rng = np.random.default_rng(3)
  current = rng.normal(size=shape).astype(np.float32)
  magnetic = rng.normal(size=shape).astype(np.float32)
  expected = yee_ade_2d_step(grid, numpy_state, current, magnetic)
  actual = jax_yee_ade_2d_step(
      jax_material_grid_2d(grid), jax_yee_ade_2d_state(numpy_state, grid),
      current, magnetic)
  _assert_state_close(actual, expected)


def test_jax_scan_matches_numpy_with_cpml_and_multipoles():
  shape = (8, 40)
  materials = [Lossless(1.0), Multipole(1.0, (
      LorentzPole(1.2, 3.0, 0.2), DrudePole(2.0, 0.4)))]
  ids = np.zeros(shape, dtype=int)
  ids[3:6, 18:23] = 1
  cpml = prepare_y_cpml(shape, 8, 0.25, 1.0)
  grid = prepare_material_grid_2d(
      materials, ids, 0.25, 1.0, 1.0, dtype=np.float32, y_cpml=cpml)
  numpy_state = initialize_yee_ade_2d(grid)
  rng = np.random.default_rng(7)
  electric_sources = (0.01 * rng.normal(size=(30,) + shape)).astype(np.float32)
  magnetic_sources = (0.01 * rng.normal(size=(30,) + shape)).astype(np.float32)
  for electric_source, magnetic_source in zip(
      electric_sources, magnetic_sources):
    numpy_state = yee_ade_2d_step(
        grid, numpy_state, electric_source, magnetic_source)
  jax_state, records = simulate_jax_yee_ade_2d(
      jax_material_grid_2d(grid),
      jax_yee_ade_2d_state(initialize_yee_ade_2d(grid), grid),
      electric_sources, magnetic_sources)
  assert len(records) == 4
  assert records[0].shape == (30,) + shape
  _assert_state_close(jax_state, numpy_state, rtol=5e-6, atol=5e-6)
  final_only = advance_jax_yee_ade_2d(
      jax_material_grid_2d(grid),
      jax_yee_ade_2d_state(initialize_yee_ade_2d(grid), grid),
      electric_sources, magnetic_sources)
  _assert_state_close(final_only, numpy_state, rtol=5e-6, atol=5e-6)


def test_compact_probe_scan_matches_full_field_scan():
  shape = (6, 30)
  num_steps = 40
  grid = prepare_material_grid_2d(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 0.25, 1.0, 1.0,
      dtype=np.float32)
  initial = initialize_yee_ade_2d(grid)
  rng = np.random.default_rng(11)
  electric_waveform = rng.normal(size=num_steps).astype(np.float32) * 0.01
  magnetic_waveform = rng.normal(size=num_steps).astype(np.float32) * 0.01
  x_profile = np.linspace(0.5, 1.0, shape[0], dtype=np.float32)
  electric_sources = np.zeros((num_steps,) + shape, dtype=np.float32)
  magnetic_sources = np.zeros_like(electric_sources)
  electric_sources[:, :, 8] = electric_waveform[:, None] * x_profile
  magnetic_sources[:, :, 8] = magnetic_waveform[:, None] * x_profile
  jax_grid = jax_material_grid_2d(grid)
  jax_initial = jax_yee_ade_2d_state(initial, grid)
  full_state, full_records = simulate_jax_yee_ade_2d(
      jax_grid, jax_initial, electric_sources, magnetic_sources)
  compact_state, compact_probes = simulate_jax_yee_ade_2d_probes(
      jax_grid, jax_initial, electric_waveform, magnetic_waveform, x_profile,
      source_y=8, reflection_probe_y=12, transmission_probe_y=20)
  _assert_state_close(compact_state, full_state)
  np.testing.assert_allclose(
      np.asarray(compact_probes[0]),
      np.asarray(full_records[0])[:, :, 12].mean(axis=1),
      rtol=5e-7, atol=1e-9)
  np.testing.assert_allclose(
      np.asarray(compact_probes[1]),
      np.asarray(full_records[0])[:, :, 20].mean(axis=1),
      rtol=5e-7, atol=1e-9)
  line_state, line_probes = simulate_jax_yee_ade_2d_line_probes(
      jax_grid, jax_initial, electric_waveform, magnetic_waveform, x_profile,
      source_y=8, reflection_probe_y=12, transmission_probe_y=20)
  _assert_state_close(line_state, full_state)
  np.testing.assert_allclose(
      np.asarray(line_probes[0]), np.asarray(full_records[0])[:, :, 12])
  np.testing.assert_allclose(
      np.asarray(line_probes[1]), np.asarray(full_records[0])[:, :, 20])
