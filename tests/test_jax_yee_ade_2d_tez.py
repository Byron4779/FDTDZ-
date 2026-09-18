import numpy as np

from fdtdz_jax.jax_yee_ade_2d import jax_material_grid_2d
from fdtdz_jax.jax_yee_ade_2d_tez import jax_yee_ade_2d_tez_state
from fdtdz_jax.jax_yee_ade_2d_tez import jax_yee_ade_2d_tez_step
from fdtdz_jax.jax_yee_ade_2d_tez import simulate_jax_yee_ade_2d_tez_probes
from fdtdz_jax.jax_yee_ade_2d_tez import simulate_jax_yee_ade_2d_tez_line_probes
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.metasurface_2d import huygens_tez_current_waveforms
from fdtdz_jax.metasurface_2d import huygens_tez_line_source
from fdtdz_jax.yee_ade_2d import prepare_material_grid_2d
from fdtdz_jax.yee_ade_2d import prepare_y_cpml
from fdtdz_jax.yee_ade_2d_tez import initialize_yee_ade_2d_tez
from fdtdz_jax.yee_ade_2d_tez import simulate_yee_ade_2d_tez
from fdtdz_jax.yee_ade_2d_tez import yee_ade_2d_tez_step


def _assert_close(actual, expected):
  for name in (
      "electric_x", "electric_y", "magnetic_z", "displacement_x",
      "displacement_y", "displacement_x_previous",
      "displacement_y_previous", "electric_x_previous",
      "electric_y_previous", "polarization_x_current",
      "polarization_x_previous", "polarization_y_current",
      "polarization_y_previous", "cpml_dx_y", "cpml_hz_y"):
    np.testing.assert_allclose(
        np.asarray(getattr(actual, name)), getattr(expected, name),
        rtol=5e-6, atol=5e-6)
  assert int(actual.step) == expected.step


def test_jax_tez_step_matches_numpy_with_cpml_and_multipoles():
  shape = (7, 36)
  materials = [Lossless(1.0), Multipole(1.0, (
      LorentzPole(1.1, 3.0, 0.2), DrudePole(2.0, 0.3)))]
  ids = np.zeros(shape, dtype=int)
  ids[2:5, 16:21] = 1
  cpml = prepare_y_cpml(shape, 7, 0.2, 1.0)
  grid = prepare_material_grid_2d(
      materials, ids, 0.2, 1.0, 1.0, dtype=np.float32, y_cpml=cpml)
  numpy_state = initialize_yee_ade_2d_tez(grid)
  rng = np.random.default_rng(17)
  sources = [rng.normal(size=shape).astype(np.float32) * 0.01
             for _ in range(3)]
  expected = yee_ade_2d_tez_step(grid, numpy_state, *sources)
  actual = jax_yee_ade_2d_tez_step(
      jax_material_grid_2d(grid),
      jax_yee_ade_2d_tez_state(numpy_state, grid), *sources)
  _assert_close(actual, expected)


def test_jax_tez_compact_probes_match_numpy_huygens_run():
  shape = (5, 80)
  num_steps = 100
  dt = 0.3
  waveform = (np.sin(np.arange(num_steps) * 0.2) *
              np.exp(-((np.arange(num_steps) - 20) / 8)**2)).astype(np.float32)
  source = huygens_tez_line_source(
      waveform, shape, 22, direction=1, dtype=np.float32)
  cpml = prepare_y_cpml(shape, 12, dt, 1.0)
  ids = np.zeros(shape, dtype=int)
  numpy_result = simulate_yee_ade_2d_tez(
      [Lossless(1.0)], ids, 1.0, 1.0, dt, num_steps,
      current_density_x=source.current_density_x,
      magnetic_current_z=source.magnetic_current_z,
      dtype=np.float32, y_cpml=cpml)
  grid = prepare_material_grid_2d(
      [Lossless(1.0)], ids, dt, 1.0, 1.0,
      dtype=np.float32, y_cpml=cpml)
  electric, magnetic = huygens_tez_current_waveforms(
      waveform, direction=1, dtype=np.float32)
  state, probes = simulate_jax_yee_ade_2d_tez_probes(
      jax_material_grid_2d(grid),
      jax_yee_ade_2d_tez_state(initialize_yee_ade_2d_tez(grid), grid),
      electric, magnetic, np.ones(shape[0], np.float32),
      source.electric_y_index, source.magnetic_y_index, 35, 60)
  np.testing.assert_allclose(
      np.asarray(state.electric_x), numpy_result.final_state.electric_x,
      rtol=5e-5, atol=5e-6)
  np.testing.assert_allclose(
      np.asarray(probes[0]), numpy_result.electric_x[:, :, 35].mean(axis=1),
      rtol=5e-5, atol=5e-6)
  np.testing.assert_allclose(
      np.asarray(probes[1]), numpy_result.electric_x[:, :, 60].mean(axis=1),
      rtol=5e-5, atol=5e-6)
  line_state, line_probes = simulate_jax_yee_ade_2d_tez_line_probes(
      jax_material_grid_2d(grid),
      jax_yee_ade_2d_tez_state(initialize_yee_ade_2d_tez(grid), grid),
      electric, magnetic, np.ones(shape[0], np.float32),
      source.electric_y_index, source.magnetic_y_index, 35, 60)
  np.testing.assert_allclose(
      np.asarray(line_state.electric_x), numpy_result.final_state.electric_x,
      rtol=5e-5, atol=5e-6)
  np.testing.assert_allclose(
      np.asarray(line_probes[0]), numpy_result.electric_x[:, :, 35],
      rtol=5e-5, atol=5e-6)
