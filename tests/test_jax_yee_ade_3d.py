import numpy as np

from fdtdz_jax.jax_yee_ade_3d import advance_jax_yee_ade_3d
from fdtdz_jax.jax_yee_ade_3d import jax_material_grid_3d
from fdtdz_jax.jax_yee_ade_3d import jax_yee_ade_3d_state
from fdtdz_jax.jax_yee_ade_3d import jax_yee_ade_3d_step
from fdtdz_jax.jax_yee_ade_3d import simulate_jax_yee_ade_3d
from fdtdz_jax.jax_yee_ade_3d import simulate_jax_yee_ade_3d_plane_probes
from fdtdz_jax.jax_yee_ade_3d import (
    simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields)
from fdtdz_jax.jax_yee_ade_3d import (
    simulate_jax_yee_ade_3d_plane_probes_with_snapshots)
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.metasurface_3d import huygens_plane_source
from fdtdz_jax.yee_ade_3d import initialize_yee_ade_3d
from fdtdz_jax.yee_ade_3d import prepare_material_grid_3d
from fdtdz_jax.yee_ade_3d import prepare_z_cpml
from fdtdz_jax.yee_ade_3d import yee_ade_3d_step


def _assert_state_close(actual, expected, rtol=8e-6, atol=8e-6):
  for name in (
      "electric", "magnetic", "displacement", "displacement_previous",
      "electric_previous", "cpml_hx_ey_z", "cpml_hy_ex_z",
      "cpml_dx_hy_z", "cpml_dy_hx_z", "polarization_current",
      "polarization_previous"):
    np.testing.assert_allclose(
        np.asarray(getattr(actual, name)), getattr(expected, name),
        rtol=rtol, atol=atol)
  assert int(actual.step) == expected.step


def test_jax_three_dimensional_step_matches_numpy_direct_ade():
  shape = (4, 5, 6)
  ids = np.arange(np.prod(shape)).reshape(shape) % 2
  grid = prepare_material_grid_3d(
      [Lossless(1.0), Debye(1.5, 3.0, 0.4)], ids,
      0.2, 1.0, 1.0, 1.0, dtype=np.float32)
  initial = initialize_yee_ade_3d(grid)
  rng = np.random.default_rng(31)
  electric_current = rng.normal(size=grid.vector_shape).astype(np.float32)
  magnetic_current = rng.normal(size=grid.vector_shape).astype(np.float32)
  expected = yee_ade_3d_step(
      grid, initial, electric_current, magnetic_current)
  actual = jax_yee_ade_3d_step(
      jax_material_grid_3d(grid), jax_yee_ade_3d_state(initial, grid),
      electric_current, magnetic_current)

  _assert_state_close(actual, expected)


def test_jax_three_dimensional_scan_matches_numpy_cpml_multipole():
  shape = (4, 5, 20)
  materials = [Lossless(1.0), Multipole(1.0, (
      LorentzPole(1.2, 3.0, 0.2), DrudePole(2.0, 0.4)))]
  ids = np.zeros(shape, dtype=int)
  ids[1:3, 1:4, 9:12] = 1
  cpml = prepare_z_cpml(shape, 4, 0.25, 1.0, dtype=np.float32)
  grid = prepare_material_grid_3d(
      materials, ids, 0.25, 1.0, 1.0, 1.0,
      dtype=np.float32, z_cpml=cpml)
  initial = initialize_yee_ade_3d(grid)
  rng = np.random.default_rng(37)
  electric_sources = (
      0.002 * rng.normal(size=(20,) + grid.vector_shape)).astype(np.float32)
  magnetic_sources = (
      0.002 * rng.normal(size=(20,) + grid.vector_shape)).astype(np.float32)
  expected = initial
  for electric_source, magnetic_source in zip(
      electric_sources, magnetic_sources):
    expected = yee_ade_3d_step(
        grid, expected, electric_source, magnetic_source)

  actual, records = simulate_jax_yee_ade_3d(
      jax_material_grid_3d(grid), jax_yee_ade_3d_state(initial, grid),
      electric_sources, magnetic_sources)
  assert len(records) == 3
  assert records[0].shape == (20,) + grid.vector_shape
  _assert_state_close(actual, expected)
  final_only = advance_jax_yee_ade_3d(
      jax_material_grid_3d(grid), jax_yee_ade_3d_state(initial, grid),
      electric_sources, magnetic_sources)
  _assert_state_close(final_only, expected)


def test_compact_jax_huygens_plane_probes_match_expanded_source_scan():
  shape = (4, 5, 40)
  num_steps = 36
  dt = 0.25
  cpml = prepare_z_cpml(shape, 8, dt, 1.0, dtype=np.float32)
  grid = prepare_material_grid_3d(
      [Lossless(1.0)], np.zeros(shape, dtype=int), dt, 1.0, 1.0, 1.0,
      dtype=np.float32, z_cpml=cpml)
  initial = initialize_yee_ade_3d(grid)
  rng = np.random.default_rng(41)
  waveform = (0.01 * rng.normal(size=num_steps)).astype(np.float32)
  profile = np.linspace(
      0.5, 1.0, shape[0] * shape[1], dtype=np.float32).reshape(shape[:2])
  source = huygens_plane_source(
      waveform, shape, 12, polarization_xy=(0.6, 0.8), direction=1,
      transverse_profile=profile, dtype=np.float32, expand=True)
  jax_grid = jax_material_grid_3d(grid)
  jax_initial = jax_yee_ade_3d_state(initial, grid)
  full_state, full_records = simulate_jax_yee_ade_3d(
      jax_grid, jax_initial, source.electric_current,
      source.magnetic_current)
  compact_state, probes = simulate_jax_yee_ade_3d_plane_probes(
      jax_grid, jax_initial, source.electric_waveform,
      source.magnetic_waveform, source.transverse_profile,
      np.asarray(source.polarization_xy, dtype=np.float32),
      electric_source_z=source.electric_z_index,
      magnetic_source_z=source.magnetic_z_index,
      reflection_probe_z=18, transmission_probe_z=27)

  _assert_state_close(compact_state, full_state)
  np.testing.assert_allclose(
      np.asarray(probes[0]), np.asarray(full_records[0])[:, :2, :, :, 18],
      rtol=2e-6, atol=2e-7)
  np.testing.assert_allclose(
      np.asarray(probes[1]), np.asarray(full_records[0])[:, :2, :, :, 27],
      rtol=2e-6, atol=2e-7)

  snapshot_steps = (5, 17, 36)
  snapshot_state, snapshot_probes, snapshots = (
      simulate_jax_yee_ade_3d_plane_probes_with_snapshots(
          jax_grid, jax_initial, source.electric_waveform,
          source.magnetic_waveform, source.transverse_profile,
          np.asarray(source.polarization_xy, dtype=np.float32),
          electric_source_z=source.electric_z_index,
          magnetic_source_z=source.magnetic_z_index,
          reflection_probe_z=18, transmission_probe_z=27,
          snapshot_steps=snapshot_steps))
  _assert_state_close(snapshot_state, full_state)
  np.testing.assert_allclose(
      np.asarray(snapshot_probes[0]), np.asarray(probes[0]),
      rtol=2e-6, atol=2e-7)
  selected = np.asarray(snapshot_steps) - 1
  for snapshot, full_history in zip(snapshots, full_records):
    np.testing.assert_allclose(
        np.asarray(snapshot), np.asarray(full_history)[selected],
        rtol=2e-6, atol=2e-7)

  frequencies = np.asarray((0.06, 0.11), dtype=np.float32)
  weights = np.hanning(num_steps).astype(np.float32)
  frequency_state, frequency_probes, frequency_snapshots, fields = (
      simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields(
          jax_grid, jax_initial, source.electric_waveform,
          source.magnetic_waveform, source.transverse_profile,
          np.asarray(source.polarization_xy, dtype=np.float32),
          frequencies, weights,
          electric_source_z=source.electric_z_index,
          magnetic_source_z=source.magnetic_z_index,
          reflection_probe_z=18, transmission_probe_z=27,
          snapshot_steps=snapshot_steps))
  _assert_state_close(frequency_state, full_state)
  np.testing.assert_allclose(
      np.asarray(frequency_probes[0]), np.asarray(probes[0]),
      rtol=2e-6, atol=2e-7)
  for snapshot, expected_snapshot in zip(frequency_snapshots, snapshots):
    np.testing.assert_allclose(
        np.asarray(snapshot), np.asarray(expected_snapshot),
        rtol=2e-6, atol=2e-7)
  electric_times = (np.arange(num_steps) + 1.0) * dt
  magnetic_times = (np.arange(num_steps) + 0.5) * dt
  phases = dt * weights[None, :] * np.exp(
      2j * np.pi * frequencies[:, None] * electric_times[None, :])
  magnetic_phases = dt * weights[None, :] * np.exp(
      2j * np.pi * frequencies[:, None] * magnetic_times[None, :])
  expected_fields = (
      np.einsum("ft,tcxyz->fcxyz", phases, np.asarray(full_records[0])),
      np.einsum("ft,tcxyz->fcxyz", magnetic_phases,
                np.asarray(full_records[1])),
      np.einsum("ft,tcxyz->fcxyz", phases, np.asarray(full_records[2])))
  for actual, expected in zip(fields, expected_fields):
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=2e-5, atol=2e-6)
