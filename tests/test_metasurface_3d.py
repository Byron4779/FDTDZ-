import numpy as np
import pytest

from fdtdz_jax.materials import Lossless
from fdtdz_jax.metasurface_3d import huygens_plane_source
from fdtdz_jax.yee_ade_3d import prepare_z_cpml
from fdtdz_jax.yee_ade_3d import simulate_yee_ade_3d


def test_huygens_plane_source_builds_staggered_vector_currents():
  waveform = np.asarray([2.0, 4.0, 6.0])
  source = huygens_plane_source(
      waveform, (3, 4, 10), z_index=5, polarization_xy=(3.0, 4.0),
      direction=1, wave_impedance=2.0, expand=True)

  assert source.electric_z_index == 5
  assert source.magnetic_z_index == 4
  assert source.polarization_xy == pytest.approx((0.6, 0.8))
  np.testing.assert_allclose(
      source.electric_current[:, 0, :, :, 5],
      np.broadcast_to(waveform[:, None, None] * 0.6, (3, 3, 4)))
  np.testing.assert_allclose(
      source.electric_current[:, 1, :, :, 5],
      np.broadcast_to(waveform[:, None, None] * 0.8, (3, 3, 4)))
  half_step = np.asarray([1.0, 3.0, 5.0])
  np.testing.assert_allclose(
      source.magnetic_current[:, 0, :, :, 4],
      np.broadcast_to(-2.0 * half_step[:, None, None] * 0.8, (3, 3, 4)))
  np.testing.assert_allclose(
      source.magnetic_current[:, 1, :, :, 4],
      np.broadcast_to(2.0 * half_step[:, None, None] * 0.6, (3, 3, 4)))
  assert not np.any(source.electric_current[:, 2])
  assert not np.any(source.magnetic_current[:, 2])


def test_huygens_plane_source_is_compact_by_default():
  source = huygens_plane_source(
      np.ones(20), (8, 9, 40), z_index=12,
      polarization_xy=(1.0, 0.0))
  assert source.electric_current is None
  assert source.magnetic_current is None
  assert source.electric_waveform.shape == (20,)
  assert source.transverse_profile.shape == (8, 9)
  assert source.shape == (8, 9, 40)


@pytest.mark.parametrize("direction", [-1, 1])
def test_huygens_plane_source_propagates_in_requested_z_direction(direction):
  shape = (2, 2, 180)
  num_steps = 260
  dt = 0.5
  time = np.arange(num_steps) * dt
  waveform = np.exp(-0.5 * ((time - 22.0) / 6.0)**2) * np.sin(
      2.0 * np.pi * 0.05 * (time - 22.0))
  source = huygens_plane_source(
      waveform, shape, z_index=90, polarization_xy=(1.0, 0.0),
      direction=direction, expand=True)
  cpml = prepare_z_cpml(shape, width=24, dt=dt, dz=1.0)
  result = simulate_yee_ade_3d(
      [Lossless(1.0)], np.zeros(shape, dtype=int), 1.0, 1.0, 1.0,
      dt, num_steps, electric_current=source.electric_current,
      magnetic_current=source.magnetic_current, z_cpml=cpml)
  positive_z_energy = np.sum(result.electric[:, 0, :, :, 110]**2)
  negative_z_energy = np.sum(result.electric[:, 0, :, :, 70]**2)
  requested = positive_z_energy if direction == 1 else negative_z_energy
  rejected = negative_z_energy if direction == 1 else positive_z_energy
  assert requested > 100.0 * rejected


def test_huygens_plane_source_validates_polarization_and_staggering():
  with pytest.raises(ValueError, match="zero vector"):
    huygens_plane_source([1.0], (2, 2, 8), 4, polarization_xy=(0.0, 0.0))
  with pytest.raises(ValueError, match="z_index >= 1"):
    huygens_plane_source([1.0], (2, 2, 8), 0, direction=1)
