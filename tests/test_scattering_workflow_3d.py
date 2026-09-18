import numpy as np
import pytest

from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.metasurface_3d import huygens_plane_source
from fdtdz_jax.scattering_workflow_3d import run_metasurface_scattering_3d


def _problem():
  shape = (4, 4, 100)
  num_steps = 220
  dt = 0.45
  time = np.arange(num_steps) * dt
  waveform = np.exp(-0.5 * ((time - 18.0) / 6.0)**2) * np.sin(
      2.0 * np.pi * 0.06 * (time - 18.0))
  source = huygens_plane_source(
      waveform, shape, z_index=22, polarization_xy=(1.0, 0.0),
      direction=1)
  reference_ids = np.zeros(shape, dtype=np.uint8)
  return shape, dt, source, reference_ids


def _peak_index(diffraction):
  amplitude = np.sqrt(np.sum(
      np.abs(diffraction.reference_incident_field)**2, axis=1))
  candidates = np.flatnonzero(diffraction.valid)
  return candidates[np.argmax(amplitude[candidates])]


def test_empty_reference_device_pair_has_unit_transmission():
  _, dt, source, reference_ids = _problem()
  result = run_metasurface_scattering_3d(
      [Lossless(1.0)], reference_ids, reference_ids.copy(),
      1.0, 1.0, 1.0, dt, source,
      reflection_probe_z=35, transmission_probe_z=72, cpml_width=14)
  peak = _peak_index(result.diffraction)

  assert result.diffraction.reflection[peak] == pytest.approx(0.0, abs=1e-14)
  assert result.diffraction.transmission[peak] == pytest.approx(1.0)
  assert result.diffraction.absorption[peak] == pytest.approx(0.0, abs=1e-14)
  assert result.reference_reflection.shape == (220, 2, 4, 4)
  assert result.final_device_state.electric.shape == (3, 4, 4, 100)
  assert result.device == "NumPy CPU reference"


def test_patterned_multipole_device_produces_finite_scattering_and_polarization():
  _, dt, source, reference_ids = _problem()
  metal = Multipole(1.0, (
      DrudePole(0.55, 0.025), LorentzPole(0.8, 0.42, 0.025)))
  device_ids = reference_ids.copy()
  device_ids[1:3, 1:3, 49:53] = 1
  result = run_metasurface_scattering_3d(
      [Lossless(1.0), metal], reference_ids, device_ids,
      1.0, 1.0, 1.0, dt, source,
      reflection_probe_z=35, transmission_probe_z=72, cpml_width=14)
  peak = _peak_index(result.diffraction)

  assert np.all(np.isfinite(result.diffraction.reflection[result.diffraction.valid]))
  assert np.all(np.isfinite(result.diffraction.transmission[result.diffraction.valid]))
  assert result.diffraction.reflection[peak] > 0.0
  assert 0.0 < result.diffraction.transmission[peak] < 1.0
  assert 0.0 < result.diffraction.absorption[peak] < 1.0
  polarization = result.final_device_state.polarization_current
  assert np.any(np.abs(polarization[:, 1:3, 1:3, 49:53]) > 0.0)
  assert not np.any(polarization[:, 0, 0, 0])


def test_scattering_workflow_rejects_probe_order_and_cpml_mismatch():
  _, dt, source, reference_ids = _problem()
  with pytest.raises(ValueError, match="source_z < reflection_probe_z"):
    run_metasurface_scattering_3d(
        [Lossless(1.0)], reference_ids, reference_ids,
        1.0, 1.0, 1.0, dt, source,
        reflection_probe_z=18, transmission_probe_z=72, cpml_width=14)

  changed_cpml = reference_ids.copy()
  changed_cpml[:, :, 2] = 1
  with pytest.raises(ValueError, match="match inside z CPML"):
    run_metasurface_scattering_3d(
        [Lossless(1.0), Lossless(2.0)], reference_ids, changed_cpml,
        1.0, 1.0, 1.0, dt, source,
        reflection_probe_z=35, transmission_probe_z=72, cpml_width=14)


def test_jax_scattering_workflow_matches_numpy_patterned_device():
  _, dt, source, reference_ids = _problem()
  metal = Multipole(1.0, (
      DrudePole(0.55, 0.025), LorentzPole(0.8, 0.42, 0.025)))
  device_ids = reference_ids.copy()
  device_ids[1:3, 1:3, 49:53] = 1
  common = dict(
      materials=[Lossless(1.0), metal],
      reference_material_ids=reference_ids,
      device_material_ids=device_ids, dx=1.0, dy=1.0, dz=1.0, dt=dt,
      source=source, reflection_probe_z=35, transmission_probe_z=72,
      cpml_width=14, dtype=np.float32, window=None, incident_floor=1e-3,
      snapshot_steps=(20, 100, 220), frequencies=(0.04, 0.06),
      frequency_window="hann")
  numpy_result = run_metasurface_scattering_3d(backend="numpy", **common)
  jax_result = run_metasurface_scattering_3d(backend="jax", **common)
  valid = numpy_result.diffraction.valid & jax_result.diffraction.valid

  assert np.any(valid)
  for name in ("reflection", "transmission", "absorption"):
    np.testing.assert_allclose(
        getattr(jax_result.diffraction, name)[valid],
        getattr(numpy_result.diffraction, name)[valid],
        rtol=1e-3, atol=5e-4)
  np.testing.assert_allclose(
      jax_result.final_device_state.electric,
      numpy_result.final_device_state.electric,
      rtol=2e-5, atol=2e-5)
  assert jax_result.backend == "jax"
  assert jax_result.device
  assert jax_result.field_snapshots.steps.tolist() == [20, 100, 220]
  for name in ("electric", "magnetic", "displacement", "polarization"):
    np.testing.assert_allclose(
        getattr(jax_result.field_snapshots, name),
        getattr(numpy_result.field_snapshots, name),
        rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        getattr(jax_result.frequency_fields, name),
        getattr(numpy_result.frequency_fields, name),
        rtol=3e-5, atol=3e-5)
