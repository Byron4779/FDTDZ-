import numpy as np

from fdtdz_jax.materials import Lossless
from fdtdz_jax.metasurface_2d import gaussian_sine_pulse
from fdtdz_jax.scattering_workflow_2d import jax_runtime_report
from fdtdz_jax.scattering_workflow_2d import run_tmz_scattering_2d
from fdtdz_jax.scattering_workflow_2d import run_tez_scattering_2d


def _empty_workflow(backend):
  shape = (3, 100)
  num_steps = 220
  dt = 0.45
  waveform = gaussian_sine_pulse(
      num_steps, dt, center_time=15.0, width=5.0, frequency=0.06,
      dtype=np.float32)
  ids = np.zeros(shape, dtype=int)
  return run_tmz_scattering_2d(
      [Lossless(1.0)], ids, ids.copy(), 1.0, 1.0, dt, waveform,
      source_y=25, reflection_probe_y=40, transmission_probe_y=65,
      cpml_width=15, backend=backend, window=None)


def test_empty_numpy_workflow_has_zero_r_unit_t_and_zero_a():
  result = _empty_workflow("numpy")
  valid = result.spectrum.valid
  np.testing.assert_allclose(result.spectrum.reflection[valid], 0.0)
  np.testing.assert_allclose(result.spectrum.transmission[valid], 1.0)
  np.testing.assert_allclose(result.spectrum.absorption[valid], 0.0, atol=1e-6)
  assert result.final_device_electric_z.shape == (3, 100)
  diffraction_valid = result.diffraction.valid
  np.testing.assert_allclose(
      result.diffraction.transmission[diffraction_valid], 1.0)
  np.testing.assert_allclose(
      result.diffraction.reflection[diffraction_valid], 0.0)


def test_jax_and_numpy_empty_workflows_match():
  numpy_result = _empty_workflow("numpy")
  jax_result = _empty_workflow("jax")
  np.testing.assert_allclose(
      jax_result.reference_transmission,
      numpy_result.reference_transmission, rtol=5e-5, atol=5e-6)
  np.testing.assert_allclose(
      jax_result.final_device_electric_z,
      numpy_result.final_device_electric_z, rtol=5e-5, atol=5e-6)


def test_jax_runtime_report_has_backend_and_devices():
  report = jax_runtime_report()
  assert report["backend"] in ("cpu", "gpu", "tpu")
  assert report["devices"]
  assert isinstance(report["has_gpu"], bool)


def _empty_tez_workflow(backend):
  shape = (3, 100)
  num_steps = 220
  dt = 0.45
  waveform = gaussian_sine_pulse(
      num_steps, dt, center_time=15.0, width=5.0, frequency=0.06,
      dtype=np.float32)
  ids = np.zeros(shape, dtype=int)
  return run_tez_scattering_2d(
      [Lossless(1.0)], ids, ids.copy(), 1.0, 1.0, dt, waveform,
      source_y=25, reflection_probe_y=40, transmission_probe_y=65,
      cpml_width=15, backend=backend, window=None)


def test_empty_tez_jax_workflow_has_unit_transmission():
  result = _empty_tez_workflow("jax")
  valid = result.spectrum.valid
  np.testing.assert_allclose(result.spectrum.reflection[valid], 0.0)
  np.testing.assert_allclose(result.spectrum.transmission[valid], 1.0)
  np.testing.assert_allclose(result.spectrum.absorption[valid], 0.0, atol=1e-6)
  diffraction_valid = result.diffraction.valid
  np.testing.assert_allclose(
      result.diffraction.transmission[diffraction_valid], 1.0)


def test_tez_jax_and_numpy_workflows_match():
  numpy_result = _empty_tez_workflow("numpy")
  jax_result = _empty_tez_workflow("jax")
  np.testing.assert_allclose(
      jax_result.reference_reflection, numpy_result.reference_reflection,
      rtol=5e-5, atol=5e-6)
  np.testing.assert_allclose(
      jax_result.final_device_electric_x,
      numpy_result.final_device_electric_x, rtol=5e-5, atol=5e-6)
