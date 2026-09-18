"""End-to-end 2D TMz reference/device scattering workflow."""

from dataclasses import dataclass

import numpy as np

from .diffraction_2d import DiffractionSpectrum2D
from .diffraction_2d import diffraction_spectrum
from .jax_yee_ade_2d import jax_material_grid_2d
from .jax_yee_ade_2d import jax_yee_ade_2d_state
from .jax_yee_ade_2d import simulate_jax_yee_ade_2d_line_probes
from .jax_yee_ade_2d_tez import jax_yee_ade_2d_tez_state
from .jax_yee_ade_2d_tez import simulate_jax_yee_ade_2d_tez_line_probes
from .metasurface_2d import ScatteringSpectrum2D
from .metasurface_2d import huygens_line_source
from .metasurface_2d import huygens_current_waveforms
from .metasurface_2d import huygens_tez_current_waveforms
from .metasurface_2d import huygens_tez_line_source
from .metasurface_2d import sample_line
from .metasurface_2d import scattering_spectrum
from .yee_ade_2d import initialize_yee_ade_2d
from .yee_ade_2d import prepare_material_grid_2d
from .yee_ade_2d import prepare_y_cpml
from .yee_ade_2d import simulate_yee_ade_2d
from .yee_ade_2d_tez import initialize_yee_ade_2d_tez
from .yee_ade_2d_tez import simulate_yee_ade_2d_tez


@dataclass(frozen=True)
class TMzScatteringResult2D:
  """Probe traces, spectrum, and final device field from a paired run."""

  reference_reflection: np.ndarray
  device_reflection: np.ndarray
  reference_transmission: np.ndarray
  device_transmission: np.ndarray
  spectrum: ScatteringSpectrum2D
  final_device_electric_z: np.ndarray
  backend: str
  device: str
  diffraction: DiffractionSpectrum2D = None


@dataclass(frozen=True)
class TEzScatteringResult2D:
  """TEz probe traces, spectrum, and final device Ex field."""

  reference_reflection: np.ndarray
  device_reflection: np.ndarray
  reference_transmission: np.ndarray
  device_transmission: np.ndarray
  spectrum: ScatteringSpectrum2D
  final_device_electric_x: np.ndarray
  backend: str
  device: str
  diffraction: DiffractionSpectrum2D = None


def jax_runtime_report():
  """Return the active JAX backend and visible devices without changing them."""
  import jax
  devices = tuple(str(device) for device in jax.devices())
  return {
      "backend": jax.default_backend(),
      "devices": devices,
      "has_gpu": any(device.platform == "gpu" for device in jax.devices()),
  }


def _record_tmz(materials, material_ids, dx, dy, dt, waveform, source_y,
                reflection_probe_y, transmission_probe_y, direction,
                wave_impedance, cpml, backend, dtype):
  shape = np.asarray(material_ids).shape
  if backend == "numpy":
    source = huygens_line_source(
        waveform, shape, source_y, direction=direction,
        wave_impedance=wave_impedance, dtype=dtype)
    result = simulate_yee_ade_2d(
        materials, material_ids, dx, dy, dt,
        source.current_density_z.shape[0],
        current_density_z=source.current_density_z,
        magnetic_current_x=source.magnetic_current_x,
        dtype=dtype, y_cpml=cpml)
    return (
        sample_line(result.electric_z, reflection_probe_y, average_x=False),
        sample_line(result.electric_z, transmission_probe_y, average_x=False),
        result.final_state.electric_z, "NumPy CPU reference")
  if backend != "jax":
    raise ValueError("backend must be 'numpy' or 'jax'.")
  grid = prepare_material_grid_2d(
      materials, material_ids, dt, dx, dy, dtype=dtype, y_cpml=cpml)
  initial = initialize_yee_ade_2d(grid)
  electric_waveform, magnetic_waveform = huygens_current_waveforms(
      waveform, direction=direction, wave_impedance=wave_impedance,
      dtype=dtype)
  final_state, probes = simulate_jax_yee_ade_2d_line_probes(
      jax_material_grid_2d(grid), jax_yee_ade_2d_state(initial, grid),
      electric_waveform, magnetic_waveform,
      np.ones(shape[0], dtype=dtype), source_y, reflection_probe_y,
      transmission_probe_y)
  final_state.electric_z.block_until_ready()
  report = jax_runtime_report()
  return (np.asarray(probes[0]), np.asarray(probes[1]),
          np.asarray(final_state.electric_z),
          ", ".join(report["devices"]))


def _record_tez(materials, material_ids, dx, dy, dt, waveform, source_y,
                reflection_probe_y, transmission_probe_y, direction,
                wave_impedance, cpml, backend, dtype):
  shape = np.asarray(material_ids).shape
  source = huygens_tez_line_source(
      waveform, shape, source_y, direction=direction,
      wave_impedance=wave_impedance, dtype=dtype)
  if backend == "numpy":
    result = simulate_yee_ade_2d_tez(
        materials, material_ids, dx, dy, dt, waveform.size,
        current_density_x=source.current_density_x,
        magnetic_current_z=source.magnetic_current_z,
        dtype=dtype, y_cpml=cpml)
    return (
        sample_line(result.electric_x, reflection_probe_y, average_x=False),
        sample_line(result.electric_x, transmission_probe_y, average_x=False),
        result.final_state.electric_x, "NumPy CPU reference")
  if backend != "jax":
    raise ValueError("backend must be 'numpy' or 'jax'.")
  grid = prepare_material_grid_2d(
      materials, material_ids, dt, dx, dy, dtype=dtype, y_cpml=cpml)
  initial = initialize_yee_ade_2d_tez(grid)
  electric_waveform, magnetic_waveform = huygens_tez_current_waveforms(
      waveform, direction=direction, wave_impedance=wave_impedance,
      dtype=dtype)
  final_state, probes = simulate_jax_yee_ade_2d_tez_line_probes(
      jax_material_grid_2d(grid), jax_yee_ade_2d_tez_state(initial, grid),
      electric_waveform, magnetic_waveform,
      np.ones(shape[0], dtype=dtype), source.electric_y_index,
      source.magnetic_y_index, reflection_probe_y, transmission_probe_y)
  final_state.electric_x.block_until_ready()
  report = jax_runtime_report()
  return (np.asarray(probes[0]), np.asarray(probes[1]),
          np.asarray(final_state.electric_x), ", ".join(report["devices"]))


def _validate_scattering_geometry(reference_ids, device_ids, source_y,
                                  reflection_probe_y, transmission_probe_y,
                                  direction, cpml):
  reference_ids = np.asarray(reference_ids)
  device_ids = np.asarray(device_ids)
  if (reference_ids.ndim != 2 or device_ids.shape != reference_ids.shape or
      not np.issubdtype(reference_ids.dtype, np.integer) or
      not np.issubdtype(device_ids.dtype, np.integer)):
    raise ValueError(
        "reference and device material maps must be matching 2D integer arrays.")
  nx, ny = reference_ids.shape
  indices = (source_y, reflection_probe_y, transmission_probe_y)
  if any(isinstance(value, (bool, np.bool_)) or
         not isinstance(value, (int, np.integer)) or
         value < 0 or value >= ny for value in indices):
    raise ValueError("source and probe y indices must lie inside the grid.")
  if direction == 1 and not (
      source_y < reflection_probe_y < transmission_probe_y):
    raise ValueError(
        "for +y propagation require source_y < reflection_probe_y < "
        "transmission_probe_y.")
  if direction == -1 and not (
      source_y > reflection_probe_y > transmission_probe_y):
    raise ValueError(
        "for -y propagation require source_y > reflection_probe_y > "
        "transmission_probe_y.")
  boundary = np.zeros((nx, ny), dtype=bool)
  boundary[:, :cpml.width] = True
  boundary[:, -cpml.width:] = True
  if np.any(reference_ids[boundary] != device_ids[boundary]):
    raise ValueError("reference and device material maps must match inside CPML.")
  return reference_ids, device_ids


def run_tmz_scattering_2d(materials, reference_material_ids,
                          device_material_ids, dx, dy, dt, waveform,
                          source_y, reflection_probe_y,
                          transmission_probe_y, cpml_width,
                          direction=1, wave_impedance=1.0,
                          backend="numpy", dtype=np.float32,
                          target_reflection=1e-8,
                          transmission_power_factor=1.0,
                          incident_floor=1e-6, window="hann",
                          background_eps_r=1.0):
  """Run matched reference/device simulations and calculate R/T/A spectra."""
  reference_ids = np.asarray(reference_material_ids)
  device_ids = np.asarray(device_material_ids)
  if reference_ids.ndim != 2:
    raise ValueError("reference material map must be two-dimensional.")
  nx, ny = reference_ids.shape
  waveform = np.asarray(waveform, dtype=dtype)
  cpml = prepare_y_cpml(
      (nx, ny), cpml_width, dt, dy,
      target_reflection=target_reflection, dtype=dtype)
  reference_ids, device_ids = _validate_scattering_geometry(
      reference_ids, device_ids, source_y, reflection_probe_y,
      transmission_probe_y, direction, cpml)

  reference_reflection_line, reference_transmission_line, _, reference_device = (
      _record_tmz(
          materials, reference_ids, dx, dy, dt, waveform, source_y,
          reflection_probe_y, transmission_probe_y, direction,
          wave_impedance, cpml, backend, dtype))
  device_reflection_line, device_transmission_line, final_device_field, actual_device = (
      _record_tmz(
          materials, device_ids, dx, dy, dt, waveform, source_y,
          reflection_probe_y, transmission_probe_y, direction,
          wave_impedance, cpml, backend, dtype))
  reference_reflection = np.mean(reference_reflection_line, axis=1)
  reference_transmission = np.mean(reference_transmission_line, axis=1)
  device_reflection = np.mean(device_reflection_line, axis=1)
  device_transmission = np.mean(device_transmission_line, axis=1)
  spectrum = scattering_spectrum(
      reference_reflection, device_reflection, reference_transmission,
      device_transmission, dt,
      transmission_power_factor=transmission_power_factor,
      incident_floor=incident_floor, window=window)
  diffraction = diffraction_spectrum(
      reference_reflection_line, device_reflection_line,
      reference_transmission_line, device_transmission_line,
      dt, dx, background_eps_r=background_eps_r, polarization="tmz",
      incident_floor=incident_floor, window=window)
  return TMzScatteringResult2D(
      reference_reflection, device_reflection, reference_transmission,
      device_transmission, spectrum, final_device_field, backend,
      actual_device or reference_device, diffraction)


def run_tez_scattering_2d(materials, reference_material_ids,
                          device_material_ids, dx, dy, dt, waveform,
                          source_y, reflection_probe_y,
                          transmission_probe_y, cpml_width,
                          direction=1, wave_impedance=1.0,
                          backend="numpy", dtype=np.float32,
                          target_reflection=1e-8,
                          transmission_power_factor=1.0,
                          incident_floor=1e-6, window="hann",
                          background_eps_r=1.0):
  """Run matched TEz reference/device simulations and calculate R/T/A."""
  reference_ids = np.asarray(reference_material_ids)
  device_ids = np.asarray(device_material_ids)
  if reference_ids.ndim != 2:
    raise ValueError("reference material map must be two-dimensional.")
  nx, ny = reference_ids.shape
  waveform = np.asarray(waveform, dtype=dtype)
  cpml = prepare_y_cpml(
      (nx, ny), cpml_width, dt, dy,
      target_reflection=target_reflection, dtype=dtype)
  reference_ids, device_ids = _validate_scattering_geometry(
      reference_ids, device_ids, source_y, reflection_probe_y,
      transmission_probe_y, direction, cpml)
  reference_reflection_line, reference_transmission_line, _, reference_device = (
      _record_tez(
          materials, reference_ids, dx, dy, dt, waveform, source_y,
          reflection_probe_y, transmission_probe_y, direction,
          wave_impedance, cpml, backend, dtype))
  device_reflection_line, device_transmission_line, final_device_field, actual_device = (
      _record_tez(
          materials, device_ids, dx, dy, dt, waveform, source_y,
          reflection_probe_y, transmission_probe_y, direction,
          wave_impedance, cpml, backend, dtype))
  reference_reflection = np.mean(reference_reflection_line, axis=1)
  reference_transmission = np.mean(reference_transmission_line, axis=1)
  device_reflection = np.mean(device_reflection_line, axis=1)
  device_transmission = np.mean(device_transmission_line, axis=1)
  spectrum = scattering_spectrum(
      reference_reflection, device_reflection, reference_transmission,
      device_transmission, dt,
      transmission_power_factor=transmission_power_factor,
      incident_floor=incident_floor, window=window)
  diffraction = diffraction_spectrum(
      reference_reflection_line, device_reflection_line,
      reference_transmission_line, device_transmission_line,
      dt, dx, background_eps_r=background_eps_r, polarization="tez",
      incident_floor=incident_floor, window=window)
  return TEzScatteringResult2D(
      reference_reflection, device_reflection, reference_transmission,
      device_transmission, spectrum, final_device_field, backend,
      actual_device or reference_device, diffraction)


__all__ = [
    "TEzScatteringResult2D", "TMzScatteringResult2D", "jax_runtime_report",
    "run_tez_scattering_2d", "run_tmz_scattering_2d",
]
