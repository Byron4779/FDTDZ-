"""Reference/device scattering workflow for a patterned x-y metasurface."""

from dataclasses import dataclass

import numpy as np

from .diffraction_3d import DiffractionSpectrum3D
from .diffraction_3d import diffraction_spectrum_3d
from .metasurface_3d import HuygensPlaneSource3D
from .jax_yee_ade_3d import jax_material_grid_3d
from .jax_yee_ade_3d import jax_yee_ade_3d_state
from .jax_yee_ade_3d import numpy_yee_ade_3d_state
from .jax_yee_ade_3d import simulate_jax_yee_ade_3d_plane_probes
from .jax_yee_ade_3d import (
    simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields)
from .jax_yee_ade_3d import simulate_jax_yee_ade_3d_plane_probes_with_snapshots
from .scattering_workflow_2d import jax_runtime_report
from .yee_ade_3d import YeeADE3DState
from .yee_ade_3d import YeeADE3DFieldSnapshots
from .yee_ade_3d import YeeADE3DFrequencyFields
from .yee_ade_3d import initialize_yee_ade_3d
from .yee_ade_3d import prepare_material_grid_3d
from .yee_ade_3d import prepare_z_cpml
from .yee_ade_3d import simulate_yee_ade_3d_compact_plane_probes


@dataclass(frozen=True)
class MetasurfaceScatteringResult3D:
  """Probe planes, diffraction spectrum and final dispersive device state."""

  reference_reflection: np.ndarray
  device_reflection: np.ndarray
  reference_transmission: np.ndarray
  device_transmission: np.ndarray
  diffraction: DiffractionSpectrum3D
  final_device_state: YeeADE3DState
  reflection_probe_z: int
  transmission_probe_z: int
  backend: str
  device: str
  field_snapshots: YeeADE3DFieldSnapshots = None
  frequency_fields: YeeADE3DFrequencyFields = None


def _component_map(name, value):
  array = np.asarray(value)
  if array.ndim == 3:
    spatial_shape = array.shape
    array = np.broadcast_to(array[None, ...], (3,) + spatial_shape)
  elif array.ndim == 4 and array.shape[0] == 3:
    spatial_shape = array.shape[1:]
  else:
    raise ValueError(
        f"{name} must have shape (nx, ny, nz) or (3, nx, ny, nz).")
  if not np.issubdtype(array.dtype, np.integer):
    raise TypeError(f"{name} must contain integers.")
  return np.asarray(array), spatial_shape


def _validate_geometry(reference_material_ids, device_material_ids, source,
                       reflection_probe_z, transmission_probe_z, cpml_width):
  if isinstance(cpml_width, (bool, np.bool_)) or not isinstance(
      cpml_width, (int, np.integer)):
    raise TypeError("cpml_width must be an integer.")
  cpml_width = int(cpml_width)
  reference, reference_shape = _component_map(
      "reference_material_ids", reference_material_ids)
  device, device_shape = _component_map(
      "device_material_ids", device_material_ids)
  if reference_shape != device_shape:
    raise ValueError("reference and device material maps must have matching shape.")
  if not isinstance(source, HuygensPlaneSource3D):
    raise TypeError("source must be a HuygensPlaneSource3D instance.")
  if tuple(source.shape) != tuple(reference_shape):
    raise ValueError("source shape must match the material-map shape.")
  nz = reference_shape[2]
  if cpml_width <= 0 or 2 * cpml_width >= nz:
    raise ValueError(
        "cpml_width must be positive and leave at least one interior z cell.")
  indices = (source.electric_z_index, source.magnetic_z_index,
             reflection_probe_z, transmission_probe_z)
  if any(isinstance(value, (bool, np.bool_)) or
         not isinstance(value, (int, np.integer)) or
         value < 0 or value >= nz for value in indices):
    raise ValueError("source and probe z indices must lie inside the grid.")
  reflection_probe_z = int(reflection_probe_z)
  transmission_probe_z = int(transmission_probe_z)
  if source.direction == 1 and not (
      source.electric_z_index < reflection_probe_z < transmission_probe_z):
    raise ValueError(
        "for +z propagation require source_z < reflection_probe_z < "
        "transmission_probe_z.")
  if source.direction == -1 and not (
      source.electric_z_index > reflection_probe_z > transmission_probe_z):
    raise ValueError(
        "for -z propagation require source_z > reflection_probe_z > "
        "transmission_probe_z.")
  if any(index < cpml_width or index >= nz - cpml_width for index in indices):
    raise ValueError("source and probes must lie outside the z CPML layers.")
  cpml_nodes = np.zeros(reference.shape, dtype=bool)
  cpml_nodes[..., :cpml_width] = True
  cpml_nodes[..., -cpml_width:] = True
  if np.any(reference[cpml_nodes] != device[cpml_nodes]):
    raise ValueError("reference and device maps must match inside z CPML.")
  return reference_shape


def _record(materials, material_ids, dx, dy, dz, dt, source,
            reflection_probe_z, transmission_probe_z, cpml, dtype, backend,
            snapshot_steps=None, frequencies=None, frequency_weights=None):
  if backend == "numpy":
    result = simulate_yee_ade_3d_compact_plane_probes(
        materials, material_ids, dx, dy, dz, dt,
        source.electric_waveform, source.magnetic_waveform,
        source.transverse_profile, source.polarization_xy,
        source.electric_z_index, source.magnetic_z_index,
        reflection_probe_z, transmission_probe_z,
        dtype=dtype, z_cpml=cpml, snapshot_steps=snapshot_steps,
        frequencies=frequencies, frequency_window=frequency_weights,
        bloch_wavevector=source.bloch_wavevector)
    return (result.electric_tangential[:, 0],
            result.electric_tangential[:, 1], result.final_state,
            "NumPy CPU reference", result.snapshots,
            result.frequency_fields)
  if backend != "jax":
    raise ValueError("backend must be 'numpy' or 'jax'.")
  grid = prepare_material_grid_3d(
      materials, material_ids, dt, dx, dy, dz, dtype=dtype, z_cpml=cpml,
      bloch_wavevector=source.bloch_wavevector)
  initial = initialize_yee_ade_3d(grid)
  common = (
      jax_material_grid_3d(grid), jax_yee_ade_3d_state(initial, grid),
      source.electric_waveform, source.magnetic_waveform,
      source.transverse_profile,
      np.asarray(source.polarization_xy, dtype=np.float32))
  snapshots = None
  frequency_fields = None
  if frequencies is not None and len(frequencies):
    final_state, probes, snapshot_values, frequency_values = (
        simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields(
            *common, np.asarray(frequencies, dtype=dtype),
            np.asarray(frequency_weights, dtype=dtype),
            electric_source_z=source.electric_z_index,
            magnetic_source_z=source.magnetic_z_index,
            reflection_probe_z=reflection_probe_z,
            transmission_probe_z=transmission_probe_z,
            snapshot_steps=tuple(int(step) for step in snapshot_steps)))
    if snapshot_steps:
      snapshots = YeeADE3DFieldSnapshots(
          electric=np.asarray(snapshot_values[0]),
          magnetic=np.asarray(snapshot_values[1]),
          displacement=np.asarray(snapshot_values[2]),
          steps=np.asarray(snapshot_steps, dtype=np.int64) + int(initial.step))
    frequency_fields = YeeADE3DFrequencyFields(
        frequencies=np.asarray(frequencies, dtype=float),
        electric=np.asarray(frequency_values[0]),
        magnetic=np.asarray(frequency_values[1]),
        displacement=np.asarray(frequency_values[2]))
  elif snapshot_steps:
    final_state, probes, snapshot_fields = (
        simulate_jax_yee_ade_3d_plane_probes_with_snapshots(
            *common,
            electric_source_z=source.electric_z_index,
            magnetic_source_z=source.magnetic_z_index,
            reflection_probe_z=reflection_probe_z,
            transmission_probe_z=transmission_probe_z,
            snapshot_steps=tuple(int(step) for step in snapshot_steps)))
    snapshots = YeeADE3DFieldSnapshots(
        electric=np.asarray(snapshot_fields[0]),
        magnetic=np.asarray(snapshot_fields[1]),
        displacement=np.asarray(snapshot_fields[2]),
        steps=np.asarray(snapshot_steps, dtype=np.int64) + int(initial.step))
  else:
    final_state, probes = simulate_jax_yee_ade_3d_plane_probes(
        *common,
        electric_source_z=source.electric_z_index,
        magnetic_source_z=source.magnetic_z_index,
        reflection_probe_z=reflection_probe_z,
        transmission_probe_z=transmission_probe_z)
  final_state.electric.block_until_ready()
  report = jax_runtime_report()
  return (np.asarray(probes[0]), np.asarray(probes[1]),
          numpy_yee_ade_3d_state(final_state), ", ".join(report["devices"]),
          snapshots, frequency_fields)


def _frequency_configuration(frequencies, window, num_steps, dt, dtype):
  if frequencies is None:
    return None, None
  values = np.asarray(frequencies)
  if values.ndim == 1 and values.size == 0:
    return None, None
  if (values.ndim != 1 or np.iscomplexobj(values) or
      not np.issubdtype(values.dtype, np.number) or
      not np.all(np.isfinite(values))):
    raise ValueError("frequencies must be a finite real one-dimensional array.")
  values = np.asarray(values, dtype=float)
  nyquist = 0.5 / dt
  if (np.any(values <= 0.0) or np.any(values > nyquist) or
      np.unique(values).size != values.size or
      (values.size > 1 and np.any(np.diff(values) <= 0.0))):
    raise ValueError(
        "frequencies must be strictly increasing, unique, positive, and no "
        f"greater than the Nyquist frequency {nyquist}.")
  if window is None:
    weights = np.ones(num_steps, dtype=dtype)
  elif isinstance(window, str):
    if window != "hann":
      raise ValueError("frequency_window string must be 'hann'.")
    weights = np.asarray(np.hanning(num_steps), dtype=dtype)
  else:
    weights = np.asarray(window, dtype=dtype)
    if weights.shape != (num_steps,) or not np.all(np.isfinite(weights)):
      raise ValueError(
          "frequency_window must be None, 'hann', or a matching finite array.")
  return values, weights


def run_metasurface_scattering_3d(
    materials, reference_material_ids, device_material_ids,
    dx, dy, dz, dt, source, reflection_probe_z, transmission_probe_z,
    cpml_width, dtype=np.float64, target_reflection=1e-8,
    background_eps_r=1.0, incident_floor=1e-6, window=None,
    backend="numpy", snapshot_steps=None, frequencies=None,
    frequency_window=None):
  """Run matched empty/device simulations and calculate total 3D R/T/A."""
  if backend not in ("numpy", "jax"):
    raise ValueError("backend must be 'numpy' or 'jax'.")
  shape = _validate_geometry(
      reference_material_ids, device_material_ids, source,
      reflection_probe_z, transmission_probe_z, cpml_width)
  if snapshot_steps is None:
    snapshot_steps = ()
  else:
    snapshot_array = np.asarray(snapshot_steps)
    if snapshot_array.ndim == 1 and snapshot_array.size == 0:
      snapshot_steps = ()
    elif (snapshot_array.ndim != 1 or
          not np.issubdtype(snapshot_array.dtype, np.integer)):
      raise ValueError(
          "snapshot_steps must be a one-dimensional integer sequence.")
    else:
      snapshot_steps = tuple(int(step) for step in snapshot_array)
      if (tuple(sorted(set(snapshot_steps))) != snapshot_steps or
          snapshot_steps[0] < 1 or
          snapshot_steps[-1] > source.electric_waveform.size):
        raise ValueError(
            "snapshot_steps must be strictly increasing, unique, and lie "
            f"in [1, {source.electric_waveform.size}].")
  frequencies, frequency_weights = _frequency_configuration(
      frequencies, frequency_window, source.electric_waveform.size, dt,
      np.dtype(dtype))
  cpml = prepare_z_cpml(
      shape, cpml_width, dt, dz, target_reflection=target_reflection,
      dtype=dtype)
  common = (materials, dx, dy, dz, dt, source, reflection_probe_z,
            transmission_probe_z, cpml, dtype, backend)
  (reference_reflection, reference_transmission, _, reference_device,
   _, _) = _record(common[0], reference_material_ids, *common[1:])
  (device_reflection, device_transmission, final_device_state,
   actual_device, field_snapshots, frequency_fields) = _record(
       common[0], device_material_ids, *common[1:],
       snapshot_steps=snapshot_steps, frequencies=frequencies,
       frequency_weights=frequency_weights)
  diffraction = diffraction_spectrum_3d(
      reference_reflection, device_reflection,
      reference_transmission, device_transmission,
      dt, dx, dy, background_eps_r=background_eps_r,
      incident_floor=incident_floor, window=window,
      bloch_wavevector=source.bloch_wavevector)
  return MetasurfaceScatteringResult3D(
      reference_reflection=reference_reflection,
      device_reflection=device_reflection,
      reference_transmission=reference_transmission,
      device_transmission=device_transmission,
      diffraction=diffraction, final_device_state=final_device_state,
      reflection_probe_z=int(reflection_probe_z),
      transmission_probe_z=int(transmission_probe_z),
      backend=backend, device=actual_device or reference_device,
      field_snapshots=field_snapshots, frequency_fields=frequency_fields)


__all__ = ["MetasurfaceScatteringResult3D", "run_metasurface_scattering_3d"]
