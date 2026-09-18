"""Portable analysis archives for ``SimulationResult3D`` (schema version 1).

NPZ keys and HDF5 dataset paths are identical. Read NPZ with
``np.load(path, allow_pickle=False)`` and HDF5 with ``h5py.File(path, 'r')``.
``metadata_json`` is a scalar JSON string; all other entries are numeric or
boolean arrays. Missing optional state/monitor arrays are omitted. These are
analysis exports, not restart files: material models and source configuration
are not stored. Coordinates and frequencies retain normalized solver units.
"""

from dataclasses import fields
import json
from pathlib import Path

import numpy as np


def _metadata(result):
  from . import __version__
  return json.dumps({
      "schema": "fdtdz.simulation_result_3d", "schema_version": 1,
      "fdtdz_version": __version__, "backend": result.backend,
      "device": result.device, "units": "normalized",
      "material_names": list(result.material_names),
      "field_axes": ["component", "x", "y", "z"],
      "components": ["x", "y", "z"],
      "snapshot_axes": ["snapshot", "component", "x", "y", "z"],
      "frequency_field_axes": ["frequency", "component", "x", "y", "z"],
      "probe_axes": ["time", "tangential_component", "x", "y"],
      "tangential_components": ["Ex", "Ey"],
      "electric_offsets_cells": [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
      "magnetic_offsets_cells": [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
      "polarization": "normalized total P = displacement - electric",
      "field_times": "E/D/P: step * dt; H: (step - 0.5) * dt",
      "frequency_field_transform": "dt * sum(window(t) * F(t) * exp(+i*omega*t))",
      "bloch_wavevector_units": "normalized radians/length; spectrum/bloch_wavevector",
      "complex_time_domain_fields": "Bloch quadratures; ADE coefficients remain real",
  }, ensure_ascii=False, sort_keys=True)


def _record_arrays(prefix, record):
  for field in fields(record):
    value = getattr(record, field.name)
    if value is not None:
      yield f"{prefix}/{field.name}", np.asarray(value)


def _arrays(result):
  yield from _record_arrays("grid", result.grid)
  yield "material_ids", np.asarray(result.material_ids)
  yield from _record_arrays("final", result.final_state)
  yield "final/polarization", np.asarray(result.final_polarization())
  yield from _record_arrays("spectrum", result.spectrum)
  for name in ("reference_reflection", "device_reflection",
               "reference_transmission", "device_transmission",
               "reflection_probe_z", "transmission_probe_z"):
    yield f"probes/{name}", np.asarray(getattr(result.scattering, name))
  if result.snapshots is not None:
    yield from _record_arrays("snapshots", result.snapshots)
    yield "snapshots/polarization", np.asarray(result.snapshots.polarization)
    yield "snapshots/times", np.asarray(result.snapshot_times)
  if result.frequency_fields is not None:
    yield from _record_arrays("frequency_fields", result.frequency_fields)
    yield "frequency_fields/polarization", np.asarray(
        result.frequency_fields.polarization)


def save_npz(result, path, *, compressed=True, overwrite=False):
  """Write numeric arrays without pickle, using the exact supplied path."""
  path = Path(path)
  arrays = dict(_arrays(result))
  arrays["metadata_json"] = np.asarray(_metadata(result))
  writer = np.savez_compressed if compressed else np.savez
  with path.open("wb" if overwrite else "xb") as stream:
    writer(stream, **arrays)
  return path


def save_hdf5(result, path, *, compression="gzip", overwrite=False):
  """Write arrays one at a time; h5py is imported only for this operation."""
  if compression not in (None, "gzip", "lzf"):
    raise ValueError("compression must be None, 'gzip', or 'lzf'.")
  try:
    import h5py
  except ImportError as error:
    raise ImportError(
        "HDF5 export requires h5py; install fdtdz[hdf5].") from error
  path = Path(path)
  metadata = _metadata(result)
  with h5py.File(path, "w" if overwrite else "x") as archive:
    archive.create_dataset(
        "metadata_json", data=metadata, dtype=h5py.string_dtype("utf-8"))
    for name, value in _arrays(result):
      # HDF5 scalar datasets do not support chunking/compression.
      options = {"compression": compression} if value.ndim and value.size else {}
      archive.create_dataset(name, data=value, **options)
  return path
