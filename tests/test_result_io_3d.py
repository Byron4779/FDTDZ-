import builtins
from dataclasses import fields
import json

import numpy as np
import pytest

from fdtdz_jax import BoundarySpec3D, FrequencyMonitor3D, GridSpec3D
from fdtdz_jax import Layer3D, Lossless, PlaneWaveSource3D
from fdtdz_jax import ScatteringMonitor3D, Simulation3D


@pytest.fixture(scope="module", params=["numpy", "jax"])
def result(request):
  grid = GridSpec3D((2, 3, 32), (1.0, 0.8, 1.0), 0.3)
  simulation = Simulation3D(grid, BoundarySpec3D(5), backend=request.param)
  simulation.add_material("薄膜", Lossless(2.25))
  simulation.add_geometry(Layer3D(16.0, 18.0, "薄膜"))
  time = np.arange(60) * grid.dt
  waveform = np.exp(-0.5 * ((time - 4.0) / 1.5)**2) * np.sin(time)
  simulation.set_source(PlaneWaveSource3D(waveform, 8))
  simulation.set_monitors(ScatteringMonitor3D(12, 24, (1, 30, 60)))
  simulation.set_frequency_monitor(FrequencyMonitor3D((0.1, 0.2)))
  return simulation.run()


@pytest.mark.parametrize("compressed", [True, False])
def test_npz_preserves_fields_spectra_and_metadata(result, tmp_path, compressed):
  path = tmp_path / "result.data"
  assert result.to_npz(path, compressed=compressed) == path
  assert not path.with_suffix(".data.npz").exists()
  with np.load(path, allow_pickle=False) as archive:
    metadata = json.loads(archive["metadata_json"].item())
    assert metadata["schema_version"] == 1
    assert metadata["units"] == "normalized"
    assert metadata["backend"] == result.backend
    assert metadata["material_names"] == ["background", "薄膜"]
    assert metadata["field_axes"] == ["component", "x", "y", "z"]
    for key in archive.files:
      assert not archive[key].dtype.hasobject
    for prefix, record in (("grid", result.grid), ("final", result.final_state),
                           ("spectrum", result.spectrum),
                           ("snapshots", result.snapshots),
                           ("frequency_fields", result.frequency_fields)):
      for field in fields(record):
        expected = getattr(record, field.name)
        if expected is not None:
          actual = archive[f"{prefix}/{field.name}"]
          np.testing.assert_array_equal(actual, expected)
          assert actual.dtype == np.asarray(expected).dtype
    np.testing.assert_array_equal(archive["material_ids"], result.material_ids)
    np.testing.assert_array_equal(
        archive["final/polarization"], result.final_polarization())
    np.testing.assert_array_equal(
        archive["snapshots/times"], result.snapshot_times)
    np.testing.assert_array_equal(
        archive["frequency_fields/polarization"],
        result.frequency_fields.polarization)
    assert np.any(archive["frequency_fields/electric"].imag != 0)
    np.testing.assert_array_equal(archive["probes/device_transmission"],
                                  result.scattering.device_transmission)


@pytest.mark.parametrize("compression", [None, "gzip", "lzf"])
def test_hdf5_matches_every_npz_dataset(result, tmp_path, compression):
  h5py = pytest.importorskip("h5py")
  npz_path = result.to_npz(tmp_path / "result.npz")
  h5_path = tmp_path / "result.h5"
  assert result.to_hdf5(h5_path, compression=compression) == h5_path
  with np.load(npz_path, allow_pickle=False) as npz, h5py.File(h5_path) as h5:
    datasets = []
    h5.visititems(lambda name, value: datasets.append(name)
                  if isinstance(value, h5py.Dataset) else None)
    assert set(datasets) == set(npz.files)
    assert json.loads(h5["metadata_json"].asstr()[()]) == json.loads(
        npz["metadata_json"].item())
    for key in set(datasets) - {"metadata_json"}:
      np.testing.assert_array_equal(h5[key][()], npz[key])
      assert h5[key].dtype == npz[key].dtype
    assert h5["final/electric"].compression == compression


@pytest.mark.parametrize("method", ["to_npz", "to_hdf5"])
def test_export_protects_existing_files(result, tmp_path, method):
  if method == "to_hdf5":
    pytest.importorskip("h5py")
  path = tmp_path / "existing"
  original = b"user data"
  path.write_bytes(original)
  with pytest.raises((FileExistsError, OSError)):
    getattr(result, method)(path)
  assert path.read_bytes() == original
  getattr(result, method)(path, overwrite=True)
  assert path.read_bytes() != original


def test_absent_optional_monitors_and_state_are_omitted(result, tmp_path):
  from dataclasses import replace
  state = replace(result.final_state, polarization_current=None,
                  polarization_previous=None)
  scattering = replace(result.scattering, final_device_state=state,
                       field_snapshots=None, frequency_fields=None)
  minimal = replace(result, scattering=scattering)
  path = minimal.to_npz(tmp_path / "minimal.npz")
  with np.load(path, allow_pickle=False) as archive:
    assert not any(key.startswith(("snapshots/", "frequency_fields/"))
                   for key in archive.files)
    assert "final/polarization_current" not in archive
    assert "final/polarization" in archive


def test_optional_imports_and_invalid_compression(result, tmp_path, monkeypatch):
  original_import = builtins.__import__

  def without_optional(name, *args, **kwargs):
    if name.split(".")[0] in ("h5py", "matplotlib"):
      raise ImportError("optional dependency deliberately unavailable")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", without_optional)
  result.to_npz(tmp_path / "without_extras.npz")
  with pytest.raises(ImportError, match=r"fdtdz\[hdf5\]"):
    result.to_hdf5(tmp_path / "missing.h5")
  assert not (tmp_path / "missing.h5").exists()
  with pytest.raises(ImportError, match=r"fdtdz\[plot\]"):
    result.plot_spectrum()
  with pytest.raises(ValueError, match="compression"):
    result.to_hdf5(tmp_path / "invalid.h5", compression="invalid")
  assert not (tmp_path / "invalid.h5").exists()


@pytest.fixture
def axes():
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  figure, ax = plt.subplots()
  yield ax
  plt.close(figure)


def test_spectrum_plot_masks_invalid_bins_without_mutation(result, axes):
  before = result.spectrum.reflection.copy()
  assert result.plot_spectrum(ax=axes) is axes
  assert [line.get_label() for line in axes.lines] == ["R", "T", "A"]
  for line, name in zip(axes.lines, ("reflection", "transmission", "absorption")):
    expected = np.where(result.spectrum.valid, getattr(result.spectrum, name),
                        np.nan)
    np.testing.assert_array_equal(line.get_ydata(), expected)
  np.testing.assert_array_equal(result.spectrum.reflection, before)


@pytest.mark.parametrize("axis,index,name,extent", [
    ("z", 10, "Ex", (0.0, 2.0, -0.4, 2.0)),
    ("x", 1, "Hx", (0.0, 2.4, 0.0, 32.0)),
    ("y", 0, "Dz", (-0.5, 1.5, 0.0, 32.0)),
])
def test_field_slice_orientation_and_yee_coordinates(
    result, axes, axis, index, name, extent):
  image = result.plot_field(name, axis=axis, index=index, ax=axes)
  expected = np.take(result.final_field(name), index, axis="xyz".index(axis)).T
  np.testing.assert_array_equal(image.get_array(), expected)
  np.testing.assert_allclose(image.get_extent(), extent)
  assert image.origin == "lower"


def test_complex_frequency_and_snapshot_plots(result, axes, tmp_path):
  image = result.plot_field("Ex", frequency=0.1, quantity="abs", index=16,
                             ax=axes)
  np.testing.assert_allclose(image.get_array(),
                             np.abs(result.frequency_field("Ex")[0, :, :, 16]).T)
  image = result.plot_field("Px", snapshot_step=30, index=16, ax=axes)
  np.testing.assert_allclose(image.get_array(),
                             result.snapshot_field("Px")[1, :, :, 16].T)
  path = tmp_path / "slice.png"
  axes.figure.savefig(path)
  assert path.stat().st_size > 1000


@pytest.mark.parametrize("options,message", [
    ({"axis": "q"}, "axis"), ({"index": -1}, "outside"),
    ({"index": 32}, "outside"), ({"index": 1.5}, "integer"),
    ({"quantity": "power"}, "quantity"),
    ({"frequency": 0.1, "snapshot_step": 1}, "either"),
    ({"snapshot_step": 5}, "not recorded"),
    ({"snapshot_step": True}, "integer"),
    ({"frequency": 0.3}, "not requested"),
])
def test_plot_rejects_invalid_selection(result, options, message):
  with pytest.raises((ValueError, TypeError), match=message):
    result.plot_field("Ex", **options)
