import numpy as np
import pytest

from fdtdz_jax.materials import Drude
from fdtdz_jax.materials import Lossless
from fdtdz_jax.simulation_3d import BoundarySpec3D
from fdtdz_jax.simulation_3d import Box3D
from fdtdz_jax.simulation_3d import CylinderZ3D
from fdtdz_jax.simulation_3d import FrequencyMonitor3D
from fdtdz_jax.simulation_3d import GridSpec3D
from fdtdz_jax.simulation_3d import Layer3D
from fdtdz_jax.simulation_3d import PlaneWaveSource3D
from fdtdz_jax.simulation_3d import ScatteringMonitor3D
from fdtdz_jax.simulation_3d import Simulation3D


def _waveform(num_steps=180, dt=0.45):
  time = np.arange(num_steps) * dt
  return np.exp(-0.5 * ((time - 15.0) / 5.0)**2) * np.sin(
      2.0 * np.pi * 0.06 * (time - 15.0))


def _empty_simulation(snapshot_steps=(), frequency_monitor=None):
  grid = GridSpec3D((4, 4, 84), (1.0, 1.0, 1.0), 0.45)
  simulation = Simulation3D(
      grid, BoundarySpec3D(12), backend="numpy", dtype=np.float64)
  simulation.set_source(PlaneWaveSource3D(_waveform(), 19))
  simulation.set_monitors(ScatteringMonitor3D(
      31, 61, snapshot_steps=snapshot_steps))
  if frequency_monitor is not None:
    simulation.set_frequency_monitor(frequency_monitor)
  return simulation


def test_grid_boundary_and_source_validation():
  grid = GridSpec3D((4, 5, 20), (0.5, 0.75, 1.0), 0.2)

  assert grid.extent == (2.0, 3.75, 20.0)
  with pytest.raises(ValueError, match="integers greater than one"):
    GridSpec3D((4, 1, 20), (1.0, 1.0, 1.0), 0.2)
  with pytest.raises(ValueError, match="leave at least one"):
    Simulation3D(grid, BoundarySpec3D(10))
  with pytest.raises(ValueError, match="zero vector"):
    PlaneWaveSource3D(_waveform(), 4, polarization_xy=(0.0, 0.0))
  with pytest.raises(ValueError, match="strictly increasing"):
    ScatteringMonitor3D(5, 10, snapshot_steps=(8, 4))
  with pytest.raises(ValueError, match="strictly increasing"):
    FrequencyMonitor3D((0.08, 0.04))


def test_layer_is_sampled_at_staggered_electric_nodes():
  grid = GridSpec3D((3, 4, 8), (1.0, 1.0, 1.0), 0.4)
  mask = Layer3D(1.25, 1.75, "film").mask(grid)

  assert mask.shape == (3, 3, 4, 8)
  assert not np.any(mask[0])
  assert not np.any(mask[1])
  assert np.count_nonzero(mask[2]) == 3 * 4


def test_periodic_cylinder_wraps_across_unit_cell_edge():
  grid = GridSpec3D((4, 4, 6), (1.0, 1.0, 1.0), 0.4)
  periodic = CylinderZ3D(
      (0.0, 0.0), 0.8, z_center=2.0, height=2.0,
      material="metal", periodic_xy=True).mask(grid)
  clipped = CylinderZ3D(
      (0.0, 0.0), 0.8, z_center=2.0, height=2.0,
      material="metal", periodic_xy=False).mask(grid)

  assert np.count_nonzero(periodic) > np.count_nonzero(clipped)
  assert np.any(periodic[:, -1])
  assert not np.any(clipped[:, -1])


def test_named_materials_and_later_geometry_override():
  grid = GridSpec3D((4, 4, 24), (1.0, 1.0, 1.0), 0.4)
  simulation = Simulation3D(grid, BoundarySpec3D(4), backend="numpy")
  simulation.add_material("substrate", Lossless(2.25))
  simulation.add_material("metal", Drude(1.0, 0.5, 0.02))
  simulation.add_geometry(Layer3D(9.0, 12.0, "substrate"))
  simulation.add_geometry(Box3D((2.0, 2.0, 10.5), (1.5, 1.5, 1.5), "metal"))
  reference, device = simulation.material_map()

  assert reference.shape == (3,) + grid.shape
  assert not np.any(reference)
  assert np.any(device == 1)
  assert np.any(device == 2)
  with pytest.raises(ValueError, match="already registered"):
    simulation.add_material("metal", Lossless(3.0))
  simulation.add_geometry(Layer3D(14.0, 15.0, "missing"))
  with pytest.raises(ValueError, match="unknown material"):
    simulation.material_map()


def test_run_requires_source_and_monitors():
  grid = GridSpec3D((4, 4, 40), (1.0, 1.0, 1.0), 0.4)
  simulation = Simulation3D(grid, BoundarySpec3D(8), backend="numpy")
  with pytest.raises(ValueError, match="set a PlaneWaveSource3D"):
    simulation.run()
  simulation.set_source(PlaneWaveSource3D(_waveform(20), 10))
  with pytest.raises(ValueError, match="set a ScatteringMonitor3D"):
    simulation.run()


def test_empty_high_level_simulation_has_unit_transmission_and_named_fields():
  result = _empty_simulation().run()
  spectrum = result.spectrum
  incident = np.sqrt(np.sum(
      np.abs(spectrum.reference_incident_field)**2, axis=1))
  candidates = np.flatnonzero(spectrum.valid)
  peak = candidates[np.argmax(incident[candidates])]

  assert spectrum.reflection[peak] == pytest.approx(0.0, abs=1e-14)
  assert spectrum.transmission[peak] == pytest.approx(1.0)
  assert spectrum.absorption[peak] == pytest.approx(0.0, abs=1e-14)
  assert result.final_field("Ex").shape == result.grid.shape
  assert result.final_field("Hz").shape == result.grid.shape
  assert result.final_field("Dy").shape == result.grid.shape
  assert result.final_field("Pz").shape == result.grid.shape
  np.testing.assert_array_equal(
      result.final_field("Pz"), result.final_polarization("z"))
  assert result.backend == "numpy"
  assert result.material_names == ("background",)
  with pytest.raises(ValueError, match="name must be"):
    result.final_field("intensity")


def test_high_level_sparse_snapshots_include_named_e_h_d_p_fields():
  result = _empty_simulation(snapshot_steps=(1, 90, 180)).run()

  assert result.snapshots.steps.tolist() == [1, 90, 180]
  assert result.snapshot_times.tolist() == pytest.approx([0.45, 40.5, 81.0])
  assert result.snapshot_field("Ex").shape == (3,) + result.grid.shape
  assert result.snapshot_field("Hz").shape == (3,) + result.grid.shape
  assert result.snapshot_field("Dx").shape == (3,) + result.grid.shape
  np.testing.assert_allclose(
      result.snapshot_field("Px"),
      result.snapshot_field("Dx") - result.snapshot_field("Ex"))
  np.testing.assert_allclose(
      result.snapshot_field("Ex")[-1], result.final_field("Ex"))


def test_high_level_frequency_monitor_exposes_complex_e_h_d_p_fields():
  monitor = FrequencyMonitor3D((0.04, 0.06, 0.08), window="hann")
  result = _empty_simulation(frequency_monitor=monitor).run()

  np.testing.assert_array_equal(
      result.frequency_fields.frequencies, monitor.frequencies)
  assert result.frequency_field("Ex").shape == (3,) + result.grid.shape
  assert result.frequency_field("Hz").shape == (3,) + result.grid.shape
  assert np.iscomplexobj(result.frequency_field("Dx"))
  np.testing.assert_allclose(
      result.frequency_field("Px"),
      result.frequency_field("Dx") - result.frequency_field("Ex"))
  assert result.frequency_index(0.06) == 1


def test_refined_simulation_preserves_physical_extent_and_configuration():
  grid = GridSpec3D((4, 5, 40), (1.0, 0.5, 0.75), 0.3)
  simulation = Simulation3D(grid, BoundarySpec3D(8), backend="numpy")
  simulation.add_material("film", Lossless(2.25))
  simulation.add_geometry(Box3D(
      center=(2.0, 1.25, 15.0), size=(1.0, 0.5, 1.5), material="film"))
  waveform = np.linspace(-0.1, 0.2, 20)
  profile = np.arange(20, dtype=float).reshape(4, 5)
  simulation.set_source(PlaneWaveSource3D(
      waveform, 12, transverse_profile=profile))
  simulation.set_monitors(ScatteringMonitor3D(
      17, 28, snapshot_steps=(5, 20)))
  simulation.set_frequency_monitor(FrequencyMonitor3D((0.04, 0.08)))

  refined = simulation.refined(2)

  assert refined.grid.shape == (8, 10, 80)
  assert refined.grid.spacing == (0.5, 0.25, 0.375)
  assert refined.grid.dt == pytest.approx(0.15)
  assert refined.grid.extent == grid.extent
  assert refined.boundary.cpml_width == 16
  assert refined.materials == simulation.materials
  assert refined.geometries == simulation.geometries
  assert refined._source.z_index == 24
  assert refined._source.waveform.shape == (40,)
  np.testing.assert_allclose(refined._source.waveform[::2], waveform)
  np.testing.assert_allclose(refined._source.transverse_profile[::2, ::2], profile)
  assert refined._monitors.reflection_z == 34
  assert refined._monitors.transmission_z == 56
  assert refined._monitors.snapshot_steps == (10, 40)


def test_automatic_refinement_study_runs_and_compares_same_physical_duration():
  grid = GridSpec3D((2, 2, 32), (1.0, 1.0, 1.0), 0.4)
  simulation = Simulation3D(grid, BoundarySpec3D(6), backend="numpy")
  time = np.arange(30) * grid.dt
  waveform = np.exp(-0.5 * ((time - 4.0) / 1.5)**2) * np.sin(
      2.0 * np.pi * 0.1 * (time - 4.0))
  simulation.set_source(PlaneWaveSource3D(waveform, 10))
  simulation.set_monitors(ScatteringMonitor3D(14, 22))

  study = simulation.run_refinement_study(
      factors=(2,), backend="numpy", incident_floor=1e-4)

  assert study.factors == (1, 2)
  assert set(study.results) == {1, 2}
  assert study.results[1].grid.extent == study.results[2].grid.extent
  assert study.results[1].final_state.step * study.results[1].grid.dt == pytest.approx(
      study.results[2].final_state.step * study.results[2].grid.dt)
  assert "factor_2" in study.comparisons.comparisons
