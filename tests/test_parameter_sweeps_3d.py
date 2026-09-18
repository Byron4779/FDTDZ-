import numpy as np
import pytest

from fdtdz_jax import BoundarySpec3D, FrequencyMonitor3D, GridSpec3D
from fdtdz_jax import Layer3D, Lossless, ParameterSweep3D, PlaneWaveSource3D
from fdtdz_jax import ScatteringMonitor3D, Simulation3D


def _pulse(times):
  return np.exp(-0.5 * ((times - 15.0) / 5.0)**2) * np.sin(
      2.0 * np.pi * 0.06 * (times - 15.0))


def _simulation(dt=0.4, width=6, duration=120.0):
  grid = GridSpec3D((2, 3, 64), (1.0, 0.8, 1.0), dt)
  simulation = Simulation3D(grid, BoundarySpec3D(width), backend="numpy",
                            dtype=np.float32)
  simulation.add_material("film", Lossless(2.25))
  simulation.add_geometry(Layer3D(29.0, 32.0, "film"))
  steps = int(round(duration / dt))
  simulation.set_source(PlaneWaveSource3D(_pulse(np.arange(steps) * dt), 12))
  simulation.set_monitors(ScatteringMonitor3D(20, 46, (steps // 2, steps)))
  simulation.set_frequency_monitor(FrequencyMonitor3D((0.05, 0.075)))
  return simulation


def test_temporal_refinement_preserves_space_cpml_and_physical_times():
  simulation = _simulation()
  profile = np.arange(6).reshape(2, 3)
  simulation.set_source(PlaneWaveSource3D(
      simulation._source.waveform, 12, transverse_profile=profile))
  variant = simulation.temporal_refined(3)
  assert variant.grid.shape == simulation.grid.shape
  assert variant.grid.spacing == simulation.grid.spacing
  assert variant.grid.dt == pytest.approx(simulation.grid.dt / 3)
  assert variant.boundary == simulation.boundary
  assert variant.materials == simulation.materials
  assert variant.geometries == simulation.geometries
  assert variant._source.z_index == 12
  assert variant._monitors.reflection_z == 20
  assert variant._monitors.transmission_z == 46
  np.testing.assert_array_equal(variant.material_map()[1], simulation.material_map()[1])
  np.testing.assert_array_equal(variant._source.transverse_profile, profile)
  np.testing.assert_allclose(variant._source.waveform[::3], simulation._source.waveform,
                             atol=1e-14)
  assert variant._source.waveform.size * variant.grid.dt == pytest.approx(120.0)
  np.testing.assert_allclose(np.array(variant._monitors.snapshot_steps) * variant.grid.dt,
                             np.array(simulation._monitors.snapshot_steps) * simulation.grid.dt)
  assert variant._frequency_monitor.frequencies == simulation._frequency_monitor.frequencies
  variant._source.waveform[:] = 0
  variant._source.transverse_profile[:] = 0
  assert np.any(simulation._source.waveform)
  np.testing.assert_array_equal(simulation._source.transverse_profile, profile)


def test_analytic_waveform_is_sampled_even_for_baseline():
  simulation = _simulation()
  for factor in (1, 2, 3):
    variant = simulation.temporal_refined(factor, waveform_factory=lambda t: t**2)
    times = np.arange(300 * factor) * (0.4 / factor)
    np.testing.assert_allclose(variant._source.waveform, times**2)
  np.testing.assert_allclose(simulation._source.waveform, _pulse(np.arange(300) * 0.4))


def test_cpml_variant_keeps_grid_source_windows_and_monitors():
  simulation = _simulation()
  weights = np.hanning(300)
  simulation.set_frequency_monitor(FrequencyMonitor3D((0.05,), weights))
  variant = simulation.with_cpml_width(8)
  assert variant.grid == simulation.grid
  assert variant.boundary.cpml_width == 8
  assert variant.boundary.target_reflection == simulation.boundary.target_reflection
  assert simulation.boundary.cpml_width == 6
  assert variant._monitors == simulation._monitors
  np.testing.assert_array_equal(variant._source.waveform, simulation._source.waveform)
  np.testing.assert_array_equal(variant._frequency_monitor.window, weights)
  np.testing.assert_array_equal(variant.material_map()[1], simulation.material_map()[1])
  variant._frequency_monitor.window[:] = 0
  np.testing.assert_array_equal(simulation._frequency_monitor.window, weights)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_temporal_sweep_matches_an_independent_smaller_dt_run(backend):
  simulation = _simulation()
  study = simulation.run_temporal_study(
      (2, 1, 2), waveform_factory=_pulse, backend=backend,
      window="hann", frequency_min=0.04, frequency_max=0.09)
  assert isinstance(study, ParameterSweep3D)
  assert study.parameter == "temporal_factor"
  assert study.values == (1, 2)
  assert study.baseline_value == 1
  candidate = study.results[2]
  independent = _simulation(dt=0.2).run(backend=backend, window="hann")
  np.testing.assert_array_equal(candidate.final_field("Ex"), independent.final_field("Ex"))
  np.testing.assert_array_equal(candidate.spectrum.reflection, independent.spectrum.reflection)
  for result in study.results.values():
    assert result.grid.shape == simulation.grid.shape
    assert result.final_state.step * result.grid.dt == pytest.approx(120.0)
    np.testing.assert_allclose(result.snapshot_times, (60.0, 120.0))
  report = study.comparisons.comparisons["temporal_factor_2"]
  assert np.all(report.frequencies[report.valid] >= 0.04)
  assert np.all(report.frequencies[report.valid] <= 0.09)
  assert 0 < report.maximum_reflection_difference < 0.1
  assert study.converged(0.1)
  assert not study.converged(0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_cpml_sweep_matches_an_independent_boundary_run(backend):
  simulation = _simulation(width=8)
  study = simulation.run_cpml_study((6, 6), backend=backend,
                                    frequency_min=0.04, frequency_max=0.09)
  assert study.parameter == "cpml_width"
  assert study.values == (6, 8)
  assert study.baseline_value == 8
  independent = _simulation(width=6).run(backend=backend)
  np.testing.assert_array_equal(study.results[6].final_field("Ex"), independent.final_field("Ex"))
  np.testing.assert_array_equal(study.results[6].spectrum.transmission,
                                independent.spectrum.transmission)
  assert study.results[6].grid == study.results[8].grid
  assert study.comparisons.baseline_name == "cpml_width_8"
  assert set(study.comparisons.comparisons) == {"cpml_width_6"}
  assert np.any(independent.spectrum.reflection[independent.spectrum.valid] > 0)
  assert study.converged(0.1)


@pytest.mark.parametrize("method,values", [
    ("run_temporal_study", ()), ("run_temporal_study", (1,)),
    ("run_temporal_study", (0, 2)), ("run_temporal_study", (True, 2)),
    ("run_temporal_study", (1.5, 2)), ("run_temporal_study", 2),
    ("run_cpml_study", ()), ("run_cpml_study", (6,)),
    ("run_cpml_study", (0, 8)), ("run_cpml_study", (True, 8)),
    ("run_cpml_study", (2.5, 8)), ("run_cpml_study", 8),
])
def test_invalid_parameter_sequences_fail_before_running(method, values, monkeypatch):
  simulation = _simulation()
  monkeypatch.setattr(Simulation3D, "run", lambda *a, **k: pytest.fail("unexpected run"))
  with pytest.raises((ValueError, TypeError)):
    getattr(simulation, method)(values)


@pytest.mark.parametrize("width,geometry,message", [
    (12, None, "outside the z CPML"),  # staggered magnetic source plane is 11
    (18, None, "outside the z CPML"),  # transmission probe also covered
    (32, None, "interior"),
    (10, Layer3D(9, 10, "film"), "match inside z CPML"),
])
def test_cpml_sweep_preflights_all_widths_before_any_run(width, geometry, message, monkeypatch):
  simulation = _simulation()
  if geometry is not None:
    simulation.add_geometry(geometry)
  monkeypatch.setattr(Simulation3D, "run", lambda *a, **k: pytest.fail("unexpected run"))
  with pytest.raises(ValueError, match=message):
    simulation.run_cpml_study((8, width))


def test_cpml_preflight_handles_negative_propagation_direction():
  simulation = _simulation()
  simulation.set_source(PlaneWaveSource3D(simulation._source.waveform, 52, direction=-1))
  simulation.set_monitors(ScatteringMonitor3D(45, 20))
  assert simulation.with_cpml_width(10)._source.direction == -1
  with pytest.raises(ValueError, match="outside the z CPML"):
    simulation.with_cpml_width(12)


def test_temporal_study_rejects_sampled_windows_before_running(monkeypatch):
  simulation = _simulation()
  monkeypatch.setattr(Simulation3D, "run", lambda *a, **k: pytest.fail("unexpected run"))
  with pytest.raises(ValueError, match="spectral window"):
    simulation.run_temporal_study(window=np.ones(300))
  simulation.set_frequency_monitor(FrequencyMonitor3D((0.05,), np.ones(300)))
  with pytest.raises(ValueError, match="frequency window"):
    simulation.run_temporal_study()


@pytest.mark.parametrize("factory,message", [
    (42, "callable"), (lambda t: np.zeros(3), "one sample"),
    (lambda t: np.full(t.shape, np.nan), "finite real"),
    (lambda t: t + 1j, "finite real"),
])
def test_invalid_analytic_waveforms_fail_before_running(factory, message, monkeypatch):
  simulation = _simulation()
  monkeypatch.setattr(Simulation3D, "run", lambda *a, **k: pytest.fail("unexpected run"))
  with pytest.raises((ValueError, TypeError), match=message):
    simulation.run_temporal_study(waveform_factory=factory)


def test_variant_builders_require_source_and_monitors():
  simulation = Simulation3D(GridSpec3D((2, 2, 32), (1, 1, 1), 0.4), BoundarySpec3D(6))
  with pytest.raises(ValueError, match="set source"):
    simulation.temporal_refined(2)
  with pytest.raises(ValueError, match="set source"):
    simulation.with_cpml_width(8)
