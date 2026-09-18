from types import SimpleNamespace

import numpy as np
import pytest

from fdtdz_jax.convergence_3d import cfl_report_3d
from fdtdz_jax.convergence_3d import compare_scattering_results_3d
from fdtdz_jax.convergence_3d import convergence_study_3d
from fdtdz_jax.convergence_3d import cpml_resolution_report_3d
from fdtdz_jax.convergence_3d import numerical_report_3d
from fdtdz_jax.convergence_3d import spatial_resolution_report_3d
from fdtdz_jax.convergence_3d import temporal_resolution_report_3d
from fdtdz_jax.materials import Drude
from fdtdz_jax.materials import Lossless
from fdtdz_jax.simulation_3d import BoundarySpec3D
from fdtdz_jax.simulation_3d import GridSpec3D
from fdtdz_jax.simulation_3d import Simulation3D


def _spectrum(reflection):
  frequencies = np.linspace(0.02, 0.12, 6)
  return SimpleNamespace(
      frequencies=frequencies,
      reflection=np.full(frequencies.shape, reflection),
      transmission=np.full(frequencies.shape, 1.0 - reflection),
      absorption=np.zeros(frequencies.shape),
      valid=np.ones(frequencies.shape, dtype=bool))


def test_cfl_temporal_spatial_and_cpml_reports_have_expected_normalized_values():
  cfl = cfl_report_3d([Lossless(1.0), Lossless(4.0)], (0.5, 1.0, 2.0), 0.2)
  expected_dt = 1.0 / np.sqrt(1.0 / 0.5**2 + 1.0 + 1.0 / 2.0**2)

  assert cfl.maximum_dt == pytest.approx(expected_dt)
  assert cfl.stable
  assert cfl.safety_factor == pytest.approx(0.2 / expected_dt)

  temporal = temporal_resolution_report_3d((0.05, 0.1), 0.25)
  np.testing.assert_allclose(temporal.samples_per_period, (80.0, 40.0))
  assert temporal.nyquist_frequency == pytest.approx(2.0)
  assert temporal.below_nyquist

  spatial = spatial_resolution_report_3d(
      {"glass": Lossless(4.0)}, (0.5, 0.5, 0.5), (0.1,))
  np.testing.assert_allclose(spatial.cells_per_phase_wavelength, 10.0)
  assert np.isinf(spatial.cells_per_skin_depth).all()

  cpml = cpml_resolution_report_3d(
      Lossless(4.0), dz=0.5, width_cells=10,
      target_reflection=1e-8, frequencies=(0.1,))
  assert cpml.thickness == pytest.approx(5.0)
  assert cpml.thickness_phase_wavelengths[0] == pytest.approx(1.0)


def test_combined_report_handles_dispersive_material_and_high_level_api():
  metal = Drude(1.0, 0.5, 0.02)
  report = numerical_report_3d(
      {"air": Lossless(1.0), "metal": metal}, (1.0, 1.0, 1.0),
      0.25, (0.04, 0.06, 0.08), cpml_width=12)

  assert report.cfl.stable
  assert report.cpml.width_cells == 12
  assert report.material_band.maximum_relative_error >= 0.0
  assert report.material_band.maximum_omega_dt == pytest.approx(2.0 * np.pi * 0.08 * 0.25)
  assert report.spatial.minimum_cells_per_skin_depth < np.inf

  simulation = Simulation3D(
      GridSpec3D((4, 4, 40), (1.0, 1.0, 1.0), 0.25),
      BoundarySpec3D(8), background=Lossless(1.0))
  simulation.add_material("metal", metal)
  high_level = simulation.numerical_report((0.04, 0.06))
  assert high_level.cfl.dt == pytest.approx(0.25)
  assert high_level.material_band.material_names == ("background", "metal")


def test_paired_3d_convergence_compares_simulation_or_raw_spectrum_results():
  baseline = SimpleNamespace(diffraction=_spectrum(0.2))
  temporal = SimpleNamespace(diffraction=_spectrum(0.205))
  cpml = SimpleNamespace(diffraction=_spectrum(0.202))

  comparison = compare_scattering_results_3d(baseline, temporal)
  assert comparison.maximum_reflection_difference == pytest.approx(0.005)
  assert comparison.maximum_transmission_difference == pytest.approx(0.005)

  study = convergence_study_3d(
      baseline, {"longer_time": temporal, "wider_cpml": cpml})
  assert set(study.comparisons) == {"longer_time", "wider_cpml"}
  assert study.converged(0.01)
  assert not study.converged(0.001)


def test_reports_reject_invalid_frequency_or_cpml_inputs():
  with pytest.raises(ValueError, match="strictly increasing"):
    temporal_resolution_report_3d((0.1, 0.05), 0.2)
  with pytest.raises(ValueError, match="positive integer"):
    cpml_resolution_report_3d(
        Lossless(1.0), 1.0, 0, 1e-8, (0.1,))
