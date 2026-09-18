import builtins
import dataclasses

import numpy as np
import pytest

from fdtdz_jax import PassiveFitResult, PermittivityTable
from fdtdz_jax import fit_passive_material, read_material_table
from fdtdz_jax.material_accuracy import material_accuracy_report
from fdtdz_jax.materials import DebyePole, DrudePole, LorentzPole, Lossless, Multipole
from fdtdz_jax.physical_units import PhysicalScale


def test_table_sorts_paired_samples_and_copies_arrays():
  omega = np.array([3., 1., 2.])
  epsilon = np.array([3+.3j, 1+.1j, 2+.2j])
  table = PermittivityTable(omega, epsilon)
  omega[:] = 9
  epsilon[:] = 9
  np.testing.assert_array_equal(table.angular_frequencies, [1, 2, 3])
  np.testing.assert_array_equal(table.permittivity, [1+.1j, 2+.2j, 3+.3j])
  with pytest.raises(ValueError, match="read-only"):
    table.permittivity[0] = 0
  with pytest.raises(dataclasses.FrozenInstanceError):
    table.angular_frequencies = omega


@pytest.mark.parametrize("axis,unit,coordinate", [
    ("omega", "normalized", lambda s, w: w),
    ("frequency", "normalized", lambda s, w: w / (2*np.pi)),
    ("omega", "rad/s", lambda s, w: s.radians_per_second(w)),
    ("frequency", "Hz", lambda s, w: s.hertz(w / (2*np.pi))),
    ("wavelength", "m", lambda s, w: 2*np.pi*s.length_unit_m / w),
    ("wavelength", "um", lambda s, w: 2*np.pi*s.length_unit_m / w * 1e6),
    ("wavelength", "nm", lambda s, w: 2*np.pi*s.length_unit_m / w * 1e9),
    ("energy", "eV", lambda s, w: w / s.angular_frequency_from_ev(1.0)),
])
def test_equivalent_spectral_units_give_same_nk_table(axis, unit, coordinate):
  scale = PhysicalScale(10e-9)
  omega = np.array([.1, .2, .3])
  n, k = np.array([1., 2., 3.]), np.array([.2, .3, .4])
  table = PermittivityTable.from_samples(
      coordinate(scale, omega), n, k, axis=axis, unit=unit, scale=scale)
  np.testing.assert_allclose(table.angular_frequencies, omega, rtol=1e-14)
  np.testing.assert_allclose(table.permittivity, (n + 1j*k)**2)


def test_csv_bom_header_selected_columns_wavelength_and_negative_real_epsilon(tmp_path):
  path = tmp_path / "metal.csv"
  path.write_text("k,label,wavelength_nm,n\n2,10,400,1\n3,20,800,1.5\n",
                  encoding="utf-8-sig")
  table = read_material_table(path, delimiter=",", skiprows=1,
                              columns=(2, 3, 0), axis="wavelength", unit="nm",
                              scale=PhysicalScale(10e-9))
  np.testing.assert_allclose(table.angular_frequencies, [np.pi/40, np.pi/20])
  np.testing.assert_allclose(table.permittivity, [-6.75+9j, -3+4j])


@pytest.mark.parametrize("delimiter", [None, "\t"])
def test_epsilon_file_comments_and_opposite_fourier_sign(tmp_path, delimiter):
  path = tmp_path / "epsilon.txt"
  path.write_text("# omega real imag\n2\t-3\t-.5\n1\t-4\t-.2\n", encoding="utf-8")
  table = read_material_table(path, representation="epsilon", delimiter=delimiter,
                              imaginary_sign=-1)
  np.testing.assert_array_equal(table.permittivity, [-4+.2j, -3+.5j])


@pytest.mark.parametrize("omega,epsilon,match", [
    ([1], [2], "at least two"),
    ([0, 1], [2, 2], "positive"),
    ([1, 1], [2, 2], "unique"),
    ([1, np.nan], [2, 2], "finite"),
    ([1, 2], [2, np.inf], "finite"),
    ([1, 2], [2], "shape"),
    ([1, 2], [2, 2-.01j], "passive"),
])
def test_invalid_table_rejected(omega, epsilon, match):
  with pytest.raises(ValueError, match=match):
    PermittivityTable(omega, epsilon)


@pytest.mark.parametrize("kwargs,match", [
    ({"axis": "wavelength", "unit": "nm"}, "PhysicalScale"),
    ({"axis": "frequency", "unit": "rad/s", "scale": PhysicalScale(1e-9)}, "unsupported"),
    ({"representation": "other"}, "representation"),
    ({"imaginary_sign": -1}, "only to epsilon"),
    ({"imaginary_sign": True}, "imaginary_sign"),
    ({"scale": 1}, "PhysicalScale"),
])
def test_invalid_sample_options_rejected(kwargs, match):
  with pytest.raises((TypeError, ValueError), match=match):
    PermittivityTable.from_samples([1, 2], [1, 2], [.1, .2], **kwargs)


@pytest.mark.parametrize("n,k", [([-1, 1], [0, 0]), ([1, 1], [-.1, 0])])
def test_negative_n_or_k_rejected(n, k):
  with pytest.raises(ValueError, match="non-negative n and k"):
    PermittivityTable.from_samples([1, 2], n, k)


@pytest.mark.parametrize("values", [[True, False], [1j, 2j], ["1", "2"]])
def test_non_real_coordinates_rejected(values):
  with pytest.raises(TypeError, match="real numbers"):
    PermittivityTable.from_samples(values, [1, 2], [0, 0])


@pytest.mark.parametrize("kwargs", [
    {"columns": (0, 1)}, {"columns": (0, 1, 1)}, {"columns": (0, 1, -1)},
    {"columns": (0, 1, True)}, {"skiprows": -1}, {"skiprows": True},
])
def test_invalid_file_options_checked_before_io(kwargs):
  with pytest.raises(ValueError):
    read_material_table("does-not-exist", **kwargs)


def _known_material():
  return Multipole(1.7, (DebyePole(.8, 2.5), LorentzPole(1.3, .65, .04),
                         DrudePole(.5, .03)))


@pytest.mark.parametrize("fixed_background", [False, True])
def test_recovers_mixed_material_and_matches_held_out_frequencies(fixed_background):
  expected = _known_material()
  omega = np.linspace(.1, 1., 100)
  table = PermittivityTable(omega, expected.relative_permittivity(omega))
  # Arbitrary candidate strengths must not constrain fitted strengths.
  candidates = (DebyePole(100, 2.5), LorentzPole(.001, .65, .04), DrudePole(9, .03))
  fit = fit_passive_material(table, candidates,
                            eps_inf=1.7 if fixed_background else None)
  assert isinstance(fit, PassiveFitResult)
  assert fit.material.eps_inf == pytest.approx(1.7, abs=1e-12)
  np.testing.assert_allclose(fit.pole_strengths, [.8, 1.3, .25], atol=1e-12)
  held_out = np.linspace(.11, .99, 61)
  np.testing.assert_allclose(fit.material.relative_permittivity(held_out),
                            expected.relative_permittivity(held_out), atol=1e-12)
  assert fit.maximum_absolute_error < 1e-11
  assert fit.rms_error < 1e-11
  assert np.all(fit.material.relative_permittivity(np.geomspace(.001, 100, 200)).imag >= 0)
  assert not fit.pole_strengths.flags.writeable
  report = material_accuracy_report(fit.material, .005, omega)
  assert report.maximum_relative_error < .001


def test_zero_strength_candidates_omitted_and_background_floor_enforced():
  table = PermittivityTable([1, 2, 3], [2, 2, 2])
  fit = fit_passive_material(table, [DebyePole(1, 1)], eps_inf=2)
  assert isinstance(fit.material, Lossless)
  np.testing.assert_array_equal(fit.pole_strengths, [0])
  floor_fit = fit_passive_material(table, [], eps_inf_min=3)
  assert floor_fit.material == Lossless(3)
  np.testing.assert_array_equal(floor_fit.absolute_error, [1, 1, 1])


def test_weighted_fit_and_report_match_analytic_constant_solution():
  table = PermittivityTable([1, 2, 3], [2+.1j, 3+.2j, 9+.3j])
  weights = np.array([2., 4., 1.])
  fit = fit_passive_material(table, [], weights=weights)
  expected = np.sum(weights**2 * table.permittivity.real) / np.sum(weights**2)
  assert isinstance(fit.material, Lossless)
  assert fit.material.eps_r == pytest.approx(expected)
  residual = np.abs(expected - table.permittivity)
  np.testing.assert_allclose(fit.absolute_error, residual)
  np.testing.assert_allclose(fit.relative_error, residual / np.abs(table.permittivity))
  assert fit.rms_error == pytest.approx(np.sqrt(np.mean(residual**2)))
  assert fit.maximum_absolute_error == pytest.approx(residual.max())
  scaled = fit_passive_material(table, [], weights=weights*1e100)
  assert scaled.material.eps_r == pytest.approx(expected)


def test_passivity_constraint_retains_error_instead_of_negative_strength():
  omega = np.linspace(.1, 1, 80)
  # This target would require a negative Lorentz oscillator strength.
  target = 3 - LorentzPole(.1, 2, 0).susceptibility(omega)
  fit = fit_passive_material(PermittivityTable(omega, target),
                            [LorentzPole(1, 2, 0)], eps_inf=3)
  assert isinstance(fit.material, Lossless)
  assert fit.pole_strengths[0] == 0
  assert fit.maximum_absolute_error > .1


@pytest.mark.parametrize("kwargs,match", [
    ({"eps_inf": 0}, "eps_inf"), ({"eps_inf_min": 0}, "eps_inf_min"),
    ({"relative_floor": 0}, "relative_floor"),
    ({"weights": [1, 0]}, "positive"), ({"weights": [1]}, "shape"),
    ({"weights": [1, np.nan]}, "finite"),
])
def test_invalid_fit_parameters_rejected(kwargs, match):
  with pytest.raises(ValueError, match=match):
    fit_passive_material(PermittivityTable([1, 2], [2, 2]), [], **kwargs)


def test_singular_undamped_resonance_and_invalid_poles_rejected():
  table = PermittivityTable([1, 2], [2, 2])
  with pytest.raises(ValueError, match="non-finite"):
    fit_passive_material(table, [LorentzPole(1, 1, 0)])
  with pytest.raises(TypeError, match=r"poles\[0\]"):
    fit_passive_material(table, [Lossless(2)])
  with pytest.raises(TypeError, match="PermittivityTable"):
    fit_passive_material(None, [])


def test_import_does_not_require_scipy_and_fit_has_actionable_error(monkeypatch):
  original = builtins.__import__

  def without_scipy(name, *args, **kwargs):
    if name == "scipy.optimize":
      raise ImportError("simulated absent SciPy")
    return original(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", without_scipy)
  table = PermittivityTable.from_samples([1, 2], [1, 1], [0, 0])
  with pytest.raises(ImportError, match=r"fdtdz\[fit\]"):
    fit_passive_material(table, [])
  assert fit_passive_material(table, [], eps_inf=1).material == Lossless(1)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_fitted_material_matches_known_material_in_3d_simulation(backend):
  from fdtdz_jax import BoundarySpec3D, GridSpec3D, Layer3D
  from fdtdz_jax import PlaneWaveSource3D, ScatteringMonitor3D, Simulation3D

  known = Multipole(1.7, (DebyePole(.8, 2.5), LorentzPole(1.3, .65, .04)))
  omega = np.linspace(.1, 1, 100)
  fit = fit_passive_material(
      PermittivityTable(omega, known.relative_permittivity(omega)), known.poles)

  def run(material):
    grid = GridSpec3D((2, 2, 28), (1., 1., 1.), .25)
    simulation = Simulation3D(grid, BoundarySpec3D(4), backend=backend,
                              dtype=np.float64 if backend == "numpy" else np.float32)
    simulation.add_material("film", material)
    simulation.add_geometry(Layer3D(12, 15, "film"))
    simulation.set_source(PlaneWaveSource3D(np.sin(np.arange(50)*.2), 6))
    simulation.set_monitors(ScatteringMonitor3D(9, 20))
    return simulation.run()

  expected, actual = run(known), run(fit.material)
  np.testing.assert_allclose(actual.final_state.electric,
                            expected.final_state.electric, rtol=2e-6, atol=1e-7)
  np.testing.assert_allclose(actual.diffraction.reflection,
                            expected.diffraction.reflection, rtol=2e-6, atol=1e-7,
                            equal_nan=True)
