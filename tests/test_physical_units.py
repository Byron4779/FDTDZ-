import numpy as np
import pytest

from fdtdz_jax.material_accuracy import material_accuracy_report
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import Multipole
from fdtdz_jax.physical_units import ELECTRON_VOLT_J
from fdtdz_jax.physical_units import PhysicalScale
from fdtdz_jax.physical_units import REDUCED_PLANCK_J_S
from fdtdz_jax.physical_units import SPEED_OF_LIGHT_M_PER_S


def test_physical_scale_round_trips_length_time_and_frequency():
  scale = PhysicalScale(10e-9)
  assert scale.length(25e-9) == pytest.approx(2.5)
  assert scale.metres(2.5) == pytest.approx(25e-9)
  assert scale.time_unit_s == pytest.approx(10e-9 / SPEED_OF_LIGHT_M_PER_S)
  frequency = 3e14
  assert scale.hertz(scale.frequency(frequency)) == pytest.approx(frequency)


def test_ev_conversion_uses_hbar_omega_and_builds_drude_pole():
  scale = PhysicalScale(5e-9)
  expected = 2.0 * ELECTRON_VOLT_J / REDUCED_PLANCK_J_S * scale.time_unit_s
  assert scale.angular_frequency_from_ev(2.0) == pytest.approx(expected)
  pole = scale.drude_pole_from_ev(9.0, 0.07)
  assert pole.plasma_frequency == pytest.approx(
      scale.angular_frequency_from_ev(9.0))
  assert pole.collision_frequency == pytest.approx(
      scale.angular_frequency_from_ev(0.07))


def test_material_accuracy_converges_as_dt_decreases():
  material = Multipole(1.0, (DrudePole(2.0, 0.2),))
  omega = np.linspace(0.5, 1.5, 20)
  coarse = material_accuracy_report(material, 0.05, omega)
  fine = material_accuracy_report(material, 0.005, omega)
  assert fine.maximum_relative_error < coarse.maximum_relative_error
  assert fine.below_nyquist
  assert fine.maximum_omega_dt == pytest.approx(0.0075)


def test_lossless_material_has_zero_discretization_error():
  report = material_accuracy_report(
      Lossless(2.25), 0.1, np.linspace(0.0, 2.0, 5))
  assert report.maximum_absolute_error == pytest.approx(0.0)
  assert report.maximum_relative_error == pytest.approx(0.0)


def test_material_report_flags_nyquist_violation():
  report = material_accuracy_report(
      Lossless(1.0), 0.5, np.asarray([1.0, 7.0]))
  assert not report.below_nyquist
