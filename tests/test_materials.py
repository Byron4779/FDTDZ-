import dataclasses

import numpy as np
import pytest

from fdtdz_jax.materials import ADECoefficients
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import Drude
from fdtdz_jax.materials import DebyePole
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import Lorentz
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Lossless
from fdtdz_jax.materials import MaterialModel
from fdtdz_jax.materials import Multipole
from fdtdz_jax.materials import ade_coefficients


def test_material_model_ids_are_stable():
  assert int(MaterialModel.LOSSLESS) == 0
  assert int(MaterialModel.DEBYE) == 1
  assert int(MaterialModel.LORENTZ) == 2
  assert int(MaterialModel.DRUDE) == 3
  assert int(MaterialModel.MULTIPOLE) == 4


def test_lossless_coefficients_and_frequency_response():
  material = Lossless(eps_r=4.0)
  coeff = material.ade_coefficients(dt=0.1)

  assert coeff == ADECoefficients(model=MaterialModel.LOSSLESS, a0=0.25)
  np.testing.assert_array_equal(
      coeff.as_array(), np.asarray([0.25, 0, 0, 0, 0], np.float32))
  np.testing.assert_array_equal(
      material.relative_permittivity(np.asarray([0.0, 1.0])),
      np.asarray([4.0, 4.0], np.complex128))


def test_debye_coefficients_match_centered_difference_formula():
  material = Debye(eps_inf=2.0, eps_static=5.0, tau=3.0)
  coeff = material.ade_coefficients(dt=0.25)
  denominator = 5.0 * 0.25 + 2.0 * 3.0 * 2.0

  assert coeff.model == MaterialModel.DEBYE
  assert coeff.a0 == pytest.approx((0.25 + 6.0) / denominator)
  assert coeff.a1 == pytest.approx((0.25 - 6.0) / denominator)
  assert coeff.a2 == 0.0
  assert coeff.b1 == pytest.approx((12.0 - 1.25) / denominator)
  assert coeff.b2 == 0.0


def test_lorentz_coefficients_match_reference_equation():
  material = Lorentz(
      eps_inf=1.0, eps_static=2.25, omega_0=4.0, damping=0.28)
  dt = 0.05
  coeff = material.ade_coefficients(dt)
  omega_dt_sq = (4.0 * dt)**2
  damping_dt = 2.0 * 0.28 * dt
  denominator = omega_dt_sq * 2.25 + damping_dt + 2.0

  assert coeff.model == MaterialModel.LORENTZ
  assert coeff.a0 == pytest.approx(
      (omega_dt_sq + damping_dt + 2.0) / denominator)
  assert coeff.a1 == pytest.approx(-4.0 / denominator)
  assert coeff.a2 == pytest.approx(
      (omega_dt_sq - damping_dt + 2.0) / denominator)
  assert coeff.b1 == pytest.approx(4.0 / denominator)
  assert coeff.b2 == pytest.approx(
      -(omega_dt_sq * 2.25 - damping_dt + 2.0) / denominator)


def test_drude_coefficients_match_centered_difference_equation():
  material = Drude(2.0, plasma_frequency=5.0, collision_frequency=0.4)
  dt = 0.05
  coeff = material.ade_coefficients(dt)
  denominator = 2.0 * (2.0 + 0.4 * dt)
  assert coeff.model == MaterialModel.DRUDE
  assert coeff.a0 == pytest.approx(0.5)
  assert coeff.a1 == pytest.approx(-4.0 / denominator)
  assert coeff.a2 == pytest.approx((2.0 - 0.4 * dt) / denominator)
  assert coeff.b1 == pytest.approx(
      (8.0 - 2.0 * (5.0 * dt)**2) / denominator)
  assert coeff.b2 == pytest.approx(-2.0 * (2.0 - 0.4 * dt) / denominator)


def test_drude_frequency_response_is_passive_and_can_be_negative():
  material = Drude(1.0, plasma_frequency=5.0, collision_frequency=0.2)
  epsilon = material.relative_permittivity(np.asarray([1.0, 10.0]))
  assert epsilon[0].real < 0.0
  assert np.all(epsilon.imag > 0.0)


def test_multipole_frequency_response_is_sum_of_poles():
  poles = (
      DebyePole(1.5, 0.4),
      LorentzPole(2.0, 4.0, 0.2),
      DrudePole(3.0, 0.3),
  )
  material = Multipole(1.2, poles)
  omega = np.asarray([0.5, 2.0, 7.0])
  expected = 1.2 + sum(pole.susceptibility(omega) for pole in poles)
  np.testing.assert_allclose(material.relative_permittivity(omega), expected)
  assert np.all(material.relative_permittivity(omega).imag > 0.0)


def test_multipole_copies_poles_to_immutable_tuple():
  poles = [DebyePole(1.0, 0.5)]
  material = Multipole(2.0, poles)
  poles.append(DrudePole(3.0, 0.2))
  assert len(material.poles) == 1


@pytest.mark.parametrize("material", [
    Lossless(2.5),
    Debye(eps_inf=2.0, eps_static=4.0, tau=0.5),
    Lorentz(eps_inf=1.0, eps_static=2.25, omega_0=4.0, damping=0.28),
])
def test_zero_frequency_response_is_static_permittivity(material):
  expected = (material.eps_r if isinstance(material, Lossless) else
              material.eps_static)
  assert material.relative_permittivity(0.0) == pytest.approx(expected + 0j)


def test_passive_frequency_responses_have_nonnegative_imaginary_part():
  omega = np.linspace(0.0, 10.0, 101)
  materials = [
      Debye(eps_inf=2.0, eps_static=4.0, tau=0.5),
      Lorentz(eps_inf=1.0, eps_static=2.25, omega_0=4.0, damping=0.28),
  ]
  for material in materials:
    assert np.all(material.relative_permittivity(omega).imag >= 0.0)


@pytest.mark.parametrize("constructor,match", [
    (lambda: Lossless(0.0), "eps_r"),
    (lambda: Lossless(np.inf), "eps_r"),
    (lambda: Debye(2.0, 1.0, 0.5), "eps_static >= eps_inf"),
    (lambda: Debye(1.0, 2.0, 0.0), "tau"),
    (lambda: Lorentz(2.0, 1.0, 4.0, 0.1), "eps_static >= eps_inf"),
    (lambda: Lorentz(1.0, 2.0, 0.0, 0.1), "omega_0"),
    (lambda: Lorentz(1.0, 2.0, 4.0, -0.1), "damping"),
    (lambda: Drude(1.0, 0.0, 0.1), "plasma_frequency"),
    (lambda: Drude(1.0, 2.0, -0.1), "collision_frequency"),
    (lambda: DebyePole(0.0, 1.0), "delta_eps"),
    (lambda: LorentzPole(1.0, 0.0, 0.1), "omega_0"),
    (lambda: DrudePole(2.0, -0.1), "collision_frequency"),
    (lambda: Multipole(1.0, ()), "poles"),
])
def test_invalid_material_parameters_are_rejected(constructor, match):
  with pytest.raises(ValueError, match=match):
    constructor()


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
def test_invalid_time_step_is_rejected(dt):
  with pytest.raises(ValueError, match="dt"):
    Lossless(2.0).ade_coefficients(dt)


def test_materials_and_coefficients_are_immutable():
  material = Debye(eps_inf=2.0, eps_static=4.0, tau=0.5)
  coefficient = material.ade_coefficients(dt=0.1)
  with pytest.raises(dataclasses.FrozenInstanceError):
    material.tau = 1.0
  with pytest.raises(dataclasses.FrozenInstanceError):
    coefficient.a0 = 1.0


def test_module_level_coefficient_dispatch_rejects_unknown_types():
  assert ade_coefficients(Lossless(2.0), 0.1).a0 == 0.5
  with pytest.raises(TypeError, match="Lossless, Debye, Lorentz, or Drude"):
    ade_coefficients(object(), 0.1)


def test_public_package_exports_material_types():
  import fdtdz_jax

  assert fdtdz_jax.Lossless is Lossless
  assert fdtdz_jax.Debye is Debye
  assert fdtdz_jax.Drude is Drude
  assert fdtdz_jax.Multipole is Multipole
  assert fdtdz_jax.Lorentz is Lorentz
  assert fdtdz_jax.Material is not None
  assert fdtdz_jax.MaterialModel is MaterialModel
