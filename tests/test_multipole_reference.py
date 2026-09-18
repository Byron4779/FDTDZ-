import numpy as np
import pytest

from fdtdz_jax.materials import DebyePole
from fdtdz_jax.materials import DrudePole
from fdtdz_jax.materials import LorentzPole
from fdtdz_jax.materials import Multipole
from fdtdz_jax.multipole_reference import discrete_multipole_permittivity
from fdtdz_jax.multipole_reference import initialize_multipole_ade
from fdtdz_jax.multipole_reference import multipole_ade_step
from fdtdz_jax.multipole_reference import pole_ade_coefficients
from fdtdz_jax.multipole_reference import simulate_multipole_ade


@pytest.mark.parametrize("pole", [
    DebyePole(1.5, 0.4),
    LorentzPole(2.0, 4.0, 0.2),
    DrudePole(3.0, 0.3),
])
def test_discrete_single_pole_converges_to_analytic_response(pole):
  material = Multipole(1.2, (pole,))
  omega = 1.3
  coarse = discrete_multipole_permittivity(material, 0.02, omega)
  fine = discrete_multipole_permittivity(material, 0.002, omega)
  analytic = material.relative_permittivity(omega)
  assert abs(fine - analytic) < abs(coarse - analytic)
  assert fine == pytest.approx(analytic, rel=3e-5, abs=3e-5)


def test_multipole_step_satisfies_displacement_identity():
  material = Multipole(1.5, (
      DebyePole(0.8, 0.5), LorentzPole(1.2, 3.0, 0.15),
      DrudePole(2.0, 0.2)))
  state = initialize_multipole_ade(material)
  electric, state = multipole_ade_step(material, 0.01, 2.5, state)
  assert 1.5 * electric + np.sum(state.polarization_current) == pytest.approx(2.5)


def test_sinusoid_matches_discrete_multipole_response():
  material = Multipole(1.3, (
      DebyePole(0.7, 0.4), LorentzPole(1.1, 4.0, 0.2),
      DrudePole(2.5, 0.3)))
  omega = 1.5
  samples_per_period = 2000
  dt = 2.0 * np.pi / (omega * samples_per_period)
  time = dt * np.arange(100 * samples_per_period)
  displacement = np.cos(omega * time)
  electric = simulate_multipole_ade(material, dt, displacement)
  start = time.size // 2
  phase = np.exp(1j * omega * time[start:])
  measured = np.sum(displacement[start:] * phase) / np.sum(
      electric[start:] * phase)
  expected = discrete_multipole_permittivity(material, dt, omega)
  assert measured == pytest.approx(expected, rel=8e-5, abs=8e-5)


def test_pole_coefficients_validate_type():
  with pytest.raises(TypeError, match="DebyePole"):
    pole_ade_coefficients(object(), 0.1)
