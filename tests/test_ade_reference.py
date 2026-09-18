import dataclasses

import numpy as np
import pytest

from fdtdz_jax.ade_reference import ADEState
from fdtdz_jax.ade_reference import ade_step
from fdtdz_jax.ade_reference import discrete_relative_permittivity
from fdtdz_jax.ade_reference import simulate_ade
from fdtdz_jax.materials import Debye
from fdtdz_jax.materials import Drude
from fdtdz_jax.materials import Lorentz
from fdtdz_jax.materials import Lossless


def _complex_amplitude(signal, omega, dt):
  """Amplitude for the exp(-1j*omega*t) convention used by the models."""
  time = dt * np.arange(signal.size)
  return (2.0 / signal.size) * np.sum(signal * np.exp(1j * omega * time))


def test_ade_step_uses_and_advances_all_history_values():
  coefficients = Lorentz(1.0, 2.25, 4.0, 0.28).ade_coefficients(0.05)
  state = ADEState(
      d_current=2.0, d_previous=3.0, e_current=5.0, e_previous=7.0)
  expected = (coefficients.a0 * 11.0 + coefficients.a1 * 2.0 +
              coefficients.a2 * 3.0 + coefficients.b1 * 5.0 +
              coefficients.b2 * 7.0)

  e_next, next_state = ade_step(coefficients, 11.0, state)

  assert e_next == pytest.approx(expected)
  assert next_state == ADEState(
      d_current=11.0,
      d_previous=2.0,
      e_current=e_next,
      e_previous=5.0,
  )


def test_lossless_sequence_is_exact_and_final_state_is_returned():
  displacement = np.asarray([1.0, -2.0, 4.0, 8.0])
  electric, state = simulate_ade(
      Lossless(4.0), 0.1, displacement, return_final_state=True)

  np.testing.assert_array_equal(electric, displacement / 4.0)
  assert state == ADEState(
      d_current=8.0,
      d_previous=4.0,
      e_current=2.0,
      e_previous=1.0,
  )


@pytest.mark.parametrize("material,omega,dt,periods,rtol", [
    (Debye(eps_inf=2.0, eps_static=4.0, tau=0.5), 1.0, 0.005, 40, 2e-5),
    (Lorentz(
        eps_inf=1.0, eps_static=2.25, omega_0=4.0, damping=0.28),
     2.0, 0.0025, 80, 3e-5),
    (Drude(eps_inf=1.0, plasma_frequency=3.0, collision_frequency=0.4),
     1.5, 0.0025, 100, 5e-5),
])
def test_sinusoidal_response_matches_discrete_transfer_function(
    material, omega, dt, periods, rtol):
  samples_per_period = int(round(2.0 * np.pi / (omega * dt)))
  dt = 2.0 * np.pi / (omega * samples_per_period)
  sample_count = periods * samples_per_period
  time = dt * np.arange(sample_count)
  displacement = np.cos(omega * time)

  electric = simulate_ade(material, dt, displacement)
  # Discard the first half so the zero-state transient has decayed.  The
  # remaining interval contains an integer number of periods.
  start = sample_count // 2
  d_amplitude = _complex_amplitude(displacement[start:], omega, dt)
  e_amplitude = _complex_amplitude(electric[start:], omega, dt)
  measured = d_amplitude / e_amplitude
  expected = discrete_relative_permittivity(material, dt, omega)

  assert measured == pytest.approx(expected, rel=rtol, abs=rtol)


@pytest.mark.parametrize("material,omega", [
    (Lossless(2.25), 2.0),
    (Debye(eps_inf=2.0, eps_static=4.0, tau=0.5), 1.0),
    (Lorentz(
        eps_inf=1.0, eps_static=2.25, omega_0=4.0, damping=0.28), 2.0),
    (Drude(eps_inf=1.0, plasma_frequency=3.0, collision_frequency=0.4), 1.5),
])
def test_discrete_response_converges_to_continuous_material(material, omega):
  coarse = discrete_relative_permittivity(material, 0.02, omega)
  fine = discrete_relative_permittivity(material, 0.002, omega)
  analytic = material.relative_permittivity(omega)

  assert abs(fine - analytic) < abs(coarse - analytic) or abs(
      fine - analytic) < 1e-14
  assert fine == pytest.approx(analytic, rel=2e-5, abs=2e-5)


def test_float32_path_quantizes_output_and_remains_close_to_float64():
  material = Debye(eps_inf=2.0, eps_static=4.0, tau=0.5)
  displacement = np.sin(np.linspace(0.0, 20.0, 1000))
  electric32 = simulate_ade(material, 0.01, displacement, dtype=np.float32)
  electric64 = simulate_ade(material, 0.01, displacement, dtype=np.float64)

  assert electric32.dtype == np.float32
  assert electric64.dtype == np.float64
  np.testing.assert_allclose(electric32, electric64, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("displacement,error", [
    (np.ones((2, 2)), ValueError),
    (np.asarray([1.0 + 1.0j]), TypeError),
    (np.asarray([1.0, np.nan]), ValueError),
])
def test_invalid_displacement_is_rejected(displacement, error):
  with pytest.raises(error):
    simulate_ade(Lossless(1.0), 0.1, displacement)


def test_state_is_immutable_and_publicly_exported():
  import fdtdz_jax

  state = ADEState()
  with pytest.raises(dataclasses.FrozenInstanceError):
    state.e_current = 1.0
  assert fdtdz_jax.ADEState is ADEState
  assert fdtdz_jax.simulate_ade is simulate_ade
