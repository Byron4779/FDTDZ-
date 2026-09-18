"""Scalar executable reference for multi-pole ADE materials."""

from dataclasses import dataclass
import math

import numpy as np

from .materials import DebyePole
from .materials import DrudePole
from .materials import LorentzPole
from .materials import Multipole


@dataclass(frozen=True)
class PoleADECoefficients:
  """Coefficients of a polarization update with shared electric history."""

  q_next: float
  q_current: float
  q_previous: float
  r_current: float
  r_previous: float


@dataclass(frozen=True)
class MultipoleADEState:
  """Electric and per-pole polarization histories through one time index."""

  electric_current: float
  electric_previous: float
  polarization_current: np.ndarray
  polarization_previous: np.ndarray


def pole_ade_coefficients(pole, dt):
  """Discretize one passive pole using centered/trapezoidal differences."""
  dt = float(dt)
  if not math.isfinite(dt) or dt <= 0.0:
    raise ValueError("dt must be finite and positive.")
  if isinstance(pole, DebyePole):
    denominator = 2.0 * pole.tau + dt
    q = pole.delta_eps * dt / denominator
    return PoleADECoefficients(
        q_next=q, q_current=q, q_previous=0.0,
        r_current=(2.0 * pole.tau - dt) / denominator,
        r_previous=0.0)
  if isinstance(pole, LorentzPole):
    omega_dt_sq = (pole.omega_0 * dt)**2
    damping_dt = 2.0 * pole.damping * dt
    denominator = omega_dt_sq + damping_dt + 2.0
    q = pole.delta_eps * omega_dt_sq / denominator
    return PoleADECoefficients(
        q_next=q, q_current=0.0, q_previous=q,
        r_current=4.0 / denominator,
        r_previous=-(omega_dt_sq - damping_dt + 2.0) / denominator)
  if isinstance(pole, DrudePole):
    collision_dt = pole.collision_frequency * dt
    denominator = 2.0 + collision_dt
    return PoleADECoefficients(
        q_next=0.0,
        q_current=2.0 * (pole.plasma_frequency * dt)**2 / denominator,
        q_previous=0.0,
        r_current=4.0 / denominator,
        r_previous=-(2.0 - collision_dt) / denominator)
  raise TypeError("pole must be DebyePole, LorentzPole, or DrudePole.")


def initialize_multipole_ade(material):
  if not isinstance(material, Multipole):
    raise TypeError("material must be a Multipole instance.")
  zeros = np.zeros(len(material.poles), dtype=float)
  return MultipoleADEState(0.0, 0.0, zeros.copy(), zeros.copy())


def multipole_ade_step(material, dt, displacement_next, state):
  """Recover E[n+1] from D[n+1] and advance all pole polarizations."""
  if not isinstance(material, Multipole):
    raise TypeError("material must be a Multipole instance.")
  if not isinstance(state, MultipoleADEState):
    raise TypeError("state must be a MultipoleADEState instance.")
  polarization_current = np.asarray(state.polarization_current, dtype=float)
  polarization_previous = np.asarray(state.polarization_previous, dtype=float)
  expected_shape = (len(material.poles),)
  if (polarization_current.shape != expected_shape or
      polarization_previous.shape != expected_shape):
    raise ValueError(f"polarization histories must have shape {expected_shape}.")
  values = np.asarray([
      displacement_next, state.electric_current, state.electric_previous,
      *polarization_current, *polarization_previous], dtype=float)
  if not np.all(np.isfinite(values)):
    raise ValueError("state and displacement_next must contain finite values.")

  coefficients = [pole_ade_coefficients(pole, dt) for pole in material.poles]
  known_polarization = np.asarray([
      coefficient.q_current * state.electric_current +
      coefficient.q_previous * state.electric_previous +
      coefficient.r_current * polarization_current[index] +
      coefficient.r_previous * polarization_previous[index]
      for index, coefficient in enumerate(coefficients)])
  denominator = material.eps_inf + sum(
      coefficient.q_next for coefficient in coefficients)
  electric_next = (
      float(displacement_next) - np.sum(known_polarization)) / denominator
  polarization_next = known_polarization + np.asarray([
      coefficient.q_next * electric_next for coefficient in coefficients])
  return electric_next, MultipoleADEState(
      electric_current=electric_next,
      electric_previous=float(state.electric_current),
      polarization_current=polarization_next,
      polarization_previous=polarization_current.copy())


def simulate_multipole_ade(material, dt, displacement, initial_state=None,
                           dtype=np.float64, return_final_state=False):
  """Simulate a real normalized-displacement sequence through all poles."""
  if not isinstance(material, Multipole):
    raise TypeError("material must be a Multipole instance.")
  displacement = np.asarray(displacement)
  if (displacement.ndim != 1 or not np.issubdtype(displacement.dtype, np.number)
      or np.iscomplexobj(displacement)):
    raise ValueError("displacement must be a one-dimensional real array.")
  if not np.all(np.isfinite(displacement)):
    raise ValueError("displacement must contain only finite values.")
  dtype = np.dtype(dtype)
  if not np.issubdtype(dtype, np.floating):
    raise TypeError("dtype must be a floating dtype.")
  state = (initialize_multipole_ade(material) if initial_state is None else
           initial_state)
  electric = np.empty(displacement.shape, dtype=dtype)
  for index, displacement_next in enumerate(displacement):
    electric_next, state = multipole_ade_step(
        material, dt, displacement_next, state)
    electric[index] = electric_next
  return (electric, state) if return_final_state else electric


def discrete_multipole_permittivity(material, dt, omega):
  """Exact z-transform frequency response of the multi-pole recurrence."""
  if not isinstance(material, Multipole):
    raise TypeError("material must be a Multipole instance.")
  omega = np.asarray(omega)
  if (not np.issubdtype(omega.dtype, np.number) or np.iscomplexobj(omega) or
      not np.all(np.isfinite(omega))):
    raise ValueError("omega must contain finite real angular frequencies.")
  z_history = np.exp(1j * omega * float(dt))
  result = np.full(omega.shape, material.eps_inf, dtype=np.complex128)
  for pole in material.poles:
    coefficient = pole_ade_coefficients(pole, dt)
    numerator = (coefficient.q_next +
                 coefficient.q_current * z_history +
                 coefficient.q_previous * z_history**2)
    denominator = (1.0 - coefficient.r_current * z_history -
                   coefficient.r_previous * z_history**2)
    result = result + numerator / denominator
  return result


__all__ = [
    "MultipoleADEState",
    "PoleADECoefficients",
    "discrete_multipole_permittivity",
    "initialize_multipole_ade",
    "multipole_ade_step",
    "pole_ade_coefficients",
    "simulate_multipole_ade",
]
