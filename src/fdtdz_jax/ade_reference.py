"""Scalar reference implementation of the D-to-E ADE recurrence.

This implementation favors clarity and validation over speed.  It is the
executable specification against which future JAX and CUDA material updates
can be compared one time step at a time.
"""

from dataclasses import dataclass
import math

import numpy as np

from .materials import ADECoefficients
from .materials import ade_coefficients


@dataclass(frozen=True)
class ADEState:
  """History immediately before one D-to-E update.

  ``d_current`` and ``e_current`` are D_r[n] and E[n]; ``d_previous`` and
  ``e_previous`` are D_r[n-1] and E[n-1].  Calling :func:`ade_step` with
  D_r[n+1] returns E[n+1] and advances all four history values.
  """

  d_current: float = 0.0
  d_previous: float = 0.0
  e_current: float = 0.0
  e_previous: float = 0.0

  def __post_init__(self):
    for name in ("d_current", "d_previous", "e_current", "e_previous"):
      value = getattr(self, name)
      if isinstance(value, (bool, np.bool_)) or not isinstance(
          value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number.")
      if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value}.")


def ade_step(coefficients, d_next, state=ADEState()):
  """Advance the common scalar ADE recurrence by one time sample.

  Args:
    coefficients: An :class:`ADECoefficients` instance.
    d_next: The normalized displacement D_r[n+1] = D[n+1] / epsilon_0.
    state: History through time index n.

  Returns:
    ``(e_next, next_state)`` where ``next_state`` contains history through
    time index n+1.
  """
  if not isinstance(coefficients, ADECoefficients):
    raise TypeError("coefficients must be an ADECoefficients instance.")
  if not isinstance(state, ADEState):
    raise TypeError("state must be an ADEState instance.")
  if isinstance(d_next, (bool, np.bool_)) or not isinstance(
      d_next, (int, float, np.integer, np.floating)):
    raise TypeError("d_next must be a real number.")
  d_next = float(d_next)
  if not math.isfinite(d_next):
    raise ValueError(f"d_next must be finite, got {d_next}.")

  e_next = (coefficients.a0 * d_next +
            coefficients.a1 * state.d_current +
            coefficients.a2 * state.d_previous +
            coefficients.b1 * state.e_current +
            coefficients.b2 * state.e_previous)
  next_state = ADEState(
      d_current=d_next,
      d_previous=state.d_current,
      e_current=e_next,
      e_previous=state.e_current,
  )
  return e_next, next_state


def simulate_ade(material, dt, displacement, initial_state=None,
                 dtype=np.float64, return_final_state=False):
  """Compute E for a real one-dimensional D_r time series.

  The first input sample is interpreted as D_r[0].  With the default zero
  state, samples at indices -1 and -2 are zero.  Supplying ``initial_state``
  allows an exact nonzero history to be used for steady-state checks.

  Args:
    material: A supported material model from :mod:`fdtdz_jax.materials`.
    dt: Positive time step used to generate the ADE coefficients.
    displacement: One-dimensional, finite, real D_r samples.
    initial_state: Optional history immediately before the first sample.
    dtype: Floating output dtype.  Use ``np.float32`` to emulate the planned
      first CUDA kernel, or ``np.float64`` for a high-accuracy reference.
    return_final_state: Also return the final :class:`ADEState` when true.

  Returns:
    The electric-field samples, or ``(samples, final_state)``.
  """
  displacement = np.asarray(displacement)
  if displacement.ndim != 1:
    raise ValueError(
        f"displacement must be one-dimensional, got shape {displacement.shape}.")
  if not np.issubdtype(displacement.dtype, np.number) or np.iscomplexobj(
      displacement):
    raise TypeError("displacement must contain real numbers.")
  if not np.all(np.isfinite(displacement)):
    raise ValueError("displacement must contain only finite values.")

  dtype = np.dtype(dtype)
  if not np.issubdtype(dtype, np.floating):
    raise TypeError(f"dtype must be a floating dtype, got {dtype}.")

  coefficients = ade_coefficients(material, dt)
  # Quantize coefficients once to emulate how a backend stores them.
  stored = coefficients.as_array(dtype=dtype)
  coefficients = ADECoefficients(
      model=coefficients.model,
      a0=float(stored[0]),
      a1=float(stored[1]),
      a2=float(stored[2]),
      b1=float(stored[3]),
      b2=float(stored[4]),
  )
  state = ADEState() if initial_state is None else initial_state
  if not isinstance(state, ADEState):
    raise TypeError("initial_state must be an ADEState instance or None.")

  electric = np.empty(displacement.shape, dtype=dtype)
  for index, d_next in enumerate(displacement):
    e_next, state = ade_step(coefficients, d_next, state)
    electric[index] = e_next

  if return_final_state:
    return electric, state
  return electric


def discrete_relative_permittivity(material, dt, omega):
  """Return the exact frequency response of the time-discrete recurrence.

  This is the z-transform of the implemented ADE update, not the continuous
  material model.  Comparing a simulated time series to this function tests
  implementation correctness; comparing this function to
  ``material.relative_permittivity`` measures time-discretization error.
  """
  coefficients = ade_coefficients(material, dt)
  omega = np.asarray(omega)
  if not np.issubdtype(omega.dtype, np.number) or np.iscomplexobj(omega):
    raise TypeError("omega must contain real angular frequencies.")
  if not np.all(np.isfinite(omega)):
    raise ValueError("omega must contain only finite angular frequencies.")

  history_phase = np.exp(1j * omega * float(dt))
  history_phase_sq = history_phase**2
  numerator = (1.0 - coefficients.b1 * history_phase -
               coefficients.b2 * history_phase_sq)
  denominator = (coefficients.a0 + coefficients.a1 * history_phase +
                 coefficients.a2 * history_phase_sq)
  with np.errstate(divide="ignore", invalid="ignore"):
    return numerator / denominator


__all__ = [
    "ADEState",
    "ade_step",
    "discrete_relative_permittivity",
    "simulate_ade",
]
