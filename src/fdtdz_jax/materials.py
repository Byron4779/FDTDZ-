"""Material models and time-discrete ADE coefficients.

The FDTD solver evolves real-valued time-domain fields.  Complex relative
permittivity is used only to describe or validate a material in the frequency
domain; it is not the dtype of the time-domain fields.

The dispersive models use the Fourier convention employed by Joseph, Hagness,
and Taflove (1991), whose inverse transform contains ``exp(-1j * omega * t)``.
Consequently, passive materials have a non-negative imaginary permittivity at
positive angular frequency.  This sign is reversed in software that uses the
opposite time-harmonic convention.

Internally, the electric displacement is normalized as ``D_r = D / epsilon_0``
so that a lossless material satisfies ``D_r = epsilon_r * E``.  Every model is
converted to the common recurrence

  E[n + 1] = a0 * D_r[n + 1] + a1 * D_r[n] + a2 * D_r[n - 1]
             + b1 * E[n] + b2 * E[n - 1].

This module contains no JAX or CUDA dependency, which keeps material creation
and validation usable before a simulation backend is initialized.
"""

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Union

import numpy as np


class MaterialModel(IntEnum):
  """Material model identifiers shared by Python and future CUDA kernels."""

  LOSSLESS = 0
  DEBYE = 1
  LORENTZ = 2
  DRUDE = 3
  MULTIPOLE = 4


@dataclass(frozen=True)
class ADECoefficients:
  """Coefficients for the common real-valued D-to-E recurrence."""

  model: MaterialModel
  a0: float
  a1: float = 0.0
  a2: float = 0.0
  b1: float = 0.0
  b2: float = 0.0

  def as_array(self, dtype=np.float32):
    """Return ``(a0, a1, a2, b1, b2)`` in a backend-friendly array."""
    return np.asarray((self.a0, self.a1, self.a2, self.b1, self.b2),
                      dtype=dtype)


def _positive_finite(name, value):
  if isinstance(value, (bool, np.bool_)) or not isinstance(
      value, (int, float, np.integer, np.floating)):
    raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
  value = float(value)
  if not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"{name} must be finite and greater than zero, got {value}.")
  return value


def _nonnegative_finite(name, value):
  if isinstance(value, (bool, np.bool_)) or not isinstance(
      value, (int, float, np.integer, np.floating)):
    raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
  value = float(value)
  if not math.isfinite(value) or value < 0.0:
    raise ValueError(f"{name} must be finite and non-negative, got {value}.")
  return value


def _angular_frequency(omega):
  omega = np.asarray(omega)
  if not np.issubdtype(omega.dtype, np.number):
    raise TypeError("omega must be a real scalar or array of angular frequencies.")
  if np.iscomplexobj(omega):
    raise TypeError("omega must contain real angular frequencies.")
  if not np.all(np.isfinite(omega)):
    raise ValueError("omega must contain only finite angular frequencies.")
  return omega


@dataclass(frozen=True)
class Lossless:
  """Frequency-independent, isotropic, lossless dielectric."""

  eps_r: float

  def __post_init__(self):
    object.__setattr__(self, "eps_r", _positive_finite("eps_r", self.eps_r))

  @property
  def model(self):
    return MaterialModel.LOSSLESS

  def ade_coefficients(self, dt):
    """Return coefficients for ``E[n + 1] = D_r[n + 1] / eps_r``."""
    _positive_finite("dt", dt)
    return ADECoefficients(model=self.model, a0=1.0 / self.eps_r)

  def relative_permittivity(self, omega):
    """Evaluate the analytic relative permittivity."""
    omega = _angular_frequency(omega)
    return np.full(omega.shape, self.eps_r, dtype=np.complex128)


@dataclass(frozen=True)
class Debye:
  """Single-pole passive Debye material.

  Args:
    eps_inf: Relative permittivity in the infinite-frequency limit.
    eps_static: Relative permittivity at zero frequency.
    tau: Positive relaxation time in the same time unit as the solver's ``dt``.
  """

  eps_inf: float
  eps_static: float
  tau: float

  def __post_init__(self):
    eps_inf = _positive_finite("eps_inf", self.eps_inf)
    eps_static = _positive_finite("eps_static", self.eps_static)
    tau = _positive_finite("tau", self.tau)
    if eps_static < eps_inf:
      raise ValueError(
          "A passive Debye pole requires eps_static >= eps_inf, got "
          f"{eps_static} < {eps_inf}.")
    object.__setattr__(self, "eps_inf", eps_inf)
    object.__setattr__(self, "eps_static", eps_static)
    object.__setattr__(self, "tau", tau)

  @property
  def model(self):
    return MaterialModel.DEBYE

  def ade_coefficients(self, dt):
    """Return centered-difference coefficients for the Debye ADE."""
    dt = _positive_finite("dt", dt)
    denominator = self.eps_static * dt + 2.0 * self.tau * self.eps_inf
    return ADECoefficients(
        model=self.model,
        a0=(dt + 2.0 * self.tau) / denominator,
        a1=(dt - 2.0 * self.tau) / denominator,
        b1=(2.0 * self.tau * self.eps_inf -
            self.eps_static * dt) / denominator,
    )

  def relative_permittivity(self, omega):
    """Evaluate ``eps_inf + (eps_static-eps_inf)/(1 - 1j*omega*tau)``."""
    omega = _angular_frequency(omega)
    return self.eps_inf + (self.eps_static - self.eps_inf) / (
        1.0 - 1j * omega * self.tau)


@dataclass(frozen=True)
class Lorentz:
  """Single-pole passive Lorentz material.

  ``damping`` is the coefficient delta in a differential equation whose first
  derivative term is ``2 * delta * d/dt``.  It therefore has units of inverse
  time, as does ``omega_0``.

  Args:
    eps_inf: Relative permittivity in the infinite-frequency limit.
    eps_static: Relative permittivity at zero frequency.
    omega_0: Positive resonant angular frequency.
    damping: Non-negative damping coefficient delta.
  """

  eps_inf: float
  eps_static: float
  omega_0: float
  damping: float

  def __post_init__(self):
    eps_inf = _positive_finite("eps_inf", self.eps_inf)
    eps_static = _positive_finite("eps_static", self.eps_static)
    omega_0 = _positive_finite("omega_0", self.omega_0)
    damping = _nonnegative_finite("damping", self.damping)
    if eps_static < eps_inf:
      raise ValueError(
          "A passive Lorentz pole requires eps_static >= eps_inf, got "
          f"{eps_static} < {eps_inf}.")
    object.__setattr__(self, "eps_inf", eps_inf)
    object.__setattr__(self, "eps_static", eps_static)
    object.__setattr__(self, "omega_0", omega_0)
    object.__setattr__(self, "damping", damping)

  @property
  def model(self):
    return MaterialModel.LORENTZ

  def ade_coefficients(self, dt):
    """Return the second-order coefficients from Joseph et al., Eq. (11)."""
    dt = _positive_finite("dt", dt)
    omega_dt_sq = (self.omega_0 * dt)**2
    damping_dt = 2.0 * self.damping * dt
    denominator = (omega_dt_sq * self.eps_static +
                   damping_dt * self.eps_inf + 2.0 * self.eps_inf)
    return ADECoefficients(
        model=self.model,
        a0=(omega_dt_sq + damping_dt + 2.0) / denominator,
        a1=-4.0 / denominator,
        a2=(omega_dt_sq - damping_dt + 2.0) / denominator,
        b1=4.0 * self.eps_inf / denominator,
        b2=-(omega_dt_sq * self.eps_static -
             damping_dt * self.eps_inf +
             2.0 * self.eps_inf) / denominator,
    )

  def relative_permittivity(self, omega):
    """Evaluate the analytic single-pole Lorentz relative permittivity."""
    omega = _angular_frequency(omega)
    numerator = self.omega_0**2 * (self.eps_static - self.eps_inf)
    denominator = (self.omega_0**2 - omega**2 -
                   2j * self.damping * omega)
    return self.eps_inf + numerator / denominator


@dataclass(frozen=True)
class Drude:
  """Single-pole free-electron Drude material.

  The polarization obeys ``P'' + gamma P' = omega_p**2 E`` in normalized
  units.  ``collision_frequency`` is gamma (not gamma/2).  With the module's
  ``exp(-1j*omega*t)`` inverse-transform convention this gives
  ``eps_inf - omega_p**2/(omega**2 + 1j*gamma*omega)``.
  """

  eps_inf: float
  plasma_frequency: float
  collision_frequency: float

  def __post_init__(self):
    object.__setattr__(self, "eps_inf", _positive_finite(
        "eps_inf", self.eps_inf))
    object.__setattr__(self, "plasma_frequency", _positive_finite(
        "plasma_frequency", self.plasma_frequency))
    object.__setattr__(self, "collision_frequency", _nonnegative_finite(
        "collision_frequency", self.collision_frequency))

  @property
  def model(self):
    return MaterialModel.DRUDE

  def ade_coefficients(self, dt):
    """Return centered-difference coefficients for the Drude ADE."""
    dt = _positive_finite("dt", dt)
    collision_dt = self.collision_frequency * dt
    plasma_dt_sq = (self.plasma_frequency * dt)**2
    denominator = self.eps_inf * (2.0 + collision_dt)
    return ADECoefficients(
        model=self.model,
        a0=1.0 / self.eps_inf,
        a1=-4.0 / denominator,
        a2=(2.0 - collision_dt) / denominator,
        b1=(4.0 * self.eps_inf - 2.0 * plasma_dt_sq) / denominator,
        b2=-self.eps_inf * (2.0 - collision_dt) / denominator,
    )

  def relative_permittivity(self, omega):
    """Evaluate the analytic Drude relative permittivity."""
    omega = _angular_frequency(omega)
    denominator = omega**2 + 1j * self.collision_frequency * omega
    with np.errstate(divide="ignore", invalid="ignore"):
      return self.eps_inf - self.plasma_frequency**2 / denominator


@dataclass(frozen=True)
class DebyePole:
  """One Debye susceptibility contribution ``delta_eps/(1-i*omega*tau)``."""

  delta_eps: float
  tau: float

  def __post_init__(self):
    object.__setattr__(self, "delta_eps", _positive_finite(
        "delta_eps", self.delta_eps))
    object.__setattr__(self, "tau", _positive_finite("tau", self.tau))

  def susceptibility(self, omega):
    return self.delta_eps / (1.0 - 1j * omega * self.tau)


@dataclass(frozen=True)
class LorentzPole:
  """One Lorentz susceptibility contribution."""

  delta_eps: float
  omega_0: float
  damping: float

  def __post_init__(self):
    object.__setattr__(self, "delta_eps", _positive_finite(
        "delta_eps", self.delta_eps))
    object.__setattr__(self, "omega_0", _positive_finite(
        "omega_0", self.omega_0))
    object.__setattr__(self, "damping", _nonnegative_finite(
        "damping", self.damping))

  def susceptibility(self, omega):
    return (self.delta_eps * self.omega_0**2 /
            (self.omega_0**2 - omega**2 - 2j * self.damping * omega))


@dataclass(frozen=True)
class DrudePole:
  """One free-electron Drude susceptibility contribution."""

  plasma_frequency: float
  collision_frequency: float

  def __post_init__(self):
    object.__setattr__(self, "plasma_frequency", _positive_finite(
        "plasma_frequency", self.plasma_frequency))
    object.__setattr__(self, "collision_frequency", _nonnegative_finite(
        "collision_frequency", self.collision_frequency))

  def susceptibility(self, omega):
    denominator = omega**2 + 1j * self.collision_frequency * omega
    with np.errstate(divide="ignore", invalid="ignore"):
      return -self.plasma_frequency**2 / denominator


Pole = Union[DebyePole, LorentzPole, DrudePole]


@dataclass(frozen=True)
class Multipole:
  """Passive multi-pole material with a positive high-frequency background."""

  eps_inf: float
  poles: tuple

  def __post_init__(self):
    eps_inf = _positive_finite("eps_inf", self.eps_inf)
    poles = tuple(self.poles)
    if not poles:
      raise ValueError("poles must contain at least one pole.")
    for index, pole in enumerate(poles):
      if not isinstance(pole, (DebyePole, LorentzPole, DrudePole)):
        raise TypeError(
            f"poles[{index}] must be DebyePole, LorentzPole, or DrudePole.")
    object.__setattr__(self, "eps_inf", eps_inf)
    object.__setattr__(self, "poles", poles)

  @property
  def model(self):
    return MaterialModel.MULTIPOLE

  def relative_permittivity(self, omega):
    omega = _angular_frequency(omega)
    result = np.full(omega.shape, self.eps_inf, dtype=np.complex128)
    for pole in self.poles:
      result = result + pole.susceptibility(omega)
    return result


Material = Union[Lossless, Debye, Lorentz, Drude, Multipole]


def ade_coefficients(material, dt):
  """Generate common ADE coefficients for one supported material."""
  if not isinstance(material, (Lossless, Debye, Lorentz, Drude)):
    raise TypeError(
        "material must be a Lossless, Debye, Lorentz, or Drude instance, got "
        f"{type(material).__name__}.")
  return material.ade_coefficients(dt)


__all__ = [
    "ADECoefficients",
    "Debye",
    "DebyePole",
    "Drude",
    "DrudePole",
    "Lorentz",
    "LorentzPole",
    "Lossless",
    "Material",
    "MaterialModel",
    "Multipole",
    "Pole",
    "ade_coefficients",
]
