"""Optical tables and passive, fixed-pole ADE material fits.

Uses normalized omega and exp(-i*omega*t). Only strengths and optionally
eps_inf are fitted; pole positions/widths are supplied by the caller.
SciPy is imported only when fitting. Fits do not certify out-of-band accuracy
or the stability/accuracy of a time-discrete FDTD simulation.
"""

from dataclasses import dataclass

import numpy as np

from .materials import DebyePole, DrudePole, LorentzPole, Lossless, Material, Multipole
from .materials import _positive_finite
from .physical_units import PhysicalScale, SPEED_OF_LIGHT_M_PER_S


def _real_vector(name, values):
  array = np.asarray(values)
  if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
    raise TypeError(f"{name} must contain real numbers.")
  array = np.array(array, dtype=np.float64, copy=True)
  if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must be a non-empty finite 1D array.")
  return array


def _readonly(values):
  result = np.array(values, copy=True)
  result.setflags(write=False)
  return result


def _normalized_omega(values, axis, unit, scale):
  values = _real_vector("spectral coordinates", values)
  if np.any(values <= 0):
    raise ValueError("spectral coordinates must be strictly positive.")
  if scale is not None and not isinstance(scale, PhysicalScale):
    raise TypeError("scale must be a PhysicalScale.")
  with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
    if axis in ("omega", "frequency") and unit == "normalized":
      result = values if axis == "omega" else 2 * np.pi * values
    else:
      if scale is None:
        raise ValueError("physical spectral units require a PhysicalScale.")
      if axis == "omega" and unit == "rad/s":
        result = scale.angular_frequency(values)
      elif axis == "frequency" and unit == "Hz":
        result = 2 * np.pi * scale.frequency(values)
      elif axis == "energy" and unit == "eV":
        result = scale.angular_frequency_from_ev(values)
      elif axis == "wavelength" and unit in ("m", "um", "nm"):
        metres = values * {"m": 1.0, "um": 1e-6, "nm": 1e-9}[unit]
        result = scale.angular_frequency(2 * np.pi * SPEED_OF_LIGHT_M_PER_S / metres)
      else:
        raise ValueError(f"unsupported spectral axis/unit: {axis!r}/{unit!r}.")
  if not np.all(np.isfinite(result)) or np.any(result <= 0):
    raise ValueError("converted angular frequencies must be finite and positive.")
  return result


@dataclass(frozen=True)
class PermittivityTable:
  """Copied, read-only samples sorted by positive normalized omega.

  Requires at least two unique frequencies and non-negative imaginary
  epsilon. No clipping, extrapolation or interpolation is performed.
  """

  angular_frequencies: np.ndarray
  permittivity: np.ndarray

  def __post_init__(self):
    omega = _real_vector("angular_frequencies", self.angular_frequencies)
    epsilon = np.asarray(self.permittivity)
    if not np.issubdtype(epsilon.dtype, np.number):
      raise TypeError("permittivity must contain numbers.")
    epsilon = np.array(epsilon, dtype=np.complex128, copy=True)
    if epsilon.shape != omega.shape or not np.all(np.isfinite(epsilon)):
      raise ValueError("permittivity must be finite and match the frequency shape.")
    if omega.size < 2 or np.any(omega <= 0):
      raise ValueError("at least two positive angular frequencies are required.")
    order = np.argsort(omega)
    omega, epsilon = omega[order], epsilon[order]
    if np.any(np.diff(omega) <= 0):
      raise ValueError("angular frequencies must be unique.")
    if np.any(epsilon.imag < 0):
      raise ValueError("passive data require non-negative imaginary permittivity.")
    object.__setattr__(self, "angular_frequencies", _readonly(omega))
    object.__setattr__(self, "permittivity", _readonly(epsilon))

  @classmethod
  def from_samples(cls, coordinates, first, second, *, representation="nk",
                   axis="omega", unit="normalized", scale=None,
                   imaginary_sign=1):
    """Convert n/k or epsilon-real/imaginary columns to normalized units.

    representation='nk' uses epsilon=(n+i*k)**2 with relative mu=1 and
    n,k >= 0. For 'epsilon', imaginary_sign=-1 converts the opposite Fourier
    convention. Supported axis/unit pairs: omega/normalized or rad/s,
    frequency/normalized or Hz, wavelength/m or um or nm, energy/eV.
    Physical units require an explicit PhysicalScale.
    """
    omega = _normalized_omega(coordinates, axis, unit, scale)
    first = _real_vector("first column", first)
    second = _real_vector("second column", second)
    if first.shape != omega.shape or second.shape != omega.shape:
      raise ValueError("all sample columns must have the same shape.")
    if (isinstance(imaginary_sign, (bool, np.bool_))
        or imaginary_sign not in (-1, 1)):
      raise ValueError("imaginary_sign must be +1 or -1.")
    with np.errstate(over="ignore", invalid="ignore"):
      if representation == "nk":
        if imaginary_sign != 1:
          raise ValueError("imaginary_sign applies only to epsilon tables.")
        if np.any(first < 0) or np.any(second < 0):
          raise ValueError("passive nk data require non-negative n and k.")
        epsilon = (first + 1j * second)**2
      elif representation == "epsilon":
        epsilon = first + 1j * imaginary_sign * second
      else:
        raise ValueError("representation must be 'nk' or 'epsilon'.")
    return cls(omega, epsilon)


def read_material_table(path, *, representation="nk", axis="omega",
                        unit="normalized", scale=None, columns=(0, 1, 2),
                        delimiter=None, skiprows=0, imaginary_sign=1):
  """Read coordinate, n/epsilon-real, k/epsilon-imag columns from text.

  Units/headers are explicit. delimiter=',' for CSV, '\\t' for TSV, or None
  for whitespace. # comments are supported. skiprows counts physical lines.
  """
  columns = tuple(columns)
  if (len(columns) != 3
      or any(isinstance(c, (bool, np.bool_))
             or not isinstance(c, (int, np.integer)) or c < 0 for c in columns)
      or len(set(columns)) != 3):
    raise ValueError("columns must contain three distinct non-negative integers.")
  if (isinstance(skiprows, (bool, np.bool_))
      or not isinstance(skiprows, (int, np.integer)) or skiprows < 0):
    raise ValueError("skiprows must be a non-negative integer.")
  samples = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows,
                       usecols=columns, ndmin=2, encoding="utf-8-sig")
  return PermittivityTable.from_samples(
      samples[:, 0], samples[:, 1], samples[:, 2],
      representation=representation, axis=axis, unit=unit, scale=scale,
      imaginary_sign=imaginary_sign)


@dataclass(frozen=True)
class PassiveFitResult:
  """Material and unweighted sample errors for a fixed-pole passive fit.

  pole_strengths follows candidate order, including zeros: delta_eps for
  Debye/Lorentz, plasma_frequency**2 for Drude. rms_error is the RMS complex
  residual. Relative errors divide by max(abs(target), relative_floor).
  """

  material: Material
  table: PermittivityTable
  candidate_poles: tuple
  pole_strengths: np.ndarray
  fitted_permittivity: np.ndarray
  absolute_error: np.ndarray
  relative_error: np.ndarray
  rms_error: float
  maximum_absolute_error: float
  maximum_relative_error: float


def fit_passive_material(table, poles, *, eps_inf=None, eps_inf_min=1e-6,
                         weights=None, relative_floor=1e-12):
  """Fit non-negative strengths for supplied Debye/Lorentz/Drude shapes.

  Candidate strengths are ignored; times, resonances and damping stay fixed.
  eps_inf fixes the positive background; None fits it above eps_inf_min.
  Empty poles fits a Lossless background; zero-strength poles are omitted.
  Positive weights multiply each complex residual, following sorted table
  order. Real/imaginary parts are fitted together. Poor fits retain their
  errors; validate ADE discretization and FDTD stability separately.
  """
  if not isinstance(table, PermittivityTable):
    raise TypeError("table must be a PermittivityTable.")
  relative_floor = _positive_finite("relative_floor", relative_floor)
  eps_inf_min = _positive_finite("eps_inf_min", eps_inf_min)
  if eps_inf is not None:
    eps_inf = _positive_finite("eps_inf", eps_inf)
  poles = tuple(poles)
  unit_poles = []
  for index, pole in enumerate(poles):
    if isinstance(pole, DebyePole):
      unit_poles.append(DebyePole(1.0, pole.tau))
    elif isinstance(pole, LorentzPole):
      unit_poles.append(LorentzPole(1.0, pole.omega_0, pole.damping))
    elif isinstance(pole, DrudePole):
      unit_poles.append(DrudePole(1.0, pole.collision_frequency))
    else:
      raise TypeError(f"poles[{index}] must be a DebyePole, LorentzPole or DrudePole.")
  omega = table.angular_frequencies
  if weights is None:
    weights = np.ones(omega.shape)
  else:
    weights = _real_vector("weights", weights)
    if weights.shape != omega.shape or np.any(weights <= 0):
      raise ValueError("weights must be positive and match the table shape.")
  # Global scaling leaves the optimum unchanged, avoiding large weights.
  weights = weights / np.max(weights)
  background = eps_inf_min if eps_inf is None else eps_inf
  basis = [] if eps_inf is not None else [np.ones(omega.shape, dtype=complex)]
  with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    basis.extend(pole.susceptibility(omega) for pole in unit_poles)
  if basis:
    matrix = np.column_stack(basis)
    if not np.all(np.isfinite(matrix)):
      raise ValueError("candidate pole response is non-finite in the data band.")
    matrix = np.concatenate((matrix.real * weights[:, None],
                             matrix.imag * weights[:, None]), axis=0)
    target = table.permittivity - background
    target = np.concatenate((target.real * weights, target.imag * weights))
    # Column scaling improves conditioning for differently sized responses.
    norms = np.max(np.abs(matrix), axis=0)
    if np.any(norms == 0):
      raise ValueError("candidate pole response vanishes in the data band.")
    try:
      from scipy.optimize import nnls
    except ImportError as error:
      raise ImportError("Material fitting requires SciPy; install fdtdz[fit].") from error
    coefficients, _ = nnls(matrix / norms, target)
    coefficients = coefficients / norms
  else:
    coefficients = np.empty(0)
  if eps_inf is None:
    background += coefficients[0]
    strengths = coefficients[1:]
  else:
    strengths = coefficients
  fitted_poles = []
  for pole, strength in zip(unit_poles, strengths):
    if strength == 0:
      continue
    if isinstance(pole, DebyePole):
      fitted_poles.append(DebyePole(strength, pole.tau))
    elif isinstance(pole, LorentzPole):
      fitted_poles.append(LorentzPole(strength, pole.omega_0, pole.damping))
    else:
      fitted_poles.append(DrudePole(np.sqrt(strength), pole.collision_frequency))
  material = (Multipole(background, tuple(fitted_poles)) if fitted_poles
              else Lossless(background))
  fitted = material.relative_permittivity(omega)
  absolute = np.abs(fitted - table.permittivity)
  relative = absolute / np.maximum(np.abs(table.permittivity), relative_floor)
  return PassiveFitResult(
      material, table, poles, _readonly(strengths), _readonly(fitted),
      _readonly(absolute), _readonly(relative),
      float(np.linalg.norm(absolute) / np.sqrt(absolute.size)),
      float(np.max(absolute)), float(np.max(relative)))


__all__ = ["PermittivityTable", "PassiveFitResult", "read_material_table",
           "fit_passive_material"]
