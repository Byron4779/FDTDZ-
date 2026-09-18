"""Sources and geometry helpers for a patterned x-y metasurface unit cell."""

from dataclasses import dataclass
import math

import numpy as np


def _shape_3d(shape):
  if len(shape) != 3 or any(
      isinstance(value, (bool, np.bool_)) or
      not isinstance(value, (int, np.integer)) or value < 2
      for value in shape):
    raise ValueError(f"shape must contain integer (nx, ny, nz) >= 2; got {shape}.")
  return tuple(int(value) for value in shape)


def _index(name, value, size):
  if isinstance(value, (bool, np.bool_)) or not isinstance(
      value, (int, np.integer)):
    raise TypeError(f"{name} must be an integer.")
  value = int(value)
  if value < 0 or value >= size:
    raise ValueError(f"{name} must be in [0, {size - 1}], got {value}.")
  return value


def _transverse_profile(shape, profile, dtype):
  nx, ny, _ = shape
  if profile is None:
    return np.ones((nx, ny), dtype=dtype)
  profile = np.asarray(profile, dtype=dtype)
  if profile.shape != (nx, ny) or not np.all(np.isfinite(profile)):
    raise ValueError(
        f"transverse_profile must be finite with shape {(nx, ny)}.")
  return profile


def _polarization_xy(polarization):
  if len(polarization) != 2:
    raise ValueError("polarization_xy must contain (Ex, Ey).")
  polarization = np.asarray(polarization, dtype=float)
  if not np.all(np.isfinite(polarization)):
    raise ValueError("polarization_xy must contain finite values.")
  magnitude = float(np.linalg.norm(polarization))
  if magnitude == 0.0:
    raise ValueError("polarization_xy cannot be the zero vector.")
  return polarization / magnitude


@dataclass(frozen=True)
class HuygensPlaneSource3D:
  """Compact paired current planes, including optional complex Bloch profiles."""

  electric_current: np.ndarray
  magnetic_current: np.ndarray
  electric_z_index: int
  magnetic_z_index: int
  polarization_xy: tuple
  direction: int
  electric_waveform: np.ndarray
  magnetic_waveform: np.ndarray
  transverse_profile: np.ndarray
  shape: tuple
  bloch_wavevector: tuple = (0.0, 0.0)
  reference_frequency: float = None


def huygens_plane_source(waveform, shape, z_index, polarization_xy=(1.0, 0.0),
                         direction=1, wave_impedance=1.0,
                         transverse_profile=None, dtype=np.float64,
                         expand=False, bloch_wavevector=(0.0, 0.0),
                         reference_frequency=None, spacing=(1.0, 1.0, 1.0),
                         background_eps_r=1.0):
  """Build an x-y Huygens plane with optional fixed transverse wavevector.

  For propagation direction ``s = +/-1`` and transverse electric unit vector
  ``p = (px, py)``, the magnetic-current orientation is
  ``s * z_hat cross p = s * (-py, px)``.  The magnetic plane is staggered half
  a z cell behind the electric plane for positive-z propagation.
  Nonzero bloch_wavevector uses complex component-wise Yee phases and
  TE/TM admittances matched at reference_frequency (ordinary cycles/time).
  It is a narrowband directional source, not a fixed-angle broadband source.
  """
  shape = _shape_3d(shape)
  from .bloch_3d import validate_bloch_wavevector
  bloch_wavevector = validate_bloch_wavevector(bloch_wavevector)
  z_index = _index("z_index", z_index, shape[2])
  if direction not in (-1, 1):
    raise ValueError("direction must be +1 or -1.")
  magnetic_z_index = z_index - 1 if direction == 1 else z_index
  if magnetic_z_index < 0:
    raise ValueError("+z propagation requires z_index >= 1.")
  wave_impedance = float(wave_impedance)
  if not math.isfinite(wave_impedance) or wave_impedance <= 0.0:
    raise ValueError("wave_impedance must be finite and positive.")
  dtype = np.dtype(dtype)
  if not np.issubdtype(dtype, np.floating):
    raise TypeError("dtype must be a floating dtype.")
  waveform = np.asarray(waveform, dtype=dtype)
  if waveform.ndim != 1 or not np.all(np.isfinite(waveform)):
    raise ValueError("waveform must be a finite one-dimensional array.")
  polarization = _polarization_xy(polarization_xy)
  profile = _transverse_profile(shape, transverse_profile, dtype)
  half_step = np.empty_like(waveform)
  if waveform.size:
    half_step[0] = 0.5 * waveform[0]
    half_step[1:] = 0.5 * (waveform[1:] + waveform[:-1])

  magnetic_scale = direction * wave_impedance * half_step
  if any(bloch_wavevector):
    # A fixed k_parallel broadband source has its specified angle only at
    # reference_frequency. These sheet-current impedances are narrowband.
    spacing = np.asarray(spacing, dtype=float)
    if (spacing.shape != (3,) or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)):
      raise ValueError("spacing must contain three finite positive values.")
    if (reference_frequency is None or not np.isfinite(reference_frequency)
        or reference_frequency <= 0):
      raise ValueError("oblique incidence requires a positive reference_frequency.")
    if not np.isfinite(background_eps_r) or background_eps_r <= 0:
      raise ValueError("background_eps_r must be finite and positive.")
    if not np.isclose(wave_impedance, 1 / np.sqrt(background_eps_r)):
      raise ValueError("oblique wave_impedance must match the lossless background.")
    kt = np.asarray(bloch_wavevector)
    if np.any(np.abs(kt * spacing[:2]) >= np.pi):
      raise ValueError("Bloch wavevector exceeds the spatial Nyquist band.")
    k = 2 * np.pi * reference_frequency * np.sqrt(background_eps_r)
    if np.dot(kt, kt) >= k*k:
      raise ValueError("reference_frequency must support a propagating incident order.")
    kz = np.sqrt(k*k - np.dot(kt, kt))
    admittance_p = kz/k * polarization + kt * np.dot(kt, polarization)/(k*kz)
    dtype = np.result_type(dtype, np.complex64)
    x = np.arange(shape[0])[:, None] * spacing[0]
    y = np.arange(shape[1])[None, :] * spacing[1]
    phase_ex = np.exp(1j * (kt[0]*(x + .5*spacing[0]) + kt[1]*y))
    phase_ey = np.exp(1j * (kt[0]*x + kt[1]*(y + .5*spacing[1])))
    # Mx sits at Ey's x/y position; My at Ex's. Their z plane sits
    # half a cell behind J for either propagation direction.
    magnetic_phase = np.exp(-.5j * kz * spacing[2])
    profile = np.asarray((admittance_p[0]*phase_ex*profile,
                          admittance_p[1]*phase_ey*profile,
                          -polarization[1]*phase_ey*profile*magnetic_phase,
                          polarization[0]*phase_ex*profile*magnetic_phase), dtype=dtype)
  if not isinstance(expand, (bool, np.bool_)):
    raise TypeError("expand must be boolean.")
  electric = magnetic = None
  if expand:
    vector_shape = (waveform.size, 3) + shape
    electric = np.zeros(vector_shape, dtype=dtype)
    magnetic = np.zeros(vector_shape, dtype=dtype)
    from .bloch_3d import plane_source_values
    values = plane_source_values(waveform[:, None, None],
                                 magnetic_scale[:, None, None], profile, polarization)
    electric[:, 0, :, :, z_index], electric[:, 1, :, :, z_index] = values[:2]
    magnetic[:, 0, :, :, magnetic_z_index], magnetic[:, 1, :, :, magnetic_z_index] = values[2:]
  return HuygensPlaneSource3D(
      electric_current=electric, magnetic_current=magnetic,
      electric_z_index=z_index, magnetic_z_index=magnetic_z_index,
      polarization_xy=tuple(float(value) for value in polarization),
      direction=int(direction), electric_waveform=waveform.copy(),
      magnetic_waveform=np.asarray(magnetic_scale, dtype=dtype),
      transverse_profile=np.asarray(profile, dtype=dtype), shape=shape,
      bloch_wavevector=bloch_wavevector, reference_frequency=reference_frequency)


__all__ = ["HuygensPlaneSource3D", "huygens_plane_source"]
