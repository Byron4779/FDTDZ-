"""Geometry, excitation, and probe helpers for 2D TMz metasurfaces.

Array axes follow :mod:`fdtdz_jax.yee_ade_2d`: ``(x, y)``.  The x direction
is the lateral, periodic direction of a unit cell and y is the propagation
direction.  These helpers deliberately contain no solver-specific state so
they can also be reused by a future JAX/CUDA implementation.
"""

from dataclasses import dataclass
import math

import numpy as np


def _shape_2d(shape):
  if len(shape) != 2:
    raise ValueError(f"shape must be (nx, ny), got {shape}.")
  result = tuple(int(value) for value in shape)
  if any(value < 2 for value in result):
    raise ValueError(f"both shape dimensions must be at least two, got {shape}.")
  return result


def _index(name, value, size, allow_endpoint=False):
  if isinstance(value, (bool, np.bool_)) or not isinstance(
      value, (int, np.integer)):
    raise TypeError(f"{name} must be an integer.")
  value = int(value)
  upper = size if allow_endpoint else size - 1
  if value < 0 or value > upper:
    raise ValueError(f"{name} must be in [0, {upper}], got {value}.")
  return value


def rectangle_mask(shape, x_start, x_stop, y_start, y_stop):
  """Return a half-open rectangular mask ``[x_start:x_stop, y_start:y_stop]``."""
  nx, ny = _shape_2d(shape)
  x_start = _index("x_start", x_start, nx, allow_endpoint=True)
  x_stop = _index("x_stop", x_stop, nx, allow_endpoint=True)
  y_start = _index("y_start", y_start, ny, allow_endpoint=True)
  y_stop = _index("y_stop", y_stop, ny, allow_endpoint=True)
  if x_stop <= x_start or y_stop <= y_start:
    raise ValueError("rectangle stops must be greater than their starts.")
  mask = np.zeros((nx, ny), dtype=bool)
  mask[x_start:x_stop, y_start:y_stop] = True
  return mask


def circle_mask(shape, center, radius, dx=1.0, dy=1.0, periodic_x=True):
  """Return a physical-distance circular mask sampled at Ez nodes.

  When ``periodic_x`` is true, distance in x uses the minimum-image
  convention, so a feature crossing a unit-cell edge wraps continuously.
  """
  nx, ny = _shape_2d(shape)
  if len(center) != 2:
    raise ValueError("center must contain (x, y) coordinates in index units.")
  center_x, center_y = (float(value) for value in center)
  radius, dx, dy = (float(value) for value in (radius, dx, dy))
  if not all(math.isfinite(value) for value in
             (center_x, center_y, radius, dx, dy)):
    raise ValueError("center, radius, dx, and dy must be finite.")
  if radius <= 0.0 or dx <= 0.0 or dy <= 0.0:
    raise ValueError("radius, dx, and dy must be greater than zero.")
  x_distance = np.abs(np.arange(nx, dtype=float) - center_x)
  if periodic_x:
    x_distance = np.minimum(x_distance, nx - x_distance)
  y_distance = np.abs(np.arange(ny, dtype=float) - center_y)
  return ((x_distance[:, None] * dx)**2 +
          (y_distance[None, :] * dy)**2 <= radius**2)


def paint_material(material_ids, mask, material_id, copy=True):
  """Assign ``material_id`` wherever ``mask`` is true."""
  material_ids = np.asarray(material_ids)
  mask = np.asarray(mask)
  if material_ids.ndim != 2:
    raise ValueError("material_ids must be two-dimensional.")
  if mask.shape != material_ids.shape or mask.dtype != np.bool_:
    raise ValueError("mask must be boolean and match material_ids.shape.")
  if isinstance(material_id, (bool, np.bool_)) or not isinstance(
      material_id, (int, np.integer)) or material_id < 0 or material_id > 255:
    raise ValueError("material_id must be an integer in [0, 255].")
  result = np.array(material_ids, copy=copy)
  result[mask] = int(material_id)
  return result


def gaussian_sine_pulse(num_steps, dt, center_time, width, frequency,
                        amplitude=1.0, phase=0.0, dtype=np.float64):
  """Create a Gaussian-envelope sinusoidal current waveform."""
  if isinstance(num_steps, (bool, np.bool_)) or not isinstance(
      num_steps, (int, np.integer)) or num_steps < 0:
    raise ValueError("num_steps must be a non-negative integer.")
  dt, center_time, width, frequency, amplitude, phase = (
      float(value) for value in
      (dt, center_time, width, frequency, amplitude, phase))
  if not all(math.isfinite(value) for value in
             (dt, center_time, width, frequency, amplitude, phase)):
    raise ValueError("pulse parameters must be finite.")
  if dt <= 0.0 or width <= 0.0 or frequency < 0.0:
    raise ValueError("dt and width must be positive; frequency cannot be negative.")
  time = np.arange(num_steps, dtype=float) * dt
  envelope = np.exp(-0.5 * ((time - center_time) / width)**2)
  carrier = np.sin(2.0 * np.pi * frequency * (time - center_time) + phase)
  return np.asarray(amplitude * envelope * carrier, dtype=dtype)


def line_current_source(waveform, shape, y_index, x_profile=None,
                        dtype=np.float64):
  """Expand a time waveform into a z-current line across the unit cell."""
  nx, ny = _shape_2d(shape)
  y_index = _index("y_index", y_index, ny)
  waveform = np.asarray(waveform, dtype=dtype)
  if waveform.ndim != 1 or not np.all(np.isfinite(waveform)):
    raise ValueError("waveform must be a finite one-dimensional array.")
  if x_profile is None:
    profile = np.ones(nx, dtype=dtype)
  else:
    profile = np.asarray(x_profile, dtype=dtype)
    if profile.shape != (nx,) or not np.all(np.isfinite(profile)):
      raise ValueError(f"x_profile must be finite with shape ({nx},).")
  source = np.zeros((waveform.size, nx, ny), dtype=dtype)
  source[:, :, y_index] = waveform[:, None] * profile[None, :]
  return source


def sample_line(field_history, y_index, average_x=False):
  """Extract a y-normal line from ``(time, x, y)`` recorded fields."""
  field_history = np.asarray(field_history)
  if field_history.ndim != 3:
    raise ValueError("field_history must have shape (time, x, y).")
  y_index = _index("y_index", y_index, field_history.shape[2])
  values = field_history[:, :, y_index]
  return np.mean(values, axis=1) if average_x else values.copy()


@dataclass(frozen=True)
class LineProbe2D:
  """Named electric-field probe on a line normal to propagation."""

  y_index: int
  average_x: bool = True
  name: str = "probe"

  def sample(self, field_history):
    return sample_line(field_history, self.y_index, self.average_x)


@dataclass(frozen=True)
class Sponge2D:
  """Matched electric/magnetic conductivity profiles for a debug absorber."""

  electric_conductivity: np.ndarray
  magnetic_conductivity: np.ndarray


@dataclass(frozen=True)
class ScatteringSpectrum2D:
  """Power spectra and the complex field ratios used to calculate them."""

  frequencies: np.ndarray
  reflection: np.ndarray
  transmission: np.ndarray
  absorption: np.ndarray
  reflected_field_ratio: np.ndarray
  transmitted_field_ratio: np.ndarray
  valid: np.ndarray


@dataclass(frozen=True)
class HuygensLineSource2D:
  """Paired electric and magnetic line currents for directional injection."""

  current_density_z: np.ndarray
  magnetic_current_x: np.ndarray


@dataclass(frozen=True)
class HuygensTEzLineSource2D:
  """Staggered Jx/Mz line currents for directional TEz injection."""

  current_density_x: np.ndarray
  magnetic_current_z: np.ndarray
  electric_y_index: int
  magnetic_y_index: int


def huygens_current_waveforms(waveform, direction=1, wave_impedance=1.0,
                              dtype=np.float64):
  """Return compact electric/magnetic waveforms for a Huygens line source."""
  if direction not in (-1, 1):
    raise ValueError("direction must be +1 or -1.")
  wave_impedance = float(wave_impedance)
  if not math.isfinite(wave_impedance) or wave_impedance <= 0.0:
    raise ValueError("wave_impedance must be finite and positive.")
  waveform = np.asarray(waveform, dtype=dtype)
  if waveform.ndim != 1 or not np.all(np.isfinite(waveform)):
    raise ValueError("waveform must be a finite one-dimensional array.")
  half_step = np.empty_like(waveform)
  if waveform.size:
    half_step[0] = 0.5 * waveform[0]
    half_step[1:] = 0.5 * (waveform[1:] + waveform[:-1])
  return waveform.copy(), direction * wave_impedance * half_step


def huygens_tez_current_waveforms(waveform, direction=1,
                                  wave_impedance=1.0,
                                  dtype=np.float64):
  """Return compact Jx/Mz waveforms for directional TEz propagation."""
  electric, staggered = huygens_current_waveforms(
      waveform, direction=direction, wave_impedance=wave_impedance,
      dtype=dtype)
  return electric, -staggered


def huygens_tez_line_source(waveform, shape, y_index, direction=1,
                            wave_impedance=1.0, x_profile=None,
                            dtype=np.float64):
  """Build the spatially staggered TEz Jx/Mz Huygens source.

  For +y propagation Mz occupies the magnetic node at ``y_index - 1``; for
  -y it occupies ``y_index``.  This half-cell staggering follows the TEz Yee
  layout and strongly suppresses the counter-propagating wave.
  """
  _, ny = _shape_2d(shape)
  y_index = _index("y_index", y_index, ny)
  if direction not in (-1, 1):
    raise ValueError("direction must be +1 or -1.")
  magnetic_y_index = y_index - 1 if direction == 1 else y_index
  if magnetic_y_index < 0:
    raise ValueError("+y TEz source requires y_index >= 1.")
  electric, magnetic = huygens_tez_current_waveforms(
      waveform, direction=direction, wave_impedance=wave_impedance,
      dtype=dtype)
  return HuygensTEzLineSource2D(
      current_density_x=line_current_source(
          electric, shape, y_index, x_profile=x_profile, dtype=dtype),
      magnetic_current_z=line_current_source(
          magnetic, shape, magnetic_y_index, x_profile=x_profile, dtype=dtype),
      electric_y_index=y_index,
      magnetic_y_index=magnetic_y_index)


def huygens_line_source(waveform, shape, y_index, direction=1,
                        wave_impedance=1.0, x_profile=None,
                        dtype=np.float64):
  """Build a colocated Jz/Mx pair targeting ``+y`` or ``-y`` propagation.

  This normalized soft Huygens source is intended for the homogeneous
  incidence region.  The magnetic waveform is offset by half a Yee time step
  using linear interpolation to respect the E/H time staggering.
  """
  electric_waveform, magnetic_waveform = huygens_current_waveforms(
      waveform, direction=direction, wave_impedance=wave_impedance,
      dtype=dtype)
  electric = line_current_source(
      electric_waveform, shape, y_index, x_profile=x_profile, dtype=dtype)
  magnetic = line_current_source(
      magnetic_waveform, shape, y_index,
      x_profile=x_profile, dtype=dtype)
  return HuygensLineSource2D(
      current_density_z=electric, magnetic_current_x=magnetic)


def scattering_spectrum(reference_reflection, device_reflection,
                        reference_transmission, device_transmission, dt,
                        transmission_power_factor=1.0,
                        incident_floor=1e-8, window="hann"):
  """Calculate R/T/A from matched empty-cell and device probe signals.

  Reflection is isolated by subtracting the empty-cell reflection-side signal.
  Transmission is normalized by the empty-cell transmission-side signal.  The
  four time traces must use exactly the same source, grid, probes, and timing.
  ``transmission_power_factor`` is the output/input wave-admittance ratio and
  equals one when both probe regions contain the same normalized medium.
  """
  traces = [np.asarray(value) for value in (
      reference_reflection, device_reflection,
      reference_transmission, device_transmission)]
  if any(trace.ndim != 1 for trace in traces):
    raise ValueError("all probe signals must be one-dimensional.")
  if not traces or len({trace.shape for trace in traces}) != 1:
    raise ValueError("all probe signals must have the same length.")
  if traces[0].size < 2:
    raise ValueError("probe signals must contain at least two samples.")
  if not all(np.all(np.isfinite(trace)) for trace in traces):
    raise ValueError("probe signals must be finite.")
  dt = float(dt)
  transmission_power_factor = float(transmission_power_factor)
  incident_floor = float(incident_floor)
  if not math.isfinite(dt) or dt <= 0.0:
    raise ValueError("dt must be finite and positive.")
  if (not math.isfinite(transmission_power_factor) or
      transmission_power_factor <= 0.0):
    raise ValueError("transmission_power_factor must be finite and positive.")
  if not math.isfinite(incident_floor) or not 0.0 < incident_floor < 1.0:
    raise ValueError("incident_floor must be strictly between zero and one.")

  if window is None:
    weights = np.ones(traces[0].size)
  elif window == "hann":
    weights = np.hanning(traces[0].size)
  else:
    weights = np.asarray(window, dtype=float)
    if weights.shape != traces[0].shape or not np.all(np.isfinite(weights)):
      raise ValueError("window must be None, 'hann', or a finite matching array.")

  spectra = [np.fft.rfft(trace * weights) for trace in traces]
  incident, device_reflected_side, incident_transmitted, device_transmitted = spectra
  reflected = device_reflected_side - incident
  scale = max(float(np.max(np.abs(incident))),
              float(np.max(np.abs(incident_transmitted))))
  valid = ((np.abs(incident) > incident_floor * scale) &
           (np.abs(incident_transmitted) > incident_floor * scale))
  reflected_ratio = np.full(incident.shape, np.nan + 0j)
  transmitted_ratio = np.full(incident.shape, np.nan + 0j)
  np.divide(reflected, incident, out=reflected_ratio, where=valid)
  np.divide(device_transmitted, incident_transmitted,
            out=transmitted_ratio, where=valid)
  reflection = np.abs(reflected_ratio)**2
  transmission = transmission_power_factor * np.abs(transmitted_ratio)**2
  absorption = 1.0 - reflection - transmission
  return ScatteringSpectrum2D(
      frequencies=np.fft.rfftfreq(traces[0].size, dt),
      reflection=reflection,
      transmission=transmission,
      absorption=absorption,
      reflected_field_ratio=reflected_ratio,
      transmitted_field_ratio=transmitted_ratio,
      valid=valid)


def graded_y_sponge(shape, width, maximum_conductivity, order=3,
                     dtype=np.float64):
  """Build symmetric polynomial conductivity layers at the two y edges.

  This is a useful reflection-reducing boundary for development and regression
  tests.  It is not CPML and should not be used for final quantitative spectra.
  The surrounding medium should be normalized vacuum (eps=mu=1) so equal
  electric and magnetic conductivities are impedance matched.
  """
  nx, ny = _shape_2d(shape)
  if isinstance(width, (bool, np.bool_)) or not isinstance(
      width, (int, np.integer)):
    raise TypeError("width must be an integer.")
  width = int(width)
  if width <= 0 or 2 * width >= ny:
    raise ValueError("width must be positive and leave at least one interior cell.")
  maximum_conductivity = float(maximum_conductivity)
  if (not math.isfinite(maximum_conductivity) or
      maximum_conductivity < 0.0):
    raise ValueError("maximum_conductivity must be finite and non-negative.")
  if isinstance(order, (bool, np.bool_)) or not isinstance(
      order, (int, np.integer)) or order <= 0:
    raise ValueError("order must be a positive integer.")
  y_profile = np.zeros(ny, dtype=dtype)
  edge = maximum_conductivity * (
      np.arange(width, 0, -1, dtype=float) / width)**int(order)
  y_profile[:width] = edge
  y_profile[-width:] = edge[::-1]
  conductivity = np.repeat(y_profile[None, :], nx, axis=0)
  return Sponge2D(
      electric_conductivity=conductivity.copy(),
      magnetic_conductivity=conductivity.copy())


__all__ = [
    "LineProbe2D",
    "Sponge2D",
    "ScatteringSpectrum2D",
    "HuygensLineSource2D",
    "HuygensTEzLineSource2D",
    "circle_mask",
    "gaussian_sine_pulse",
    "graded_y_sponge",
    "huygens_line_source",
    "huygens_current_waveforms",
    "huygens_tez_current_waveforms",
    "huygens_tez_line_source",
    "line_current_source",
    "paint_material",
    "rectangle_mask",
    "sample_line",
    "scattering_spectrum",
]
