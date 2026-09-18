"""Vector diffraction-order power analysis for periodic x-y unit cells."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DiffractionSpectrum3D:
  """Complex transverse fields and power in every periodic order (m, n)."""

  frequencies: np.ndarray
  orders_x: np.ndarray
  orders_y: np.ndarray
  transverse_wavevectors_x: np.ndarray
  transverse_wavevectors_y: np.ndarray
  longitudinal_wavevectors: np.ndarray
  propagating: np.ndarray
  reference_incident_field: np.ndarray
  reference_transmitted_field: np.ndarray
  reflected_tangential_field: np.ndarray
  transmitted_tangential_field: np.ndarray
  reflection_orders: np.ndarray
  transmission_orders: np.ndarray
  reflection: np.ndarray
  transmission: np.ndarray
  absorption: np.ndarray
  valid: np.ndarray
  bloch_wavevector: tuple = (0.0, 0.0)


def _probe_history(name, value):
  array = np.asarray(value)
  if array.ndim != 4 or array.shape[1] != 2:
    raise ValueError(
        f"{name} must have shape (time, 2, nx, ny), got {array.shape}.")
  if not np.issubdtype(array.dtype, np.number):
    raise TypeError(f"{name} must contain numeric time-domain samples.")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values.")
  return np.asarray(array, dtype=np.complex128 if np.iscomplexobj(array) else float)


def diffraction_spectrum_3d(
    reference_reflection, device_reflection, reference_transmission,
    device_transmission, dt, dx, dy, background_eps_r=1.0,
    incident_floor=1e-8, window=None, bloch_wavevector=(0.0, 0.0)):
  """Resolve transverse probe planes into propagating (m, n) orders.

  Each probe history contains Ex and Ey on a homogeneous constant-z plane.
  Nonzero bloch_wavevector shifts every reciprocal-lattice order. Complex
  Bloch histories use exp(+i*omega*t) and component-wise Yee de-staggering.
  Bins where the incident order is evanescent are invalid. The reflected
  field is isolated by subtracting the
  empty-cell reflection-side recording; transmission is normalized against
  the empty-cell transmission-side zero order.
  """
  names = (
      "reference_reflection", "device_reflection",
      "reference_transmission", "device_transmission")
  from .bloch_3d import validate_bloch_wavevector
  bloch_wavevector = validate_bloch_wavevector(bloch_wavevector)
  planes = tuple(_probe_history(name, value) for name, value in zip(
      names, (reference_reflection, device_reflection,
              reference_transmission, device_transmission)))
  if len({plane.shape for plane in planes}) != 1:
    raise ValueError("all probe histories must have the same shape.")
  num_steps, _, nx, ny = planes[0].shape
  if num_steps < 2 or nx < 2 or ny < 2:
    raise ValueError("probe histories need at least two time, x and y samples.")
  dt, dx, dy, background_eps_r, incident_floor = (
      float(value) for value in
      (dt, dx, dy, background_eps_r, incident_floor))
  if not all(math.isfinite(value) and value > 0.0 for value in
             (dt, dx, dy, background_eps_r, incident_floor)):
    raise ValueError(
        "dt, dx, dy, background_eps_r and incident_floor must be positive.")
  if incident_floor >= 1.0:
    raise ValueError("incident_floor must be less than one.")
  if window is None:
    weights = np.ones(num_steps)
  elif isinstance(window, str):
    if window != "hann":
      raise ValueError("window string must be 'hann'.")
    weights = np.hanning(num_steps)
  else:
    weights = np.asarray(window, dtype=float)
    if weights.shape != (num_steps,) or not np.all(np.isfinite(weights)):
      raise ValueError("window must be None, 'hann', or a matching finite array.")

  complex_samples = any(np.iscomplexobj(plane) for plane in planes)
  if any(bloch_wavevector) and not complex_samples:
    raise ValueError("Bloch probe histories must contain complex quadratures.")
  temporal = [
      (np.fft.ifft(plane * weights[:, None, None, None], axis=0) * num_steps)[:num_steps//2+1]
      if complex_samples else
      np.fft.rfft(plane * weights[:, None, None, None], axis=0)
      for plane in planes]
  gx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  gy = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
  if any(bloch_wavevector):
    x, y = np.arange(nx)[:, None]*dx, np.arange(ny)[None, :]*dy
    carrier = np.asarray((
        np.exp(-1j*(bloch_wavevector[0]*(x+.5*dx) + bloch_wavevector[1]*y)),
        np.exp(-1j*(bloch_wavevector[0]*x + bloch_wavevector[1]*(y+.5*dy)))))
    temporal = [value * carrier[None, ...] for value in temporal]
  order_fields = [
      np.fft.fft2(value, axes=(2, 3)) / (nx * ny) for value in temporal
  ]
  if any(bloch_wavevector):
    # Collocate Ex and Ey from their component-wise Yee sampling positions.
    for field in order_fields:
      field[:, 0] *= np.exp(-.5j*gx*dx)[None, :, None]
      field[:, 1] *= np.exp(-.5j*gy*dy)[None, None, :]
  reference_incident = order_fields[0][:, :, 0, 0]
  reference_transmitted = order_fields[2][:, :, 0, 0]
  reflected = order_fields[1] - order_fields[0]
  transmitted = order_fields[3]
  incident_norm = np.sqrt(np.sum(np.abs(reference_incident)**2, axis=1))
  transmitted_reference_norm = np.sqrt(
      np.sum(np.abs(reference_transmitted)**2, axis=1))
  scale = max(float(np.max(incident_norm)),
              float(np.max(transmitted_reference_norm)))
  frequencies = np.fft.rfftfreq(num_steps, dt)
  valid = (
      (frequencies > 0.0) &
      (incident_norm > incident_floor * scale) &
      (transmitted_reference_norm > incident_floor * scale))

  orders_x = np.rint(np.fft.fftfreq(nx) * nx).astype(int)
  orders_y = np.rint(np.fft.fftfreq(ny) * ny).astype(int)
  kx = gx + bloch_wavevector[0]
  ky = gy + bloch_wavevector[1]
  medium_wavenumber = (
      2.0 * np.pi * frequencies * math.sqrt(background_eps_r))
  kz_squared = (
      medium_wavenumber[:, None, None]**2 - kx[None, :, None]**2 -
      ky[None, None, :]**2)
  propagating = (kz_squared > 0.0) & valid[:, None, None]
  kz = np.sqrt(np.maximum(kz_squared, 0.0))
  valid &= kz_squared[:, 0, 0] > 0.0
  propagating &= valid[:, None, None]

  def reference_power(field):
    if not any(bloch_wavevector):
      return np.sum(np.abs(field)**2, axis=1)
    longitudinal = kz[:, 0, 0]
    power = np.zeros(num_steps//2+1)
    dot = kx[0]*field[:, 0] + ky[0]*field[:, 1]
    np.divide(longitudinal * np.sum(np.abs(field)**2, axis=1),
              medium_wavenumber, out=power, where=valid)
    correction = np.zeros_like(power)
    np.divide(np.abs(dot)**2, medium_wavenumber*longitudinal,
              out=correction, where=valid)
    return power + correction

  def order_power(field, reference_flux):
    transverse_squared = np.sum(np.abs(field)**2, axis=1)
    transverse_dot = (
        kx[None, :, None] * field[:, 0] +
        ky[None, None, :] * field[:, 1])
    numerator = np.zeros_like(transverse_squared, dtype=float)
    np.divide(
        kz * transverse_squared,
        medium_wavenumber[:, None, None], out=numerator,
        where=propagating)
    longitudinal = np.zeros_like(numerator)
    np.divide(
        np.abs(transverse_dot)**2,
        medium_wavenumber[:, None, None] * kz, out=longitudinal,
        where=propagating)
    numerator += longitudinal
    power = np.zeros_like(numerator)
    np.divide(
        numerator, reference_flux[:, None, None], out=power,
        where=propagating)
    return power

  reflection_orders = order_power(reflected, reference_power(reference_incident))
  transmission_orders = order_power(
      transmitted, reference_power(reference_transmitted))
  reflection = np.sum(reflection_orders, axis=(1, 2))
  transmission = np.sum(transmission_orders, axis=(1, 2))
  absorption = 1.0 - reflection - transmission
  reflection[~valid] = np.nan
  transmission[~valid] = np.nan
  absorption[~valid] = np.nan
  return DiffractionSpectrum3D(
      frequencies=frequencies, orders_x=orders_x, orders_y=orders_y,
      transverse_wavevectors_x=kx, transverse_wavevectors_y=ky,
      longitudinal_wavevectors=kz, propagating=propagating,
      reference_incident_field=reference_incident,
      reference_transmitted_field=reference_transmitted,
      reflected_tangential_field=reflected,
      transmitted_tangential_field=transmitted,
      reflection_orders=reflection_orders,
      transmission_orders=transmission_orders,
      reflection=reflection, transmission=transmission,
      absorption=absorption, valid=valid, bloch_wavevector=bloch_wavevector)


__all__ = ["DiffractionSpectrum3D", "diffraction_spectrum_3d"]
