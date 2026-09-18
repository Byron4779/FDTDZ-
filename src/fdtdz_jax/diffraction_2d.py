"""Diffraction-order power analysis for periodic 2D metasurfaces."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DiffractionSpectrum2D:
  """Per-order and total power spectra for a periodic unit cell."""

  frequencies: np.ndarray
  orders: np.ndarray
  transverse_wavevectors: np.ndarray
  longitudinal_wavevectors: np.ndarray
  propagating: np.ndarray
  reflected_field_ratio: np.ndarray
  transmitted_field_ratio: np.ndarray
  reflection_orders: np.ndarray
  transmission_orders: np.ndarray
  reflection: np.ndarray
  transmission: np.ndarray
  absorption: np.ndarray
  valid: np.ndarray


def diffraction_spectrum(reference_reflection, device_reflection,
                         reference_transmission, device_transmission,
                         dt, dx, background_eps_r=1.0,
                         polarization="tmz", incident_floor=1e-8,
                         window="hann"):
  """Resolve periodic probe lines into propagating diffraction orders.

  Probe arrays have shape ``(time, x)`` and must come from matched reference
  and device runs.  Incidence is assumed normal and the input/output probe
  regions must contain the same homogeneous lossless background.

  ``polarization='tmz'`` expects Ez probe lines and uses the power weight
  ``ky/k``.  ``polarization='tez'`` expects Ex probe lines and uses ``k/ky``.
  """
  lines = [np.asarray(value) for value in (
      reference_reflection, device_reflection,
      reference_transmission, device_transmission)]
  if any(line.ndim != 2 for line in lines):
    raise ValueError("all probe lines must have shape (time, x).")
  if len({line.shape for line in lines}) != 1:
    raise ValueError("all probe lines must have the same shape.")
  num_steps, nx = lines[0].shape
  if num_steps < 2 or nx < 2:
    raise ValueError("probe lines need at least two time and x samples.")
  if not all(np.all(np.isfinite(line)) for line in lines):
    raise ValueError("probe lines must contain only finite values.")
  dt, dx, background_eps_r, incident_floor = (
      float(value) for value in (dt, dx, background_eps_r, incident_floor))
  if not all(math.isfinite(value) and value > 0.0 for value in
             (dt, dx, background_eps_r, incident_floor)):
    raise ValueError("dt, dx, background_eps_r, and incident_floor must be positive.")
  if incident_floor >= 1.0:
    raise ValueError("incident_floor must be less than one.")
  polarization = str(polarization).lower()
  if polarization not in ("tmz", "tez"):
    raise ValueError("polarization must be 'tmz' or 'tez'.")
  if window is None:
    weights = np.ones(num_steps)
  elif window == "hann":
    weights = np.hanning(num_steps)
  else:
    weights = np.asarray(window, dtype=float)
    if weights.shape != (num_steps,) or not np.all(np.isfinite(weights)):
      raise ValueError("window must be None, 'hann', or a matching finite array.")

  # Temporal rFFT preserves the positive-frequency complex amplitude; the
  # normalized spatial FFT then maps it onto integer periodic orders.
  temporal = [np.fft.rfft(line * weights[:, None], axis=0) for line in lines]
  order_spectra = [np.fft.fft(value, axis=1) / nx for value in temporal]
  reference_incident = order_spectra[0][:, 0]
  reference_transmitted = order_spectra[2][:, 0]
  reflected = order_spectra[1] - order_spectra[0]
  transmitted = order_spectra[3]

  scale = max(float(np.max(np.abs(reference_incident))),
              float(np.max(np.abs(reference_transmitted))))
  frequencies = np.fft.rfftfreq(num_steps, dt)
  valid = ((frequencies > 0.0) &
           (np.abs(reference_incident) > incident_floor * scale) &
           (np.abs(reference_transmitted) > incident_floor * scale))
  reflected_ratio = np.full(reflected.shape, np.nan + 0j)
  transmitted_ratio = np.full(transmitted.shape, np.nan + 0j)
  np.divide(reflected, reference_incident[:, None], out=reflected_ratio,
            where=valid[:, None])
  np.divide(transmitted, reference_transmitted[:, None],
            out=transmitted_ratio, where=valid[:, None])

  orders = np.rint(np.fft.fftfreq(nx) * nx).astype(int)
  kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  angular_frequency = 2.0 * np.pi * frequencies
  medium_wavenumber = angular_frequency * math.sqrt(background_eps_r)
  ky_squared = medium_wavenumber[:, None]**2 - kx[None, :]**2
  propagating = (ky_squared > 0.0) & valid[:, None]
  ky = np.sqrt(np.maximum(ky_squared, 0.0))
  power_weight = np.zeros_like(ky)
  if polarization == "tmz":
    np.divide(ky, medium_wavenumber[:, None], out=power_weight,
              where=propagating)
  else:
    np.divide(medium_wavenumber[:, None], ky, out=power_weight,
              where=propagating)
  reflection_orders = np.where(
      propagating, power_weight * np.abs(reflected_ratio)**2, 0.0)
  transmission_orders = np.where(
      propagating, power_weight * np.abs(transmitted_ratio)**2, 0.0)
  reflection = np.sum(reflection_orders, axis=1)
  transmission = np.sum(transmission_orders, axis=1)
  absorption = 1.0 - reflection - transmission
  reflection[~valid] = np.nan
  transmission[~valid] = np.nan
  absorption[~valid] = np.nan
  return DiffractionSpectrum2D(
      frequencies=frequencies, orders=orders,
      transverse_wavevectors=kx, longitudinal_wavevectors=ky,
      propagating=propagating,
      reflected_field_ratio=reflected_ratio,
      transmitted_field_ratio=transmitted_ratio,
      reflection_orders=reflection_orders,
      transmission_orders=transmission_orders,
      reflection=reflection, transmission=transmission,
      absorption=absorption, valid=valid)


__all__ = ["DiffractionSpectrum2D", "diffraction_spectrum"]
