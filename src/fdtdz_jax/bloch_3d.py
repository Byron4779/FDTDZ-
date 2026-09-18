"""Shared validation and compact source handling for x/y Bloch boundaries."""

import numpy as np


def validate_bloch_wavevector(value):
  array = np.asarray(value)
  if (array.shape != (2,) or not np.issubdtype(array.dtype, np.number)
      or np.iscomplexobj(array)):
    raise ValueError("bloch_wavevector must contain two real values (kx, ky).")
  if not np.all(np.isfinite(array)):
    raise ValueError("bloch_wavevector must contain finite values.")
  return tuple(float(v) for v in array)


def plane_source_values(electric_amplitude, magnetic_amplitude, profile,
                        polarization):
  """Return Jx/Jy and Mx/My planes; works with NumPy and JAX arrays.

  Legacy profiles have shape (nx,ny); Bloch profiles (4,nx,ny) contain the
  separate Jx,Jy,Mx,My amplitudes at their own Yee coordinates.
  """
  if profile.ndim == 3:
    return (electric_amplitude * profile[0], electric_amplitude * profile[1],
            magnetic_amplitude * profile[2], magnetic_amplitude * profile[3])
  px, py = polarization
  return (electric_amplitude * px * profile, electric_amplitude * py * profile,
          -magnetic_amplitude * py * profile, magnetic_amplitude * px * profile)
