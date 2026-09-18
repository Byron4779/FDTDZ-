"""Three-dimensional periodic Yee-ADE reference solver.

This module is the executable NumPy specification for a patterned metasurface
solver.  Normalized units use ``c0 = epsilon0 = mu0 = 1`` and array axes are
ordered ``(component, x, y, z)``.  Electric components follow the original
``fdtdz`` convention: Ex, Ey and Ez are located at ``(1/2, 0, 0)``,
``(0, 1/2, 0)`` and ``(0, 0, 1/2)`` within a Yee cell.

x/y support periodic or Bloch boundaries; z supports periodic or CPML.
Bloch fields are complex (two real quadratures); ADE coefficients stay real.
"""

from dataclasses import dataclass
import math

import numpy as np

from .materials import Debye
from .materials import Drude
from .materials import Lorentz
from .materials import Lossless
from .materials import Multipole
from .materials import ade_coefficients
from .multipole_reference import pole_ade_coefficients
from .bloch_3d import plane_source_values, validate_bloch_wavevector


_SUPPORTED_MATERIALS = (Lossless, Debye, Lorentz, Drude, Multipole)
_NUM_COMPONENTS = 3


def _positive_finite(name, value):
  if isinstance(value, (bool, np.bool_)) or not isinstance(
      value, (int, float, np.integer, np.floating)):
    raise TypeError(f"{name} must be a real number.")
  value = float(value)
  if not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"{name} must be finite and greater than zero, got {value}.")
  return value


def _floating_dtype(dtype):
  dtype = np.dtype(dtype)
  if not np.issubdtype(dtype, np.floating):
    raise TypeError(f"dtype must be a floating dtype, got {dtype}.")
  return dtype


def _field_array(name, value, shape, dtype):
  if np.iscomplexobj(value) and not np.issubdtype(dtype, np.complexfloating):
    raise TypeError(f"{name}: complex quadratures require a nonzero Bloch wavevector.")
  array = np.asarray(value, dtype=dtype)
  if array.shape != shape:
    raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values.")
  return np.array(array, dtype=dtype, copy=True)


@dataclass(frozen=True)
class MaterialGrid3D:
  """ADE coefficients sampled independently at Ex, Ey and Ez nodes."""

  coefficients: np.ndarray
  material_ids: np.ndarray
  multipole_mask: np.ndarray
  multipole_eps_inf: np.ndarray
  pole_coefficients: np.ndarray
  z_cpml: object
  dt: float
  dx: float
  dy: float
  dz: float
  bloch_wavevector: tuple = (0.0, 0.0)

  @property
  def field_dtype(self):
    return (np.result_type(self.coefficients.dtype, np.complex64)
            if any(self.bloch_wavevector) else self.coefficients.dtype)

  @property
  def shape(self):
    return self.material_ids.shape[1:]

  @property
  def vector_shape(self):
    return self.material_ids.shape

  @property
  def maximum_poles(self):
    return self.pole_coefficients.shape[-2]

  @property
  def courant_x(self):
    return self.dt / self.dx

  @property
  def courant_y(self):
    return self.dt / self.dy

  @property
  def courant_z(self):
    return self.dt / self.dz


@dataclass(frozen=True)
class YeeADE3DState:
  """Full-vector fields and material histories after ``step`` updates."""

  electric: np.ndarray
  magnetic: np.ndarray
  displacement: np.ndarray
  displacement_previous: np.ndarray
  electric_previous: np.ndarray
  cpml_hx_ey_z: np.ndarray = None
  cpml_hy_ex_z: np.ndarray = None
  cpml_dx_hy_z: np.ndarray = None
  cpml_dy_hx_z: np.ndarray = None
  polarization_current: np.ndarray = None
  polarization_previous: np.ndarray = None
  step: int = 0


@dataclass(frozen=True)
class YeeADE3DResult:
  """Sparse full-field recordings and the final three-dimensional state."""

  electric: np.ndarray
  magnetic: np.ndarray
  displacement: np.ndarray
  steps: np.ndarray
  final_state: YeeADE3DState


@dataclass(frozen=True)
class YeeADE3DFieldSnapshots:
  """Sparse full-vector E/H/D recordings at selected completed steps."""

  electric: np.ndarray
  magnetic: np.ndarray
  displacement: np.ndarray
  steps: np.ndarray

  @property
  def polarization(self):
    """Normalized total polarization ``P / epsilon_0 = D_r - E``."""
    return self.displacement - self.electric


@dataclass(frozen=True)
class YeeADE3DFrequencyFields:
  """Complex full fields evaluated online at selected positive frequencies.

  The transform is ``dt * sum_t window(t) F(t) exp(+i omega t)``, matching
  this project's inverse-transform convention ``exp(-i omega t)``.  Electric
  and displacement fields use integer-step times; magnetic fields use their
  native Yee half-step times.
  """

  frequencies: np.ndarray
  electric: np.ndarray
  magnetic: np.ndarray
  displacement: np.ndarray

  @property
  def polarization(self):
    return self.displacement - self.electric


@dataclass(frozen=True)
class YeeADE3DPlaneProbeResult:
  """Tangential electric histories on selected constant-z planes."""

  electric_tangential: np.ndarray
  z_indices: np.ndarray
  steps: np.ndarray
  final_state: YeeADE3DState
  snapshots: YeeADE3DFieldSnapshots = None
  frequency_fields: YeeADE3DFrequencyFields = None


@dataclass(frozen=True)
class ZCPML3D:
  """Convolutional PML coefficients for z-directed derivatives."""

  inverse_kappa_e: np.ndarray
  b_e: np.ndarray
  c_e: np.ndarray
  inverse_kappa_h: np.ndarray
  b_h: np.ndarray
  c_h: np.ndarray
  width: int


def prepare_z_cpml(shape, width, dt, dz, target_reflection=1e-8,
                   grading_order=3, kappa_max=5.0, alpha_fraction=0.05,
                   dtype=np.float64):
  """Create normalized z-directed CPML coefficients for a 3D domain."""
  if len(shape) != 3 or any(int(value) < 2 for value in shape):
    raise ValueError(
        f"shape must be (nx, ny, nz), all at least two; got {shape}.")
  _, _, nz = (int(value) for value in shape)
  if isinstance(width, (bool, np.bool_)) or not isinstance(
      width, (int, np.integer)):
    raise TypeError("width must be an integer.")
  width = int(width)
  if width <= 0 or 2 * width >= nz:
    raise ValueError("width must be positive and leave at least one interior cell.")
  dt = _positive_finite("dt", dt)
  dz = _positive_finite("dz", dz)
  dtype = _floating_dtype(dtype)
  target_reflection = float(target_reflection)
  kappa_max = float(kappa_max)
  alpha_fraction = float(alpha_fraction)
  if not 0.0 < target_reflection < 1.0:
    raise ValueError("target_reflection must be strictly between zero and one.")
  if (isinstance(grading_order, (bool, np.bool_)) or
      not isinstance(grading_order, (int, np.integer)) or grading_order <= 0):
    raise ValueError("grading_order must be a positive integer.")
  if not math.isfinite(kappa_max) or kappa_max < 1.0:
    raise ValueError("kappa_max must be finite and at least one.")
  if not math.isfinite(alpha_fraction) or alpha_fraction < 0.0:
    raise ValueError("alpha_fraction must be finite and non-negative.")

  sigma_max = -(int(grading_order) + 1) * math.log(target_reflection) / (
      2.0 * width * dz)
  alpha_max = alpha_fraction * sigma_max

  def depth_at(offset):
    positions = np.arange(nz, dtype=float) + offset
    distance = np.minimum(positions + 0.5, nz - 0.5 - positions)
    return np.minimum(np.maximum((width - distance) / width, 0.0), 1.0)

  def coefficients(depth):
    active = depth > 0.0
    graded = depth**int(grading_order)
    sigma = sigma_max * graded
    kappa = 1.0 + (kappa_max - 1.0) * graded
    alpha = np.where(active, alpha_max * (1.0 - depth), 0.0)
    b = np.where(active, np.exp(-(sigma / kappa + alpha) * dt), 0.0)
    denominator = sigma * kappa + kappa * kappa * alpha
    c = np.zeros_like(depth)
    np.divide(sigma * (b - 1.0), denominator, out=c,
              where=denominator != 0.0)
    return (np.asarray(1.0 / kappa, dtype=dtype),
            np.asarray(b, dtype=dtype), np.asarray(c, dtype=dtype))

  inverse_kappa_e, b_e, c_e = coefficients(depth_at(0.0))
  inverse_kappa_h, b_h, c_h = coefficients(depth_at(0.5))
  return ZCPML3D(
      inverse_kappa_e=inverse_kappa_e, b_e=b_e, c_e=c_e,
      inverse_kappa_h=inverse_kappa_h, b_h=b_h, c_h=c_h, width=width)


def _component_material_map(material_ids):
  material_ids = np.asarray(material_ids)
  if material_ids.ndim == 3:
    if min(material_ids.shape, default=0) < 2:
      raise ValueError(
          "material_ids spatial dimensions must each contain at least two "
          f"nodes, got {material_ids.shape}.")
    material_ids = np.broadcast_to(
        material_ids[None, ...], (_NUM_COMPONENTS,) + material_ids.shape)
  elif material_ids.ndim == 4 and material_ids.shape[0] == _NUM_COMPONENTS:
    if min(material_ids.shape[1:], default=0) < 2:
      raise ValueError(
          "material_ids spatial dimensions must each contain at least two "
          f"nodes, got {material_ids.shape}.")
  else:
    raise ValueError(
        "material_ids must have shape (nx, ny, nz) or (3, nx, ny, nz), "
        f"got {material_ids.shape}.")
  if not np.issubdtype(material_ids.dtype, np.integer):
    raise TypeError("material_ids must contain integers.")
  return np.array(material_ids, copy=True)


def prepare_material_grid_3d(materials, material_ids, dt, dx, dy, dz,
                             dtype=np.float64, z_cpml=None,
                             bloch_wavevector=(0.0, 0.0)):
  """Validate a 3D material map and build per-electric-node ADE data.

  A scalar cell map with shape ``(nx, ny, nz)`` is copied to all electric
  components.  Passing ``(3, nx, ny, nz)`` permits geometry to be sampled at
  the staggered Ex, Ey and Ez positions independently.
  """
  dt = _positive_finite("dt", dt)
  dx = _positive_finite("dx", dx)
  dy = _positive_finite("dy", dy)
  dz = _positive_finite("dz", dz)
  dtype = _floating_dtype(dtype)
  bloch_wavevector = validate_bloch_wavevector(bloch_wavevector)
  if any(abs(k * spacing) >= np.pi for k, spacing in
         zip(bloch_wavevector, (dx, dy))):
    raise ValueError("Bloch wavevector must lie inside the spatial Nyquist band.")
  materials = tuple(materials)
  if not materials:
    raise ValueError("materials must contain at least one material.")
  if len(materials) > 256:
    raise ValueError(
        "materials may contain at most 256 entries because material_ids are "
        "stored as uint8.")
  for index, material in enumerate(materials):
    if not isinstance(material, _SUPPORTED_MATERIALS):
      raise TypeError(
          f"materials[{index}] must be Lossless, Debye, Lorentz, Drude, or "
          f"Multipole, got {type(material).__name__}.")

  material_ids = _component_material_map(material_ids)
  if np.any(material_ids < 0) or np.any(material_ids >= len(materials)):
    raise ValueError(f"material_ids must be in [0, {len(materials) - 1}].")
  if z_cpml is not None:
    if not isinstance(z_cpml, ZCPML3D):
      raise TypeError("z_cpml must be a ZCPML3D instance.")
    if z_cpml.inverse_kappa_e.shape != (material_ids.shape[-1],):
      raise ValueError(
          "z_cpml coefficient length must match the material map z extent.")
    lossless_material = np.asarray([
        isinstance(material, Lossless) for material in materials], dtype=bool)
    dispersive_nodes = ~lossless_material[material_ids]
    cpml_nodes = np.zeros(material_ids.shape, dtype=bool)
    cpml_nodes[..., :z_cpml.width] = True
    cpml_nodes[..., -z_cpml.width:] = True
    if np.any(dispersive_nodes & cpml_nodes):
      raise ValueError(
          "CPML is restricted to Lossless boundary nodes; keep dispersive "
          "geometry outside the z CPML layers.")

  material_coefficients = np.stack([
      (ade_coefficients(material, dt).as_array(dtype)
       if not isinstance(material, Multipole) else
       ade_coefficients(Lossless(material.eps_inf), dt).as_array(dtype))
      for material in materials
  ])
  coefficients = material_coefficients[material_ids]
  high_frequency_eps = np.asarray([
      material.eps_r if isinstance(material, Lossless) else material.eps_inf
      for material in materials
  ], dtype=float)
  minimum_eps = float(np.min(high_frequency_eps[material_ids]))
  inverse_spacing = math.sqrt(
      1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2)
  maximum_dt = math.sqrt(minimum_eps) / inverse_spacing
  if dt > maximum_dt * (1.0 + 1e-12):
    raise ValueError(
        "The three-dimensional CFL condition is violated: dt must be <= "
        f"{maximum_dt} using min(eps_inf)={minimum_eps}, got {dt}.")

  maximum_poles = max(
      (len(material.poles) if isinstance(material, Multipole) else 0)
      for material in materials)
  material_pole_coefficients = np.zeros(
      (len(materials), maximum_poles, 5), dtype=dtype)
  material_multipole_eps_inf = np.ones(len(materials), dtype=dtype)
  material_is_multipole = np.zeros(len(materials), dtype=bool)
  for material_index, material in enumerate(materials):
    if isinstance(material, Multipole):
      material_is_multipole[material_index] = True
      material_multipole_eps_inf[material_index] = material.eps_inf
      for pole_index, pole in enumerate(material.poles):
        coefficient = pole_ade_coefficients(pole, dt)
        material_pole_coefficients[material_index, pole_index] = (
            coefficient.q_next, coefficient.q_current,
            coefficient.q_previous, coefficient.r_current,
            coefficient.r_previous)

  return MaterialGrid3D(
      coefficients=np.asarray(coefficients, dtype=dtype),
      material_ids=np.asarray(material_ids, dtype=np.uint8),
      multipole_mask=material_is_multipole[material_ids],
      multipole_eps_inf=material_multipole_eps_inf[material_ids],
      pole_coefficients=material_pole_coefficients[material_ids],
      z_cpml=z_cpml, dt=dt, dx=dx, dy=dy, dz=dz,
      bloch_wavevector=bloch_wavevector)


def initialize_yee_ade_3d(grid, dtype=None):
  """Create a zero-field, zero-polarization full-vector state."""
  if not isinstance(grid, MaterialGrid3D):
    raise TypeError("grid must be a MaterialGrid3D instance.")
  dtype = (grid.field_dtype if dtype is None else
           np.result_type(grid.field_dtype, _floating_dtype(dtype)))
  zeros = np.zeros(grid.vector_shape, dtype=dtype)
  pole_zeros = np.zeros(
      grid.vector_shape + (grid.maximum_poles,), dtype=dtype)
  return YeeADE3DState(
      electric=zeros.copy(), magnetic=zeros.copy(), displacement=zeros.copy(),
      displacement_previous=zeros.copy(), electric_previous=zeros.copy(),
      cpml_hx_ey_z=zeros[0].copy(), cpml_hy_ex_z=zeros[0].copy(),
      cpml_dx_hy_z=zeros[0].copy(), cpml_dy_hx_z=zeros[0].copy(),
      polarization_current=pole_zeros.copy(),
      polarization_previous=pole_zeros.copy(), step=0)


def _validate_state(grid, state):
  if not isinstance(state, YeeADE3DState):
    raise TypeError("state must be a YeeADE3DState instance.")
  if isinstance(state.step, (bool, np.bool_)) or not isinstance(
      state.step, (int, np.integer)) or state.step < 0:
    raise ValueError("state.step must be a non-negative integer.")
  dtype = grid.field_dtype
  fields = tuple(
      _field_array(name, getattr(state, name), grid.vector_shape, dtype)
      for name in ("electric", "magnetic", "displacement",
                   "displacement_previous", "electric_previous"))
  cpml_fields = tuple(
      np.zeros(grid.shape, dtype=dtype) if getattr(state, name) is None else
      _field_array(name, getattr(state, name), grid.shape, dtype)
      for name in ("cpml_hx_ey_z", "cpml_hy_ex_z",
                   "cpml_dx_hy_z", "cpml_dy_hx_z"))
  pole_shape = grid.vector_shape + (grid.maximum_poles,)
  polarizations = tuple(
      np.zeros(pole_shape, dtype=dtype) if getattr(state, name) is None else
      _field_array(name, getattr(state, name), pole_shape, dtype)
      for name in ("polarization_current", "polarization_previous"))
  return fields + cpml_fields + polarizations


def _vector_source(name, value, grid):
  if value is None:
    return np.zeros(grid.vector_shape, dtype=grid.field_dtype)
  if np.iscomplexobj(value) and not np.issubdtype(grid.field_dtype, np.complexfloating):
    raise TypeError(f"{name}: complex quadratures require a nonzero Bloch wavevector.")
  array = np.asarray(value, dtype=grid.field_dtype)
  if array.ndim == 0:
    array = np.full(grid.vector_shape, array, dtype=grid.field_dtype)
  elif array.shape == (_NUM_COMPONENTS,):
    array = np.broadcast_to(
        array.reshape((_NUM_COMPONENTS,) + (1,) * 3), grid.vector_shape)
  elif array.shape == grid.shape:
    array = np.broadcast_to(array[None, ...], grid.vector_shape)
  elif array.shape != grid.vector_shape:
    raise ValueError(
        f"{name} must be scalar or have shape (3,), {grid.shape}, or "
        f"{grid.vector_shape}; got {array.shape}.")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values.")
  return np.asarray(array, dtype=grid.field_dtype)


def _forward(field, axis, spacing, phase=1.0):
  neighbor = np.roll(field, -1, axis=axis)
  if phase != 1.0:
    edge = [slice(None)] * field.ndim
    edge[axis] = -1
    neighbor[tuple(edge)] *= phase
  return (neighbor - field) / spacing


def _backward(field, axis, spacing, phase=1.0):
  neighbor = np.roll(field, 1, axis=axis)
  if phase != 1.0:
    edge = [slice(None)] * field.ndim
    edge[axis] = 0
    neighbor[tuple(edge)] /= phase
  return (field - neighbor) / spacing


def yee_ade_3d_step(grid, state, electric_current=None,
                    magnetic_current=None):
  """Advance one periodic full-vector Yee-ADE step.

  ``electric_current`` is J at the electric nodes and ``magnetic_current`` is
  the normalized M term in Faraday's law.  Their leading component axis is
  ordered x, y, z.  Positive currents subtract from D or H respectively.
  """
  if not isinstance(grid, MaterialGrid3D):
    raise TypeError("grid must be a MaterialGrid3D instance.")
  (electric, magnetic, displacement, displacement_previous,
   electric_previous, cpml_hx_ey_z, cpml_hy_ex_z,
   cpml_dx_hy_z, cpml_dy_hx_z, polarization_current,
   polarization_previous) = _validate_state(grid, state)
  electric_current = _vector_source(
      "electric_current", electric_current, grid)
  magnetic_current = _vector_source(
      "magnetic_current", magnetic_current, grid)
  ex, ey, ez = electric
  hx, hy, hz = magnetic
  phase_x, phase_y = (np.exp(1j * k * n * d) if k else 1.0 for k, n, d in
                       zip(grid.bloch_wavevector, grid.shape, (grid.dx, grid.dy)))

  ey_forward_z = _forward(ey, 2, grid.dz)
  ex_forward_z = _forward(ex, 2, grid.dz)
  if grid.z_cpml is None:
    cpml_hx_ey_z_next = np.zeros_like(cpml_hx_ey_z)
    cpml_hy_ex_z_next = np.zeros_like(cpml_hy_ex_z)
    ey_forward_z_effective = ey_forward_z
    ex_forward_z_effective = ex_forward_z
  else:
    b_h = grid.z_cpml.b_h[None, None, :]
    c_h = grid.z_cpml.c_h[None, None, :]
    inverse_kappa_h = grid.z_cpml.inverse_kappa_h[None, None, :]
    cpml_hx_ey_z_next = b_h * cpml_hx_ey_z + c_h * ey_forward_z
    cpml_hy_ex_z_next = b_h * cpml_hy_ex_z + c_h * ex_forward_z
    ey_forward_z_effective = (
        inverse_kappa_h * ey_forward_z + cpml_hx_ey_z_next)
    ex_forward_z_effective = (
        inverse_kappa_h * ex_forward_z + cpml_hy_ex_z_next)

  hx_next = hx - grid.dt * (
      _forward(ez, 1, grid.dy, phase_y) - ey_forward_z_effective)
  hy_next = hy - grid.dt * (
      ex_forward_z_effective - _forward(ez, 0, grid.dx, phase_x))
  hz_next = hz - grid.dt * (
      _forward(ey, 0, grid.dx, phase_x) - _forward(ex, 1, grid.dy, phase_y))
  magnetic_next = np.stack((hx_next, hy_next, hz_next))
  magnetic_next -= grid.dt * magnetic_current
  hx_next, hy_next, hz_next = magnetic_next

  hy_backward_z = _backward(hy_next, 2, grid.dz)
  hx_backward_z = _backward(hx_next, 2, grid.dz)
  if grid.z_cpml is None:
    cpml_dx_hy_z_next = np.zeros_like(cpml_dx_hy_z)
    cpml_dy_hx_z_next = np.zeros_like(cpml_dy_hx_z)
    hy_backward_z_effective = hy_backward_z
    hx_backward_z_effective = hx_backward_z
  else:
    b_e = grid.z_cpml.b_e[None, None, :]
    c_e = grid.z_cpml.c_e[None, None, :]
    inverse_kappa_e = grid.z_cpml.inverse_kappa_e[None, None, :]
    cpml_dx_hy_z_next = b_e * cpml_dx_hy_z + c_e * hy_backward_z
    cpml_dy_hx_z_next = b_e * cpml_dy_hx_z + c_e * hx_backward_z
    hy_backward_z_effective = (
        inverse_kappa_e * hy_backward_z + cpml_dx_hy_z_next)
    hx_backward_z_effective = (
        inverse_kappa_e * hx_backward_z + cpml_dy_hx_z_next)

  dx_next = displacement[0] + grid.dt * (
      _backward(hz_next, 1, grid.dy, phase_y) -
      hy_backward_z_effective)
  dy_next = displacement[1] + grid.dt * (
      hx_backward_z_effective -
      _backward(hz_next, 0, grid.dx, phase_x))
  dz_next = displacement[2] + grid.dt * (
      _backward(hy_next, 0, grid.dx, phase_x) -
      _backward(hx_next, 1, grid.dy, phase_y))
  displacement_next = np.stack((dx_next, dy_next, dz_next))
  displacement_next -= grid.dt * electric_current

  a0, a1, a2, b1, b2 = np.moveaxis(grid.coefficients, -1, 0)
  electric_next = (
      a0 * displacement_next + a1 * displacement +
      a2 * displacement_previous + b1 * electric +
      b2 * electric_previous)
  if grid.maximum_poles:
    q_next, q_current, q_previous, r_current, r_previous = np.moveaxis(
        grid.pole_coefficients, -1, 0)
    known_polarization = (
        q_current * electric[..., None] +
        q_previous * electric_previous[..., None] +
        r_current * polarization_current +
        r_previous * polarization_previous)
    multipole_denominator = (
        grid.multipole_eps_inf + np.sum(q_next, axis=-1))
    multipole_electric = (
        displacement_next - np.sum(known_polarization, axis=-1)) / (
            multipole_denominator)
    polarization_next = (
        known_polarization + q_next * multipole_electric[..., None])
    electric_next = np.where(
        grid.multipole_mask, multipole_electric, electric_next)
    polarization_next = np.where(
        grid.multipole_mask[..., None], polarization_next, 0.0)
  else:
    polarization_next = polarization_current

  dtype = grid.field_dtype
  return YeeADE3DState(
      electric=np.asarray(electric_next, dtype=dtype),
      magnetic=np.asarray(magnetic_next, dtype=dtype),
      displacement=np.asarray(displacement_next, dtype=dtype),
      displacement_previous=displacement,
      electric_previous=electric,
      cpml_hx_ey_z=np.asarray(cpml_hx_ey_z_next, dtype=dtype),
      cpml_hy_ex_z=np.asarray(cpml_hy_ex_z_next, dtype=dtype),
      cpml_dx_hy_z=np.asarray(cpml_dx_hy_z_next, dtype=dtype),
      cpml_dy_hx_z=np.asarray(cpml_dy_hx_z_next, dtype=dtype),
      polarization_current=np.asarray(polarization_next, dtype=dtype),
      polarization_previous=polarization_current,
      step=int(state.step) + 1)


def _source_history(name, value, grid, num_steps):
  if value is None:
    return None, False
  if np.iscomplexobj(value) and not np.issubdtype(grid.field_dtype, np.complexfloating):
    raise TypeError(f"{name}: complex quadratures require a nonzero Bloch wavevector.")
  array = np.asarray(value, dtype=grid.field_dtype)
  static_shapes = ((), (_NUM_COMPONENTS,), grid.shape, grid.vector_shape)
  history_shapes = (
      (num_steps,), (num_steps, _NUM_COMPONENTS),
      (num_steps,) + grid.shape, (num_steps,) + grid.vector_shape)
  if array.shape not in static_shapes + history_shapes:
    raise ValueError(
        f"{name} has unsupported shape {array.shape}; expected one of "
        f"{static_shapes + history_shapes}.")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values.")
  return array, array.shape in history_shapes


def _snapshot_steps(value, num_steps):
  if value is None:
    return np.empty(0, dtype=np.int64)
  array = np.asarray(value)
  if array.ndim == 1 and array.size == 0:
    return np.empty(0, dtype=np.int64)
  if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
    raise ValueError("snapshot_steps must be a one-dimensional integer sequence.")
  array = np.asarray(array, dtype=np.int64)
  if (np.any(array < 1) or np.any(array > num_steps) or
      np.unique(array).size != array.size or
      (array.size > 1 and np.any(np.diff(array) <= 0))):
    raise ValueError(
        "snapshot_steps must be strictly increasing, unique, and lie in "
        f"[1, {num_steps}].")
  return array


def _frequency_monitor(frequencies, window, num_steps, dt, dtype):
  if frequencies is None:
    return np.empty(0, dtype=float), np.empty(0, dtype=dtype)
  values = np.asarray(frequencies)
  if values.ndim == 1 and values.size == 0:
    return np.empty(0, dtype=float), np.empty(0, dtype=dtype)
  if (values.ndim != 1 or np.iscomplexobj(values) or
      not np.issubdtype(values.dtype, np.number) or
      not np.all(np.isfinite(values))):
    raise ValueError("frequencies must be a finite real one-dimensional array.")
  values = np.asarray(values, dtype=float)
  nyquist = 0.5 / dt
  if (np.any(values <= 0.0) or np.any(values > nyquist) or
      np.unique(values).size != values.size or
      (values.size > 1 and np.any(np.diff(values) <= 0.0))):
    raise ValueError(
        "frequencies must be strictly increasing, unique, positive, and no "
        f"greater than the Nyquist frequency {nyquist}.")
  if window is None:
    weights = np.ones(num_steps, dtype=dtype)
  elif isinstance(window, str):
    if window != "hann":
      raise ValueError("frequency_window string must be 'hann'.")
    weights = np.asarray(np.hanning(num_steps), dtype=dtype)
  else:
    weights = np.asarray(window, dtype=dtype)
    if weights.shape != (num_steps,) or not np.all(np.isfinite(weights)):
      raise ValueError(
          "frequency_window must be None, 'hann', or a finite array with one "
          "weight per time step.")
  return values, weights


def simulate_yee_ade_3d(materials, material_ids, dx, dy, dz, dt, num_steps,
                        electric_current=None, magnetic_current=None,
                        dtype=np.float64, initial_state=None,
                        record_interval=1, z_cpml=None, bloch_wavevector=(0.0, 0.0)):
  """Run the periodic 3D reference and sparsely record full E/H/D fields."""
  if isinstance(num_steps, (bool, np.bool_)) or not isinstance(
      num_steps, (int, np.integer)) or num_steps < 0:
    raise ValueError("num_steps must be a non-negative integer.")
  if isinstance(record_interval, (bool, np.bool_)) or not isinstance(
      record_interval, (int, np.integer)) or record_interval <= 0:
    raise ValueError("record_interval must be a positive integer.")
  num_steps = int(num_steps)
  record_interval = int(record_interval)
  grid = prepare_material_grid_3d(
      materials, material_ids, dt, dx, dy, dz, dtype=dtype, z_cpml=z_cpml, bloch_wavevector=bloch_wavevector)
  state = initialize_yee_ade_3d(grid) if initial_state is None else initial_state
  _validate_state(grid, state)
  electric_source, electric_history = _source_history(
      "electric_current", electric_current, grid, num_steps)
  magnetic_source, magnetic_history = _source_history(
      "magnetic_current", magnetic_current, grid, num_steps)
  final_step = int(state.step) + num_steps

  recorded_electric = []
  recorded_magnetic = []
  recorded_displacement = []
  recorded_steps = []
  for time_index in range(num_steps):
    step_electric = (
        electric_source[time_index] if electric_history else electric_source)
    step_magnetic = (
        magnetic_source[time_index] if magnetic_history else magnetic_source)
    state = yee_ade_3d_step(
        grid, state, step_electric, step_magnetic)
    if state.step % record_interval == 0 or state.step == final_step:
      recorded_electric.append(state.electric.copy())
      recorded_magnetic.append(state.magnetic.copy())
      recorded_displacement.append(state.displacement.copy())
      recorded_steps.append(state.step)

  empty_shape = (0,) + grid.vector_shape
  def stack_or_empty(values):
    return (np.stack(values) if values else
            np.empty(empty_shape, dtype=grid.field_dtype))

  return YeeADE3DResult(
      electric=stack_or_empty(recorded_electric),
      magnetic=stack_or_empty(recorded_magnetic),
      displacement=stack_or_empty(recorded_displacement),
      steps=np.asarray(recorded_steps, dtype=np.int64),
      final_state=state)


def simulate_yee_ade_3d_plane_probes(
    materials, material_ids, dx, dy, dz, dt, num_steps, probe_z_indices,
    electric_current=None, magnetic_current=None, dtype=np.float64,
    initial_state=None, z_cpml=None, bloch_wavevector=(0.0, 0.0)):
  """Run the 3D reference while retaining only Ex/Ey monitor planes.

  The returned history has shape ``(time, probe, 2, nx, ny)``.  This is the
  compact time-domain input required by the two-dimensional diffraction-order
  transform and avoids storing full ``(time, 3, nx, ny, nz)`` field histories.
  """
  if isinstance(num_steps, (bool, np.bool_)) or not isinstance(
      num_steps, (int, np.integer)) or num_steps < 0:
    raise ValueError("num_steps must be a non-negative integer.")
  num_steps = int(num_steps)
  grid = prepare_material_grid_3d(
      materials, material_ids, dt, dx, dy, dz, dtype=dtype, z_cpml=z_cpml, bloch_wavevector=bloch_wavevector)
  probe_z_indices = np.asarray(probe_z_indices)
  if (probe_z_indices.ndim != 1 or probe_z_indices.size == 0 or
      not np.issubdtype(probe_z_indices.dtype, np.integer)):
    raise ValueError("probe_z_indices must be a non-empty 1D integer array.")
  if (np.any(probe_z_indices < 0) or
      np.any(probe_z_indices >= grid.shape[2])):
    raise ValueError(
        f"probe_z_indices must lie in [0, {grid.shape[2] - 1}].")
  if np.unique(probe_z_indices).size != probe_z_indices.size:
    raise ValueError("probe_z_indices must not contain duplicates.")
  probe_z_indices = np.asarray(probe_z_indices, dtype=np.int64)
  state = initialize_yee_ade_3d(grid) if initial_state is None else initial_state
  _validate_state(grid, state)
  electric_source, electric_history = _source_history(
      "electric_current", electric_current, grid, num_steps)
  magnetic_source, magnetic_history = _source_history(
      "magnetic_current", magnetic_current, grid, num_steps)

  history = np.empty(
      (num_steps, probe_z_indices.size, 2, grid.shape[0], grid.shape[1]),
      dtype=grid.field_dtype)
  steps = np.empty(num_steps, dtype=np.int64)
  for time_index in range(num_steps):
    step_electric = (
        electric_source[time_index] if electric_history else electric_source)
    step_magnetic = (
        magnetic_source[time_index] if magnetic_history else magnetic_source)
    state = yee_ade_3d_step(
        grid, state, step_electric, step_magnetic)
    # Advanced indexing places the probe axis first; keep component next.
    history[time_index] = np.moveaxis(
        state.electric[:2, :, :, probe_z_indices], -1, 0)
    steps[time_index] = state.step
  return YeeADE3DPlaneProbeResult(
      electric_tangential=history, z_indices=probe_z_indices,
      steps=steps, final_state=state)


def simulate_yee_ade_3d_compact_plane_probes(
    materials, material_ids, dx, dy, dz, dt,
    electric_waveform, magnetic_waveform, transverse_profile,
    polarization_xy, electric_source_z, magnetic_source_z,
    reflection_probe_z, transmission_probe_z, dtype=np.float64,
    initial_state=None, z_cpml=None, snapshot_steps=None,
    frequencies=None, frequency_window=None,
    bloch_wavevector=(0.0, 0.0)):
  """Run compact probes with optional sparse time/frequency full fields."""
  grid = prepare_material_grid_3d(
      materials, material_ids, dt, dx, dy, dz, dtype=dtype, z_cpml=z_cpml, bloch_wavevector=bloch_wavevector)
  if (not any(grid.bloch_wavevector) and any(np.iscomplexobj(value) for value in
      (electric_waveform, magnetic_waveform, transverse_profile, polarization_xy))):
    raise TypeError("complex source quadratures require a nonzero Bloch wavevector.")
  electric_waveform = np.asarray(
      electric_waveform, dtype=grid.field_dtype)
  magnetic_waveform = np.asarray(
      magnetic_waveform, dtype=grid.field_dtype)
  if (electric_waveform.ndim != 1 or
      magnetic_waveform.shape != electric_waveform.shape or
      not np.all(np.isfinite(electric_waveform)) or
      not np.all(np.isfinite(magnetic_waveform))):
    raise ValueError("electric and magnetic waveforms must be matching finite 1D arrays.")
  profile = np.asarray(transverse_profile, dtype=grid.field_dtype)
  if profile.shape not in (grid.shape[:2], (4,) + grid.shape[:2]) or not np.all(np.isfinite(profile)):
    raise ValueError(
        f"transverse_profile must be finite with shape {grid.shape[:2]}.")
  polarization = np.asarray(polarization_xy, dtype=grid.field_dtype)
  if polarization.shape != (2,) or not np.all(np.isfinite(polarization)):
    raise ValueError("polarization_xy must be a finite two-component array.")
  indices = (electric_source_z, magnetic_source_z,
             reflection_probe_z, transmission_probe_z)
  if any(isinstance(index, (bool, np.bool_)) or
         not isinstance(index, (int, np.integer)) or
         index < 0 or index >= grid.shape[2] for index in indices):
    raise ValueError("source and probe indices must lie inside the z extent.")
  state = initialize_yee_ade_3d(grid) if initial_state is None else initial_state
  _validate_state(grid, state)
  num_steps = electric_waveform.size
  requested_snapshots = _snapshot_steps(snapshot_steps, num_steps)
  monitored_frequencies, frequency_weights = _frequency_monitor(
      frequencies, frequency_window, num_steps, grid.dt,
      grid.coefficients.dtype)
  history = np.empty(
      (num_steps, 2, 2, grid.shape[0], grid.shape[1]),
      dtype=grid.field_dtype)
  steps = np.empty(num_steps, dtype=np.int64)
  snapshot_shape = (requested_snapshots.size,) + grid.vector_shape
  snapshot_electric = np.empty(snapshot_shape, dtype=grid.field_dtype)
  snapshot_magnetic = np.empty(snapshot_shape, dtype=grid.field_dtype)
  snapshot_displacement = np.empty(
      snapshot_shape, dtype=grid.field_dtype)
  snapshot_actual_steps = np.empty(requested_snapshots.size, dtype=np.int64)
  snapshot_index = 0
  complex_dtype = np.result_type(grid.coefficients.dtype, np.complex64)
  frequency_shape = (monitored_frequencies.size,) + grid.vector_shape
  frequency_electric = np.zeros(frequency_shape, dtype=complex_dtype)
  frequency_magnetic = np.zeros(frequency_shape, dtype=complex_dtype)
  frequency_displacement = np.zeros(frequency_shape, dtype=complex_dtype)
  px, py = polarization
  for time_index in range(num_steps):
    electric_source = np.zeros(grid.vector_shape, dtype=grid.field_dtype)
    magnetic_source = np.zeros(grid.vector_shape, dtype=grid.field_dtype)
    jx, jy, mx, my = plane_source_values(
        electric_waveform[time_index], magnetic_waveform[time_index],
        profile, polarization)
    electric_source[0, :, :, electric_source_z] = jx
    electric_source[1, :, :, electric_source_z] = jy
    magnetic_source[0, :, :, magnetic_source_z] = mx
    magnetic_source[1, :, :, magnetic_source_z] = my
    state = yee_ade_3d_step(grid, state, electric_source, magnetic_source)
    history[time_index, 0] = state.electric[
        :2, :, :, reflection_probe_z]
    history[time_index, 1] = state.electric[
        :2, :, :, transmission_probe_z]
    steps[time_index] = state.step
    relative_step = time_index + 1
    if (snapshot_index < requested_snapshots.size and
        relative_step == requested_snapshots[snapshot_index]):
      snapshot_electric[snapshot_index] = state.electric
      snapshot_magnetic[snapshot_index] = state.magnetic
      snapshot_displacement[snapshot_index] = state.displacement
      snapshot_actual_steps[snapshot_index] = state.step
      snapshot_index += 1
    if monitored_frequencies.size:
      electric_time = state.step * grid.dt
      magnetic_time = (state.step - 0.5) * grid.dt
      scale = grid.dt * frequency_weights[time_index]
      electric_phases = scale * np.exp(
          2j * np.pi * monitored_frequencies * electric_time)
      magnetic_phases = scale * np.exp(
          2j * np.pi * monitored_frequencies * magnetic_time)
      reshape = (monitored_frequencies.size,) + (1,) * 4
      frequency_electric += electric_phases.reshape(reshape) * state.electric
      frequency_magnetic += magnetic_phases.reshape(reshape) * state.magnetic
      frequency_displacement += (
          electric_phases.reshape(reshape) * state.displacement)
  snapshots = None
  if requested_snapshots.size:
    snapshots = YeeADE3DFieldSnapshots(
        electric=snapshot_electric, magnetic=snapshot_magnetic,
        displacement=snapshot_displacement, steps=snapshot_actual_steps)
  frequency_fields = None
  if monitored_frequencies.size:
    frequency_fields = YeeADE3DFrequencyFields(
        frequencies=monitored_frequencies,
        electric=frequency_electric, magnetic=frequency_magnetic,
        displacement=frequency_displacement)
  return YeeADE3DPlaneProbeResult(
      electric_tangential=history,
      z_indices=np.asarray(
          (reflection_probe_z, transmission_probe_z), dtype=np.int64),
      steps=steps, final_state=state, snapshots=snapshots,
      frequency_fields=frequency_fields)


__all__ = [
    "MaterialGrid3D", "YeeADE3DFieldSnapshots", "YeeADE3DFrequencyFields",
    "YeeADE3DPlaneProbeResult", "YeeADE3DResult",
    "YeeADE3DState", "ZCPML3D",
    "initialize_yee_ade_3d", "prepare_material_grid_3d",
    "prepare_z_cpml", "simulate_yee_ade_3d",
    "simulate_yee_ade_3d_compact_plane_probes",
    "simulate_yee_ade_3d_plane_probes", "yee_ade_3d_step",
]
