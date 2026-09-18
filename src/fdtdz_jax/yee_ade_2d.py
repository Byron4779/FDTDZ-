"""Two-dimensional TMz Yee-ADE reference solver.

The solver uses normalized units ``c0 = epsilon0 = mu0 = 1``.  Its TMz fields
are ``E_z``, ``D_z / epsilon0``, ``H_x``, and ``H_y``.  Array axes are ordered
``(x, y)``.  Electric quantities live at integer nodes, ``H_x`` is staggered
half a cell in y, and ``H_y`` is staggered half a cell in x.

Both axes are periodic in this first reference implementation.  This matches a
periodic metasurface unit cell laterally and provides a clean numerical baseline
before an absorbing boundary is introduced along the propagation direction.
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


_SUPPORTED_MATERIALS = (Lossless, Debye, Lorentz, Drude, Multipole)


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
  array = np.asarray(value, dtype=dtype)
  if array.shape != shape:
    raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values.")
  return np.array(array, dtype=dtype, copy=True)


@dataclass(frozen=True)
class MaterialGrid2D:
  """Per-Ez-node material coefficients for a two-dimensional unit cell."""

  coefficients: np.ndarray
  material_ids: np.ndarray
  electric_conductivity: np.ndarray
  magnetic_conductivity: np.ndarray
  y_cpml: object
  multipole_mask: np.ndarray
  multipole_eps_inf: np.ndarray
  pole_coefficients: np.ndarray
  dt: float
  dx: float
  dy: float

  @property
  def shape(self):
    return self.material_ids.shape

  @property
  def courant_x(self):
    return self.dt / self.dx

  @property
  def courant_y(self):
    return self.dt / self.dy

  @property
  def maximum_poles(self):
    return self.pole_coefficients.shape[-2]


@dataclass(frozen=True)
class YeeADE2DState:
  """TMz fields and material histories after ``step`` complete updates."""

  electric_z: np.ndarray
  magnetic_x: np.ndarray
  magnetic_y: np.ndarray
  displacement_z: np.ndarray
  displacement_z_previous: np.ndarray
  electric_z_previous: np.ndarray
  cpml_hx_y: np.ndarray = None
  cpml_dz_y: np.ndarray = None
  polarization_current: np.ndarray = None
  polarization_previous: np.ndarray = None
  step: int = 0


@dataclass(frozen=True)
class YeeADE2DResult:
  """Recorded TMz fields and final simulation state."""

  electric_z: np.ndarray
  magnetic_x: np.ndarray
  magnetic_y: np.ndarray
  displacement_z: np.ndarray
  steps: np.ndarray
  final_state: YeeADE2DState


@dataclass(frozen=True)
class YCPML2D:
  """Convolutional PML coefficients for derivatives in the y direction."""

  inverse_kappa_e: np.ndarray
  b_e: np.ndarray
  c_e: np.ndarray
  inverse_kappa_h: np.ndarray
  b_h: np.ndarray
  c_h: np.ndarray
  width: int


def prepare_y_cpml(shape, width, dt, dy, target_reflection=1e-8,
                   grading_order=3, kappa_max=5.0, alpha_fraction=0.05,
                   dtype=np.float64):
  """Create normalized y-directed CPML coefficients.

  The coefficient estimate assumes normalized vacuum impedance at both y
  boundaries.  Keep material geometry outside the CPML region.
  """
  if len(shape) != 2 or any(int(value) < 2 for value in shape):
    raise ValueError(f"shape must be (nx, ny), both at least two; got {shape}.")
  nx, ny = (int(value) for value in shape)
  if isinstance(width, (bool, np.bool_)) or not isinstance(
      width, (int, np.integer)):
    raise TypeError("width must be an integer.")
  width = int(width)
  if width <= 0 or 2 * width >= ny:
    raise ValueError("width must be positive and leave at least one interior cell.")
  dt = _positive_finite("dt", dt)
  dy = _positive_finite("dy", dy)
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
      2.0 * width * dy)
  alpha_max = alpha_fraction * sigma_max

  def depth_at(offset):
    positions = np.arange(ny, dtype=float) + offset
    distance_to_boundary = np.minimum(positions + 0.5, ny - 0.5 - positions)
    return np.minimum(
        np.maximum((width - distance_to_boundary) / width, 0.0), 1.0)

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
  return YCPML2D(
      inverse_kappa_e=inverse_kappa_e,
      b_e=b_e,
      c_e=c_e,
      inverse_kappa_h=inverse_kappa_h,
      b_h=b_h,
      c_h=c_h,
      width=width)


def prepare_material_grid_2d(materials, material_ids, dt, dx, dy,
                             dtype=np.float64, electric_conductivity=None,
                             magnetic_conductivity=None, y_cpml=None):
  """Validate a 2D material map and generate five ADE coefficients per Ez node."""
  dt = _positive_finite("dt", dt)
  dx = _positive_finite("dx", dx)
  dy = _positive_finite("dy", dy)
  dtype = _floating_dtype(dtype)
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
          "Multipole, got "
          f"{type(material).__name__}.")

  material_ids = np.asarray(material_ids)
  if material_ids.ndim != 2 or min(material_ids.shape, default=0) < 2:
    raise ValueError(
        "material_ids must have shape (nx, ny) with both dimensions at least "
        f"two, got {material_ids.shape}.")
  if not np.issubdtype(material_ids.dtype, np.integer):
    raise TypeError("material_ids must contain integers.")
  if np.any(material_ids < 0) or np.any(material_ids >= len(materials)):
    raise ValueError(f"material_ids must be in [0, {len(materials) - 1}].")

  def conductivity_array(name, value):
    if value is None:
      return np.zeros(material_ids.shape, dtype=dtype)
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
      array = np.full(material_ids.shape, array, dtype=dtype)
    if array.shape != material_ids.shape:
      raise ValueError(
          f"{name} must be scalar or have shape {material_ids.shape}, got "
          f"{array.shape}.")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
      raise ValueError(f"{name} must contain finite non-negative values.")
    return np.array(array, dtype=dtype, copy=True)

  electric_conductivity = conductivity_array(
      "electric_conductivity", electric_conductivity)
  magnetic_conductivity = conductivity_array(
      "magnetic_conductivity", magnetic_conductivity)
  if y_cpml is not None:
    if not isinstance(y_cpml, YCPML2D):
      raise TypeError("y_cpml must be a YCPML2D instance.")
    if y_cpml.inverse_kappa_e.shape != (material_ids.shape[1],):
      raise ValueError("y_cpml coefficient length must match material_ids.shape[1].")
    if (np.any(electric_conductivity) or np.any(magnetic_conductivity)):
      raise ValueError("conductivity sponge and CPML cannot be enabled together.")
  lossless_material = np.asarray([
      isinstance(material, Lossless) for material in materials], dtype=bool)
  dispersive_nodes = ~lossless_material[material_ids]
  if (np.any(electric_conductivity[dispersive_nodes] > 0.0) or
      np.any(magnetic_conductivity[dispersive_nodes] > 0.0)):
    raise ValueError(
        "absorbing conductivity is currently restricted to Lossless material "
        "nodes; keep dispersive geometry outside the sponge layer.")
  if y_cpml is not None:
    cpml_nodes = np.zeros(material_ids.shape, dtype=bool)
    cpml_nodes[:, :y_cpml.width] = True
    cpml_nodes[:, -y_cpml.width:] = True
    if np.any(dispersive_nodes & cpml_nodes):
      raise ValueError(
          "CPML is currently restricted to Lossless boundary nodes; keep "
          "dispersive geometry outside the CPML layer.")

  material_coefficients = np.stack([
      (ade_coefficients(material, dt).as_array(dtype)
       if not isinstance(material, Multipole) else
       ade_coefficients(Lossless(material.eps_inf), dt).as_array(dtype))
      for material in materials])
  coefficients = material_coefficients[material_ids]
  high_frequency_eps = np.asarray([
      material.eps_r if isinstance(material, Lossless) else material.eps_inf
      for material in materials
  ])
  minimum_eps = float(np.min(high_frequency_eps[material_ids]))
  inverse_spacing = math.sqrt(1.0 / dx**2 + 1.0 / dy**2)
  maximum_dt = math.sqrt(minimum_eps) / inverse_spacing
  if dt > maximum_dt * (1.0 + 1e-12):
    raise ValueError(
        "The two-dimensional CFL condition is violated: dt must be <= "
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

  return MaterialGrid2D(
      coefficients=np.asarray(coefficients, dtype=dtype),
      material_ids=np.asarray(material_ids, dtype=np.uint8),
      electric_conductivity=electric_conductivity,
      magnetic_conductivity=magnetic_conductivity,
      y_cpml=y_cpml,
      multipole_mask=material_is_multipole[material_ids],
      multipole_eps_inf=material_multipole_eps_inf[material_ids],
      pole_coefficients=material_pole_coefficients[material_ids],
      dt=dt,
      dx=dx,
      dy=dy,
  )


def initialize_yee_ade_2d(grid, dtype=None):
  """Create a zero-field, zero-polarization TMz state."""
  if not isinstance(grid, MaterialGrid2D):
    raise TypeError("grid must be a MaterialGrid2D instance.")
  dtype = grid.coefficients.dtype if dtype is None else _floating_dtype(dtype)
  zeros = np.zeros(grid.shape, dtype=dtype)
  pole_zeros = np.zeros(grid.shape + (grid.maximum_poles,), dtype=dtype)
  return YeeADE2DState(
      electric_z=zeros.copy(),
      magnetic_x=zeros.copy(),
      magnetic_y=zeros.copy(),
      displacement_z=zeros.copy(),
      displacement_z_previous=zeros.copy(),
      electric_z_previous=zeros.copy(),
      cpml_hx_y=zeros.copy(),
      cpml_dz_y=zeros.copy(),
      polarization_current=pole_zeros.copy(),
      polarization_previous=pole_zeros.copy(),
      step=0,
  )


def _validate_state(grid, state):
  if not isinstance(state, YeeADE2DState):
    raise TypeError("state must be a YeeADE2DState instance.")
  if isinstance(state.step, (bool, np.bool_)) or not isinstance(
      state.step, (int, np.integer)) or state.step < 0:
    raise ValueError("state.step must be a non-negative integer.")
  dtype = grid.coefficients.dtype
  fields = tuple(
      _field_array(name, getattr(state, name), grid.shape, dtype)
      for name in ("electric_z", "magnetic_x", "magnetic_y",
                   "displacement_z", "displacement_z_previous",
                   "electric_z_previous"))
  cpml_fields = tuple(
      np.zeros(grid.shape, dtype=dtype) if getattr(state, name) is None else
      _field_array(name, getattr(state, name), grid.shape, dtype)
      for name in ("cpml_hx_y", "cpml_dz_y"))
  pole_shape = grid.shape + (grid.maximum_poles,)
  polarization_fields = tuple(
      np.zeros(pole_shape, dtype=dtype) if getattr(state, name) is None else
      _field_array(name, getattr(state, name), pole_shape, dtype)
      for name in ("polarization_current", "polarization_previous"))
  return fields + cpml_fields + polarization_fields


def yee_ade_2d_step(grid, state, current_density_z=None,
                    magnetic_current_x=None):
  """Advance one periodic TMz Yee-ADE step.

  ``current_density_z`` is J_z[n+1/2] and ``magnetic_current_x`` is the
  normalized M_x source in Faraday's law.  Either may be scalar or ``(nx, ny)``.
  Positive currents subtract from their respective D or H updates.
  """
  if not isinstance(grid, MaterialGrid2D):
    raise TypeError("grid must be a MaterialGrid2D instance.")
  (electric_z, magnetic_x, magnetic_y, displacement_z,
   displacement_z_previous, electric_z_previous, cpml_hx_y,
   cpml_dz_y, polarization_current,
   polarization_previous) = _validate_state(grid, state)
  dtype = grid.coefficients.dtype

  electric_forward_y = (
      np.roll(electric_z, -1, axis=1) - electric_z) / grid.dy
  if grid.y_cpml is None:
    cpml_hx_y_next = np.zeros_like(cpml_hx_y)
    electric_forward_y_effective = electric_forward_y
  else:
    b_h = grid.y_cpml.b_h[None, :]
    c_h = grid.y_cpml.c_h[None, :]
    cpml_hx_y_next = b_h * cpml_hx_y + c_h * electric_forward_y
    electric_forward_y_effective = (
        grid.y_cpml.inverse_kappa_h[None, :] * electric_forward_y +
        cpml_hx_y_next)

  magnetic_denominator = 1.0 + 0.5 * grid.dt * grid.magnetic_conductivity
  magnetic_decay = (1.0 - 0.5 * grid.dt *
                    grid.magnetic_conductivity) / magnetic_denominator
  magnetic_x_next = magnetic_decay * magnetic_x - (
      grid.dt / magnetic_denominator) * electric_forward_y_effective
  magnetic_y_next = magnetic_decay * magnetic_y + (
      grid.courant_x / magnetic_denominator) * (
          np.roll(electric_z, -1, axis=0) - electric_z)
  if magnetic_current_x is not None:
    magnetic_current_x = np.asarray(magnetic_current_x, dtype=dtype)
    if magnetic_current_x.ndim == 0:
      magnetic_current_x = np.full(
          grid.shape, magnetic_current_x, dtype=dtype)
    if magnetic_current_x.shape != grid.shape:
      raise ValueError(
          "magnetic_current_x must be scalar or have shape "
          f"{grid.shape}, got {magnetic_current_x.shape}.")
    if not np.all(np.isfinite(magnetic_current_x)):
      raise ValueError("magnetic_current_x must contain only finite values.")
    magnetic_x_next -= grid.dt * magnetic_current_x / magnetic_denominator
  displacement_z_raw = displacement_z + grid.courant_x * (
      magnetic_y_next - np.roll(magnetic_y_next, 1, axis=0))
  magnetic_backward_y = (
      magnetic_x_next - np.roll(magnetic_x_next, 1, axis=1)) / grid.dy
  if grid.y_cpml is None:
    cpml_dz_y_next = np.zeros_like(cpml_dz_y)
    magnetic_backward_y_effective = magnetic_backward_y
  else:
    b_e = grid.y_cpml.b_e[None, :]
    c_e = grid.y_cpml.c_e[None, :]
    cpml_dz_y_next = b_e * cpml_dz_y + c_e * magnetic_backward_y
    magnetic_backward_y_effective = (
        grid.y_cpml.inverse_kappa_e[None, :] * magnetic_backward_y +
        cpml_dz_y_next)
  displacement_z_raw -= grid.dt * magnetic_backward_y_effective

  if current_density_z is not None:
    current_density_z = np.asarray(current_density_z, dtype=dtype)
    if current_density_z.ndim == 0:
      current_density_z = np.full(grid.shape, current_density_z, dtype=dtype)
    if current_density_z.shape != grid.shape:
      raise ValueError(
          "current_density_z must be scalar or have shape "
          f"{grid.shape}, got {current_density_z.shape}.")
    if not np.all(np.isfinite(current_density_z)):
      raise ValueError("current_density_z must contain only finite values.")
    displacement_z_raw -= grid.dt * current_density_z

  a0, a1, a2, b1, b2 = np.moveaxis(grid.coefficients, -1, 0)
  electric_without_new_displacement = (
      a1 * displacement_z + a2 * displacement_z_previous +
      b1 * electric_z + b2 * electric_z_previous)
  half_conductivity_step = 0.5 * grid.dt * grid.electric_conductivity
  displacement_z_next = (
      displacement_z_raw - half_conductivity_step *
      (electric_z + electric_without_new_displacement)) / (
          1.0 + half_conductivity_step * a0)
  electric_z_next = (a0 * displacement_z_next + a1 * displacement_z +
                     a2 * displacement_z_previous + b1 * electric_z +
                     b2 * electric_z_previous)
  if grid.maximum_poles:
    q_next, q_current, q_previous, r_current, r_previous = np.moveaxis(
        grid.pole_coefficients, -1, 0)
    known_polarization = (
        q_current * electric_z[..., None] +
        q_previous * electric_z_previous[..., None] +
        r_current * polarization_current +
        r_previous * polarization_previous)
    multipole_denominator = (
        grid.multipole_eps_inf + np.sum(q_next, axis=-1))
    multipole_electric = (
        displacement_z_next - np.sum(known_polarization, axis=-1)) / (
            multipole_denominator)
    polarization_next = (
        known_polarization + q_next * multipole_electric[..., None])
    electric_z_next = np.where(
        grid.multipole_mask, multipole_electric, electric_z_next)
    polarization_next = np.where(
        grid.multipole_mask[..., None], polarization_next, 0.0)
  else:
    polarization_next = polarization_current

  return YeeADE2DState(
      electric_z=np.asarray(electric_z_next, dtype=dtype),
      magnetic_x=np.asarray(magnetic_x_next, dtype=dtype),
      magnetic_y=np.asarray(magnetic_y_next, dtype=dtype),
      displacement_z=np.asarray(displacement_z_next, dtype=dtype),
      displacement_z_previous=displacement_z,
      electric_z_previous=electric_z,
      cpml_hx_y=np.asarray(cpml_hx_y_next, dtype=dtype),
      cpml_dz_y=np.asarray(cpml_dz_y_next, dtype=dtype),
      polarization_current=np.asarray(polarization_next, dtype=dtype),
      polarization_previous=polarization_current,
      step=int(state.step) + 1,
  )


def simulate_yee_ade_2d(materials, material_ids, dx, dy, dt, num_steps,
                        current_density_z=None, dtype=np.float64,
                        initial_state=None, record_interval=1,
                        electric_conductivity=None,
                        magnetic_conductivity=None, y_cpml=None,
                        magnetic_current_x=None):
  """Run the periodic TMz reference solver and record fields after updates.

  The source may be scalar, ``(nx, ny)``, ``(num_steps,)``, or
  ``(num_steps, nx, ny)``.  A ``(num_steps,)`` source is spatially uniform;
  a ``(num_steps, nx, ny)`` source supports point, line, or shaped excitation.
  """
  if isinstance(num_steps, (bool, np.bool_)) or not isinstance(
      num_steps, (int, np.integer)) or num_steps < 0:
    raise ValueError("num_steps must be a non-negative integer.")
  if isinstance(record_interval, (bool, np.bool_)) or not isinstance(
      record_interval, (int, np.integer)) or record_interval <= 0:
    raise ValueError("record_interval must be a positive integer.")

  grid = prepare_material_grid_2d(
      materials, material_ids, dt, dx, dy, dtype,
      electric_conductivity=electric_conductivity,
      magnetic_conductivity=magnetic_conductivity, y_cpml=y_cpml)
  state = initialize_yee_ade_2d(grid) if initial_state is None else initial_state
  _validate_state(grid, state)
  final_step = int(state.step) + int(num_steps)

  source = None
  if current_density_z is not None:
    source = np.asarray(current_density_z, dtype=grid.coefficients.dtype)
    valid_shapes = ((), grid.shape, (num_steps,), (num_steps,) + grid.shape)
    if source.shape not in valid_shapes:
      raise ValueError(
          "current_density_z must be scalar or have shape "
          f"{grid.shape}, ({num_steps},), or {(num_steps,) + grid.shape}; "
          f"got {source.shape}.")
    if not np.all(np.isfinite(source)):
      raise ValueError("current_density_z must contain only finite values.")

  magnetic_source = None
  if magnetic_current_x is not None:
    magnetic_source = np.asarray(
        magnetic_current_x, dtype=grid.coefficients.dtype)
    valid_shapes = ((), grid.shape, (num_steps,), (num_steps,) + grid.shape)
    if magnetic_source.shape not in valid_shapes:
      raise ValueError(
          "magnetic_current_x must be scalar or have shape "
          f"{grid.shape}, ({num_steps},), or {(num_steps,) + grid.shape}; "
          f"got {magnetic_source.shape}.")
    if not np.all(np.isfinite(magnetic_source)):
      raise ValueError("magnetic_current_x must contain only finite values.")

  recorded_electric_z = []
  recorded_magnetic_x = []
  recorded_magnetic_y = []
  recorded_displacement_z = []
  recorded_steps = []
  for time_index in range(num_steps):
    step_source = source
    if source is not None and source.shape == (num_steps,):
      step_source = source[time_index]
    elif source is not None and source.shape == (num_steps,) + grid.shape:
      step_source = source[time_index]
    step_magnetic_source = magnetic_source
    if magnetic_source is not None and magnetic_source.shape == (num_steps,):
      step_magnetic_source = magnetic_source[time_index]
    elif (magnetic_source is not None and
          magnetic_source.shape == (num_steps,) + grid.shape):
      step_magnetic_source = magnetic_source[time_index]
    state = yee_ade_2d_step(
        grid, state, step_source, step_magnetic_source)
    if state.step % record_interval == 0 or state.step == final_step:
      recorded_electric_z.append(state.electric_z.copy())
      recorded_magnetic_x.append(state.magnetic_x.copy())
      recorded_magnetic_y.append(state.magnetic_y.copy())
      recorded_displacement_z.append(state.displacement_z.copy())
      recorded_steps.append(state.step)

  empty_shape = (0,) + grid.shape
  def stack_or_empty(values):
    return (np.stack(values) if values else
            np.empty(empty_shape, dtype=grid.coefficients.dtype))

  return YeeADE2DResult(
      electric_z=stack_or_empty(recorded_electric_z),
      magnetic_x=stack_or_empty(recorded_magnetic_x),
      magnetic_y=stack_or_empty(recorded_magnetic_y),
      displacement_z=stack_or_empty(recorded_displacement_z),
      steps=np.asarray(recorded_steps, dtype=np.int64),
      final_state=state,
  )


__all__ = [
    "MaterialGrid2D",
    "YeeADE2DResult",
    "YeeADE2DState",
    "YCPML2D",
    "initialize_yee_ade_2d",
    "prepare_material_grid_2d",
    "prepare_y_cpml",
    "simulate_yee_ade_2d",
    "yee_ade_2d_step",
]
