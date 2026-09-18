"""Two-dimensional periodic TEz Yee-ADE CPU reference solver."""

from dataclasses import dataclass

import numpy as np

from .yee_ade_2d import MaterialGrid2D
from .yee_ade_2d import _field_array
from .yee_ade_2d import _floating_dtype
from .yee_ade_2d import prepare_material_grid_2d


@dataclass(frozen=True)
class YeeADE2DTEzState:
  electric_x: np.ndarray
  electric_y: np.ndarray
  magnetic_z: np.ndarray
  displacement_x: np.ndarray
  displacement_y: np.ndarray
  displacement_x_previous: np.ndarray
  displacement_y_previous: np.ndarray
  electric_x_previous: np.ndarray
  electric_y_previous: np.ndarray
  polarization_x_current: np.ndarray = None
  polarization_x_previous: np.ndarray = None
  polarization_y_current: np.ndarray = None
  polarization_y_previous: np.ndarray = None
  cpml_dx_y: np.ndarray = None
  cpml_hz_y: np.ndarray = None
  step: int = 0


@dataclass(frozen=True)
class YeeADE2DTEzResult:
  electric_x: np.ndarray
  electric_y: np.ndarray
  magnetic_z: np.ndarray
  displacement_x: np.ndarray
  displacement_y: np.ndarray
  steps: np.ndarray
  final_state: YeeADE2DTEzState


def initialize_yee_ade_2d_tez(grid, dtype=None):
  if not isinstance(grid, MaterialGrid2D):
    raise TypeError("grid must be a MaterialGrid2D instance.")
  if np.any(grid.electric_conductivity) or np.any(grid.magnetic_conductivity):
    raise ValueError("TEz reference does not yet support conductivity sponge.")
  dtype = grid.coefficients.dtype if dtype is None else _floating_dtype(dtype)
  zeros = np.zeros(grid.shape, dtype=dtype)
  pole_zeros = np.zeros(grid.shape + (grid.maximum_poles,), dtype=dtype)
  return YeeADE2DTEzState(
      electric_x=zeros.copy(), electric_y=zeros.copy(), magnetic_z=zeros.copy(),
      displacement_x=zeros.copy(), displacement_y=zeros.copy(),
      displacement_x_previous=zeros.copy(),
      displacement_y_previous=zeros.copy(), electric_x_previous=zeros.copy(),
      electric_y_previous=zeros.copy(),
      polarization_x_current=pole_zeros.copy(),
      polarization_x_previous=pole_zeros.copy(),
      polarization_y_current=pole_zeros.copy(),
      polarization_y_previous=pole_zeros.copy(),
      cpml_dx_y=zeros.copy(), cpml_hz_y=zeros.copy())


def _validate_state(grid, state):
  if not isinstance(state, YeeADE2DTEzState):
    raise TypeError("state must be a YeeADE2DTEzState instance.")
  dtype = grid.coefficients.dtype
  fields = tuple(_field_array(name, getattr(state, name), grid.shape, dtype)
                 for name in (
                     "electric_x", "electric_y", "magnetic_z",
                     "displacement_x", "displacement_y",
                     "displacement_x_previous", "displacement_y_previous",
                     "electric_x_previous", "electric_y_previous"))
  pole_shape = grid.shape + (grid.maximum_poles,)
  poles = tuple(
      np.zeros(pole_shape, dtype=dtype) if getattr(state, name) is None else
      _field_array(name, getattr(state, name), pole_shape, dtype)
      for name in (
          "polarization_x_current", "polarization_x_previous",
          "polarization_y_current", "polarization_y_previous"))
  cpml = tuple(
      np.zeros(grid.shape, dtype=dtype) if getattr(state, name) is None else
      _field_array(name, getattr(state, name), grid.shape, dtype)
      for name in ("cpml_dx_y", "cpml_hz_y"))
  return fields + poles + cpml


def _source(value, shape, dtype, name):
  if value is None:
    return None
  value = np.asarray(value, dtype=dtype)
  if value.ndim == 0:
    value = np.full(shape, value, dtype=dtype)
  if value.shape != shape or not np.all(np.isfinite(value)):
    raise ValueError(f"{name} must be finite and scalar or have shape {shape}.")
  return value


def _recover_electric(grid, displacement_next, displacement_current,
                      displacement_previous, electric_current,
                      electric_previous, polarization_current,
                      polarization_previous):
  a0, a1, a2, b1, b2 = np.moveaxis(grid.coefficients, -1, 0)
  electric_next = (
      a0 * displacement_next + a1 * displacement_current +
      a2 * displacement_previous + b1 * electric_current +
      b2 * electric_previous)
  if not grid.maximum_poles:
    return electric_next, polarization_current
  q_next, q_current, q_previous, r_current, r_previous = np.moveaxis(
      grid.pole_coefficients, -1, 0)
  known = (q_current * electric_current[..., None] +
           q_previous * electric_previous[..., None] +
           r_current * polarization_current +
           r_previous * polarization_previous)
  multipole_electric = (
      displacement_next - np.sum(known, axis=-1)) / (
          grid.multipole_eps_inf + np.sum(q_next, axis=-1))
  polarization_next = known + q_next * multipole_electric[..., None]
  return (np.where(grid.multipole_mask, multipole_electric, electric_next),
          np.where(grid.multipole_mask[..., None], polarization_next, 0.0))


def yee_ade_2d_tez_step(grid, state, current_density_x=None,
                        current_density_y=None, magnetic_current_z=None):
  """Advance one periodic TEz Yee-ADE step."""
  if not isinstance(grid, MaterialGrid2D):
    raise TypeError("grid must be a MaterialGrid2D instance.")
  if np.any(grid.electric_conductivity) or np.any(grid.magnetic_conductivity):
    raise ValueError("TEz reference does not yet support conductivity sponge.")
  (ex, ey, hz, dx_field, dy_field, dx_previous, dy_previous,
   ex_previous, ey_previous, px, px_previous, py,
   py_previous, cpml_dx_y, cpml_hz_y) = _validate_state(grid, state)
  dtype = grid.coefficients.dtype
  jx = _source(current_density_x, grid.shape, dtype, "current_density_x")
  jy = _source(current_density_y, grid.shape, dtype, "current_density_y")
  mz = _source(magnetic_current_z, grid.shape, dtype, "magnetic_current_z")

  magnetic_backward_y = (hz - np.roll(hz, 1, axis=1)) / grid.dy
  if grid.y_cpml is None:
    cpml_dx_y_next = np.zeros_like(cpml_dx_y)
    magnetic_backward_y_effective = magnetic_backward_y
  else:
    cpml_dx_y_next = (
        grid.y_cpml.b_e[None, :] * cpml_dx_y +
        grid.y_cpml.c_e[None, :] * magnetic_backward_y)
    magnetic_backward_y_effective = (
        grid.y_cpml.inverse_kappa_e[None, :] * magnetic_backward_y +
        cpml_dx_y_next)
  dx_next = dx_field + grid.dt * magnetic_backward_y_effective
  dy_next = dy_field - grid.courant_x * (
      hz - np.roll(hz, 1, axis=0))
  if jx is not None:
    dx_next -= grid.dt * jx
  if jy is not None:
    dy_next -= grid.dt * jy
  ex_next, px_next = _recover_electric(
      grid, dx_next, dx_field, dx_previous, ex, ex_previous, px, px_previous)
  ey_next, py_next = _recover_electric(
      grid, dy_next, dy_field, dy_previous, ey, ey_previous, py, py_previous)
  electric_forward_y = (
      np.roll(ex_next, -1, axis=1) - ex_next) / grid.dy
  if grid.y_cpml is None:
    cpml_hz_y_next = np.zeros_like(cpml_hz_y)
    electric_forward_y_effective = electric_forward_y
  else:
    cpml_hz_y_next = (
        grid.y_cpml.b_h[None, :] * cpml_hz_y +
        grid.y_cpml.c_h[None, :] * electric_forward_y)
    electric_forward_y_effective = (
        grid.y_cpml.inverse_kappa_h[None, :] * electric_forward_y +
        cpml_hz_y_next)
  hz_next = hz + grid.dt * electric_forward_y_effective
  hz_next -= grid.courant_x * (
      np.roll(ey_next, -1, axis=0) - ey_next)
  if mz is not None:
    hz_next -= grid.dt * mz
  return YeeADE2DTEzState(
      electric_x=np.asarray(ex_next, dtype=dtype),
      electric_y=np.asarray(ey_next, dtype=dtype),
      magnetic_z=np.asarray(hz_next, dtype=dtype),
      displacement_x=np.asarray(dx_next, dtype=dtype),
      displacement_y=np.asarray(dy_next, dtype=dtype),
      displacement_x_previous=dx_field,
      displacement_y_previous=dy_field,
      electric_x_previous=ex, electric_y_previous=ey,
      polarization_x_current=np.asarray(px_next, dtype=dtype),
      polarization_x_previous=px,
      polarization_y_current=np.asarray(py_next, dtype=dtype),
      polarization_y_previous=py,
      cpml_dx_y=np.asarray(cpml_dx_y_next, dtype=dtype),
      cpml_hz_y=np.asarray(cpml_hz_y_next, dtype=dtype),
      step=int(state.step) + 1)


def simulate_yee_ade_2d_tez(materials, material_ids, dx, dy, dt, num_steps,
                            current_density_x=None, current_density_y=None,
                            dtype=np.float64, initial_state=None,
                            record_interval=1, y_cpml=None,
                            magnetic_current_z=None):
  """Run and record the periodic TEz CPU reference solver."""
  if not isinstance(num_steps, (int, np.integer)) or isinstance(
      num_steps, (bool, np.bool_)) or num_steps < 0:
    raise ValueError("num_steps must be a non-negative integer.")
  if not isinstance(record_interval, (int, np.integer)) or isinstance(
      record_interval, (bool, np.bool_)) or record_interval <= 0:
    raise ValueError("record_interval must be a positive integer.")
  grid = prepare_material_grid_2d(
      materials, material_ids, dt, dx, dy, dtype, y_cpml=y_cpml)
  state = initialize_yee_ade_2d_tez(grid) if initial_state is None else initial_state
  _validate_state(grid, state)

  def source_history(value, name):
    if value is None:
      return None
    value = np.asarray(value, dtype=grid.coefficients.dtype)
    valid = ((), grid.shape, (num_steps,), (num_steps,) + grid.shape)
    if value.shape not in valid or not np.all(np.isfinite(value)):
      raise ValueError(f"{name} has an invalid shape or non-finite values.")
    return value

  sources = (source_history(current_density_x, "current_density_x"),
             source_history(current_density_y, "current_density_y"),
             source_history(magnetic_current_z, "magnetic_current_z"))
  records = [[], [], [], [], []]
  steps = []
  final_step = state.step + num_steps
  for time_index in range(num_steps):
    current = []
    for source in sources:
      if source is not None and source.shape in ((num_steps,),
                                                  (num_steps,) + grid.shape):
        current.append(source[time_index])
      else:
        current.append(source)
    state = yee_ade_2d_tez_step(grid, state, *current)
    if state.step % record_interval == 0 or state.step == final_step:
      for values, field in zip(records, (
          state.electric_x, state.electric_y, state.magnetic_z,
          state.displacement_x, state.displacement_y)):
        values.append(field.copy())
      steps.append(state.step)
  empty = (0,) + grid.shape
  stacked = [np.stack(values) if values else np.empty(
      empty, dtype=grid.coefficients.dtype) for values in records]
  return YeeADE2DTEzResult(
      *stacked, steps=np.asarray(steps, dtype=np.int64), final_state=state)


__all__ = [
    "YeeADE2DTEzResult", "YeeADE2DTEzState",
    "initialize_yee_ade_2d_tez", "simulate_yee_ade_2d_tez",
    "yee_ade_2d_tez_step",
]
