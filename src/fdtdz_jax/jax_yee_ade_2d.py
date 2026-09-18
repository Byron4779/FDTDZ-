"""JAX-compiled 2D TMz Yee-ADE backend.

The array update mirrors :mod:`fdtdz_jax.yee_ade_2d` and is device agnostic:
JAX places it on CPU, GPU, or accelerator according to the active backend.
"""

import functools
from typing import NamedTuple

import numpy as np

import jax
from jax import lax
from jax import numpy as jnp

from .yee_ade_2d import MaterialGrid2D
from .yee_ade_2d import YeeADE2DState


class JAXMaterialGrid2D(NamedTuple):
  coefficients: object
  electric_conductivity: object
  magnetic_conductivity: object
  multipole_mask: object
  multipole_eps_inf: object
  pole_coefficients: object
  inverse_kappa_e: object
  b_e: object
  c_e: object
  inverse_kappa_h: object
  b_h: object
  c_h: object
  dt: object
  dx: object
  dy: object


class JAXYeeADE2DState(NamedTuple):
  electric_z: object
  magnetic_x: object
  magnetic_y: object
  displacement_z: object
  displacement_z_previous: object
  electric_z_previous: object
  cpml_hx_y: object
  cpml_dz_y: object
  polarization_current: object
  polarization_previous: object
  step: object


def jax_material_grid_2d(grid, dtype=jnp.float32):
  """Copy a validated NumPy material grid into JAX device arrays."""
  if not isinstance(grid, MaterialGrid2D):
    raise TypeError("grid must be a MaterialGrid2D instance.")
  if grid.y_cpml is None:
    inverse_kappa_e = np.ones(grid.shape[1])
    inverse_kappa_h = np.ones(grid.shape[1])
    b_e = c_e = b_h = c_h = np.zeros(grid.shape[1])
  else:
    inverse_kappa_e = grid.y_cpml.inverse_kappa_e
    inverse_kappa_h = grid.y_cpml.inverse_kappa_h
    b_e, c_e = grid.y_cpml.b_e, grid.y_cpml.c_e
    b_h, c_h = grid.y_cpml.b_h, grid.y_cpml.c_h
  as_device = lambda value: jnp.asarray(value, dtype=dtype)
  return JAXMaterialGrid2D(
      coefficients=as_device(grid.coefficients),
      electric_conductivity=as_device(grid.electric_conductivity),
      magnetic_conductivity=as_device(grid.magnetic_conductivity),
      multipole_mask=jnp.asarray(grid.multipole_mask, dtype=bool),
      multipole_eps_inf=as_device(grid.multipole_eps_inf),
      pole_coefficients=as_device(grid.pole_coefficients),
      inverse_kappa_e=as_device(inverse_kappa_e), b_e=as_device(b_e),
      c_e=as_device(c_e), inverse_kappa_h=as_device(inverse_kappa_h),
      b_h=as_device(b_h), c_h=as_device(c_h),
      dt=as_device(grid.dt), dx=as_device(grid.dx), dy=as_device(grid.dy))


def jax_yee_ade_2d_state(state, grid, dtype=jnp.float32):
  """Copy a NumPy reference state into the JAX state PyTree."""
  if not isinstance(state, YeeADE2DState):
    raise TypeError("state must be a YeeADE2DState instance.")
  if not isinstance(grid, MaterialGrid2D):
    raise TypeError("grid must be a MaterialGrid2D instance.")
  zeros = np.zeros(grid.shape)
  pole_zeros = np.zeros(grid.shape + (grid.maximum_poles,))
  value = lambda name, fallback: (
      fallback if getattr(state, name) is None else getattr(state, name))
  as_device = lambda array: jnp.asarray(array, dtype=dtype)
  return JAXYeeADE2DState(
      electric_z=as_device(state.electric_z),
      magnetic_x=as_device(state.magnetic_x),
      magnetic_y=as_device(state.magnetic_y),
      displacement_z=as_device(state.displacement_z),
      displacement_z_previous=as_device(state.displacement_z_previous),
      electric_z_previous=as_device(state.electric_z_previous),
      cpml_hx_y=as_device(value("cpml_hx_y", zeros)),
      cpml_dz_y=as_device(value("cpml_dz_y", zeros)),
      polarization_current=as_device(value(
          "polarization_current", pole_zeros)),
      polarization_previous=as_device(value(
          "polarization_previous", pole_zeros)),
      step=jnp.asarray(state.step, dtype=jnp.int32))


@jax.jit
def jax_yee_ade_2d_step(grid, state, current_density_z,
                        magnetic_current_x):
  """Advance one compiled TMz step; both sources are full-shape arrays."""
  electric_forward_y = (
      jnp.roll(state.electric_z, -1, axis=1) - state.electric_z) / grid.dy
  cpml_hx_y_next = (
      grid.b_h[None, :] * state.cpml_hx_y +
      grid.c_h[None, :] * electric_forward_y)
  electric_forward_y_effective = (
      grid.inverse_kappa_h[None, :] * electric_forward_y + cpml_hx_y_next)

  magnetic_denominator = 1.0 + 0.5 * grid.dt * grid.magnetic_conductivity
  magnetic_decay = (
      1.0 - 0.5 * grid.dt * grid.magnetic_conductivity) / (
          magnetic_denominator)
  magnetic_x_next = (
      magnetic_decay * state.magnetic_x -
      grid.dt / magnetic_denominator * electric_forward_y_effective -
      grid.dt / magnetic_denominator * magnetic_current_x)
  magnetic_y_next = (
      magnetic_decay * state.magnetic_y +
      (grid.dt / grid.dx) / magnetic_denominator *
      (jnp.roll(state.electric_z, -1, axis=0) - state.electric_z))

  displacement_raw = state.displacement_z + grid.dt / grid.dx * (
      magnetic_y_next - jnp.roll(magnetic_y_next, 1, axis=0))
  magnetic_backward_y = (
      magnetic_x_next - jnp.roll(magnetic_x_next, 1, axis=1)) / grid.dy
  cpml_dz_y_next = (
      grid.b_e[None, :] * state.cpml_dz_y +
      grid.c_e[None, :] * magnetic_backward_y)
  magnetic_backward_y_effective = (
      grid.inverse_kappa_e[None, :] * magnetic_backward_y + cpml_dz_y_next)
  displacement_raw = (
      displacement_raw - grid.dt * magnetic_backward_y_effective -
      grid.dt * current_density_z)

  a0, a1, a2, b1, b2 = jnp.moveaxis(grid.coefficients, -1, 0)
  electric_without_new_displacement = (
      a1 * state.displacement_z + a2 * state.displacement_z_previous +
      b1 * state.electric_z + b2 * state.electric_z_previous)
  half_conductivity_step = 0.5 * grid.dt * grid.electric_conductivity
  displacement_next = (
      displacement_raw - half_conductivity_step *
      (state.electric_z + electric_without_new_displacement)) / (
          1.0 + half_conductivity_step * a0)
  electric_next = (
      a0 * displacement_next + a1 * state.displacement_z +
      a2 * state.displacement_z_previous + b1 * state.electric_z +
      b2 * state.electric_z_previous)

  q_next, q_current, q_previous, r_current, r_previous = jnp.moveaxis(
      grid.pole_coefficients, -1, 0)
  known_polarization = (
      q_current * state.electric_z[..., None] +
      q_previous * state.electric_z_previous[..., None] +
      r_current * state.polarization_current +
      r_previous * state.polarization_previous)
  multipole_electric = (
      displacement_next - jnp.sum(known_polarization, axis=-1)) / (
          grid.multipole_eps_inf + jnp.sum(q_next, axis=-1))
  polarization_next = (
      known_polarization + q_next * multipole_electric[..., None])
  electric_next = jnp.where(
      grid.multipole_mask, multipole_electric, electric_next)
  polarization_next = jnp.where(
      grid.multipole_mask[..., None], polarization_next, 0.0)

  return JAXYeeADE2DState(
      electric_z=electric_next, magnetic_x=magnetic_x_next,
      magnetic_y=magnetic_y_next, displacement_z=displacement_next,
      displacement_z_previous=state.displacement_z,
      electric_z_previous=state.electric_z,
      cpml_hx_y=cpml_hx_y_next, cpml_dz_y=cpml_dz_y_next,
      polarization_current=polarization_next,
      polarization_previous=state.polarization_current,
      step=state.step + jnp.asarray(1, dtype=jnp.int32))


@jax.jit
def simulate_jax_yee_ade_2d(grid, state, current_density_z,
                            magnetic_current_x):
  """Run a compiled scan over full ``(time, x, y)`` source histories."""
  def scan_step(current_state, sources):
    next_state = jax_yee_ade_2d_step(
        grid, current_state, sources[0], sources[1])
    recorded = (next_state.electric_z, next_state.magnetic_x,
                next_state.magnetic_y, next_state.displacement_z)
    return next_state, recorded
  return lax.scan(scan_step, state, (current_density_z, magnetic_current_x))


@jax.jit
def advance_jax_yee_ade_2d(grid, state, current_density_z,
                           magnetic_current_x):
  """Run source histories and return only the final state.

  Use this for long production runs or benchmarks when full time histories are
  not required; it avoids allocating ``time * field`` output arrays.
  """
  def body(index, current_state):
    return jax_yee_ade_2d_step(
        grid, current_state, current_density_z[index],
        magnetic_current_x[index])
  return lax.fori_loop(0, current_density_z.shape[0], body, state)


@functools.partial(
    jax.jit,
    static_argnames=("source_y", "reflection_probe_y",
                     "transmission_probe_y"))
def simulate_jax_yee_ade_2d_probes(
    grid, state, electric_waveform, magnetic_waveform, x_profile,
    source_y, reflection_probe_y, transmission_probe_y):
  """Run a line source while recording only two x-averaged Ez probes.

  Unlike :func:`simulate_jax_yee_ade_2d`, this production-oriented path never
  materializes ``(time, x, y)`` source or field histories.  Its dynamic memory
  is the final solver state plus two ``(time,)`` probe arrays.
  """
  nx, ny = state.electric_z.shape
  if not (0 <= source_y < ny and 0 <= reflection_probe_y < ny and
          0 <= transmission_probe_y < ny):
    raise ValueError("source and probe indices must lie inside the y extent.")

  def scan_step(current_state, amplitudes):
    electric_source = jnp.zeros_like(current_state.electric_z).at[
        :, source_y].set(amplitudes[0] * x_profile)
    magnetic_source = jnp.zeros_like(current_state.electric_z).at[
        :, source_y].set(amplitudes[1] * x_profile)
    next_state = jax_yee_ade_2d_step(
        grid, current_state, electric_source, magnetic_source)
    probes = (
        jnp.mean(next_state.electric_z[:, reflection_probe_y]),
        jnp.mean(next_state.electric_z[:, transmission_probe_y]))
    return next_state, probes

  return lax.scan(
      scan_step, state, (electric_waveform, magnetic_waveform))


@functools.partial(
    jax.jit,
    static_argnames=("source_y", "reflection_probe_y",
                     "transmission_probe_y"))
def simulate_jax_yee_ade_2d_line_probes(
    grid, state, electric_waveform, magnetic_waveform, x_profile,
    source_y, reflection_probe_y, transmission_probe_y):
  """Run compact TMz sources and retain two complete x-directed Ez lines."""
  ny = state.electric_z.shape[1]
  if any(index < 0 or index >= ny for index in (
      source_y, reflection_probe_y, transmission_probe_y)):
    raise ValueError("source and probe indices must lie inside the y extent.")

  def scan_step(current_state, amplitudes):
    electric_source = jnp.zeros_like(current_state.electric_z).at[
        :, source_y].set(amplitudes[0] * x_profile)
    magnetic_source = jnp.zeros_like(current_state.electric_z).at[
        :, source_y].set(amplitudes[1] * x_profile)
    next_state = jax_yee_ade_2d_step(
        grid, current_state, electric_source, magnetic_source)
    return next_state, (
        next_state.electric_z[:, reflection_probe_y],
        next_state.electric_z[:, transmission_probe_y])
  return lax.scan(
      scan_step, state, (electric_waveform, magnetic_waveform))


__all__ = [
    "JAXMaterialGrid2D", "JAXYeeADE2DState", "advance_jax_yee_ade_2d",
    "jax_material_grid_2d",
    "jax_yee_ade_2d_state", "jax_yee_ade_2d_step",
    "simulate_jax_yee_ade_2d", "simulate_jax_yee_ade_2d_line_probes",
    "simulate_jax_yee_ade_2d_probes",
]
