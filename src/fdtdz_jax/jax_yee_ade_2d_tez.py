"""JAX-compiled 2D TEz Yee-ADE backend."""

import functools
from typing import NamedTuple

import numpy as np
import jax
from jax import lax
from jax import numpy as jnp

from .jax_yee_ade_2d import JAXMaterialGrid2D
from .jax_yee_ade_2d import jax_material_grid_2d
from .yee_ade_2d import MaterialGrid2D
from .yee_ade_2d_tez import YeeADE2DTEzState


class JAXYeeADE2DTEzState(NamedTuple):
  electric_x: object
  electric_y: object
  magnetic_z: object
  displacement_x: object
  displacement_y: object
  displacement_x_previous: object
  displacement_y_previous: object
  electric_x_previous: object
  electric_y_previous: object
  polarization_x_current: object
  polarization_x_previous: object
  polarization_y_current: object
  polarization_y_previous: object
  cpml_dx_y: object
  cpml_hz_y: object
  step: object


def jax_yee_ade_2d_tez_state(state, grid, dtype=jnp.float32):
  """Copy a NumPy TEz reference state into JAX device arrays."""
  if not isinstance(state, YeeADE2DTEzState):
    raise TypeError("state must be a YeeADE2DTEzState instance.")
  if not isinstance(grid, MaterialGrid2D):
    raise TypeError("grid must be a MaterialGrid2D instance.")
  zeros = np.zeros(grid.shape)
  pole_zeros = np.zeros(grid.shape + (grid.maximum_poles,))
  value = lambda name, fallback: (
      fallback if getattr(state, name) is None else getattr(state, name))
  device = lambda array: jnp.asarray(array, dtype=dtype)
  return JAXYeeADE2DTEzState(
      electric_x=device(state.electric_x), electric_y=device(state.electric_y),
      magnetic_z=device(state.magnetic_z),
      displacement_x=device(state.displacement_x),
      displacement_y=device(state.displacement_y),
      displacement_x_previous=device(state.displacement_x_previous),
      displacement_y_previous=device(state.displacement_y_previous),
      electric_x_previous=device(state.electric_x_previous),
      electric_y_previous=device(state.electric_y_previous),
      polarization_x_current=device(value(
          "polarization_x_current", pole_zeros)),
      polarization_x_previous=device(value(
          "polarization_x_previous", pole_zeros)),
      polarization_y_current=device(value(
          "polarization_y_current", pole_zeros)),
      polarization_y_previous=device(value(
          "polarization_y_previous", pole_zeros)),
      cpml_dx_y=device(value("cpml_dx_y", zeros)),
      cpml_hz_y=device(value("cpml_hz_y", zeros)),
      step=jnp.asarray(state.step, dtype=jnp.int32))


def _recover_electric(grid, displacement_next, displacement_current,
                      displacement_previous, electric_current,
                      electric_previous, polarization_current,
                      polarization_previous):
  a0, a1, a2, b1, b2 = jnp.moveaxis(grid.coefficients, -1, 0)
  electric_next = (
      a0 * displacement_next + a1 * displacement_current +
      a2 * displacement_previous + b1 * electric_current +
      b2 * electric_previous)
  q_next, q_current, q_previous, r_current, r_previous = jnp.moveaxis(
      grid.pole_coefficients, -1, 0)
  known = (q_current * electric_current[..., None] +
           q_previous * electric_previous[..., None] +
           r_current * polarization_current +
           r_previous * polarization_previous)
  multipole_electric = (
      displacement_next - jnp.sum(known, axis=-1)) / (
          grid.multipole_eps_inf + jnp.sum(q_next, axis=-1))
  polarization_next = known + q_next * multipole_electric[..., None]
  return (jnp.where(grid.multipole_mask, multipole_electric, electric_next),
          jnp.where(grid.multipole_mask[..., None], polarization_next, 0.0))


@jax.jit
def jax_yee_ade_2d_tez_step(grid, state, current_density_x,
                            current_density_y, magnetic_current_z):
  """Advance one compiled TEz step using full-shape source arrays."""
  magnetic_backward_y = (
      state.magnetic_z - jnp.roll(state.magnetic_z, 1, axis=1)) / grid.dy
  cpml_dx_y_next = (
      grid.b_e[None, :] * state.cpml_dx_y +
      grid.c_e[None, :] * magnetic_backward_y)
  magnetic_backward_y_effective = (
      grid.inverse_kappa_e[None, :] * magnetic_backward_y + cpml_dx_y_next)
  displacement_x_next = (
      state.displacement_x + grid.dt * magnetic_backward_y_effective -
      grid.dt * current_density_x)
  displacement_y_next = (
      state.displacement_y - grid.dt / grid.dx *
      (state.magnetic_z - jnp.roll(state.magnetic_z, 1, axis=0)) -
      grid.dt * current_density_y)
  electric_x_next, polarization_x_next = _recover_electric(
      grid, displacement_x_next, state.displacement_x,
      state.displacement_x_previous, state.electric_x,
      state.electric_x_previous, state.polarization_x_current,
      state.polarization_x_previous)
  electric_y_next, polarization_y_next = _recover_electric(
      grid, displacement_y_next, state.displacement_y,
      state.displacement_y_previous, state.electric_y,
      state.electric_y_previous, state.polarization_y_current,
      state.polarization_y_previous)
  electric_forward_y = (
      jnp.roll(electric_x_next, -1, axis=1) - electric_x_next) / grid.dy
  cpml_hz_y_next = (
      grid.b_h[None, :] * state.cpml_hz_y +
      grid.c_h[None, :] * electric_forward_y)
  electric_forward_y_effective = (
      grid.inverse_kappa_h[None, :] * electric_forward_y + cpml_hz_y_next)
  magnetic_z_next = (
      state.magnetic_z + grid.dt * electric_forward_y_effective -
      grid.dt / grid.dx *
      (jnp.roll(electric_y_next, -1, axis=0) - electric_y_next) -
      grid.dt * magnetic_current_z)
  return JAXYeeADE2DTEzState(
      electric_x=electric_x_next, electric_y=electric_y_next,
      magnetic_z=magnetic_z_next, displacement_x=displacement_x_next,
      displacement_y=displacement_y_next,
      displacement_x_previous=state.displacement_x,
      displacement_y_previous=state.displacement_y,
      electric_x_previous=state.electric_x,
      electric_y_previous=state.electric_y,
      polarization_x_current=polarization_x_next,
      polarization_x_previous=state.polarization_x_current,
      polarization_y_current=polarization_y_next,
      polarization_y_previous=state.polarization_y_current,
      cpml_dx_y=cpml_dx_y_next, cpml_hz_y=cpml_hz_y_next,
      step=state.step + jnp.asarray(1, dtype=jnp.int32))


@functools.partial(
    jax.jit,
    static_argnames=("electric_source_y", "magnetic_source_y",
                     "reflection_probe_y", "transmission_probe_y"))
def simulate_jax_yee_ade_2d_tez_probes(
    grid, state, electric_waveform, magnetic_waveform, x_profile,
    electric_source_y, magnetic_source_y, reflection_probe_y,
    transmission_probe_y):
  """Run compact TEz Jx/Mz line sources and record two averaged Ex probes."""
  ny = state.electric_x.shape[1]
  indices = (electric_source_y, magnetic_source_y, reflection_probe_y,
             transmission_probe_y)
  if any(index < 0 or index >= ny for index in indices):
    raise ValueError("source and probe indices must lie inside the y extent.")

  def scan_step(current_state, amplitudes):
    jx = jnp.zeros_like(current_state.electric_x).at[
        :, electric_source_y].set(amplitudes[0] * x_profile)
    zero = jnp.zeros_like(jx)
    mz = zero.at[:, magnetic_source_y].set(amplitudes[1] * x_profile)
    next_state = jax_yee_ade_2d_tez_step(grid, current_state, jx, zero, mz)
    return next_state, (
        jnp.mean(next_state.electric_x[:, reflection_probe_y]),
        jnp.mean(next_state.electric_x[:, transmission_probe_y]))
  return lax.scan(scan_step, state, (electric_waveform, magnetic_waveform))


@functools.partial(
    jax.jit,
    static_argnames=("electric_source_y", "magnetic_source_y",
                     "reflection_probe_y", "transmission_probe_y"))
def simulate_jax_yee_ade_2d_tez_line_probes(
    grid, state, electric_waveform, magnetic_waveform, x_profile,
    electric_source_y, magnetic_source_y, reflection_probe_y,
    transmission_probe_y):
  """Run compact TEz sources and retain two complete x-directed Ex lines."""
  ny = state.electric_x.shape[1]
  indices = (electric_source_y, magnetic_source_y, reflection_probe_y,
             transmission_probe_y)
  if any(index < 0 or index >= ny for index in indices):
    raise ValueError("source and probe indices must lie inside the y extent.")

  def scan_step(current_state, amplitudes):
    jx = jnp.zeros_like(current_state.electric_x).at[
        :, electric_source_y].set(amplitudes[0] * x_profile)
    zero = jnp.zeros_like(jx)
    mz = zero.at[:, magnetic_source_y].set(amplitudes[1] * x_profile)
    next_state = jax_yee_ade_2d_tez_step(grid, current_state, jx, zero, mz)
    return next_state, (
        next_state.electric_x[:, reflection_probe_y],
        next_state.electric_x[:, transmission_probe_y])
  return lax.scan(scan_step, state, (electric_waveform, magnetic_waveform))


__all__ = [
    "JAXYeeADE2DTEzState", "jax_material_grid_2d",
    "jax_yee_ade_2d_tez_state", "jax_yee_ade_2d_tez_step",
    "simulate_jax_yee_ade_2d_tez_line_probes",
    "simulate_jax_yee_ade_2d_tez_probes",
]
