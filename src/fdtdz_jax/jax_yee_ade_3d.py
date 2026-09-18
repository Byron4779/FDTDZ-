"""JAX-compiled full-vector three-dimensional Yee-ADE backend."""

import functools
from typing import NamedTuple

import numpy as np

import jax
from jax import lax
from jax import numpy as jnp

from .yee_ade_3d import MaterialGrid3D
from .yee_ade_3d import YeeADE3DState
from .bloch_3d import plane_source_values


class JAXMaterialGrid3D(NamedTuple):
  coefficients: object
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
  dz: object
  phase_x: object = 1.0
  phase_y: object = 1.0


class JAXYeeADE3DState(NamedTuple):
  electric: object
  magnetic: object
  displacement: object
  displacement_previous: object
  electric_previous: object
  cpml_hx_ey_z: object
  cpml_hy_ex_z: object
  cpml_dx_hy_z: object
  cpml_dy_hx_z: object
  polarization_current: object
  polarization_previous: object
  step: object


def jax_material_grid_3d(grid, dtype=jnp.float32):
  """Copy a validated NumPy 3D material grid to the active JAX device."""
  if not isinstance(grid, MaterialGrid3D):
    raise TypeError("grid must be a MaterialGrid3D instance.")
  if grid.z_cpml is None:
    inverse_kappa_e = inverse_kappa_h = np.ones(grid.shape[2])
    b_e = c_e = b_h = c_h = np.zeros(grid.shape[2])
  else:
    inverse_kappa_e = grid.z_cpml.inverse_kappa_e
    inverse_kappa_h = grid.z_cpml.inverse_kappa_h
    b_e, c_e = grid.z_cpml.b_e, grid.z_cpml.c_e
    b_h, c_h = grid.z_cpml.b_h, grid.z_cpml.c_h
  device = lambda value: jnp.asarray(value, dtype=dtype)
  return JAXMaterialGrid3D(
      coefficients=device(grid.coefficients),
      multipole_mask=jnp.asarray(grid.multipole_mask, dtype=bool),
      multipole_eps_inf=device(grid.multipole_eps_inf),
      pole_coefficients=device(grid.pole_coefficients),
      inverse_kappa_e=device(inverse_kappa_e), b_e=device(b_e), c_e=device(c_e),
      inverse_kappa_h=device(inverse_kappa_h), b_h=device(b_h), c_h=device(c_h),
      dt=device(grid.dt), dx=device(grid.dx), dy=device(grid.dy),
      dz=device(grid.dz),
      phase_x=jnp.asarray(np.exp(1j * grid.bloch_wavevector[0] * grid.shape[0] * grid.dx)
                          if any(grid.bloch_wavevector) else 1.0,
                          dtype=jnp.result_type(dtype, jnp.complex64)
                          if any(grid.bloch_wavevector) else dtype),
      phase_y=jnp.asarray(np.exp(1j * grid.bloch_wavevector[1] * grid.shape[1] * grid.dy)
                          if any(grid.bloch_wavevector) else 1.0,
                          dtype=jnp.result_type(dtype, jnp.complex64)
                          if any(grid.bloch_wavevector) else dtype))


def jax_yee_ade_3d_state(state, grid, dtype=jnp.float32):
  """Copy a NumPy 3D reference state into the JAX state PyTree."""
  if not isinstance(state, YeeADE3DState):
    raise TypeError("state must be a YeeADE3DState instance.")
  if not isinstance(grid, MaterialGrid3D):
    raise TypeError("grid must be a MaterialGrid3D instance.")
  if any(grid.bloch_wavevector) or np.iscomplexobj(state.electric):
    dtype = jnp.result_type(dtype, jnp.complex64)
  spatial_zeros = np.zeros(grid.shape)
  pole_zeros = np.zeros(grid.vector_shape + (grid.maximum_poles,))
  value = lambda name, fallback: (
      fallback if getattr(state, name) is None else getattr(state, name))
  device = lambda array: jnp.asarray(array, dtype=dtype)
  return JAXYeeADE3DState(
      electric=device(state.electric), magnetic=device(state.magnetic),
      displacement=device(state.displacement),
      displacement_previous=device(state.displacement_previous),
      electric_previous=device(state.electric_previous),
      cpml_hx_ey_z=device(value("cpml_hx_ey_z", spatial_zeros)),
      cpml_hy_ex_z=device(value("cpml_hy_ex_z", spatial_zeros)),
      cpml_dx_hy_z=device(value("cpml_dx_hy_z", spatial_zeros)),
      cpml_dy_hx_z=device(value("cpml_dy_hx_z", spatial_zeros)),
      polarization_current=device(value("polarization_current", pole_zeros)),
      polarization_previous=device(value("polarization_previous", pole_zeros)),
      step=jnp.asarray(state.step, dtype=jnp.int32))


def numpy_yee_ade_3d_state(state):
  """Materialize a JAX state as the public NumPy reference state type."""
  if not isinstance(state, JAXYeeADE3DState):
    raise TypeError("state must be a JAXYeeADE3DState instance.")
  return YeeADE3DState(
      electric=np.asarray(state.electric), magnetic=np.asarray(state.magnetic),
      displacement=np.asarray(state.displacement),
      displacement_previous=np.asarray(state.displacement_previous),
      electric_previous=np.asarray(state.electric_previous),
      cpml_hx_ey_z=np.asarray(state.cpml_hx_ey_z),
      cpml_hy_ex_z=np.asarray(state.cpml_hy_ex_z),
      cpml_dx_hy_z=np.asarray(state.cpml_dx_hy_z),
      cpml_dy_hx_z=np.asarray(state.cpml_dy_hx_z),
      polarization_current=np.asarray(state.polarization_current),
      polarization_previous=np.asarray(state.polarization_previous),
      step=int(state.step))


def _forward(field, axis, spacing, phase=1.0):
  neighbor = jnp.roll(field, -1, axis=axis)
  if jnp.issubdtype(field.dtype, jnp.complexfloating) and axis < 2:
    edge = [slice(None)] * field.ndim
    edge[axis] = -1
    neighbor = neighbor.at[tuple(edge)].multiply(phase)
  return (neighbor - field) / spacing


def _backward(field, axis, spacing, phase=1.0):
  neighbor = jnp.roll(field, 1, axis=axis)
  if jnp.issubdtype(field.dtype, jnp.complexfloating) and axis < 2:
    edge = [slice(None)] * field.ndim
    edge[axis] = 0
    neighbor = neighbor.at[tuple(edge)].divide(phase)
  return (field - neighbor) / spacing


@jax.jit
def jax_yee_ade_3d_step(grid, state, electric_current, magnetic_current):
  """Advance one compiled full-vector step with full-shape J and M arrays."""
  ex, ey, ez = state.electric
  hx, hy, hz = state.magnetic
  ey_forward_z = _forward(ey, 2, grid.dz)
  ex_forward_z = _forward(ex, 2, grid.dz)
  cpml_hx_ey_z_next = (
      grid.b_h[None, None, :] * state.cpml_hx_ey_z +
      grid.c_h[None, None, :] * ey_forward_z)
  cpml_hy_ex_z_next = (
      grid.b_h[None, None, :] * state.cpml_hy_ex_z +
      grid.c_h[None, None, :] * ex_forward_z)
  ey_forward_z_effective = (
      grid.inverse_kappa_h[None, None, :] * ey_forward_z +
      cpml_hx_ey_z_next)
  ex_forward_z_effective = (
      grid.inverse_kappa_h[None, None, :] * ex_forward_z +
      cpml_hy_ex_z_next)

  hx_next = hx - grid.dt * (
      _forward(ez, 1, grid.dy, grid.phase_y) - ey_forward_z_effective)
  hy_next = hy - grid.dt * (
      ex_forward_z_effective - _forward(ez, 0, grid.dx, grid.phase_x))
  hz_next = hz - grid.dt * (
      _forward(ey, 0, grid.dx, grid.phase_x) - _forward(ex, 1, grid.dy, grid.phase_y))
  magnetic_next = jnp.stack((hx_next, hy_next, hz_next))
  magnetic_next = magnetic_next - grid.dt * magnetic_current
  hx_next, hy_next, hz_next = magnetic_next

  hy_backward_z = _backward(hy_next, 2, grid.dz)
  hx_backward_z = _backward(hx_next, 2, grid.dz)
  cpml_dx_hy_z_next = (
      grid.b_e[None, None, :] * state.cpml_dx_hy_z +
      grid.c_e[None, None, :] * hy_backward_z)
  cpml_dy_hx_z_next = (
      grid.b_e[None, None, :] * state.cpml_dy_hx_z +
      grid.c_e[None, None, :] * hx_backward_z)
  hy_backward_z_effective = (
      grid.inverse_kappa_e[None, None, :] * hy_backward_z +
      cpml_dx_hy_z_next)
  hx_backward_z_effective = (
      grid.inverse_kappa_e[None, None, :] * hx_backward_z +
      cpml_dy_hx_z_next)
  displacement_next = jnp.stack((
      state.displacement[0] + grid.dt * (
          _backward(hz_next, 1, grid.dy, grid.phase_y) - hy_backward_z_effective),
      state.displacement[1] + grid.dt * (
          hx_backward_z_effective - _backward(hz_next, 0, grid.dx, grid.phase_x)),
      state.displacement[2] + grid.dt * (
          _backward(hy_next, 0, grid.dx, grid.phase_x) -
          _backward(hx_next, 1, grid.dy, grid.phase_y))))
  displacement_next = displacement_next - grid.dt * electric_current

  a0, a1, a2, b1, b2 = jnp.moveaxis(grid.coefficients, -1, 0)
  electric_next = (
      a0 * displacement_next + a1 * state.displacement +
      a2 * state.displacement_previous + b1 * state.electric +
      b2 * state.electric_previous)
  q_next, q_current, q_previous, r_current, r_previous = jnp.moveaxis(
      grid.pole_coefficients, -1, 0)
  known_polarization = (
      q_current * state.electric[..., None] +
      q_previous * state.electric_previous[..., None] +
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
  return JAXYeeADE3DState(
      electric=electric_next, magnetic=magnetic_next,
      displacement=displacement_next,
      displacement_previous=state.displacement,
      electric_previous=state.electric,
      cpml_hx_ey_z=cpml_hx_ey_z_next,
      cpml_hy_ex_z=cpml_hy_ex_z_next,
      cpml_dx_hy_z=cpml_dx_hy_z_next,
      cpml_dy_hx_z=cpml_dy_hx_z_next,
      polarization_current=polarization_next,
      polarization_previous=state.polarization_current,
      step=state.step + jnp.asarray(1, dtype=jnp.int32))


@jax.jit
def simulate_jax_yee_ade_3d(grid, state, electric_current, magnetic_current):
  """Run a compiled scan and record full E/H/D histories."""
  def scan_step(current_state, sources):
    next_state = jax_yee_ade_3d_step(
        grid, current_state, sources[0], sources[1])
    return next_state, (
        next_state.electric, next_state.magnetic, next_state.displacement)
  return lax.scan(scan_step, state, (electric_current, magnetic_current))


@jax.jit
def advance_jax_yee_ade_3d(grid, state, electric_current, magnetic_current):
  """Run full source histories while retaining only the final state."""
  def body(index, current_state):
    return jax_yee_ade_3d_step(
        grid, current_state, electric_current[index], magnetic_current[index])
  return lax.fori_loop(0, electric_current.shape[0], body, state)


@functools.partial(
    jax.jit,
    static_argnames=("electric_source_z", "magnetic_source_z",
                     "reflection_probe_z", "transmission_probe_z"))
def simulate_jax_yee_ade_3d_plane_probes(
    grid, state, electric_waveform, magnetic_waveform,
    transverse_profile, polarization_xy, electric_source_z,
    magnetic_source_z, reflection_probe_z, transmission_probe_z):
  """Run compact Huygens planes and retain two complete Ex/Ey planes."""
  nz = state.electric.shape[3]
  indices = (electric_source_z, magnetic_source_z,
             reflection_probe_z, transmission_probe_z)
  if any(index < 0 or index >= nz for index in indices):
    raise ValueError("source and probe indices must lie inside the z extent.")
  px, py = polarization_xy

  def scan_step(current_state, amplitudes):
    jx, jy, mx, my = plane_source_values(
        amplitudes[0], amplitudes[1], transverse_profile, polarization_xy)
    electric_source = jnp.zeros_like(current_state.electric)
    electric_source = electric_source.at[0, :, :, electric_source_z].set(jx)
    electric_source = electric_source.at[1, :, :, electric_source_z].set(jy)
    magnetic_source = jnp.zeros_like(current_state.magnetic)
    magnetic_source = magnetic_source.at[0, :, :, magnetic_source_z].set(mx)
    magnetic_source = magnetic_source.at[1, :, :, magnetic_source_z].set(my)
    next_state = jax_yee_ade_3d_step(
        grid, current_state, electric_source, magnetic_source)
    return next_state, (
        next_state.electric[:2, :, :, reflection_probe_z],
        next_state.electric[:2, :, :, transmission_probe_z])
  return lax.scan(
      scan_step, state, (electric_waveform, magnetic_waveform))


@functools.partial(
    jax.jit,
    static_argnames=("electric_source_z", "magnetic_source_z",
                     "reflection_probe_z", "transmission_probe_z",
                     "snapshot_steps"))
def simulate_jax_yee_ade_3d_plane_probes_with_snapshots(
    grid, state, electric_waveform, magnetic_waveform,
    transverse_profile, polarization_xy, electric_source_z,
    magnetic_source_z, reflection_probe_z, transmission_probe_z,
    snapshot_steps):
  """Run compact plane probes and retain full E/H/D only at selected steps."""
  snapshot_steps = tuple(snapshot_steps)
  num_steps = electric_waveform.shape[0]
  if (not snapshot_steps or
      any(isinstance(step, (bool, np.bool_)) or
          not isinstance(step, (int, np.integer))
          for step in snapshot_steps) or
      tuple(sorted(set(snapshot_steps))) != snapshot_steps or
      snapshot_steps[0] < 1 or snapshot_steps[-1] > num_steps):
    raise ValueError(
        "snapshot_steps must be a non-empty, strictly increasing tuple of "
        f"integers in [1, {num_steps}].")
  nz = state.electric.shape[3]
  indices = (electric_source_z, magnetic_source_z,
             reflection_probe_z, transmission_probe_z)
  if any(index < 0 or index >= nz for index in indices):
    raise ValueError("source and probe indices must lie inside the z extent.")
  px, py = polarization_xy
  num_snapshots = len(snapshot_steps)
  snapshot_values = jnp.asarray(snapshot_steps, dtype=jnp.int32)
  empty_snapshots = (
      jnp.zeros((num_snapshots,) + state.electric.shape,
                dtype=state.electric.dtype),
      jnp.zeros((num_snapshots,) + state.magnetic.shape,
                dtype=state.magnetic.dtype),
      jnp.zeros((num_snapshots,) + state.displacement.shape,
                dtype=state.displacement.dtype))

  def scan_step(carry, inputs):
    current_state, snapshot_buffers = carry
    time_index, electric_amplitude, magnetic_amplitude = inputs
    jx, jy, mx, my = plane_source_values(
        electric_amplitude, magnetic_amplitude, transverse_profile, polarization_xy)
    electric_source = jnp.zeros_like(current_state.electric)
    electric_source = electric_source.at[0, :, :, electric_source_z].set(jx)
    electric_source = electric_source.at[1, :, :, electric_source_z].set(jy)
    magnetic_source = jnp.zeros_like(current_state.magnetic)
    magnetic_source = magnetic_source.at[0, :, :, magnetic_source_z].set(mx)
    magnetic_source = magnetic_source.at[1, :, :, magnetic_source_z].set(my)
    next_state = jax_yee_ade_3d_step(
        grid, current_state, electric_source, magnetic_source)
    relative_step = time_index + jnp.asarray(1, dtype=jnp.int32)
    matches = relative_step == snapshot_values
    should_record = jnp.any(matches)
    slot = jnp.argmax(matches)

    def record(buffers):
      electric, magnetic, displacement = buffers
      return (
          electric.at[slot].set(next_state.electric),
          magnetic.at[slot].set(next_state.magnetic),
          displacement.at[slot].set(next_state.displacement))

    next_buffers = lax.cond(
        should_record, record, lambda buffers: buffers, snapshot_buffers)
    probes = (
        next_state.electric[:2, :, :, reflection_probe_z],
        next_state.electric[:2, :, :, transmission_probe_z])
    return (next_state, next_buffers), probes

  indices = jnp.arange(num_steps, dtype=jnp.int32)
  (final_state, snapshots), probes = lax.scan(
      scan_step, (state, empty_snapshots),
      (indices, electric_waveform, magnetic_waveform))
  return final_state, probes, snapshots


@functools.partial(
    jax.jit,
    static_argnames=("electric_source_z", "magnetic_source_z",
                     "reflection_probe_z", "transmission_probe_z",
                     "snapshot_steps"))
def simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields(
    grid, state, electric_waveform, magnetic_waveform,
    transverse_profile, polarization_xy, frequencies, frequency_weights,
    electric_source_z, magnetic_source_z, reflection_probe_z,
    transmission_probe_z, snapshot_steps=()):
  """Retain plane probes, selected time snapshots, and online complex fields."""
  snapshot_steps = tuple(snapshot_steps)
  num_steps = electric_waveform.shape[0]
  if frequencies.ndim != 1 or frequencies.shape[0] == 0:
    raise ValueError("frequencies must be a non-empty one-dimensional array.")
  if frequency_weights.shape != (num_steps,):
    raise ValueError("frequency_weights must contain one value per time step.")
  if (snapshot_steps and
      (any(isinstance(step, (bool, np.bool_)) or
           not isinstance(step, (int, np.integer))
           for step in snapshot_steps) or
       tuple(sorted(set(snapshot_steps))) != snapshot_steps or
       snapshot_steps[0] < 1 or snapshot_steps[-1] > num_steps)):
    raise ValueError(
        "snapshot_steps must be strictly increasing integers in the run.")
  nz = state.electric.shape[3]
  source_probe_indices = (
      electric_source_z, magnetic_source_z,
      reflection_probe_z, transmission_probe_z)
  if any(index < 0 or index >= nz for index in source_probe_indices):
    raise ValueError("source and probe indices must lie inside the z extent.")
  px, py = polarization_xy
  num_snapshots = len(snapshot_steps)
  snapshot_values = jnp.asarray(snapshot_steps, dtype=jnp.int32)
  snapshot_buffers = (
      jnp.zeros((num_snapshots,) + state.electric.shape,
                dtype=state.electric.dtype),
      jnp.zeros((num_snapshots,) + state.magnetic.shape,
                dtype=state.magnetic.dtype),
      jnp.zeros((num_snapshots,) + state.displacement.shape,
                dtype=state.displacement.dtype))
  num_frequencies = frequencies.shape[0]
  complex_dtype = jnp.result_type(state.electric.dtype, jnp.complex64)
  frequency_buffers = (
      jnp.zeros((num_frequencies,) + state.electric.shape,
                dtype=complex_dtype),
      jnp.zeros((num_frequencies,) + state.magnetic.shape,
                dtype=complex_dtype),
      jnp.zeros((num_frequencies,) + state.displacement.shape,
                dtype=complex_dtype))

  def scan_step(carry, inputs):
    current_state, current_snapshots, current_frequencies = carry
    time_index, electric_amplitude, magnetic_amplitude = inputs
    jx, jy, mx, my = plane_source_values(
        electric_amplitude, magnetic_amplitude, transverse_profile, polarization_xy)
    electric_source = jnp.zeros_like(current_state.electric)
    electric_source = electric_source.at[0, :, :, electric_source_z].set(jx)
    electric_source = electric_source.at[1, :, :, electric_source_z].set(jy)
    magnetic_source = jnp.zeros_like(current_state.magnetic)
    magnetic_source = magnetic_source.at[0, :, :, magnetic_source_z].set(mx)
    magnetic_source = magnetic_source.at[1, :, :, magnetic_source_z].set(my)
    next_state = jax_yee_ade_3d_step(
        grid, current_state, electric_source, magnetic_source)
    relative_step = time_index + jnp.asarray(1, dtype=jnp.int32)

    if snapshot_steps:
      matches = relative_step == snapshot_values
      should_record = jnp.any(matches)
      slot = jnp.argmax(matches)

      def record(buffers):
        electric, magnetic, displacement = buffers
        return (
            electric.at[slot].set(next_state.electric),
            magnetic.at[slot].set(next_state.magnetic),
            displacement.at[slot].set(next_state.displacement))
      next_snapshots = lax.cond(
          should_record, record, lambda buffers: buffers, current_snapshots)
    else:
      next_snapshots = current_snapshots

    electric_time = next_state.step * grid.dt
    magnetic_time = (next_state.step - 0.5) * grid.dt
    scale = grid.dt * frequency_weights[time_index]
    electric_phase = scale * jnp.exp(
        2j * jnp.pi * frequencies * electric_time)
    magnetic_phase = scale * jnp.exp(
        2j * jnp.pi * frequencies * magnetic_time)
    phase_shape = (num_frequencies,) + (1,) * 4
    frequency_electric, frequency_magnetic, frequency_displacement = (
        current_frequencies)
    next_frequencies = (
        frequency_electric +
        electric_phase.reshape(phase_shape) * next_state.electric,
        frequency_magnetic +
        magnetic_phase.reshape(phase_shape) * next_state.magnetic,
        frequency_displacement +
        electric_phase.reshape(phase_shape) * next_state.displacement)
    probes = (
        next_state.electric[:2, :, :, reflection_probe_z],
        next_state.electric[:2, :, :, transmission_probe_z])
    return (next_state, next_snapshots, next_frequencies), probes

  time_indices = jnp.arange(num_steps, dtype=jnp.int32)
  (final_state, snapshots, frequency_fields), probes = lax.scan(
      scan_step, (state, snapshot_buffers, frequency_buffers),
      (time_indices, electric_waveform, magnetic_waveform))
  return final_state, probes, snapshots, frequency_fields


__all__ = [
    "JAXMaterialGrid3D", "JAXYeeADE3DState", "advance_jax_yee_ade_3d",
    "jax_material_grid_3d", "jax_yee_ade_3d_state", "jax_yee_ade_3d_step",
    "numpy_yee_ade_3d_state", "simulate_jax_yee_ade_3d",
    "simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields",
    "simulate_jax_yee_ade_3d_plane_probes",
    "simulate_jax_yee_ade_3d_plane_probes_with_snapshots",
]
