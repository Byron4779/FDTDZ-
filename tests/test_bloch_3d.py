from dataclasses import replace

import numpy as np
import pytest

from fdtdz_jax import BoundarySpec3D, GridSpec3D, Layer3D, Lossless
from fdtdz_jax import PlaneWaveSource3D, ScatteringMonitor3D, Simulation3D
from fdtdz_jax import Debye, Drude, Lorentz, Multipole, DebyePole, LorentzPole
from fdtdz_jax import huygens_plane_source, prepare_material_grid_3d
from fdtdz_jax import initialize_yee_ade_3d, yee_ade_3d_step
from fdtdz_jax import jax_material_grid_3d, jax_yee_ade_3d_state, jax_yee_ade_3d_step
from fdtdz_jax.diffraction_3d import diffraction_spectrum_3d
from fdtdz_jax.yee_ade_3d import _forward, _backward


@pytest.mark.parametrize("axis,k", [(0, .27), (1, -.41)])
def test_bloch_differences_have_exact_plane_wave_symbols_at_seams(axis, k):
  shape, spacing = (5, 6, 7), .8
  coordinates = np.arange(shape[axis])*spacing
  profile_shape = [1, 1, 1]
  profile_shape[axis] = shape[axis]
  field = np.broadcast_to(np.exp(1j*k*coordinates).reshape(profile_shape), shape)
  phase = np.exp(1j*k*shape[axis]*spacing)
  np.testing.assert_allclose(_forward(field, axis, spacing, phase),
                            (np.exp(1j*k*spacing)-1)/spacing*field, atol=1e-14)
  np.testing.assert_allclose(_backward(field, axis, spacing, phase),
                            (1-np.exp(-1j*k*spacing))/spacing*field, atol=1e-14)


def test_zero_bloch_retains_real_arrays_and_original_step_exactly():
  args = ([Lossless(1)], np.zeros((3, 4, 5), np.uint8), .2, 1, 1, 1)
  old = prepare_material_grid_3d(*args, dtype=np.float32)
  new = prepare_material_grid_3d(*args, dtype=np.float32, bloch_wavevector=(0, 0))
  source = np.random.default_rng(9).normal(size=old.vector_shape).astype(np.float32)
  a = yee_ade_3d_step(old, initialize_yee_ade_3d(old), source)
  b = yee_ade_3d_step(new, initialize_yee_ade_3d(new), source)
  assert b.electric.dtype == np.float32
  np.testing.assert_array_equal(a.electric, b.electric)


@pytest.mark.parametrize("material", [Lossless(1.7), Debye(1.2, 2.1, 3),
    Lorentz(1.2, 2.1, .7, .05), Drude(1.2, .5, .04),
    Multipole(1.2, (DebyePole(.7, 3), LorentzPole(.8, .7, .05)))])
def test_complex_cpml_ade_numpy_and_jax_match_step_by_step(material):
  from fdtdz_jax import prepare_z_cpml
  shape = (4, 5, 16)
  ids = np.zeros(shape, np.uint8)
  ids[..., 6:9] = 1
  cpml = prepare_z_cpml(shape, 3, .2, 1, dtype=np.float32)
  grid = prepare_material_grid_3d([Lossless(1), material], ids, .2, 1, 1, 1,
                                  dtype=np.float32, z_cpml=cpml,
                                  bloch_wavevector=(.23, -.17))
  assert grid.coefficients.dtype == np.float32
  reference = initialize_yee_ade_3d(grid)
  assert reference.electric.dtype == np.complex64
  device_grid = jax_material_grid_3d(grid)
  device = jax_yee_ade_3d_state(reference, grid)
  rng = np.random.default_rng(17)
  for _ in range(12):
    sources = [(rng.normal(size=grid.vector_shape) +
                1j*rng.normal(size=grid.vector_shape)).astype(np.complex64)*.01
               for _ in range(2)]
    reference = yee_ade_3d_step(grid, reference, *sources)
    device = jax_yee_ade_3d_step(device_grid, device, *sources)
    for field in ('electric', 'magnetic', 'displacement', 'polarization_current',
                  'cpml_hx_ey_z', 'cpml_dx_hy_z'):
      np.testing.assert_allclose(np.asarray(getattr(device, field)),
                                getattr(reference, field), rtol=2e-5, atol=1e-7)


def test_oblique_sheet_currents_have_yee_phases_and_te_tm_admittance():
  frequency, kx, ky = .1, .2, -.1
  omega = 2*np.pi*frequency
  kz = np.sqrt(omega**2-kx**2-ky**2)
  source = huygens_plane_source(np.ones(3), (4, 5, 8), 3,
      polarization_xy=(1, 0), bloch_wavevector=(kx, ky),
      reference_frequency=frequency, spacing=(.5, .6, .7), expand=True)
  x, y = np.arange(4)[:, None]*.5, np.arange(5)[None, :]*.6
  phase_ex = np.exp(1j*(kx*(x+.25)+ky*y))
  phase_ey = np.exp(1j*(kx*x+ky*(y+.3)))
  np.testing.assert_allclose(source.electric_current[1, 0, :, :, 3],
                            (kz/omega+kx*kx/(omega*kz))*phase_ex)
  np.testing.assert_allclose(source.electric_current[1, 1, :, :, 3],
                            kx*ky/(omega*kz)*phase_ey)
  np.testing.assert_allclose(source.magnetic_current[1, 1, :, :, 2],
                            phase_ex*np.exp(-.5j*kz*.7))


def _synthetic_bloch_plane(n=256, nx=8, ny=6, kxy=(.2, -.1), order=(0, 0),
                           polarization=(1, 0), time_bin=24, dt=.2, spacing=.5):
  omega = 2*np.pi*time_bin/(n*dt)
  kx = kxy[0] + 2*np.pi*order[0]/(nx*spacing)
  ky = kxy[1] + 2*np.pi*order[1]/(ny*spacing)
  x, y = np.arange(nx)[:, None]*spacing, np.arange(ny)[None, :]*spacing
  spatial = np.asarray((polarization[0]*np.exp(1j*(kx*(x+.5*spacing)+ky*y)),
                        polarization[1]*np.exp(1j*(kx*x+ky*(y+.5*spacing)))))
  return np.exp(-1j*omega*np.arange(n)*dt)[:, None, None, None]*spatial[None, ...]


@pytest.mark.parametrize("polarization", [(1, 0), (0, 1), (1, 1)])
def test_bloch_diffraction_zero_order_flux_normalization(polarization):
  reference = _synthetic_bloch_plane(polarization=polarization)
  result = diffraction_spectrum_3d(reference, reference*.8, reference,
      reference*np.sqrt(.7), .2, .5, .5, bloch_wavevector=(.2, -.1))
  assert result.reflection[24] == pytest.approx(.04)
  assert result.transmission[24] == pytest.approx(.7)
  assert result.absorption[24] == pytest.approx(.26)
  assert result.transverse_wavevectors_x[0] == .2
  assert result.transverse_wavevectors_y[0] == -.1


def test_shifted_diffraction_order_flux_and_staggered_component_collocation():
  kxy, polarization, order = (.2, -.1), (1., .4), (1, 0)
  reference = _synthetic_bloch_plane(kxy=kxy, polarization=polarization)
  scattered = _synthetic_bloch_plane(kxy=kxy, polarization=polarization, order=order)*.3
  result = diffraction_spectrum_3d(reference, reference+scattered, reference,
                                   reference, .2, .5, .5, bloch_wavevector=kxy)
  omega = 2*np.pi*24/(256*.2)

  def flux(kx, ky):
    kz = np.sqrt(omega**2-kx**2-ky**2)
    return kz/omega*np.dot(polarization, polarization) + (
        kx*polarization[0]+ky*polarization[1])**2/(omega*kz)

  expected = .09*flux(.2+2*np.pi/4, -.1)/flux(.2, -.1)
  assert result.reflection_orders[24, 1, 0] == pytest.approx(expected)
  assert result.reflection[24] == pytest.approx(expected)
  assert np.count_nonzero(result.reflection_orders[24] > 1e-12) == 1


def _film_simulation(h, polarization, theta=np.pi/6, backend='numpy', direction=1):
  f = .06
  kx = 2*np.pi*f*np.sin(theta)
  nz = round(80/h)
  simulation = Simulation3D(GridSpec3D((2, 2, nz), (h, h, h), .4*h),
      BoundarySpec3D(round(8/h), bloch_wavevector=(kx, 0)), backend=backend,
      dtype=np.float64 if backend == 'numpy' else np.float32)
  simulation.add_material('film', Lossless(2.25))
  simulation.add_geometry(Layer3D(40, 44, 'film'))
  time = np.arange(round(140/(.4*h)))*.4*h
  waveform = np.exp(-.5*((time-30)/8)**2)*np.sin(2*np.pi*f*(time-30))
  source_z = round((16 if direction == 1 else 64)/h)
  simulation.set_source(PlaneWaveSource3D(waveform, source_z,
      polarization_xy=polarization, direction=direction, reference_frequency=f))
  simulation.set_monitors(ScatteringMonitor3D(
      round((25 if direction == 1 else 55)/h),
      round((60 if direction == 1 else 25)/h)))
  return simulation


@pytest.mark.parametrize('polarization', [(0, 1), (1, 0)])
def test_oblique_dielectric_film_converges_to_fresnel_te_and_tm(polarization):
  coarse = _film_simulation(.25, polarization).run()
  fine = _film_simulation(.125, polarization).run()

  def error(result):
    index = np.argmin(abs(result.spectrum.frequencies-.06))
    omega = 2*np.pi*result.spectrum.frequencies[index]
    kx = 2*np.pi*.06*.5
    kz1, kz2 = np.sqrt(omega**2-kx**2), np.sqrt(2.25*omega**2-kx**2)
    y1, y2 = ((kz1, kz2) if polarization[0] == 0 else (1/kz1, 2.25/kz2))
    r = (y1-y2)/(y1+y2)
    phase = np.exp(2j*kz2*4)
    expected = abs(r*(1-phase)/(1-r*r*phase))**2
    return abs(result.spectrum.reflection[index]-expected)

  a, b = error(coarse), error(fine)
  assert b < .01, (a, b)
  assert b < a, (a, b)


@pytest.mark.parametrize('kxy', [(np.nan, 0), (1j, 0), (True, False), (1,), (np.pi, 0)])
def test_invalid_bloch_wavevector_rejected(kxy):
  with pytest.raises(ValueError, match='bloch_wavevector|Bloch wavevector'):
    prepare_material_grid_3d([Lossless(1)], np.zeros((3, 3, 4), np.uint8),
                             .2, 1, 1, 1, bloch_wavevector=kxy)


def test_oblique_source_requires_propagating_reference_frequency():
  simulation = _film_simulation(.5, (1, 0))
  with pytest.raises(ValueError, match='reference_frequency'):
    replace(simulation._source, reference_frequency=None).build(
        simulation.grid, bloch_wavevector=simulation.boundary.bloch_wavevector)
  with pytest.raises(ValueError, match='propagating'):
    replace(simulation._source, reference_frequency=.001).build(
        simulation.grid, bloch_wavevector=simulation.boundary.bloch_wavevector)


def test_refinement_and_parameter_variants_preserve_bloch_and_reference_frequency():
  simulation = _film_simulation(.5, (1, 0))
  for variant in (simulation.refined(2), simulation.temporal_refined(2),
                  simulation.with_cpml_width(18)):
    assert variant.boundary.bloch_wavevector == simulation.boundary.bloch_wavevector
    assert variant._source.reference_frequency == .06


@pytest.mark.parametrize('direction', [1, -1])
def test_empty_oblique_cell_and_full_monitors_match_between_backends(direction, tmp_path):
  from fdtdz_jax import FrequencyMonitor3D
  import json

  simulation = _film_simulation(.5, (1, .4), direction=direction)
  # Include both nonzero transverse phases and all monitor variants.
  simulation.boundary = replace(simulation.boundary,
                                 bloch_wavevector=(.15, -.08))
  simulation.set_monitors(replace(simulation._monitors, snapshot_steps=(25, 100)))
  simulation.set_frequency_monitor(FrequencyMonitor3D((.05, .06)))
  numpy_result = simulation.run()
  jax_result = simulation.run(backend='jax')
  for name in ('electric', 'magnetic', 'displacement', 'cpml_dx_hy_z'):
    np.testing.assert_allclose(getattr(jax_result.final_state, name),
        getattr(numpy_result.final_state, name), rtol=3e-4, atol=2e-5)
  np.testing.assert_allclose(jax_result.snapshots.electric,
                            numpy_result.snapshots.electric, rtol=3e-4, atol=2e-5)
  np.testing.assert_allclose(jax_result.frequency_fields.electric,
                            numpy_result.frequency_fields.electric, rtol=3e-4, atol=2e-5)
  bins = np.flatnonzero(numpy_result.spectrum.valid &
      (abs(numpy_result.spectrum.frequencies-.06) < .01))
  np.testing.assert_allclose(jax_result.spectrum.reflection[bins],
                            numpy_result.spectrum.reflection[bins], atol=1e-5)
  path = jax_result.to_npz(tmp_path / 'bloch.npz')
  with np.load(path, allow_pickle=False) as archive:
    assert np.iscomplexobj(archive['final/electric'])
    np.testing.assert_array_equal(archive['spectrum/bloch_wavevector'], [.15, -.08])
    metadata = json.loads(str(archive['metadata_json']))
    assert 'Bloch quadratures' in metadata['complex_time_domain_fields']
  simulation._geometries.clear()
  empty = simulation.run()
  np.testing.assert_array_equal(empty.spectrum.reflection[empty.spectrum.valid], 0)
  np.testing.assert_allclose(empty.spectrum.transmission[empty.spectrum.valid], 1)


def test_evanescent_incident_order_invalid_even_if_other_orders_propagate():
  reference = _synthetic_bloch_plane(time_bin=1)
  spectrum = diffraction_spectrum_3d(reference, reference, reference, reference,
      .2, .5, .5, bloch_wavevector=(.2, -.1))
  assert not spectrum.valid[1]
  assert np.isnan(spectrum.reflection[1])
  assert np.isnan(spectrum.transmission[1])
  assert not np.any(spectrum.propagating[1])


def test_real_grid_rejects_complex_state_and_source_instead_of_dropping_imaginary_part():
  from fdtdz_jax import simulate_yee_ade_3d
  grid = prepare_material_grid_3d([Lossless(1)], np.zeros((3, 3, 4), np.uint8),
                                  .2, 1, 1, 1)
  state = initialize_yee_ade_3d(grid)
  with pytest.raises(TypeError, match='complex quadratures'):
    yee_ade_3d_step(grid, state, np.ones(grid.vector_shape, dtype=complex))
  with pytest.raises(TypeError, match='complex quadratures'):
    yee_ade_3d_step(grid, replace(state, electric=state.electric.astype(complex)))
  with pytest.raises(TypeError, match='complex quadratures'):
    simulate_yee_ade_3d([Lossless(1)], np.zeros((3, 3, 4), np.uint8),
                        1, 1, 1, .2, 8, electric_current=np.ones(8, dtype=complex))
