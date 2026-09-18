"""High-level configuration API for patterned x-y metasurface simulations.

Coordinates use the same normalized units as the lower-level Yee solver.  A
geometry is sampled independently at the Ex, Ey and Ez nodes; later geometry
objects override earlier ones where their masks overlap.
"""

from collections import OrderedDict
from dataclasses import dataclass, replace
import math

import numpy as np

from .materials import Debye
from .materials import Drude
from .materials import Lorentz
from .materials import Lossless
from .materials import Multipole
from .metasurface_3d import huygens_plane_source
from .scattering_workflow_3d import run_metasurface_scattering_3d


_SUPPORTED_MATERIALS = (Lossless, Debye, Lorentz, Drude, Multipole)
_COMPONENT_NAMES = {"Ex": ("electric", 0), "Ey": ("electric", 1),
                    "Ez": ("electric", 2), "Hx": ("magnetic", 0),
                    "Hy": ("magnetic", 1), "Hz": ("magnetic", 2),
                    "Dx": ("displacement", 0), "Dy": ("displacement", 1),
                    "Dz": ("displacement", 2),
                    "Px": ("polarization", 0), "Py": ("polarization", 1),
                    "Pz": ("polarization", 2)}
_YEE_OFFSETS = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))


def _positive_finite(name, value):
  if isinstance(value, (bool, np.bool_)) or not isinstance(
      value, (int, float, np.integer, np.floating)):
    raise TypeError(f"{name} must be a real number.")
  value = float(value)
  if not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"{name} must be finite and greater than zero.")
  return value


def _real_tuple(name, value, length, positive=False):
  try:
    values = tuple(value)
  except TypeError as error:
    raise TypeError(f"{name} must contain {length} real values.") from error
  if len(values) != length:
    raise ValueError(f"{name} must contain {length} values.")
  result = []
  for item in values:
    if isinstance(item, (bool, np.bool_)) or not isinstance(
        item, (int, float, np.integer, np.floating)):
      raise TypeError(f"{name} must contain real values.")
    item = float(item)
    if not math.isfinite(item) or (positive and item <= 0.0):
      qualifier = "finite positive" if positive else "finite"
      raise ValueError(f"{name} must contain {qualifier} values.")
    result.append(item)
  return tuple(result)


def _material_name(value):
  if not isinstance(value, str) or not value.strip():
    raise ValueError("material must be a non-empty material name.")
  return value.strip()


def _refinement_factor(value):
  if isinstance(value, (bool, np.bool_)) or not isinstance(
      value, (int, np.integer)):
    raise TypeError("factor must be an integer.")
  value = int(value)
  if value < 1:
    raise ValueError("factor must be at least one.")
  return value


def _resample_waveform(waveform, old_dt, factor):
  """Linearly resample source samples while preserving total run time."""
  waveform = np.asarray(waveform)
  new_dt = old_dt / factor
  old_times = np.arange(waveform.size, dtype=float) * old_dt
  new_times = np.arange(waveform.size * factor, dtype=float) * new_dt
  return np.interp(new_times, old_times, waveform, left=0.0, right=0.0)


def _resample_periodic_profile(profile, factor):
  """Periodically resample a transverse source profile on a refined grid."""
  if profile is None:
    return None
  profile = np.asarray(profile, dtype=float)
  nx, ny = profile.shape
  new_nx, new_ny = nx * factor, ny * factor
  x_old = np.arange(nx + 1, dtype=float)
  x_new = np.arange(new_nx, dtype=float) / factor
  along_x = np.empty((new_nx, ny), dtype=float)
  for index in range(ny):
    along_x[:, index] = np.interp(
        x_new, x_old, np.concatenate((profile[:, index], profile[:1, index])))
  y_old = np.arange(ny + 1, dtype=float)
  y_new = np.arange(new_ny, dtype=float) / factor
  result = np.empty((new_nx, new_ny), dtype=float)
  for index in range(new_nx):
    result[index] = np.interp(
        y_new, y_old, np.concatenate((along_x[index], along_x[index, :1])))
  return result


@dataclass(frozen=True)
class GridSpec3D:
  """Uniform 3D Yee grid in normalized solver units."""

  shape: tuple
  spacing: tuple
  dt: float

  def __post_init__(self):
    try:
      shape = tuple(self.shape)
    except TypeError as error:
      raise TypeError("shape must contain (nx, ny, nz).") from error
    if len(shape) != 3 or any(
        isinstance(value, (bool, np.bool_)) or
        not isinstance(value, (int, np.integer)) or value < 2
        for value in shape):
      raise ValueError("shape must contain three integers greater than one.")
    object.__setattr__(self, "shape", tuple(int(value) for value in shape))
    object.__setattr__(
        self, "spacing", _real_tuple("spacing", self.spacing, 3, positive=True))
    object.__setattr__(self, "dt", _positive_finite("dt", self.dt))

  @property
  def dx(self):
    return self.spacing[0]

  @property
  def dy(self):
    return self.spacing[1]

  @property
  def dz(self):
    return self.spacing[2]

  @property
  def extent(self):
    return tuple(size * step for size, step in zip(self.shape, self.spacing))


@dataclass(frozen=True)
class BoundarySpec3D:
  """x/y Bloch wavevector (radians/length) and CPML on both z ends."""

  cpml_width: int
  target_reflection: float = 1e-8
  bloch_wavevector: tuple = (0.0, 0.0)

  def __post_init__(self):
    if isinstance(self.cpml_width, (bool, np.bool_)) or not isinstance(
        self.cpml_width, (int, np.integer)):
      raise TypeError("cpml_width must be an integer.")
    width = int(self.cpml_width)
    if width <= 0:
      raise ValueError("cpml_width must be positive.")
    target = float(self.target_reflection)
    if not math.isfinite(target) or not 0.0 < target < 1.0:
      raise ValueError("target_reflection must lie strictly between zero and one.")
    object.__setattr__(self, "cpml_width", width)
    object.__setattr__(self, "target_reflection", target)
    from .bloch_3d import validate_bloch_wavevector
    object.__setattr__(self, "bloch_wavevector",
                       validate_bloch_wavevector(self.bloch_wavevector))


@dataclass(frozen=True)
class PlaneWaveSource3D:
  """Plane wave; nonzero boundary kx/ky requires reference_frequency.

  Oblique Huygens impedance is matched at that normalized ordinary frequency.
  Use narrowband pulses or independent frequencies for fixed-angle spectra.
  """

  waveform: np.ndarray
  z_index: int
  polarization_xy: tuple = (1.0, 0.0)
  direction: int = 1
  wave_impedance: float = 1.0
  transverse_profile: np.ndarray = None
  reference_frequency: float = None

  def __post_init__(self):
    waveform = np.asarray(self.waveform)
    if (waveform.ndim != 1 or waveform.size < 2 or
        np.iscomplexobj(waveform) or not np.all(np.isfinite(waveform))):
      raise ValueError("waveform must be a finite real 1D array with >= 2 samples.")
    if isinstance(self.z_index, (bool, np.bool_)) or not isinstance(
        self.z_index, (int, np.integer)):
      raise TypeError("z_index must be an integer.")
    polarization = _real_tuple("polarization_xy", self.polarization_xy, 2)
    if math.hypot(*polarization) == 0.0:
      raise ValueError("polarization_xy cannot be the zero vector.")
    if self.direction not in (-1, 1):
      raise ValueError("direction must be +1 or -1.")
    impedance = _positive_finite("wave_impedance", self.wave_impedance)
    profile = self.transverse_profile
    if profile is not None:
      profile = np.asarray(profile)
      if profile.ndim != 2 or np.iscomplexobj(profile) or not np.all(
          np.isfinite(profile)):
        raise ValueError("transverse_profile must be a finite real 2D array.")
      profile = np.array(profile, copy=True)
    object.__setattr__(self, "waveform", np.array(waveform, copy=True))
    object.__setattr__(self, "z_index", int(self.z_index))
    object.__setattr__(self, "polarization_xy", polarization)
    object.__setattr__(self, "direction", int(self.direction))
    object.__setattr__(self, "wave_impedance", impedance)
    object.__setattr__(self, "transverse_profile", profile)
    if self.reference_frequency is not None:
      object.__setattr__(self, "reference_frequency", _positive_finite(
          "reference_frequency", self.reference_frequency))

  def build(self, grid, dtype=np.float32, bloch_wavevector=(0.0, 0.0),
            background_eps_r=1.0):
    if not isinstance(grid, GridSpec3D):
      raise TypeError("grid must be a GridSpec3D instance.")
    if self.transverse_profile is not None and (
        self.transverse_profile.shape != grid.shape[:2]):
      raise ValueError(
          "transverse_profile shape must match the grid's (nx, ny).")
    return huygens_plane_source(
        self.waveform, grid.shape, self.z_index,
        polarization_xy=self.polarization_xy, direction=self.direction,
        wave_impedance=self.wave_impedance,
        transverse_profile=self.transverse_profile, dtype=dtype, expand=False,
        bloch_wavevector=bloch_wavevector, reference_frequency=self.reference_frequency,
        spacing=grid.spacing, background_eps_r=background_eps_r)


@dataclass(frozen=True)
class ScatteringMonitor3D:
  """Reflection and transmission constant-z monitor planes."""

  reflection_z: int
  transmission_z: int
  snapshot_steps: tuple = ()

  def __post_init__(self):
    for name in ("reflection_z", "transmission_z"):
      value = getattr(self, name)
      if isinstance(value, (bool, np.bool_)) or not isinstance(
          value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
      object.__setattr__(self, name, int(value))
    try:
      steps = tuple(self.snapshot_steps)
    except TypeError as error:
      raise TypeError("snapshot_steps must be an integer sequence.") from error
    if any(isinstance(step, (bool, np.bool_)) or not isinstance(
        step, (int, np.integer)) for step in steps):
      raise TypeError("snapshot_steps must contain integers.")
    steps = tuple(int(step) for step in steps)
    if (tuple(sorted(set(steps))) != steps or
        (steps and steps[0] < 1)):
      raise ValueError(
          "snapshot_steps must be strictly increasing positive integers.")
    object.__setattr__(self, "snapshot_steps", steps)


@dataclass(frozen=True)
class FrequencyMonitor3D:
  """Online complex full-field monitor at selected normalized frequencies."""

  frequencies: tuple
  window: object = "hann"

  def __post_init__(self):
    try:
      raw_frequencies = tuple(self.frequencies)
    except TypeError as error:
      raise TypeError("frequencies must be a sequence of real values.") from error
    frequencies = _real_tuple(
        "frequencies", raw_frequencies, len(raw_frequencies),
        positive=True)
    if not frequencies:
      raise ValueError("frequencies must contain at least one value.")
    if tuple(sorted(set(frequencies))) != frequencies:
      raise ValueError("frequencies must be strictly increasing and unique.")
    window = self.window
    if isinstance(window, str):
      if window != "hann":
        raise ValueError("window string must be 'hann'.")
    elif window is not None:
      window = np.asarray(window)
      if window.ndim != 1 or np.iscomplexobj(window) or not np.all(
          np.isfinite(window)):
        raise ValueError("window must be None, 'hann', or a finite real array.")
      window = np.array(window, copy=True)
    object.__setattr__(self, "frequencies", frequencies)
    object.__setattr__(self, "window", window)


def _yee_coordinates(grid, component):
  offsets = _YEE_OFFSETS[component]
  return tuple(
      (np.arange(count, dtype=float) + offset) * spacing
      for count, spacing, offset in zip(grid.shape, grid.spacing, offsets))


def _axis_distance(coordinates, center, extent, periodic):
  if periodic:
    center = center % extent
    distance = np.abs(coordinates - center)
    return np.minimum(distance, extent - distance)
  return np.abs(coordinates - center)


@dataclass(frozen=True)
class Box3D:
  """Axis-aligned box; x/y can wrap across periodic unit-cell edges."""

  center: tuple
  size: tuple
  material: str
  periodic_xy: bool = True

  def __post_init__(self):
    object.__setattr__(self, "center", _real_tuple("center", self.center, 3))
    object.__setattr__(
        self, "size", _real_tuple("size", self.size, 3, positive=True))
    object.__setattr__(self, "material", _material_name(self.material))
    if not isinstance(self.periodic_xy, (bool, np.bool_)):
      raise TypeError("periodic_xy must be boolean.")
    object.__setattr__(self, "periodic_xy", bool(self.periodic_xy))

  def mask(self, grid):
    result = np.zeros((3,) + grid.shape, dtype=bool)
    for component in range(3):
      coordinates = _yee_coordinates(grid, component)
      distances = tuple(_axis_distance(
          coordinates[axis], self.center[axis], grid.extent[axis],
          self.periodic_xy and axis < 2) for axis in range(3))
      result[component] = (
          (distances[0][:, None, None] <= 0.5 * self.size[0]) &
          (distances[1][None, :, None] <= 0.5 * self.size[1]) &
          (distances[2][None, None, :] <= 0.5 * self.size[2]))
    return result


@dataclass(frozen=True)
class Layer3D:
  """Uniform x-y layer occupying ``z_min <= z <= z_max``."""

  z_min: float
  z_max: float
  material: str

  def __post_init__(self):
    z_min, z_max = _real_tuple("z range", (self.z_min, self.z_max), 2)
    if z_max <= z_min:
      raise ValueError("z_max must be greater than z_min.")
    object.__setattr__(self, "z_min", z_min)
    object.__setattr__(self, "z_max", z_max)
    object.__setattr__(self, "material", _material_name(self.material))

  def mask(self, grid):
    result = np.zeros((3,) + grid.shape, dtype=bool)
    for component in range(3):
      z = _yee_coordinates(grid, component)[2]
      result[component] = (
          (z[None, None, :] >= self.z_min) &
          (z[None, None, :] <= self.z_max))
    return np.broadcast_to(result, (3,) + grid.shape).copy()


@dataclass(frozen=True)
class CylinderZ3D:
  """Circular cylinder extruded along z with optional periodic x/y wrap."""

  center_xy: tuple
  radius: float
  z_center: float
  height: float
  material: str
  periodic_xy: bool = True

  def __post_init__(self):
    object.__setattr__(
        self, "center_xy", _real_tuple("center_xy", self.center_xy, 2))
    object.__setattr__(self, "radius", _positive_finite("radius", self.radius))
    object.__setattr__(
        self, "z_center", _real_tuple("z_center", (self.z_center,), 1)[0])
    object.__setattr__(self, "height", _positive_finite("height", self.height))
    object.__setattr__(self, "material", _material_name(self.material))
    if not isinstance(self.periodic_xy, (bool, np.bool_)):
      raise TypeError("periodic_xy must be boolean.")
    object.__setattr__(self, "periodic_xy", bool(self.periodic_xy))

  def mask(self, grid):
    result = np.zeros((3,) + grid.shape, dtype=bool)
    for component in range(3):
      x, y, z = _yee_coordinates(grid, component)
      x_distance = _axis_distance(
          x, self.center_xy[0], grid.extent[0], self.periodic_xy)
      y_distance = _axis_distance(
          y, self.center_xy[1], grid.extent[1], self.periodic_xy)
      radial = (x_distance[:, None]**2 + y_distance[None, :]**2 <=
                self.radius**2)
      axial = np.abs(z - self.z_center) <= 0.5 * self.height
      result[component] = radial[:, :, None] & axial[None, None, :]
    return result


@dataclass(frozen=True)
class SimulationResult3D:
  """Named access to final fields, material layout and scattering spectra."""

  scattering: object
  grid: GridSpec3D
  material_names: tuple
  material_ids: np.ndarray

  @property
  def diffraction(self):
    return self.scattering.diffraction

  @property
  def spectrum(self):
    return self.scattering.diffraction

  @property
  def final_state(self):
    return self.scattering.final_device_state

  @property
  def backend(self):
    return self.scattering.backend

  @property
  def device(self):
    return self.scattering.device

  @property
  def snapshots(self):
    """Sparse full-field recordings, or ``None`` when none were requested."""
    return self.scattering.field_snapshots

  @property
  def frequency_fields(self):
    """Selected complex-frequency full fields, or ``None`` if not requested."""
    return self.scattering.frequency_fields

  @property
  def snapshot_times(self):
    if self.snapshots is None:
      return np.empty(0, dtype=float)
    return self.snapshots.steps * self.grid.dt

  def final_field(self, name):
    """Return one final E, H, normalized D or normalized total P component."""
    try:
      attribute, component = _COMPONENT_NAMES[name]
    except (KeyError, TypeError) as error:
      raise ValueError(
          "name must be Ex/Ey/Ez, Hx/Hy/Hz, Dx/Dy/Dz, or Px/Py/Pz.") from error
    if attribute == "polarization":
      return (self.final_state.displacement[component] -
              self.final_state.electric[component])
    return getattr(self.final_state, attribute)[component]

  def final_polarization(self, component=None, pole=None):
    """Return total normalized P, or an auxiliary polarization for one pole."""
    if pole is None:
      value = self.final_state.displacement - self.final_state.electric
    else:
      value = self.final_state.polarization_current
      if value is None:
        return None
      if isinstance(pole, (bool, np.bool_)) or not isinstance(
          pole, (int, np.integer)):
        raise TypeError("pole must be an integer.")
      if pole < 0 or pole >= value.shape[-1]:
        raise ValueError("pole index is outside the stored polarization axis.")
      value = value[..., int(pole)]
    if component is not None:
      if component not in ("x", "y", "z", 0, 1, 2):
        raise ValueError("component must be x, y, z, 0, 1, or 2.")
      index = {"x": 0, "y": 1, "z": 2}.get(component, component)
      value = value[index]
    return value

  def snapshot_field(self, name):
    """Return a named sparse field history shaped ``(snapshot, nx, ny, nz)``."""
    if self.snapshots is None:
      raise ValueError(
          "no snapshots were requested; set ScatteringMonitor3D.snapshot_steps.")
    try:
      attribute, component = _COMPONENT_NAMES[name]
    except (KeyError, TypeError) as error:
      raise ValueError(
          "name must be Ex/Ey/Ez, Hx/Hy/Hz, Dx/Dy/Dz, or Px/Py/Pz.") from error
    if attribute == "polarization":
      return self.snapshots.polarization[:, component]
    return getattr(self.snapshots, attribute)[:, component]

  def frequency_field(self, name):
    """Return a complex field shaped ``(frequency, nx, ny, nz)``."""
    if self.frequency_fields is None:
      raise ValueError(
          "no frequency fields were requested; set a FrequencyMonitor3D.")
    try:
      attribute, component = _COMPONENT_NAMES[name]
    except (KeyError, TypeError) as error:
      raise ValueError(
          "name must be Ex/Ey/Ez, Hx/Hy/Hz, Dx/Dy/Dz, or Px/Py/Pz.") from error
    if attribute == "polarization":
      return self.frequency_fields.polarization[:, component]
    return getattr(self.frequency_fields, attribute)[:, component]

  def frequency_index(self, frequency, rtol=1e-7, atol=1e-12):
    """Return the monitor index matching one requested frequency."""
    if self.frequency_fields is None:
      raise ValueError("no frequency fields were requested.")
    frequency = float(frequency)
    matches = np.flatnonzero(np.isclose(
        self.frequency_fields.frequencies, frequency, rtol=rtol, atol=atol))
    if matches.size != 1:
      raise ValueError(f"frequency {frequency} was not requested exactly once.")
    return int(matches[0])

  def compare_spectrum(self, candidate, frequency_min=None, frequency_max=None):
    """Compare this result's total R/T/A with a rerun candidate."""
    from .convergence_3d import compare_scattering_results_3d
    return compare_scattering_results_3d(
        self, candidate, frequency_min=frequency_min,
        frequency_max=frequency_max)

  def to_npz(self, path, *, compressed=True, overwrite=False):
    """Export fields, monitors, spectra and metadata to an analysis NPZ.

    Read with ``np.load(path, allow_pickle=False)``. Keys use slash-separated
    names, e.g. ``final/electric``, ``spectrum/reflection`` and
    ``frequency_fields/electric``. ``metadata_json`` describes schema version
    1, normalized units and axis conventions. Optional monitors are omitted
    when absent. Parent directories must exist; existing files are protected
    unless ``overwrite=True``. Returns the exact output Path (no added suffix).
    """
    from .result_io_3d import save_npz
    return save_npz(self, path, compressed=compressed, overwrite=overwrite)

  def to_hdf5(self, path, *, compression="gzip", overwrite=False):
    """Export the same analysis schema as to_npz, with optional h5py.

    Install ``fdtdz[hdf5]`` to enable this method. Complex arrays retain
    their native dtype. Compression is 'gzip', 'lzf', or None. Returns a Path.
    """
    from .result_io_3d import save_hdf5
    return save_hdf5(self, path, compression=compression, overwrite=overwrite)

  def plot_spectrum(self, *, ax=None):
    """Plot valid total R/T/A with optional fdtdz[plot]; return the Axes."""
    from .plotting_3d import plot_spectrum
    return plot_spectrum(self, ax=ax)

  def plot_field(self, name, *, axis="z", index=None, frequency=None,
                 snapshot_step=None, quantity="real", ax=None, **kwargs):
    """Plot a final/snapshot/frequency field slice; return a Matplotlib image.

    Select a node along ``axis`` with ``index`` (default: midpoint).
    ``snapshot_step`` and ``frequency`` are mutually exclusive. ``quantity``
    is 'real', 'imag', 'abs', or 'phase' in radians. Axes use normalized Yee
    coordinates. Extra keywords go to imshow; the figure is never shown
    automatically. Requires ``fdtdz[plot]``.
    """
    from .plotting_3d import plot_field
    return plot_field(
        self, name, axis=axis, index=index, frequency=frequency,
        snapshot_step=snapshot_step, quantity=quantity, ax=ax, **kwargs)


class Simulation3D:
  """High-level builder for one periodic metasurface unit-cell simulation."""

  def __init__(self, grid, boundary, background=None,
               background_name="background", backend="jax", dtype=np.float32):
    if not isinstance(grid, GridSpec3D):
      raise TypeError("grid must be a GridSpec3D instance.")
    if not isinstance(boundary, BoundarySpec3D):
      raise TypeError("boundary must be a BoundarySpec3D instance.")
    if 2 * boundary.cpml_width >= grid.shape[2]:
      raise ValueError("CPML must leave at least one interior z cell.")
    if backend not in ("numpy", "jax"):
      raise ValueError("backend must be 'numpy' or 'jax'.")
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
      raise TypeError("dtype must be a floating dtype.")
    background = Lossless(1.0) if background is None else background
    if not isinstance(background, Lossless):
      raise TypeError(
          "background must currently be Lossless so diffraction wavevectors "
          "have one frequency-independent refractive index.")
    background_name = _material_name(background_name)
    self.grid = grid
    self.boundary = boundary
    self.backend = backend
    self.dtype = dtype
    self._materials = OrderedDict(((background_name, background),))
    self._geometries = []
    self._source = None
    self._monitors = None
    self._frequency_monitor = None

  @property
  def materials(self):
    return dict(self._materials)

  @property
  def geometries(self):
    return tuple(self._geometries)

  def add_material(self, name, material):
    name = _material_name(name)
    if name in self._materials:
      raise ValueError(f"material name {name!r} is already registered.")
    if not isinstance(material, _SUPPORTED_MATERIALS):
      raise TypeError(
          "material must be Lossless, Debye, Lorentz, Drude, or Multipole.")
    if len(self._materials) >= 256:
      raise ValueError("the uint8 material map supports at most 256 materials.")
    self._materials[name] = material
    return self

  def add_geometry(self, geometry):
    """Append one geometry, or expand a LayerStack3D in layer order."""
    from .geometry_3d import LayerStack3D
    if isinstance(geometry, LayerStack3D):
      self._geometries.extend(geometry.geometries)
      return self
    if not callable(getattr(geometry, "mask", None)) or not hasattr(
        geometry, "material"):
      raise TypeError("geometry must provide a material name and mask(grid).")
    self._geometries.append(geometry)
    return self

  def set_source(self, source):
    if not isinstance(source, PlaneWaveSource3D):
      raise TypeError("source must be a PlaneWaveSource3D instance.")
    self._source = source
    return self

  def set_monitors(self, monitors):
    if not isinstance(monitors, ScatteringMonitor3D):
      raise TypeError("monitors must be a ScatteringMonitor3D instance.")
    self._monitors = monitors
    return self

  def set_frequency_monitor(self, monitor):
    if monitor is not None and not isinstance(monitor, FrequencyMonitor3D):
      raise TypeError("monitor must be a FrequencyMonitor3D instance or None.")
    self._frequency_monitor = monitor
    return self

  def material_map(self):
    """Build component-aware reference and patterned material-ID maps."""
    reference = np.zeros((3,) + self.grid.shape, dtype=np.uint8)
    device = reference.copy()
    indices = {name: index for index, name in enumerate(self._materials)}
    for geometry in self._geometries:
      if geometry.material not in indices:
        raise ValueError(
            f"geometry references unknown material {geometry.material!r}.")
      mask = np.asarray(geometry.mask(self.grid))
      if mask.shape != device.shape or mask.dtype != np.bool_:
        raise ValueError(
            f"geometry mask must be boolean with shape {device.shape}.")
      device[mask] = indices[geometry.material]
    return reference, device

  def numerical_report(self, frequencies):
    """Diagnose CFL, ADE band, spatial, temporal, and CPML resolution."""
    from .convergence_3d import numerical_report_3d
    return numerical_report_3d(
        self._materials, self.grid.spacing, self.grid.dt, frequencies,
        cpml_width=self.boundary.cpml_width,
        target_reflection=self.boundary.target_reflection)

  def refined(self, factor):
    """Return a physically equivalent integer-refined simulation setup.

    Physical geometry coordinates are reused.  Grid spacing, time step, CPML
    thickness in cells, z-plane indices, and sparse-snapshot times are scaled
    by ``factor``.  The stored discrete waveform and optional transverse
    profile are linearly resampled; for a fitted/analytic source waveform,
    regenerating it directly at the refined ``dt`` remains preferable.
    """
    factor = _refinement_factor(factor)
    if self._source is None or self._monitors is None:
      raise ValueError(
          "set source and scattering monitors before creating a refinement.")
    if (self._frequency_monitor is not None and
        self._frequency_monitor.window is not None and
        not isinstance(self._frequency_monitor.window, str)):
      raise ValueError(
          "automatic refinement supports FrequencyMonitor3D.window=None or "
          "'hann'; regenerate a custom window for the refined run manually.")
    grid = GridSpec3D(
        shape=tuple(value * factor for value in self.grid.shape),
        spacing=tuple(value / factor for value in self.grid.spacing),
        dt=self.grid.dt / factor)
    boundary = BoundarySpec3D(
        cpml_width=self.boundary.cpml_width * factor,
        target_reflection=self.boundary.target_reflection,
        bloch_wavevector=self.boundary.bloch_wavevector)
    names = tuple(self._materials)
    refined = Simulation3D(
        grid, boundary, background=self._materials[names[0]],
        background_name=names[0], backend=self.backend, dtype=self.dtype)
    for name in names[1:]:
      refined.add_material(name, self._materials[name])
    for geometry in self._geometries:
      refined.add_geometry(geometry)
    source = self._source
    refined.set_source(PlaneWaveSource3D(
        waveform=_resample_waveform(source.waveform, self.grid.dt, factor),
        z_index=source.z_index * factor,
        polarization_xy=source.polarization_xy, direction=source.direction,
        wave_impedance=source.wave_impedance,
        transverse_profile=_resample_periodic_profile(
            source.transverse_profile, factor),
        reference_frequency=source.reference_frequency))
    monitors = self._monitors
    refined.set_monitors(ScatteringMonitor3D(
        reflection_z=monitors.reflection_z * factor,
        transmission_z=monitors.transmission_z * factor,
        snapshot_steps=tuple(step * factor
                             for step in monitors.snapshot_steps)))
    if self._frequency_monitor is not None:
      refined.set_frequency_monitor(FrequencyMonitor3D(
          self._frequency_monitor.frequencies,
          window=self._frequency_monitor.window))
    return refined

  def run_refinement_study(self, factors=(1, 2), backend=None,
                           incident_floor=1e-6, window=None,
                           frequency_min=None, frequency_max=None):
    """Run physical-coordinate refinements and compare each with factor one."""
    try:
      requested = tuple(factors)
    except TypeError as error:
      raise TypeError("factors must be an iterable of positive integers.") from error
    requested = tuple(_refinement_factor(factor) for factor in requested)
    if not requested:
      raise ValueError("factors must contain at least one refinement factor.")
    factors = tuple(sorted(set((1,) + requested)))
    if len(factors) < 2:
      raise ValueError("at least one refinement factor greater than one is required.")
    results = {}
    for factor in factors:
      simulation = self if factor == 1 else self.refined(factor)
      results[factor] = simulation.run(
          backend=backend, incident_floor=incident_floor, window=window)
    from .convergence_3d import refinement_study_3d
    return refinement_study_3d(
        results, frequency_min=frequency_min, frequency_max=frequency_max)

  def _copy_configuration(self, *, grid=None, boundary=None):
    """Copy builder containers and source arrays, retaining geometry objects."""
    names = tuple(self._materials)
    variant = Simulation3D(
        self.grid if grid is None else grid,
        self.boundary if boundary is None else boundary,
        background=self._materials[names[0]], background_name=names[0],
        backend=self.backend, dtype=self.dtype)
    for name in names[1:]:
      variant.add_material(name, self._materials[name])
    for geometry in self._geometries:
      variant.add_geometry(geometry)
    if self._source is not None:
      variant.set_source(replace(self._source))
    if self._monitors is not None:
      variant.set_monitors(replace(self._monitors))
    if self._frequency_monitor is not None:
      variant.set_frequency_monitor(replace(self._frequency_monitor))
    return variant

  def temporal_refined(self, factor, *, waveform_factory=None):
    """Reduce only dt by an integer factor, preserving N*dt and spatial layout.

    Source samples and snapshot steps are scaled in time. By default the
    stored waveform is linearly interpolated and zero after its last sample;
    this includes source interpolation error in the comparison. For an
    analytic pulse, supply ``waveform_factory(times)`` returning one real
    sample per normalized time. It is used even for factor one, so every
    case can sample the same function. Huygens half-step currents are rebuilt.
    Custom frequency-monitor windows must be regenerated manually; automatic
    temporal refinement supports only None/'hann'.
    """
    factor = _refinement_factor(factor)
    if self._source is None or self._monitors is None:
      raise ValueError("set source and scattering monitors before temporal refinement.")
    if (self._frequency_monitor is not None and
        self._frequency_monitor.window is not None and
        not isinstance(self._frequency_monitor.window, str)):
      raise ValueError("temporal refinement requires frequency window None or 'hann'.")
    grid = replace(self.grid, dt=self.grid.dt / factor)
    count = self._source.waveform.size * factor
    if waveform_factory is None:
      waveform = _resample_waveform(self._source.waveform, self.grid.dt, factor)
    else:
      if not callable(waveform_factory):
        raise TypeError("waveform_factory must be callable or None.")
      waveform = np.asarray(waveform_factory(np.arange(count) * grid.dt))
      if waveform.shape != (count,):
        raise ValueError("waveform_factory must return one sample per supplied time.")
    variant = self._copy_configuration(grid=grid)
    variant.set_source(replace(self._source, waveform=waveform))
    variant.set_monitors(replace(
        self._monitors, snapshot_steps=tuple(
            step * factor for step in self._monitors.snapshot_steps)))
    return variant

  def with_cpml_width(self, width):
    """Change only the z CPML width in cells, with a fixed outer domain.

    Increasing width moves the CPML interfaces inward. Grid shape/spacing/dt,
    target reflection, material layout, source, monitor planes and all time
    samples remain fixed. The new CPML must contain background only, and
    leave both Huygens source planes and the probes outside the absorber.
    Source and scattering monitors must already be configured.
    """
    if self._source is None or self._monitors is None:
      raise ValueError("set source and scattering monitors before changing CPML.")
    boundary = replace(self.boundary, cpml_width=width)
    variant = self._copy_configuration(boundary=boundary)
    # Reuse the solver's layout rules before a sweep starts any time stepping.
    from .scattering_workflow_3d import _validate_geometry
    reference, device = variant.material_map()
    source = variant._source.build(
        variant.grid, variant.dtype, variant.boundary.bloch_wavevector,
        next(iter(variant._materials.values())).eps_r)
    _validate_geometry(
        reference, device, source, variant._monitors.reflection_z,
        variant._monitors.transmission_z, boundary.cpml_width)
    return variant

  def _run_parameter_sweep(self, parameter, baseline_value, variants, *,
                           backend, incident_floor, window,
                           frequency_min, frequency_max):
    from .convergence_3d import ParameterSweep3D, convergence_study_3d
    results = {
        value: variant.run(backend=backend, incident_floor=incident_floor,
                           window=window)
        for value, variant in variants.items()}
    comparisons = convergence_study_3d(
        results[baseline_value],
        {f"{parameter}_{value}": result for value, result in results.items()
         if value != baseline_value},
        baseline_name=f"{parameter}_{baseline_value}",
        frequency_min=frequency_min, frequency_max=frequency_max)
    return ParameterSweep3D(
        parameter=parameter, values=tuple(variants),
        baseline_value=baseline_value, results=results, comparisons=comparisons)

  def run_temporal_study(self, factors=(1, 2), *, waveform_factory=None,
                         backend=None, incident_floor=1e-6, window=None,
                         frequency_min=None, frequency_max=None):
    """Run dt-only refinements and compare total R/T/A with factor one.

    Factor one is always included; duplicates are removed and values sorted.
    The comparison band is restricted to jointly valid spectrum samples.
    Spectral windows must be None/'hann', since sample counts change. See
    temporal_refined for waveform resampling and waveform_factory semantics.
    Returns ParameterSweep3D with results keyed by integer factors.
    """
    try:
      requested = tuple(_refinement_factor(value) for value in factors)
    except TypeError as error:
      raise TypeError("factors must be an iterable of positive integers.") from error
    if not requested:
      raise ValueError("factors must not be empty.")
    values = tuple(sorted(set((1,) + requested)))
    if len(values) < 2:
      raise ValueError("at least one factor greater than one is required.")
    if window is not None and not (isinstance(window, str) and window == "hann"):
      raise ValueError("temporal study requires spectral window None or 'hann'.")
    variants = {value: self.temporal_refined(
        value, waveform_factory=waveform_factory) for value in values}
    return self._run_parameter_sweep(
        "temporal_factor", 1, variants, backend=backend,
        incident_floor=incident_floor, window=window,
        frequency_min=frequency_min, frequency_max=frequency_max)

  def run_cpml_study(self, widths, *, backend=None, incident_floor=1e-6,
                     window=None, frequency_min=None, frequency_max=None):
    """Run width-only CPML variants at fixed space/time resolution and domain.

    The current width is always included as baseline. All requested layouts
    are checked before any run; unsafe widths fail rather than moving source,
    probes or geometry. Returns ParameterSweep3D keyed by width in cells.
    Agreement measures boundary sensitivity, not temporal/spatial convergence.
    """
    try:
      requested = tuple(widths)
    except TypeError as error:
      raise TypeError("widths must be an iterable of positive integers.") from error
    if not requested:
      raise ValueError("widths must not be empty.")
    requested = tuple(replace(self.boundary, cpml_width=value).cpml_width
                      for value in requested)
    baseline = self.boundary.cpml_width
    values = tuple(sorted(set((baseline,) + requested)))
    if len(values) < 2:
      raise ValueError("at least one width different from the baseline is required.")
    variants = {value: self.with_cpml_width(value) for value in values}
    return self._run_parameter_sweep(
        "cpml_width", baseline, variants, backend=backend,
        incident_floor=incident_floor, window=window,
        frequency_min=frequency_min, frequency_max=frequency_max)

  def run(self, backend=None, incident_floor=1e-6, window=None):
    """Run matched reference/device simulations and return named results."""
    if self._source is None:
      raise ValueError("set a PlaneWaveSource3D before run().")
    if self._monitors is None:
      raise ValueError("set a ScatteringMonitor3D before run().")
    selected_backend = self.backend if backend is None else backend
    if selected_backend not in ("numpy", "jax"):
      raise ValueError("backend must be 'numpy' or 'jax'.")
    reference, device = self.material_map()
    source = self._source.build(self.grid, self.dtype, self.boundary.bloch_wavevector,
                                next(iter(self._materials.values())).eps_r)
    material_values = tuple(self._materials.values())
    scattering = run_metasurface_scattering_3d(
        material_values, reference, device,
        self.grid.dx, self.grid.dy, self.grid.dz, self.grid.dt, source,
        reflection_probe_z=self._monitors.reflection_z,
        transmission_probe_z=self._monitors.transmission_z,
        cpml_width=self.boundary.cpml_width, dtype=self.dtype,
        target_reflection=self.boundary.target_reflection,
        background_eps_r=material_values[0].eps_r,
        incident_floor=incident_floor, window=window,
        backend=selected_backend,
        snapshot_steps=self._monitors.snapshot_steps,
        frequencies=(None if self._frequency_monitor is None else
                     self._frequency_monitor.frequencies),
        frequency_window=(None if self._frequency_monitor is None else
                          self._frequency_monitor.window))
    return SimulationResult3D(
        scattering=scattering, grid=self.grid,
        material_names=tuple(self._materials), material_ids=device)


__all__ = [
    "BoundarySpec3D", "Box3D", "CylinderZ3D", "FrequencyMonitor3D",
    "GridSpec3D", "Layer3D",
    "PlaneWaveSource3D", "ScatteringMonitor3D", "Simulation3D",
    "SimulationResult3D",
]
