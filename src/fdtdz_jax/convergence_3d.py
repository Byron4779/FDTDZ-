"""Numerical diagnostics and paired convergence comparisons for 3D FDTD.

The reports use normalized solver units.  Frequencies are ordinary cycles per
normalized time unit; material models receive angular frequencies internally.
They diagnose a configuration before a costly run and compare the diffraction
spectra from independently run spatial, temporal, or CPML variants.
"""

from dataclasses import dataclass
import math
from collections.abc import Mapping

import numpy as np

from .convergence_2d import SpectrumConvergenceReport
from .convergence_2d import compare_spectra
from .material_accuracy import MaterialAccuracyReport
from .material_accuracy import material_accuracy_report
from .materials import Lossless


def _positive_scalar(name, value):
  if isinstance(value, (bool, np.bool_)) or not isinstance(
      value, (int, float, np.integer, np.floating)):
    raise TypeError(f"{name} must be a real number.")
  value = float(value)
  if not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"{name} must be finite and positive.")
  return value


def _spacing(value):
  try:
    values = tuple(value)
  except TypeError as error:
    raise TypeError("spacing must contain (dx, dy, dz).") from error
  if len(values) != 3:
    raise ValueError("spacing must contain exactly three values.")
  return tuple(_positive_scalar(name, item) for name, item in zip(
      ("dx", "dy", "dz"), values))


def _frequencies(value):
  frequencies = np.asarray(value)
  if (frequencies.ndim != 1 or frequencies.size == 0 or
      np.iscomplexobj(frequencies) or not np.issubdtype(
          frequencies.dtype, np.number) or
      not np.all(np.isfinite(frequencies))):
    raise ValueError("frequencies must be a non-empty finite real 1D array.")
  frequencies = np.asarray(frequencies, dtype=float)
  if (np.any(frequencies <= 0.0) or np.unique(frequencies).size !=
      frequencies.size or (frequencies.size > 1 and
                           np.any(np.diff(frequencies) <= 0.0))):
    raise ValueError("frequencies must be strictly increasing and positive.")
  return frequencies


def _materials(value):
  if isinstance(value, Mapping):
    names = tuple(value)
    materials = tuple(value.values())
  else:
    materials = tuple(value)
    names = tuple(f"material_{index}" for index in range(len(materials)))
  if not materials:
    raise ValueError("materials must contain at least one material.")
  for name, material in zip(names, materials):
    if not (hasattr(material, "relative_permittivity") and
            hasattr(material, "eps_inf")):
      if not isinstance(material, Lossless):
        raise TypeError(
            f"{name!r} must provide relative_permittivity and eps_inf.")
  return names, materials


def _eps_inf(material):
  return material.eps_r if isinstance(material, Lossless) else material.eps_inf


def _minimum_finite(values):
  values = np.asarray(values, dtype=float)
  finite = values[np.isfinite(values)]
  return float(np.min(finite)) if finite.size else math.inf


@dataclass(frozen=True)
class CFLReport3D:
  """Three-dimensional CFL limit derived from the fastest material."""

  dt: float
  maximum_dt: float
  safety_factor: float
  stable: bool
  minimum_high_frequency_eps_r: float
  spacing: tuple


@dataclass(frozen=True)
class TemporalResolutionReport3D:
  """Samples per optical period for selected ordinary frequencies."""

  frequencies: np.ndarray
  samples_per_period: np.ndarray
  minimum_samples_per_period: float
  nyquist_frequency: float
  below_nyquist: bool


@dataclass(frozen=True)
class SpatialResolutionReport3D:
  """Phase-wavelength and skin-depth resolution of every material."""

  frequencies: np.ndarray
  material_names: tuple
  refractive_indices: np.ndarray
  cells_per_phase_wavelength: np.ndarray
  cells_per_skin_depth: np.ndarray
  minimum_cells_per_phase_wavelength: float
  minimum_cells_per_skin_depth: float
  spacing: tuple


@dataclass(frozen=True)
class CPMLResolutionReport3D:
  """CPML thickness expressed in background phase wavelengths."""

  frequencies: np.ndarray
  width_cells: int
  thickness: float
  target_reflection: float
  thickness_phase_wavelengths: np.ndarray
  minimum_thickness_phase_wavelengths: float


@dataclass(frozen=True)
class MaterialBandReport3D:
  """ADE recurrence errors over the requested ordinary-frequency band."""

  frequencies: np.ndarray
  material_names: tuple
  reports: tuple
  maximum_relative_error: float
  maximum_omega_dt: float


@dataclass(frozen=True)
class NumericalReport3D:
  """Combined pre-run numerical diagnostics for a 3D simulation."""

  cfl: CFLReport3D
  temporal: TemporalResolutionReport3D
  spatial: SpatialResolutionReport3D
  material_band: MaterialBandReport3D
  cpml: CPMLResolutionReport3D | None


@dataclass(frozen=True)
class ConvergenceStudy3D:
  """Spectral differences between a baseline and named rerun variants."""

  baseline_name: str
  comparisons: dict

  def converged(self, absolute_tolerance):
    """True only when every candidate lies within the supplied R/T/A error."""
    return all(report.converged(absolute_tolerance)
               for report in self.comparisons.values())


@dataclass(frozen=True)
class RefinementStudy3D:
  """Results and spectral comparisons from physical-coordinate refinements."""

  factors: tuple
  results: dict
  comparisons: ConvergenceStudy3D

  def converged(self, absolute_tolerance):
    """True when every refined case matches the unrefined baseline."""
    return self.comparisons.converged(absolute_tolerance)


@dataclass(frozen=True)
class ParameterSweep3D:
  """Independent parameter variants and their differences from one baseline.

  ``parameter`` is 'temporal_factor' (dt divided by an integer) or
  'cpml_width' (cells). ``results`` is keyed by entries in ``values``.
  Agreement with the baseline does not establish convergence in other
  parameters, or certify absolute physical accuracy.
  """

  parameter: str
  values: tuple
  baseline_value: int
  results: dict
  comparisons: ConvergenceStudy3D

  def converged(self, absolute_tolerance):
    """True when every variant matches the baseline within R/T/A tolerance."""
    return self.comparisons.converged(absolute_tolerance)


def cfl_report_3d(materials, spacing, dt):
  """Calculate the exact CFL bound used by the 3D material-grid builder."""
  _, materials = _materials(materials)
  spacing = _spacing(spacing)
  dt = _positive_scalar("dt", dt)
  eps_min = min(_eps_inf(material) for material in materials)
  inverse_spacing = math.sqrt(sum(1.0 / value**2 for value in spacing))
  maximum_dt = math.sqrt(eps_min) / inverse_spacing
  safety = dt / maximum_dt
  return CFLReport3D(
      dt=dt, maximum_dt=maximum_dt, safety_factor=safety,
      stable=safety <= 1.0 + 1e-12,
      minimum_high_frequency_eps_r=eps_min, spacing=spacing)


def temporal_resolution_report_3d(frequencies, dt):
  """Report temporal samples/period and the Nyquist limit."""
  frequencies = _frequencies(frequencies)
  dt = _positive_scalar("dt", dt)
  samples = 1.0 / (frequencies * dt)
  return TemporalResolutionReport3D(
      frequencies=frequencies, samples_per_period=samples,
      minimum_samples_per_period=float(np.min(samples)),
      nyquist_frequency=0.5 / dt,
      below_nyquist=bool(np.max(frequencies) < 0.5 / dt))


def spatial_resolution_report_3d(materials, spacing, frequencies):
  """Calculate cells per phase wavelength and attenuation skin depth.

  ``cells_per_phase_wavelength`` and ``cells_per_skin_depth`` have shape
  ``(material, frequency, axis)``.  Infinite entries mean a vanishing phase
  or attenuation index and are not included in the corresponding minimum.
  """
  names, materials = _materials(materials)
  spacing = np.asarray(_spacing(spacing), dtype=float)
  frequencies = _frequencies(frequencies)
  omega = 2.0 * np.pi * frequencies
  epsilon = np.stack([
      np.asarray(material.relative_permittivity(omega), dtype=np.complex128)
      for material in materials])
  refractive = np.sqrt(epsilon)
  phase_index = np.abs(np.real(refractive))
  attenuation_index = np.abs(np.imag(refractive))
  phase_cells = np.full(
      (len(materials), frequencies.size, 3), math.inf, dtype=float)
  skin_cells = np.full_like(phase_cells, math.inf)
  valid_phase = phase_index > np.finfo(float).eps
  valid_skin = attenuation_index > np.finfo(float).eps
  frequency_grid = np.broadcast_to(frequencies[None, :], phase_index.shape)
  omega_grid = np.broadcast_to(omega[None, :], phase_index.shape)
  for axis, step in enumerate(spacing):
    phase_cells[..., axis][valid_phase] = 1.0 / (
        frequency_grid[valid_phase] * phase_index[valid_phase] * step)
    skin_cells[..., axis][valid_skin] = 1.0 / (
        omega_grid[valid_skin] * attenuation_index[valid_skin] * step)
  return SpatialResolutionReport3D(
      frequencies=frequencies, material_names=names,
      refractive_indices=refractive,
      cells_per_phase_wavelength=phase_cells,
      cells_per_skin_depth=skin_cells,
      minimum_cells_per_phase_wavelength=_minimum_finite(phase_cells),
      minimum_cells_per_skin_depth=_minimum_finite(skin_cells),
      spacing=tuple(spacing))


def material_band_report_3d(materials, dt, frequencies):
  """Compare continuous and discrete ADE response over selected frequencies."""
  names, materials = _materials(materials)
  frequencies = _frequencies(frequencies)
  dt = _positive_scalar("dt", dt)
  omega = 2.0 * np.pi * frequencies
  reports = tuple(
      material_accuracy_report(material, dt, omega) for material in materials)
  return MaterialBandReport3D(
      frequencies=frequencies, material_names=names, reports=reports,
      maximum_relative_error=max(report.maximum_relative_error
                                 for report in reports),
      maximum_omega_dt=max(report.maximum_omega_dt for report in reports))


def cpml_resolution_report_3d(background, dz, width_cells,
                              target_reflection, frequencies):
  """Express a z-directed CPML thickness in background phase wavelengths."""
  if not isinstance(background, Lossless):
    raise TypeError("background must be Lossless for the CPML wavelength report.")
  dz = _positive_scalar("dz", dz)
  if isinstance(width_cells, (bool, np.bool_)) or not isinstance(
      width_cells, (int, np.integer)) or width_cells <= 0:
    raise ValueError("width_cells must be a positive integer.")
  target_reflection = float(target_reflection)
  if not math.isfinite(target_reflection) or not 0.0 < target_reflection < 1.0:
    raise ValueError("target_reflection must lie strictly between zero and one.")
  frequencies = _frequencies(frequencies)
  refractive_index = math.sqrt(background.eps_r)
  thickness = int(width_cells) * dz
  wavelengths = thickness * frequencies * refractive_index
  return CPMLResolutionReport3D(
      frequencies=frequencies, width_cells=int(width_cells),
      thickness=thickness, target_reflection=target_reflection,
      thickness_phase_wavelengths=wavelengths,
      minimum_thickness_phase_wavelengths=float(np.min(wavelengths)))


def numerical_report_3d(materials, spacing, dt, frequencies,
                        cpml_width=None, target_reflection=1e-8):
  """Create all pre-run diagnostics for a 3D periodic-x/y, CPML-z setup."""
  names, values = _materials(materials)
  spacing = _spacing(spacing)
  frequencies = _frequencies(frequencies)
  cpml = None
  if cpml_width is not None:
    cpml = cpml_resolution_report_3d(
        values[0], spacing[2], cpml_width, target_reflection, frequencies)
  return NumericalReport3D(
      cfl=cfl_report_3d(materials, spacing, dt),
      temporal=temporal_resolution_report_3d(frequencies, dt),
      spatial=spatial_resolution_report_3d(materials, spacing, frequencies),
      material_band=material_band_report_3d(materials, dt, frequencies),
      cpml=cpml)


def compare_scattering_results_3d(baseline, candidate, frequency_min=None,
                                  frequency_max=None):
  """Compare total 3D R/T/A from two SimulationResult3D or workflow results."""
  base_spectrum = getattr(baseline, "diffraction", baseline)
  candidate_spectrum = getattr(candidate, "diffraction", candidate)
  return compare_spectra(
      base_spectrum, candidate_spectrum, frequency_min=frequency_min,
      frequency_max=frequency_max)


def convergence_study_3d(baseline, candidates, baseline_name="baseline",
                         frequency_min=None, frequency_max=None):
  """Compare named temporal, CPML, or spatial reruns to one baseline result."""
  if not isinstance(candidates, Mapping) or not candidates:
    raise ValueError("candidates must be a non-empty mapping of names to results.")
  comparisons = {
      str(name): compare_scattering_results_3d(
          baseline, candidate, frequency_min=frequency_min,
          frequency_max=frequency_max)
      for name, candidate in candidates.items()
  }
  return ConvergenceStudy3D(
      baseline_name=str(baseline_name), comparisons=comparisons)


def refinement_study_3d(results, frequency_min=None, frequency_max=None):
  """Build a convergence study from an ``{integer_factor: result}`` mapping.

  Factor one is the baseline.  Higher factors must represent the same physical
  domain with proportionally reduced space and time steps.
  """
  if not isinstance(results, Mapping) or 1 not in results:
    raise ValueError("results must map refinement factors and include factor 1.")
  factors = tuple(sorted(results))
  if (not factors or any(isinstance(factor, (bool, np.bool_)) or
                         not isinstance(factor, (int, np.integer)) or
                         factor < 1 for factor in factors)):
    raise ValueError("refinement factors must be positive integers.")
  candidates = {
      f"factor_{factor}": results[factor]
      for factor in factors if factor != 1
  }
  if not candidates:
    raise ValueError("at least one refinement factor greater than one is required.")
  comparisons = convergence_study_3d(
      results[1], candidates, baseline_name="factor_1",
      frequency_min=frequency_min, frequency_max=frequency_max)
  return RefinementStudy3D(
      factors=factors, results=dict(results), comparisons=comparisons)


__all__ = [
    "CFLReport3D", "CPMLResolutionReport3D", "ConvergenceStudy3D",
    "RefinementStudy3D", "ParameterSweep3D",
    "MaterialBandReport3D", "NumericalReport3D", "SpatialResolutionReport3D",
    "TemporalResolutionReport3D", "cfl_report_3d",
    "compare_scattering_results_3d", "convergence_study_3d",
    "cpml_resolution_report_3d", "material_band_report_3d",
    "numerical_report_3d", "refinement_study_3d",
    "spatial_resolution_report_3d",
    "temporal_resolution_report_3d",
]
