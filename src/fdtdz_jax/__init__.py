# -*- coding: utf-8 -*-

"""Public API for fdtdz_jax.

This module exposes a lightweight public surface and lazily imports the
heavy submodules on attribute access to avoid circular-import problems
when compiled extensions import the package during their initialization.
"""

from importlib.metadata import PackageNotFoundError, version

try:
  from .fdtdz_jax_version import version as __version__
except ImportError:
  # setuptools-scm writes this module during package builds.  A source tree
  # remains importable before that build has happened.
  try:
    __version__ = version("fdtdz")
  except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
	"__version__",
	"ADECoefficients",
	"ADEState",
	"BoundarySpec3D",
	"Box3D",
	"BitmapZ3D",
	"PolygonZ3D",
	"LayerStack3D",
	"CFLReport3D",
	"CPMLResolutionReport3D",
	"ConvergenceStudy3D",
	"CylinderZ3D",
	"Debye",
	"DebyePole",
	"Drude",
	"DrudePole",
	"DiffractionSpectrum2D",
	"DiffractionSpectrum3D",
	"FrequencyMonitor3D",
	"Lorentz",
	"LorentzPole",
	"Lossless",
	"Material",
	"MaterialGrid2D",
	"MaterialGrid3D",
	"MultipoleADEState",
	"MaterialModel",
	"MetasurfaceScatteringResult3D",
	"NumericalReport3D",
	"MaterialAccuracyReport",
	"MaterialBandReport3D",
	"Multipole",
	"Pole",
	"PhysicalScale",
	"PermittivityTable",
	"PassiveFitResult",
	"read_material_table",
	"fit_passive_material",
	"RefinementStudy3D",
	"ParameterSweep3D",
	"LineProbe2D",
	"JAXMaterialGrid2D",
	"JAXMaterialGrid3D",
	"JAXYeeADE2DState",
	"JAXYeeADE2DTEzState",
	"JAXYeeADE3DState",
	"Sponge2D",
	"ScatteringSpectrum2D",
	"SpectrumConvergenceReport",
	"SpatialResolutionReport3D",
	"TemporalResolutionReport3D",
	"TMzScatteringResult2D",
	"TEzScatteringResult2D",
	"HuygensLineSource2D",
	"HuygensTEzLineSource2D",
	"HuygensPlaneSource3D",
	"GridSpec3D",
	"Layer3D",
	"PlaneWaveSource3D",
	"ScatteringMonitor3D",
	"Simulation3D",
	"SimulationResult3D",
	"YeeADE2DResult",
	"YeeADE2DState",
	"YeeADE2DTEzResult",
	"YeeADE2DTEzState",
	"YeeADE3DResult",
	"YeeADE3DFieldSnapshots",
	"YeeADE3DFrequencyFields",
	"YeeADE3DPlaneProbeResult",
	"YeeADE3DState",
	"ZCPML3D",
	"YCPML2D",
	"PoleADECoefficients",
	"ade_coefficients",
	"ade_step",
	"advance_jax_yee_ade_2d",
	"advance_jax_yee_ade_3d",
	"circle_mask",
	"cells_per_wavelength",
	"cfl_report_3d",
	"compare_scattering_results_3d",
	"compare_spectra",
	"convergence_study_3d",
	"cpml_resolution_report_3d",
	"discrete_relative_permittivity",
	"discrete_multipole_permittivity",
	"diffraction_spectrum",
	"diffraction_spectrum_3d",
	"fdtdz",
	"gaussian_sine_pulse",
	"graded_y_sponge",
	"huygens_line_source",
	"huygens_current_waveforms",
	"huygens_tez_current_waveforms",
	"huygens_tez_line_source",
	"huygens_plane_source",
	"initialize_yee_ade_2d",
	"initialize_yee_ade_2d_tez",
	"initialize_yee_ade_3d",
	"initialize_multipole_ade",
	"jax_material_grid_2d",
	"jax_material_grid_3d",
	"jax_runtime_report",
	"jax_yee_ade_2d_state",
	"jax_yee_ade_2d_step",
	"jax_yee_ade_2d_tez_state",
	"jax_yee_ade_2d_tez_step",
	"jax_yee_ade_3d_state",
	"jax_yee_ade_3d_step",
	"line_current_source",
	"material_accuracy_report",
	"material_band_report_3d",
	"numerical_report_3d",
	"refinement_study_3d",
	"paint_material",
	"multipole_ade_step",
	"pole_ade_coefficients",
	"prepare_material_grid_2d",
	"prepare_material_grid_3d",
	"prepare_z_cpml",
	"prepare_y_cpml",
	"rectangle_mask",
	"run_tmz_scattering_2d",
	"run_tez_scattering_2d",
	"run_metasurface_scattering_3d",
	"residual",
	"simulate_ade",
	"simulate_multipole_ade",
	"simulate_jax_yee_ade_2d",
	"simulate_jax_yee_ade_2d_probes",
	"simulate_jax_yee_ade_2d_line_probes",
	"simulate_jax_yee_ade_2d_tez_probes",
	"simulate_jax_yee_ade_3d",
	"simulate_jax_yee_ade_3d_plane_probes",
	"simulate_jax_yee_ade_3d_plane_probes_with_snapshots",
	"simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields",
	"simulate_jax_yee_ade_2d_tez_line_probes",
	"simulate_yee_ade_2d",
	"simulate_yee_ade_2d_tez",
	"simulate_yee_ade_3d",
	"simulate_yee_ade_3d_compact_plane_probes",
	"simulate_yee_ade_3d_plane_probes",
	"sample_line",
	"scattering_spectrum",
	"spatial_resolution_report_3d",
	"temporal_resolution_report_3d",
	"yee_ade_2d_step",
	"yee_ade_2d_tez_step",
	"yee_ade_3d_step",
]

def __getattr__(name: str):
	if name in {"PermittivityTable", "PassiveFitResult", "read_material_table",
			"fit_passive_material"}:
		from . import material_fitting
		return getattr(material_fitting, name)
	if name in {"BitmapZ3D", "PolygonZ3D", "LayerStack3D"}:
		from . import geometry_3d
		return getattr(geometry_3d, name)
	if name in {
		"SpectrumConvergenceReport", "cells_per_wavelength", "compare_spectra"
	}:
		from . import convergence_2d
		return getattr(convergence_2d, name)
	if name in {
		"CFLReport3D", "CPMLResolutionReport3D", "ConvergenceStudy3D",
		"RefinementStudy3D", "ParameterSweep3D",
		"MaterialBandReport3D", "NumericalReport3D", "SpatialResolutionReport3D",
		"TemporalResolutionReport3D", "cfl_report_3d",
		"compare_scattering_results_3d", "convergence_study_3d",
		"cpml_resolution_report_3d", "material_band_report_3d",
		"numerical_report_3d", "refinement_study_3d",
		"spatial_resolution_report_3d",
		"temporal_resolution_report_3d"
	}:
		from . import convergence_3d
		return getattr(convergence_3d, name)
	if name in {"MaterialAccuracyReport", "material_accuracy_report"}:
		from . import material_accuracy
		return getattr(material_accuracy, name)
	if name == "PhysicalScale":
		from .physical_units import PhysicalScale
		return PhysicalScale
	if name in {"DiffractionSpectrum2D", "diffraction_spectrum"}:
		from . import diffraction_2d
		return getattr(diffraction_2d, name)
	if name in {"DiffractionSpectrum3D", "diffraction_spectrum_3d"}:
		from . import diffraction_3d
		return getattr(diffraction_3d, name)
	if name in {
		"JAXMaterialGrid3D", "JAXYeeADE3DState", "advance_jax_yee_ade_3d",
		"jax_material_grid_3d", "jax_yee_ade_3d_state",
		"jax_yee_ade_3d_step", "simulate_jax_yee_ade_3d",
		"simulate_jax_yee_ade_3d_plane_probes",
		"simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields",
		"simulate_jax_yee_ade_3d_plane_probes_with_snapshots"
	}:
		from . import jax_yee_ade_3d
		return getattr(jax_yee_ade_3d, name)
	if name in {
		"JAXYeeADE2DTEzState", "jax_yee_ade_2d_tez_state",
		"jax_yee_ade_2d_tez_step", "simulate_jax_yee_ade_2d_tez_line_probes",
		"simulate_jax_yee_ade_2d_tez_probes"
	}:
		from . import jax_yee_ade_2d_tez
		return getattr(jax_yee_ade_2d_tez, name)
	if name in {
		"MetasurfaceScatteringResult3D", "run_metasurface_scattering_3d"
	}:
		from . import scattering_workflow_3d
		return getattr(scattering_workflow_3d, name)
	if name in {
		"BoundarySpec3D", "Box3D", "CylinderZ3D", "FrequencyMonitor3D",
		"GridSpec3D", "Layer3D",
		"PlaneWaveSource3D", "ScatteringMonitor3D", "Simulation3D",
		"SimulationResult3D"
	}:
		from . import simulation_3d
		return getattr(simulation_3d, name)
	if name in {
		"TEzScatteringResult2D", "TMzScatteringResult2D", "jax_runtime_report",
		"run_tez_scattering_2d", "run_tmz_scattering_2d"
	}:
		from . import scattering_workflow_2d
		return getattr(scattering_workflow_2d, name)
	if name == "fdtdz":
		from .fdtdz_jax import fdtdz
		return fdtdz
	if name == "residual":
		from .residual import residual
		# Importing a same-named submodule installs it on its parent package.
		# Keep subsequent public API accesses pointing to the callable.
		globals()[name] = residual
		return residual
	if name in {
		"JAXMaterialGrid2D", "JAXYeeADE2DState", "advance_jax_yee_ade_2d",
		"jax_material_grid_2d",
		"jax_yee_ade_2d_state", "jax_yee_ade_2d_step",
		"simulate_jax_yee_ade_2d", "simulate_jax_yee_ade_2d_line_probes",
		"simulate_jax_yee_ade_2d_probes"
	}:
		from . import jax_yee_ade_2d
		return getattr(jax_yee_ade_2d, name)
	if name in {
		"MaterialGrid3D", "YeeADE3DFieldSnapshots", "YeeADE3DFrequencyFields",
		"YeeADE3DPlaneProbeResult", "YeeADE3DResult",
		"YeeADE3DState", "ZCPML3D",
		"initialize_yee_ade_3d", "prepare_material_grid_3d",
		"prepare_z_cpml", "simulate_yee_ade_3d",
		"simulate_yee_ade_3d_compact_plane_probes",
		"simulate_yee_ade_3d_plane_probes", "yee_ade_3d_step"
	}:
		from . import yee_ade_3d
		return getattr(yee_ade_3d, name)
	if name in {
		"YeeADE2DTEzResult", "YeeADE2DTEzState",
		"initialize_yee_ade_2d_tez", "simulate_yee_ade_2d_tez",
		"yee_ade_2d_tez_step"
	}:
		from . import yee_ade_2d_tez
		return getattr(yee_ade_2d_tez, name)
	if name in {
		"MultipoleADEState", "PoleADECoefficients",
		"discrete_multipole_permittivity", "initialize_multipole_ade",
		"multipole_ade_step", "pole_ade_coefficients", "simulate_multipole_ade"
	}:
		from . import multipole_reference
		return getattr(multipole_reference, name)
	if name in {
		"HuygensPlaneSource3D", "huygens_plane_source"
	}:
		from . import metasurface_3d
		return getattr(metasurface_3d, name)
	if name in {
		"HuygensLineSource2D", "HuygensTEzLineSource2D", "LineProbe2D", "Sponge2D",
		"ScatteringSpectrum2D", "circle_mask",
		"gaussian_sine_pulse",
		"graded_y_sponge", "huygens_line_source", "huygens_current_waveforms",
		"huygens_tez_current_waveforms",
		"huygens_tez_line_source",
		"line_current_source", "paint_material", "rectangle_mask", "sample_line",
		"scattering_spectrum"
	}:
		from . import metasurface_2d
		return getattr(metasurface_2d, name)
	if name in {
		"ADECoefficients", "Debye", "DebyePole", "Drude", "DrudePole",
		"Lorentz", "LorentzPole", "Lossless", "Material", "MaterialModel",
		"Multipole", "Pole", "ade_coefficients"
	}:
		from . import materials
		return getattr(materials, name)
	if name in {
			"ADEState", "ade_step", "discrete_relative_permittivity",
			"simulate_ade"
	}:
		from . import ade_reference
		return getattr(ade_reference, name)
	if name in {
		"MaterialGrid2D", "YeeADE2DResult", "YeeADE2DState", "YCPML2D",
		"initialize_yee_ade_2d", "prepare_material_grid_2d",
		"prepare_y_cpml", "simulate_yee_ade_2d", "yee_ade_2d_step"
	}:
		from . import yee_ade_2d
		return getattr(yee_ade_2d, name)
	raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
	return __all__
