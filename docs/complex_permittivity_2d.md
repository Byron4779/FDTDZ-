# 2D complex-permittivity metasurface workflow

The implemented solver keeps time-domain fields real. Complex permittivity is
represented by auxiliary differential equation (ADE) histories for Debye,
Lorentz, Drude, and multi-pole materials.

## Current scope

- TMz fields: `Ez`, `Dz`, `Hx`, `Hy`
- TEz fields: `Ex`, `Ey`, `Dx`, `Dy`, `Hz`
- x-periodic unit cell and y-directed CPML
- normalized units: `c0 = epsilon0 = mu0 = 1`
- NumPy CPU reference and JAX-compiled TMz backend
- paired reference/device runs with R/T/A spectra
- low-memory JAX probe scan: sources stay as one-dimensional waveforms and
  only two `(time, x)` probe lines plus the final field are retained
- periodic diffraction-order decomposition for total reflected, transmitted,
  and absorbed power

This is a two-dimensional cross-section solver. It is appropriate for a
structure invariant in z, including a one-dimensionally periodic grating. A
planar metasurface patterned independently in both in-plane directions needs a
three-dimensional solver.

## Material choice

- `Lossless(eps_r)` for nondispersive dielectrics.
- `Debye(eps_inf, eps_static, tau)` for one relaxation pole.
- `Lorentz(eps_inf, eps_static, omega_0, damping)` for one bound resonance.
- `Drude(eps_inf, plasma_frequency, collision_frequency)` for one free-electron
  metal pole.
- `Multipole(eps_inf, poles)` for fitted real materials. Its poles may be
  `DebyePole`, `LorentzPole`, or `DrudePole`.

All frequencies in material definitions are angular frequencies. The waveform
helper takes ordinary cycles-per-time frequency. Parameters and grid/time units
must therefore be converted consistently before simulation.

Use `PhysicalScale(length_unit_m)` for real material fits. If one normalized
length represents `L0` metres, normalized angular frequencies are
`omega_SI * L0 / c0`, and normalized times are `time_SI * c0 / L0`. The helper
also converts the common `hbar*omega` values quoted in electron volts.

Run the unit-conversion example with:

```powershell
.\.venv\Scripts\python.exe examples\physical_material_units.py
```

The example parameters are illustrative, not a built-in claim for a particular
metal. Replace them with a fit valid over the intended wavelength band and
confirm the damping convention. In this package a Lorentz pole uses `2*delta`
as the coefficient of the first time derivative.

Use `material_accuracy_report(material, dt, angular_frequencies)` to compare
the analytic permittivity with the exact time-discrete ADE response before
running the full FDTD model.

## Run without a notebook

The example locates the repository's local `src` directory automatically, so
it can be run directly without installing `fdtdz_jax` and without setting
`PYTHONPATH`:

```powershell
.\.venv\Scripts\python.exe examples\metasurface_2d_tmz.py
```

The absolute script path also works when the current directory is elsewhere.

The example prints JAX's actual execution device and an R/T/A sample near the
source center frequency. On a correctly configured GPU JAX environment, the
reported backend/device must be GPU rather than CPU.

The JAX scattering workflow does not save full `(time, x, y)` field histories.
This is intentional: quantitative spectra need probe histories, while complete
field movies can exceed GPU memory rapidly. Use the full-history reference API
only for small debugging grids.

## Zero order versus total power

`result.spectrum` uses the x-averaged probes and therefore reports only the
zero diffraction order. `result.diffraction` Fourier transforms the complete x
probe lines, applies the TMz or TEz wave-admittance factor to every propagating
order, and reports their sum.

Use `result.diffraction.reflection`, `.transmission`, and `.absorption` for
quantitative energy accounting whenever the unit-cell period permits nonzero
orders. Otherwise power scattered into higher orders would incorrectly appear
as absorption. The current diffraction normalization assumes normal incidence
and the same homogeneous, lossless background at the input and output probes.

Run the numerical comparisons with:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
.\.venv\Scripts\python.exe -m pytest tests\test_jax_yee_ade_2d.py `
  tests\test_scattering_workflow_2d.py -q
```

For a strict GPU/device validation and cached-execution benchmark, run:

```powershell
.\.venv\Scripts\python.exe examples\validate_jax_gpu_2d.py
```

This command intentionally exits if JAX is not actually exposing a GPU. For a
functional check on a CPU-only machine, use `--allow-cpu`.

## Quantitative-result requirements

Before treating a spectrum as physical output:

1. Fit material poles over the simulation band and compare the fitted analytic
   permittivity with source data.
2. Keep all dispersive geometry outside CPML.
3. Place source and probes in homogeneous regions in propagation order.
4. Use identical source, probes, grid, CPML, and timing for reference and device
   runs.
5. Repeat with finer `dx`, `dy`, smaller `dt`, thicker CPML, and longer runtime.
6. Confirm R/T/A convergence over frequencies where the reference incident
   spectrum exceeds the validity threshold.
7. For propagating media with different input/output admittances, provide the
   correct `transmission_power_factor`.
8. Use `compare_spectra` on at least two grid/time-step/CPML settings and set a
   numerical tolerance appropriate for the intended conclusion.

The older custom CUDA kernel remains a separate three-dimensional systolic
solver with an incompatible field/history layout. It has deliberately not been
modified until JAX GPU profiling shows that a dedicated 2D kernel is needed.
