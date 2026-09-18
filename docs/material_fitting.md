# Optical tables and passive ADE fitting

`PermittivityTable`, `read_material_table`, `fit_passive_material` and
`PassiveFitResult` are public exports from `fdtdz_jax`. Importing tables uses
only NumPy. Fitting requires SciPy (`fdtdz[fit]` when installing a wheel).
For a source installation without CUDA:

```powershell
$env:FDTDZ_BUILD_CUDA = "0"
python -m pip install ".[fit]"
```

## Import measured data

For nonmagnetic materials, n–k samples are converted as
`epsilon_r = (n + 1j*k)**2`. The solver uses `exp(-1j*omega*t)` and passive
epsilon has a non-negative imaginary part. n and k must be non-negative.
For epsilon tables with the opposite sign convention, explicitly pass
`imaginary_sign=-1`; this option applies only to epsilon tables.

```python
from fdtdz_jax import PhysicalScale, read_material_table

scale = PhysicalScale(length_unit_m=10e-9)
table = read_material_table(
    "measured.csv", representation="nk", delimiter=",", skiprows=1,
    axis="wavelength", unit="nm", scale=scale)
```

The CSV above contains `wavelength_nm,n,k`, with one header line.
For epsilon data use `representation="epsilon"` and columns
`wavelength_nm,epsilon_real,epsilon_imag`. `columns=(2, 0, 1)`, for example,
selects/reorders coordinate, first optical value, second optical value.
`delimiter=None` reads whitespace; `delimiter="\t"` reads TSV. UTF-8 BOM
and `#` comments are accepted. Missing/invalid values are rejected.
Units and headers are never inferred.

| axis | unit | coordinates |
| --- | --- | --- |
| `omega` | `normalized` or `rad/s` | Angular frequency |
| `frequency` | `normalized` or `Hz` | Ordinary frequency, converted using 2π |
| `wavelength` | `m`, `um`, `nm` | Vacuum wavelength |
| `energy` | `eV` | Photon energy ℏω |

Physical units require a `PhysicalScale`; choose the same length unit as
your simulation. Tables store **normalized angular frequencies**, not the
ordinary frequencies used by scattering monitors. Samples are sorted together
by increasing omega, copied and made read-only. At least two unique positive
frequencies are required; duplicate points and negative imaginary epsilon are
rejected. Negative real epsilon is allowed. No smoothing, clipping,
interpolation or extrapolation is performed. For existing arrays use
`PermittivityTable.from_samples(...)` with the same options, or
`PermittivityTable(omega, complex_epsilon)` in normalized units.

## Fit a fixed passive basis

```python
from fdtdz_jax import DrudePole, LorentzPole, fit_passive_material

candidates = (DrudePole(1.0, 0.025), LorentzPole(1.0, 0.42, 0.025))
fit = fit_passive_material(table, candidates)
print(fit.material, fit.maximum_relative_error)
simulation.add_material("measured-film", fit.material)
```

These candidate frequencies are illustrative normalized values. Choose
relaxation times, resonances and damping for the material/band you need.
The fitter keeps these shapes fixed and ignores the candidate strengths.
It fits non-negative Debye/Lorentz `delta_eps` and Drude
`plasma_frequency**2`, together with a positive real `eps_inf`. Lorentz damping
is delta, half the usual gamma linewidth, matching existing material classes.
Use `eps_inf=...` to fix the background; otherwise it is fitted with lower
bound `eps_inf_min=1e-6`. Zero-strength candidates are omitted from the
material. An empty basis, or all zero strengths, produces `Lossless`.

Real and imaginary residuals are solved together using
[SciPy non-negative least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html).
This constrains the returned analytic model to the existing passive ADE
family. Optional positive `weights` multiply the complex residual before
squaring; they follow the **sorted table order**. For relative weighting,
pass `1 / np.maximum(np.abs(table.permittivity), floor)`.

`fit.pole_strengths` retains input candidate order and zero entries.
`fitted_permittivity`, `absolute_error` and `relative_error` also follow sorted
omega and are read-only. The report includes unweighted complex `rms_error`,
`maximum_absolute_error` and `maximum_relative_error`. Relative errors divide
by `max(abs(target), relative_floor)` (default floor `1e-12`). A poor fit is
returned with its errors; it does not become an accuracy guarantee. Incorrect
pole choices or noncausal/inconsistent data may not fit well.

This is an experimental **fixed-pole strength fit**, not automatic nonlinear
resonance/linewidth discovery. A finite-band fit does not certify the measured
data's global causality or out-of-band response. Check held-out data and expand
the basis/band where needed. Independently check time-discretization with
`material_accuracy_report(fit.material, dt, table.angular_frequencies)` and
perform the existing spatial/time/CPML convergence studies for spectra.

## Reproduce the complete workflow

```powershell
python examples/fit_material_3d.py --backend numpy
python examples/fit_material_3d.py --backend jax
python examples/fit_material_3d.py --input measured.csv --representation nk
```

With no input, the example creates and imports a synthetic n–k CSV, recovers
a known Drude/Lorentz material, reports both fitting and ADE errors, and runs
a small 3D film simulation. This checks the workflow; it is not a converged
physical material or GPU accuracy benchmark. JAX may execute on CPU; check
the printed device.
