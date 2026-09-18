# Oblique incidence and x/y Bloch boundaries

The 3D NumPy and JAX ADE solvers accept a transverse Bloch wavevector in
**normalized radians per length**, with the boundary convention
`F(x+Lx,y,z)=exp(1j*kx*Lx)*F(x,y,z)` (and similarly for y). Positive/negative
wraps use opposite phases. z remains periodic at the low level or has the
existing z CPML in the scattering workflow.

```python
import numpy as np
from fdtdz_jax import (
    BoundarySpec3D, GridSpec3D, PlaneWaveSource3D, Simulation3D,
)

f0 = 0.06                        # Ordinary frequency, cycles/normalized time
theta, phi = np.deg2rad([30, 20])
kt = 2*np.pi*f0*np.sin(theta)     # Vacuum; multiply by sqrt(epsilon_background)
kxy = (kt*np.cos(phi), kt*np.sin(phi))
grid = GridSpec3D((8, 8, 160), (.5, .5, .5), .2)
simulation = Simulation3D(grid, BoundarySpec3D(16, bloch_wavevector=kxy))
time = np.arange(700)*grid.dt
waveform = np.exp(-.5*((time-30)/8)**2)*np.sin(2*np.pi*f0*(time-30))
s = (-np.sin(phi), np.cos(phi))   # TE: tangent to the plane of the interface
simulation.set_source(PlaneWaveSource3D(
    waveform, 32, polarization_xy=s, reference_frequency=f0))
# Add geometry and scattering monitors as in the other 3D examples.
```

For TM use `polarization_xy=(cos(phi),sin(phi))`. This specifies the
**tangential electric direction**; the longitudinal component emerges from
Maxwell's update. Arbitrary linear combinations are also supported.
`direction=-1` reverses z propagation. The polarization is normalized.

The source has separate Jx/Jy/Mx/My profiles at their own Yee x/y coordinates.
Its TE/TM sheet admittances and half-cell magnetic-plane phase are matched
at `reference_frequency`. This is a **narrowband directional source**; the
match is approximate away from the reference frequency and at coarse Yee
resolution. In a lossless background, set `wave_impedance=1/sqrt(eps_r)`;
oblique sources reject an inconsistent impedance. Reference and device use
identical sources and Bloch phases.

## Frequency and field conventions

A single broadband run fixes **kx and ky**, so
`sin(theta(f))=sqrt(kx**2+ky**2)/(2*pi*f*sqrt(eps_r))`. The angle is specified
only at f0, as in other
[Bloch FDTD solvers](https://meep.readthedocs.io/en/latest/Scheme_Tutorials/Basics/).
For spectra at one fixed angle, run separately at each carrier frequency,
changing kxy and using sufficiently narrow pulses. There is no fixed-angle
broadband coordinate transformation in this implementation. Frequencies
below the zero-order light line are not propagating incidence and are marked
invalid in R/T/A.

With nonzero kxy the solver automatically allocates complex64 fields for
float32 precision and complex128 fields for float64 precision (JAX float64
still requires its usual x64 setting). These complex arrays contain two real
time-domain quadratures. **All material coefficients and CPML coefficients
stay real.** Complex epsilon continues to be modeled through the causal ADE,
not by inserting complex epsilon into the time-domain constitutive equation.
Zero kxy preserves the original real-field path. Histories, sparse snapshots,
online frequency E/H/D/P fields and NPZ/HDF5 exports retain both quadratures.

At the low level pass `bloch_wavevector=kxy` to `prepare_material_grid_3d` or
the NumPy `simulate_yee_ade_3d*` functions. `jax_material_grid_3d` carries the
phase factors into the JAX grid; `jax_yee_ade_3d_state` preserves quadratures.
For compact source generation pass `bloch_wavevector`, `reference_frequency`,
`spacing` and `background_eps_r` to `huygens_plane_source`. The caller must
use the same wavevector, spacing and background in the grid and source.
Boundary wavevectors are validated against the spatial Nyquist limit.

## Diffraction and convergence

`diffraction_spectrum_3d(..., bloch_wavevector=kxy)` removes the carrier at
each component's Yee node, transforms the periodic envelope and collocates
Ex/Ey amplitudes. Every order has `kx=kx0+2*pi*m/Lx`,
`ky=ky0+2*pi*n/Ly`, and `kz=sqrt(eps_r*omega**2-kx**2-ky**2)`.
Complex histories use the `exp(+i*omega*t)` forward transform. Only positive
frequency bins are returned; the Nyquist bin should not be used quantitatively.
Power includes the longitudinal electric contribution, and is normalized
by the actual incident/transmitted-reference zero-order normal flux, rather
than the normal-incidence transverse amplitude alone. Evanescent orders have
zero power; invalid incident bins have NaN total R/T/A. This workflow assumes
the same homogeneous lossless medium at both monitor planes.

Physical spatial refinement, temporal refinement and CPML-width variants
preserve kxy and the source reference frequency. Refine source sampling,
space, time and CPML independently for accuracy, especially near grazing
incidence. Do not interpret residual negative absorption as physical gain.
The sheet is directional only within its narrowband/discrete matching accuracy.

```powershell
python examples/oblique_film_3d.py --backend numpy --polarization te
python examples/oblique_film_3d.py --backend jax --polarization tm --azimuth 30
python examples/oblique_film_3d.py --polarization tm --spacing 0.125 --output film.npz
```

The example prints the actual angle at the reported frequency, R/T/A and an
independent dielectric-film Fresnel reflection. It is a convergence example,
not an accuracy certificate. A nonuniform periodic metasurface can use the
same API and geometry builders; include all propagating diffraction orders.
