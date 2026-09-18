"""Run with python -I after wheel installation to exclude source-tree imports."""

from importlib import metadata, resources
from pathlib import Path
import tempfile

import numpy as np
import fdtdz_jax as fdtdz


installed_root = Path(metadata.distribution("fdtdz").locate_file("fdtdz_jax"))
assert Path(fdtdz.__file__).resolve().parent == installed_root.resolve()
assert fdtdz.__version__ == metadata.version("fdtdz")
assert fdtdz.__version__ != "0+unknown"
for name in fdtdz.__all__:
  getattr(fdtdz, name)
assert callable(fdtdz.residual)
assert any(p.name.endswith(".ptx") for p in resources.files("fdtdz_jax.ptx").iterdir())

# Check installed optical-table imports and passive fitting as well.
known = fdtdz.Multipole(1.7, (fdtdz.DebyePole(.8, 2.5),))
omega = np.linspace(.1, 1, 40)
table = fdtdz.PermittivityTable(omega, known.relative_permittivity(omega))
fit = fdtdz.fit_passive_material(table, known.poles)
assert fit.maximum_absolute_error < 1e-12
with tempfile.TemporaryDirectory() as directory:
  path = Path(directory) / "optical.csv"
  np.savetxt(path, np.column_stack((omega, table.permittivity.real,
                                   table.permittivity.imag)), delimiter=",")
  imported = fdtdz.read_material_table(path, representation="epsilon", delimiter=",")
  np.testing.assert_array_equal(imported.permittivity, table.permittivity)

grid = fdtdz.prepare_material_grid_3d(
    [fdtdz.Lossless(1.0)], np.zeros((3, 3, 4), dtype=np.uint8),
    dt=0.2, dx=1.0, dy=1.0, dz=1.0, dtype=np.float32)
state = fdtdz.initialize_yee_ade_3d(grid)
current = np.ones(grid.vector_shape, dtype=np.float32)
expected = fdtdz.yee_ade_3d_step(grid, state, current)
actual = fdtdz.jax_yee_ade_3d_step(
    fdtdz.jax_material_grid_3d(grid),
    fdtdz.jax_yee_ade_3d_state(state, grid), current, np.zeros_like(current))
np.testing.assert_allclose(actual.electric, expected.electric, atol=1e-6)
assert np.any(np.asarray(actual.electric) != 0)

# Installed Bloch updates must retain complex quadratures on both backends.
bloch_grid = fdtdz.prepare_material_grid_3d(
    [fdtdz.Lossless(1.0)], np.zeros((3, 3, 4), dtype=np.uint8),
    dt=.2, dx=1., dy=1., dz=1., dtype=np.float32, bloch_wavevector=(.2, -.1))
bloch_state = fdtdz.initialize_yee_ade_3d(bloch_grid)
current = (np.ones(bloch_grid.vector_shape)*(1+.3j)).astype(np.complex64)
expected = fdtdz.yee_ade_3d_step(bloch_grid, bloch_state, current)
actual = fdtdz.jax_yee_ade_3d_step(fdtdz.jax_material_grid_3d(bloch_grid),
    fdtdz.jax_yee_ade_3d_state(bloch_state, bloch_grid), current, np.zeros_like(current))
assert np.asarray(actual.electric).dtype == np.complex64
np.testing.assert_allclose(actual.electric, expected.electric, atol=1e-6)

# Exercise the installed result-export modules using only core dependencies.
simulation = fdtdz.Simulation3D(
    fdtdz.GridSpec3D((2, 2, 24), (1.0, 1.0, 1.0), 0.3),
    fdtdz.BoundarySpec3D(4), backend="numpy")
simulation.set_source(fdtdz.PlaneWaveSource3D(np.sin(np.arange(30)), 6))
simulation.set_monitors(fdtdz.ScatteringMonitor3D(9, 17))
simulation.add_material("film", fdtdz.Lossless(2.25))
simulation.add_geometry(fdtdz.LayerStack3D(11.2, ((0.5, "film"),)))
simulation.add_geometry(fdtdz.PolygonZ3D(
    ((0.2, 0.2), (1.8, 0.2), (0.2, 1.8)), 12, 13, "film"))
simulation.add_geometry(fdtdz.BitmapZ3D(
    [[True]], (0, 0), (1, 1), 14, 15, "film"))
result = simulation.run()
temporal = simulation.run_temporal_study((2,))
cpml = simulation.run_cpml_study((5,))
assert isinstance(temporal, fdtdz.ParameterSweep3D)
assert temporal.results[2].grid.dt == simulation.grid.dt / 2
assert cpml.results[5].grid == simulation.grid
with tempfile.TemporaryDirectory() as directory:
  path = result.to_npz(Path(directory) / "result.npz")
  with np.load(path, allow_pickle=False) as archive:
    np.testing.assert_array_equal(archive["final/electric"],
                                  result.final_state.electric)
    assert "metadata_json" in archive
print(f"Installed wheel smoke passed: {fdtdz.__version__} at {fdtdz.__file__}")
