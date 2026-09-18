import numpy as np
import pytest

from fdtdz_jax import BitmapZ3D, BoundarySpec3D, Box3D, GridSpec3D
from fdtdz_jax import Layer3D, LayerStack3D, Lossless, PolygonZ3D
from fdtdz_jax import PlaneWaveSource3D, ScatteringMonitor3D, Simulation3D


@pytest.fixture
def grid():
  return GridSpec3D((8, 6, 20), (0.5, 0.75, 0.5), 0.2)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_rectangle_polygon_matches_box_across_periodic_seams(grid, periodic, reverse):
  points = ((-0.5, -0.75), (1.0, -0.75), (1.0, 1.5), (-0.5, 1.5))
  if reverse:
    points = tuple(reversed(points)) + (points[-1],)
  polygon = PolygonZ3D(points, 3.25, 4.25, "film", periodic)
  box = Box3D((0.25, 0.375, 3.75), (1.5, 2.25, 1.0), "film", periodic)
  np.testing.assert_array_equal(polygon.mask(grid), box.mask(grid))


def test_concave_polygon_matches_union_of_boxes(grid):
  polygon = PolygonZ3D(((0, 0), (3, 0), (3, 0.75), (1, 0.75),
                        (1, 3), (0, 3)), 3, 4, "film", False)
  horizontal = Box3D((1.5, 0.375, 3.5), (3, 0.75, 1), "film", False)
  vertical = Box3D((0.5, 1.5, 3.5), (1, 3, 1), "film", False)
  np.testing.assert_array_equal(polygon.mask(grid),
                                horizontal.mask(grid) | vertical.mask(grid))


def test_triangle_contains_diagonal_edges_and_staggered_z_nodes():
  grid = GridSpec3D((5, 5, 8), (1, 1, 1), 0.4)
  mask = PolygonZ3D(((0, 0), (4, 0), (0, 4)), 2.25, 2.75, "film", False).mask(grid)
  assert not np.any(mask[:2])
  x, y = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
  np.testing.assert_array_equal(mask[2, :, :, 2], x + y <= 4)
  assert np.count_nonzero(mask) == 15


@pytest.mark.parametrize("offset", [(0, 0), (12, -9), (-8, 13.5)])
def test_polygon_periodic_translation_invariance(grid, offset):
  points = np.array(((-0.5, -0.5), (1, 0), (0, 1.5)))
  baseline = PolygonZ3D(points, 3, 4, "film").mask(grid)
  translated = PolygonZ3D(points + offset, 3, 4, "film").mask(grid)
  np.testing.assert_array_equal(translated, baseline)


@pytest.mark.parametrize("points", [
    ((0, 0), (1, 1)), ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 1), (0, 1), (1, 0)),
    ((0, 0), (3, 0), (0, 2), (2, 2), (0, 1)),
    ((0, 0), (1, 0), (1, 0), (0, 1)),
    ((0, 0), (2, 0), (1, 0), (1, 1), (0, 1)),
    ((0, 0), (1, np.inf), (0, 1)),
])
def test_invalid_polygon_is_rejected(points):
  with pytest.raises(ValueError):
    PolygonZ3D(points, 1, 2, "film")


def test_polygon_accepts_collinear_forward_edges_and_small_scale():
  points = np.array(((0, 0), (1, 0), (2, 0), (2, 2), (0, 2))) * 1e-9
  polygon = PolygonZ3D(points, 1e-9, 2e-9, "film", False)
  grid = GridSpec3D((3, 3, 4), (1e-9,) * 3, 1e-10)
  mask = polygon.mask(grid)
  for component, offsets in enumerate(((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5))):
    x, y, z = [(np.arange(count) + offset) * 1e-9
               for count, offset in zip(grid.shape, offsets)]
    expected = ((x[:, None, None] <= 2e-9) & (y[None, :, None] <= 2e-9) &
                (z[None, None, :] >= 1e-9) & (z[None, None, :] <= 2e-9))
    np.testing.assert_array_equal(mask[component], expected)


def test_periodic_geometries_larger_than_cell_use_union_of_copies(grid):
  polygon = PolygonZ3D(((-5, -5), (5, -5), (5, 5), (-5, 5)), 3, 4, "film")
  np.testing.assert_array_equal(polygon.mask(grid), Layer3D(3, 4, "film").mask(grid))
  bitmap = BitmapZ3D([[True], [False]], (0, 0), (4, 4.5), 3, 4, "film")
  np.testing.assert_array_equal(bitmap.mask(grid), Layer3D(3, 4, "film").mask(grid))


def test_bitmap_pixel_axes_and_half_open_boundaries():
  grid = GridSpec3D((4, 4, 8), (0.5, 0.5, 0.5), 0.2)
  pixels = np.array([[True, False, False], [False, True, False]])
  geometry = BitmapZ3D(pixels, (0, 0), (0.5, 0.5), 1.25, 1.75, "film", False)
  mask = geometry.mask(grid)
  for component, offsets in enumerate(((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5))):
    for i, j, k in np.ndindex(grid.shape):
      x, y, z = (np.array((i, j, k)) + offsets) * 0.5
      ix, iy = int(x / 0.5), int(y / 0.5)
      expected = (ix < 2 and iy < 3 and pixels[ix, iy] and 1.25 <= z <= 1.75)
      assert mask[component, i, j, k] == expected
  pixels[:] = False
  assert np.any(geometry.bitmap)
  with pytest.raises(ValueError):
    geometry.bitmap[0, 0] = False


def test_bitmap_periodic_copy_uses_grid_period_and_preserves_false_pixels(grid):
  pixels = np.array([[True, False], [False, True]])
  periodic = BitmapZ3D(pixels, (-0.5, -0.75), (0.5, 0.75), 3, 4, "film")
  shifted = BitmapZ3D(pixels, (3.5, 3.75), (0.5, 0.75), 3, 4, "film")
  np.testing.assert_array_equal(periodic.mask(grid), shifted.mask(grid))
  assert np.any(periodic.mask(grid)[:, -1, -1])
  assert not np.any(periodic.mask(grid)[:, 3, 3])
  clipped = BitmapZ3D(pixels, (-0.5, -0.75), (0.5, 0.75), 3, 4, "film", False)
  assert np.count_nonzero(periodic.mask(grid)) > np.count_nonzero(clipped.mask(grid))
  simulation = Simulation3D(grid, BoundarySpec3D(3), backend="numpy")
  simulation.add_material("base", Lossless(2))
  simulation.add_material("film", Lossless(3))
  simulation.add_geometry(Layer3D(3, 4, "base"))
  simulation.add_geometry(periodic)
  ids = simulation.material_map()[1]
  assert np.all(ids[periodic.mask(grid)] == 2)
  assert ids[0, 3, 3, 6] == 1


@pytest.mark.parametrize("pixels", [[], [[0, 1]], [True, False], np.zeros((2, 0), dtype=bool)])
def test_invalid_bitmaps_are_rejected(pixels):
  with pytest.raises(ValueError, match="2D boolean"):
    BitmapZ3D(pixels, (0, 0), (1, 1), 1, 2, "film")


def test_stack_layer_interfaces_and_later_overrides(grid):
  simulation = Simulation3D(grid, BoundarySpec3D(3), backend="numpy")
  simulation.add_material("lower", Lossless(2))
  simulation.add_material("upper", Lossless(3))
  stack = LayerStack3D(3, ((1, "lower"), (1.5, "upper")))
  assert simulation.add_geometry(stack) is simulation
  assert simulation.geometries == stack.geometries
  ids = simulation.material_map()[1]
  assert ids[0, 0, 0, 7] == 1
  assert ids[0, 0, 0, 8] == 2  # upper layer owns z=4 interface
  simulation.add_geometry(Layer3D(4, 4.5, "lower"))
  assert simulation.material_map()[1][0, 0, 0, 8] == 1


@pytest.mark.parametrize("layers", [(), ((0, "film"),), ((-1, "film"),),
                                    ((np.inf, "film"),), ((1, ""),), ((1,),)])
def test_invalid_stacks_are_rejected(layers):
  with pytest.raises((ValueError, TypeError)):
    LayerStack3D(0, layers)


def test_new_geometries_preserve_physical_sampling_on_refinement(grid):
  simulation = Simulation3D(grid, BoundarySpec3D(3), backend="numpy")
  simulation.add_material("film", Lossless(2))
  simulation.add_geometry(LayerStack3D(3, ((0.5, "film"),)))
  simulation.add_geometry(PolygonZ3D(((0, 0), (2, 0), (0, 2)), 4, 5, "film"))
  simulation.add_geometry(BitmapZ3D([[True]], (1, 1), (0.8, 0.8), 5, 6, "film"))
  simulation.set_source(PlaneWaveSource3D(np.sin(np.arange(30)), 4))
  simulation.set_monitors(ScatteringMonitor3D(6, 15))
  refined = simulation.refined(3)
  coarse_ids = simulation.material_map()[1]
  fine_ids = refined.material_map()[1]
  # For odd factor 3, each staggered node has an exact fine-grid counterpart.
  for component, starts in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    selection = tuple(slice(start, None, 3) for start in starts)
    np.testing.assert_array_equal(fine_ids[(component,) + selection], coarse_ids[component])


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_new_geometry_run_matches_equivalent_existing_builders(backend):
  grid = GridSpec3D((4, 4, 40), (1, 1, 1), 0.4)
  simulations = [Simulation3D(grid, BoundarySpec3D(5), backend=backend) for _ in range(2)]
  for simulation in simulations:
    simulation.add_material("film", Lossless(2.25))
    simulation.set_source(PlaneWaveSource3D(np.sin(np.arange(100) * 0.2), 8))
    simulation.set_monitors(ScatteringMonitor3D(12, 30))
  new, old = simulations
  new.add_geometry(LayerStack3D(18.2, ((0.6, "film"), (0.6, "film"))))
  old.add_geometry(Layer3D(18.2, 19.4, "film"))
  new.add_geometry(PolygonZ3D(((0.2, 0.2), (1.8, 0.2), (1.8, 1.8), (0.2, 1.8)),
                             20.2, 21.8, "film"))
  old.add_geometry(Box3D((1, 1, 21), (1.6, 1.6, 1.6), "film"))
  new.add_geometry(BitmapZ3D([[True]], (2.2, 2.2), (1.6, 1.6), 22.2, 23.8, "film"))
  old.add_geometry(Box3D((3, 3, 23), (1.6, 1.6, 1.6), "film"))
  np.testing.assert_array_equal(new.material_map()[1], old.material_map()[1])
  actual, expected = new.run(), old.run()
  np.testing.assert_array_equal(actual.final_field("Ex"), expected.final_field("Ex"))
  np.testing.assert_array_equal(actual.spectrum.reflection, expected.spectrum.reflection)
