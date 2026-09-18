"""Physical-coordinate geometries sampled independently at Ex/Ey/Ez nodes.

All lengths use normalized solver units. Masks use point sampling, without
subpixel averaging. Periodic geometries are unions of copies translated by
the grid's x/y periods; input coordinates describe an unwrapped shape.
"""

from dataclasses import dataclass
import math

import numpy as np

from .simulation_3d import Layer3D, _material_name, _positive_finite
from .simulation_3d import _real_tuple, _yee_coordinates


_TOL = 64 * np.finfo(float).eps


def _periodic_flag(value):
  if not isinstance(value, (bool, np.bool_)):
    raise TypeError("periodic_xy must be boolean.")
  return bool(value)


def _cross(a, b):
  return a[0] * b[1] - a[1] * b[0]


def _on_segment(a, b, p):
  return (abs(_cross(b - a, p - a)) <= _TOL and
          np.all(p >= np.minimum(a, b) - _TOL) and
          np.all(p <= np.maximum(a, b) + _TOL))


def _intersects(a, b, c, d):
  values = (_cross(b - a, c - a), _cross(b - a, d - a),
            _cross(d - c, a - c), _cross(d - c, b - c))
  if ((values[0] > _TOL and values[1] < -_TOL or
       values[0] < -_TOL and values[1] > _TOL) and
      (values[2] > _TOL and values[3] < -_TOL or
       values[2] < -_TOL and values[3] > _TOL)):
    return True
  return any((_on_segment(a, b, c), _on_segment(a, b, d),
              _on_segment(c, d, a), _on_segment(c, d, b)))


def _polygon_vertices(vertices):
  try:
    points = tuple(_real_tuple("vertex", point, 2) for point in vertices)
  except TypeError as error:
    raise TypeError("vertices_xy must be a sequence of real (x, y) pairs.") from error
  if len(points) > 1 and points[-1] == points[0]:
    points = points[:-1]
  if len(points) < 3:
    raise ValueError("a polygon requires at least three vertices.")
  array = np.asarray(points)
  shifted = array - array[0]
  scale = np.max(np.abs(shifted))
  if not np.isfinite(scale) or scale == 0:
    raise ValueError("polygon must have finite extent and nonzero area.")
  normalized = shifted / scale
  edges = np.roll(normalized, -1, axis=0) - normalized
  if np.any(np.linalg.norm(edges, axis=1) <= _TOL):
    raise ValueError("polygon must not contain duplicate adjacent vertices.")
  area = sum(_cross(a, b) for a, b in zip(
      normalized, np.roll(normalized, -1, axis=0)))
  if abs(area) <= _TOL:
    raise ValueError("polygon must have nonzero area and be simple (no self-intersections).")
  count = len(points)
  for i in range(count):
    previous, following = edges[i - 1], edges[i]
    if abs(_cross(previous, following)) <= _TOL and np.dot(previous, following) < 0:
      raise ValueError("polygon must be simple; adjacent edges cannot overlap.")
    for j in range(i + 1, count):
      if j == i + 1 or (i == 0 and j == count - 1):
        continue
      if _intersects(normalized[i], normalized[(i + 1) % count],
                     normalized[j], normalized[(j + 1) % count]):
        raise ValueError("polygon must be simple (no self-intersections or touching edges).")
  return points


def _coordinate_copies(coordinates, lower, upper, period, periodic):
  """Coordinates shifted into every possibly intersecting unwrapped copy."""
  if not periodic:
    yield coordinates
    return
  start = math.ceil((lower - coordinates[-1]) / period)
  stop = math.floor((upper - coordinates[0]) / period)
  for index in range(start, stop + 1):
    yield coordinates + index * period


def _inside_polygon(x, y, vertices):
  origin = vertices[0]
  scale = np.max(np.abs(vertices - origin))
  points = (vertices - origin) / scale
  x = (x[:, None] - origin[0]) / scale
  y = (y[None, :] - origin[1]) / scale
  inside = np.zeros((x.shape[0], y.shape[1]), dtype=bool)
  boundary = inside.copy()
  for a, b in zip(points, np.roll(points, -1, axis=0)):
    cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
    boundary |= ((np.abs(cross) <= _TOL) &
                 (x >= min(a[0], b[0]) - _TOL) &
                 (x <= max(a[0], b[0]) + _TOL) &
                 (y >= min(a[1], b[1]) - _TOL) &
                 (y <= max(a[1], b[1]) + _TOL))
    if b[1] != a[1]:
      crossing_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
      inside ^= ((a[1] > y) != (b[1] > y)) & (x < crossing_x)
  return inside | boundary


@dataclass(frozen=True)
class PolygonZ3D:
  """Closed simple polygon extruded over z_min <= z <= z_max.

  Vertices may be clockwise or counterclockwise; repeating the first vertex
  at the end is optional. Concave shapes are supported, holes are not. Use
  unwrapped vertices across periodic seams (e.g. x=-1 to x=1 around x=0).
  """

  vertices_xy: tuple
  z_min: float
  z_max: float
  material: str
  periodic_xy: bool = True

  def __post_init__(self):
    layer = Layer3D(self.z_min, self.z_max, self.material)
    object.__setattr__(self, "vertices_xy", _polygon_vertices(self.vertices_xy))
    for name in ("z_min", "z_max", "material"):
      object.__setattr__(self, name, getattr(layer, name))
    object.__setattr__(self, "periodic_xy", _periodic_flag(self.periodic_xy))

  def mask(self, grid):
    vertices = np.asarray(self.vertices_xy)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    result = np.zeros((3,) + grid.shape, dtype=bool)
    for component in range(3):
      x, y, z = _yee_coordinates(grid, component)
      plane = np.zeros(grid.shape[:2], dtype=bool)
      for xx in _coordinate_copies(x, lower[0], upper[0], grid.extent[0], self.periodic_xy):
        for yy in _coordinate_copies(y, lower[1], upper[1], grid.extent[1], self.periodic_xy):
          plane |= _inside_polygon(xx, yy, vertices)
      result[component] = plane[:, :, None] & (
          (z >= self.z_min) & (z <= self.z_max))[None, None, :]
    return result


@dataclass(frozen=True, eq=False)
class BitmapZ3D:
  """Extrude a boolean bitmap with explicit origin and physical pixel size.

  bitmap[i, j] selects the x/y pixel, so conventional image rows/columns
  require a transpose first. Pixels occupy half-open x/y intervals; z uses
  inclusive endpoints. False pixels leave previously painted materials alone.
  Periodic copies repeat with the simulation period, not the bitmap's size.
  """

  bitmap: np.ndarray
  origin_xy: tuple
  pixel_size: tuple
  z_min: float
  z_max: float
  material: str
  periodic_xy: bool = True

  def __post_init__(self):
    bitmap = np.asarray(self.bitmap)
    if bitmap.ndim != 2 or 0 in bitmap.shape or bitmap.dtype != np.bool_:
      raise ValueError("bitmap must be a non-empty 2D boolean array.")
    origin = _real_tuple("origin_xy", self.origin_xy, 2)
    spacing = _real_tuple("pixel_size", self.pixel_size, 2, positive=True)
    if not all(math.isfinite(start + size * step) and start + size * step > start
               for start, size, step in zip(origin, bitmap.shape, spacing)):
      raise ValueError("bitmap physical extent must be finite and positive.")
    layer = Layer3D(self.z_min, self.z_max, self.material)
    bitmap = np.array(bitmap, copy=True)
    bitmap.setflags(write=False)
    object.__setattr__(self, "bitmap", bitmap)
    object.__setattr__(self, "origin_xy", origin)
    object.__setattr__(self, "pixel_size", spacing)
    object.__setattr__(self, "periodic_xy", _periodic_flag(self.periodic_xy))
    for name in ("z_min", "z_max", "material"):
      object.__setattr__(self, name, getattr(layer, name))

  def mask(self, grid):
    lower = np.asarray(self.origin_xy)
    upper = lower + np.asarray(self.bitmap.shape) * self.pixel_size
    result = np.zeros((3,) + grid.shape, dtype=bool)
    for component in range(3):
      x, y, z = _yee_coordinates(grid, component)
      plane = np.zeros(grid.shape[:2], dtype=bool)
      for xx in _coordinate_copies(x, lower[0], upper[0], grid.extent[0], self.periodic_xy):
        x_valid = (xx >= lower[0]) & (xx < upper[0])
        ix = np.floor(np.clip((xx - lower[0]) / self.pixel_size[0],
                             0, self.bitmap.shape[0] - 1)).astype(int)
        for yy in _coordinate_copies(y, lower[1], upper[1], grid.extent[1], self.periodic_xy):
          y_valid = (yy >= lower[1]) & (yy < upper[1])
          iy = np.floor(np.clip((yy - lower[1]) / self.pixel_size[1],
                               0, self.bitmap.shape[1] - 1)).astype(int)
          plane |= self.bitmap[ix[:, None], iy[None, :]] & x_valid[:, None] & y_valid[None, :]
      result[component] = plane[:, :, None] & (
          (z >= self.z_min) & (z <= self.z_max))[None, None, :]
    return result


@dataclass(frozen=True)
class LayerStack3D:
  """Contiguous layers starting at z_min, ordered toward increasing z.

  ``layers`` is a sequence of (thickness, material_name) pairs. Adding the
  stack to Simulation3D expands it into Layer3D objects; the upper layer
  wins at a shared interface, consistent with later-geometry precedence.
  """

  z_min: float
  layers: tuple

  def __post_init__(self):
    start = _real_tuple("z_min", (self.z_min,), 1)[0]
    try:
      raw = tuple(tuple(layer) for layer in self.layers)
    except TypeError as error:
      raise TypeError("layers must contain (thickness, material) pairs.") from error
    if not raw or any(len(layer) != 2 for layer in raw):
      raise ValueError("layers must contain at least one (thickness, material) pair.")
    layers = tuple((_positive_finite("thickness", thickness), _material_name(material))
                   for thickness, material in raw)
    end = start
    for thickness, _ in layers:
      next_end = end + thickness
      if not math.isfinite(next_end) or next_end <= end:
        raise ValueError("layer endpoints must be finite and strictly increasing.")
      end = next_end
    object.__setattr__(self, "z_min", start)
    object.__setattr__(self, "layers", layers)

  @property
  def geometries(self):
    start = self.z_min
    result = []
    for thickness, material in self.layers:
      end = start + thickness
      result.append(Layer3D(start, end, material))
      start = end
    return tuple(result)


__all__ = ["BitmapZ3D", "LayerStack3D", "PolygonZ3D"]
