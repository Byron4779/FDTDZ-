"""Optional Matplotlib views of the named 3D result API."""

import numpy as np


def _axes(ax):
  if ax is not None:
    return ax
  try:
    import matplotlib.pyplot as plt
  except ImportError as error:
    raise ImportError(
        "Plotting requires matplotlib; install fdtdz[plot].") from error
  return plt.subplots()[1]


def plot_spectrum(result, *, ax=None):
  """Plot total R/T/A, leaving gaps at invalid incident-normalization bins."""
  ax = _axes(ax)
  spectrum = result.spectrum
  for name, label in (("reflection", "R"), ("transmission", "T"),
                      ("absorption", "A")):
    values = np.where(spectrum.valid, getattr(spectrum, name), np.nan)
    ax.plot(spectrum.frequencies, values, label=label)
  ax.set_xlabel("Frequency (normalized)")
  ax.set_ylabel("Power fraction")
  ax.legend()
  return ax


def plot_field(result, name, *, axis="z", index=None, frequency=None,
               snapshot_step=None, quantity="real", ax=None, **kwargs):
  """Plot a Yee-node slice; return the image for colorbars or figure saving.

  Choose the final field (default), a recorded ``snapshot_step``, or a
  monitored ``frequency``. ``index`` selects a node along the slice normal,
  defaulting to the middle node. No spatial interpolation is applied.
  ``quantity`` is real, imag, abs, or phase (radians); additional keywords
  are passed to ``Axes.imshow``. This function never calls ``show``.
  """
  if axis not in ("x", "y", "z"):
    raise ValueError("axis must be 'x', 'y', or 'z'.")
  transforms = {"real": np.real, "imag": np.imag,
                "abs": np.abs, "phase": np.angle}
  if quantity not in transforms:
    raise ValueError("quantity must be 'real', 'imag', 'abs', or 'phase'.")
  if frequency is not None and snapshot_step is not None:
    raise ValueError("choose either frequency or snapshot_step, not both.")
  if frequency is not None:
    values = result.frequency_field(name)[result.frequency_index(frequency)]
    label = f"frequency={frequency:g}"
  elif snapshot_step is not None:
    if isinstance(snapshot_step, (bool, np.bool_)) or not isinstance(
        snapshot_step, (int, np.integer)):
      raise TypeError("snapshot_step must be an integer.")
    history = result.snapshot_field(name)
    matches = np.flatnonzero(result.snapshots.steps == snapshot_step)
    if matches.size != 1:
      raise ValueError("snapshot_step was not recorded exactly once.")
    values = history[matches[0]]
    label = f"step={snapshot_step}"
  else:
    values = result.final_field(name)
    label = f"step={result.final_state.step}"
  normal = "xyz".index(axis)
  if index is None:
    index = result.grid.shape[normal] // 2
  if isinstance(index, (bool, np.bool_)) or not isinstance(
      index, (int, np.integer)):
    raise TypeError("index must be an integer.")
  if not 0 <= index < result.grid.shape[normal]:
    raise ValueError("index is outside the grid along the selected axis.")
  component = "xyz".index(name[1].lower())
  offsets = np.zeros(3)
  offsets[component] = 0.5
  if name.startswith("H"):
    offsets = 0.5 - offsets
  remaining = [value for value in range(3) if value != normal]
  extent = []
  for dimension in remaining:
    step = result.grid.spacing[dimension]
    extent.extend(((offsets[dimension] - 0.5) * step,
                   (result.grid.shape[dimension] + offsets[dimension] - 0.5)
                   * step))
  plane = transforms[quantity](np.take(np.asarray(values), index, axis=normal))
  ax = _axes(ax)
  image = ax.imshow(plane.T, origin="lower", extent=extent, **kwargs)
  ax.set_xlabel(f"{'xyz'[remaining[0]]} (normalized)")
  ax.set_ylabel(f"{'xyz'[remaining[1]]} (normalized)")
  position = (index + offsets[normal]) * result.grid.spacing[normal]
  ax.set_title(f"{quantity}({name}), {axis}={position:g}, {label}")
  return image
