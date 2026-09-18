"""Profile the 3D JAX ADE backend before deciding on CUDA kernel/ABI work.

GPU is required unless --allow-cpu explicitly requests a functional smoke run.
Outputs JSON, Markdown, compiled-memory estimates, pprof memory and JAX traces.
"""

import argparse
import ctypes
from collections import defaultdict
from datetime import datetime, timezone
import functools
import gzip
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
  sys.path.insert(0, str(SOURCE))

import jax  # noqa: E402
import jaxlib  # noqa: E402
from jax import lax, numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fdtdz_jax import DrudePole, LorentzPole, Lossless, Multipole  # noqa: E402
from fdtdz_jax import huygens_plane_source, initialize_yee_ade_3d  # noqa: E402
from fdtdz_jax import jax_material_grid_3d, jax_yee_ade_3d_state  # noqa: E402
from fdtdz_jax import prepare_material_grid_3d, prepare_z_cpml  # noqa: E402
from fdtdz_jax.jax_yee_ade_3d import jax_yee_ade_3d_step  # noqa: E402
from fdtdz_jax.jax_yee_ade_3d import simulate_jax_yee_ade_3d_plane_probes  # noqa: E402
from fdtdz_jax.jax_yee_ade_3d import (  # noqa: E402
    simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields,
    simulate_jax_yee_ade_3d_plane_probes_with_snapshots)
from fdtdz_jax.bloch_3d import plane_source_values  # noqa: E402
from fdtdz_jax.yee_ade_3d import simulate_yee_ade_3d_compact_plane_probes  # noqa: E402

CASES = ("vacuum-planes", "ade-final", "ade-planes", "ade-snapshots",
         "ade-frequency", "bloch-planes")


@functools.partial(jax.jit, static_argnames=("electric_source_z", "magnetic_source_z"))
def _compact_final(grid, state, electric_waveform, magnetic_waveform, profile,
                   polarization, electric_source_z, magnetic_source_z):
  """Same update and compact sources with no history/monitor output."""
  def body(index, current):
    jx, jy, mx, my = plane_source_values(
        electric_waveform[index], magnetic_waveform[index], profile, polarization)
    electric = jnp.zeros_like(current.electric)
    magnetic = jnp.zeros_like(current.magnetic)
    electric = electric.at[0, :, :, electric_source_z].set(jx)
    electric = electric.at[1, :, :, electric_source_z].set(jy)
    magnetic = magnetic.at[0, :, :, magnetic_source_z].set(mx)
    magnetic = magnetic.at[1, :, :, magnetic_source_z].set(my)
    return jax_yee_ade_3d_step(grid, current, electric, magnetic)
  return lax.fori_loop(0, electric_waveform.shape[0], body, state)


def make_workload(case, shape, steps):
  dt = .3
  width = max(4, shape[2] // 8)
  source_z, reflection_z, transmission_z = width+3, width+7, shape[2]-width-7
  material = Multipole(1.0, (DrudePole(.55, .025), LorentzPole(.8, .42, .025)))
  materials = [Lossless(1)] if case == "vacuum-planes" else [Lossless(1), material]
  ids = np.zeros(shape, np.uint8)
  if len(materials) > 1:
    ids[shape[0]//4:3*shape[0]//4, shape[1]//4:3*shape[1]//4,
        shape[2]//2-2:shape[2]//2+2] = 1
  kxy = (.1, -.07) if case == "bloch-planes" else (0., 0.)
  cpml = prepare_z_cpml(shape, width, dt, 1, dtype=np.float32)
  grid = prepare_material_grid_3d(materials, ids, dt, 1, 1, 1,
                                  dtype=np.float32, z_cpml=cpml, bloch_wavevector=kxy)
  time_axis = np.arange(steps)*dt
  waveform = (np.exp(-.5*((time_axis-10)/3)**2)*
              np.sin(2*np.pi*.06*(time_axis-10))).astype(np.float32)
  source = huygens_plane_source(waveform, shape, source_z, dtype=np.float32,
                                bloch_wavevector=kxy, reference_frequency=.06)
  args = (jax_material_grid_3d(grid),
          jax_yee_ade_3d_state(initialize_yee_ade_3d(grid), grid),
          source.electric_waveform, source.magnetic_waveform,
          source.transverse_profile, np.asarray(source.polarization_xy, np.float32))
  kwargs = dict(electric_source_z=source.electric_z_index,
                magnetic_source_z=source.magnetic_z_index)
  snapshots, frequencies = (), None
  if case == "ade-final":
    function = _compact_final
  else:
    kwargs.update(reflection_probe_z=reflection_z, transmission_probe_z=transmission_z)
    function = simulate_jax_yee_ade_3d_plane_probes
    if case == "ade-snapshots":
      snapshots = (steps//2, steps)
      kwargs["snapshot_steps"] = snapshots
      function = simulate_jax_yee_ade_3d_plane_probes_with_snapshots
    elif case == "ade-frequency":
      frequencies = np.array([.04, .06, .08], np.float32)
      args += (frequencies, np.hanning(steps).astype(np.float32))
      function = simulate_jax_yee_ade_3d_plane_probes_with_frequency_fields
  args = jax.device_put(args)
  jax.block_until_ready(args)
  reference = dict(materials=materials, material_ids=ids, dx=1, dy=1, dz=1, dt=dt,
      electric_waveform=source.electric_waveform, magnetic_waveform=source.magnetic_waveform,
      transverse_profile=source.transverse_profile, polarization_xy=source.polarization_xy,
      electric_source_z=source.electric_z_index, magnetic_source_z=source.magnetic_z_index,
      reflection_probe_z=reflection_z, transmission_probe_z=transmission_z,
      dtype=np.float32, z_cpml=cpml, bloch_wavevector=kxy,
      snapshot_steps=snapshots, frequencies=frequencies,
      frequency_window="hann" if frequencies is not None else None)
  return function, args, kwargs, reference


def validate_small(case):
  function, args, kwargs, reference = make_workload(case, (4, 4, 48), 128)
  actual = jax.block_until_ready(function(*args, **kwargs))
  expected = simulate_yee_ade_3d_compact_plane_probes(**reference)
  state = actual if case == "ade-final" else actual[0]
  pairs = [(name, getattr(state, name), getattr(expected.final_state, name))
           for name in ("electric", "magnetic", "displacement", "polarization_current")]
  if case != "ade-final":
    pairs += [("reflection", actual[1][0], expected.electric_tangential[:, 0]),
              ("transmission", actual[1][1], expected.electric_tangential[:, 1])]
  if case == "ade-snapshots":
    pairs += [("snapshots E", actual[2][0], expected.snapshots.electric)]
  if case == "ade-frequency":
    pairs += [("frequency E", actual[3][0], expected.frequency_fields.electric),
              ("frequency H", actual[3][1], expected.frequency_fields.magnetic),
              ("frequency D", actual[3][2], expected.frequency_fields.displacement)]
  errors = {}
  for name, device, host in pairs:
    device = np.asarray(device)
    difference = float(np.max(np.abs(device-host), initial=0))
    errors[name] = difference
    np.testing.assert_allclose(device, host, rtol=3e-4, atol=5e-5)
  polarization_peak = float(np.max(np.abs(expected.final_state.polarization_current), initial=0))
  if case != "vacuum-planes" and polarization_peak < 1e-5:
    raise RuntimeError("Validation pulse did not sufficiently excite ADE polarization.")
  return dict(shape=[4, 4, 48], steps=128, maximum_absolute_errors=errors,
              maximum_polarization_magnitude=polarization_peak)


def summarize_trace(directory):
  """Count only GPU-device complete events, excluding host CUDA API events."""
  traces = list(Path(directory).rglob("*.trace.json.gz"))
  if not traces:
    return {"status": "no Chrome trace found", "gpu_kernel_count": 0}
  events = []
  for path in traces:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
      events.extend(json.load(stream).get("traceEvents", []))
  gpu_pids = {event.get("pid") for event in events if event.get("ph") == "M"
              and event.get("name") == "process_name" and
              re.search(r"device:gpu|^gpu[ :]*\d|^cuda[ :]*\d",
                        str(event.get("args", {}).get("name", "")), re.I)}
  kernels = [event for event in events if event.get("ph") == "X"
             and event.get("pid") in gpu_pids and event.get("dur", 0) > 0]
  groups = defaultdict(lambda: {"count": 0, "total_duration_us": 0.0})
  for event in kernels:
    group = groups[event.get("name", "unknown")]
    group["count"] += 1
    group["total_duration_us"] += event["dur"]
  top = sorted((dict(name=name, **stats) for name, stats in groups.items()),
               key=lambda row: row["total_duration_us"], reverse=True)[:15]
  return {"status": "GPU device events found" if kernels else "no GPU device events found",
          "trace_files": [str(path) for path in traces], "gpu_kernel_count": len(kernels),
          "summed_gpu_duration_us": sum(event["dur"] for event in kernels),
          "top_gpu_events": top,
          "duration_note": "Summed kernel durations may overlap; not wall time."}


def profile_case(case, shape, steps, repeats, output, capture_trace,
                 cuda_profiler_range=False):
  validation = validate_small(case)
  setup_start = time.perf_counter()
  function, args, kwargs, _ = make_workload(case, shape, steps)
  setup_seconds = time.perf_counter()-setup_start
  start = time.perf_counter()
  lowered = function.lower(*args, **kwargs)
  lower_seconds = time.perf_counter()-start
  start = time.perf_counter()
  compiled = lowered.compile()
  compile_seconds = time.perf_counter()-start
  start = time.perf_counter()
  result = jax.block_until_ready(compiled(*args))
  first_seconds = time.perf_counter()-start
  samples = []
  for _ in range(repeats):
    start = time.perf_counter()
    result = jax.block_until_ready(compiled(*args))
    samples.append(time.perf_counter()-start)
  row = dict(case=case, shape=list(shape), steps=steps, repeats=repeats,
      validation=validation, setup_and_upload_seconds=setup_seconds,
      lower_seconds=lower_seconds, compile_seconds=compile_seconds,
      first_execution_seconds=first_seconds, cached_seconds=samples,
      median_seconds=float(np.median(samples)), minimum_seconds=min(samples),
      mcell_updates_per_second=float(np.prod(shape)*steps/np.median(samples)/1e6),
      output_bytes=sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(result)),
      field_dtype=str((result if case == "ade-final" else result[0]).electric.dtype))
  try:
    memory = compiled.memory_analysis()
    row["compiled_memory"] = {name: int(getattr(memory, name)) for name in
        ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
         "alias_size_in_bytes", "generated_code_size_in_bytes")}
  except (AttributeError, RuntimeError, TypeError) as error:
    row["compiled_memory"] = {"unavailable": str(error)}
  try:
    row["cost_analysis"] = compiled.cost_analysis()
  except (AttributeError, RuntimeError) as error:
    row["cost_analysis"] = {"unavailable": str(error)}
  row["device_memory_stats"] = jax.devices()[0].memory_stats()
  case_output = output / case
  case_output.mkdir()
  try:
    hlo_path = case_output / "optimized_hlo.txt"
    hlo_path.write_text(compiled.as_text(), encoding="utf-8")
    row["optimized_hlo"] = str(hlo_path)
  except (AttributeError, RuntimeError, TypeError) as error:
    row["optimized_hlo"] = {"unavailable": str(error)}
  if cuda_profiler_range:
    runtime = ctypes.CDLL("libcudart.so.12")
    status = runtime.cudaProfilerStart()
    if status != 0:
      raise RuntimeError(f"cudaProfilerStart failed with status {status}")
    try:
      jax.block_until_ready(compiled(*args))
    finally:
      status = runtime.cudaProfilerStop()
    if status != 0:
      raise RuntimeError(f"cudaProfilerStop failed with status {status}")
    row["cuda_profiler_range"] = "One additional cached execution, excluding setup/compile/timing."
  if capture_trace:
    try:
      with jax.profiler.trace(str(case_output / "trace"), create_perfetto_trace=True):
        with jax.profiler.StepTraceAnnotation(case, step_num=0):
          jax.block_until_ready(compiled(*args))
      row["trace"] = summarize_trace(case_output / "trace")
    except Exception as error:
      row["trace"] = {"unavailable": str(error)}
  else:
    row["trace"] = {"status": "disabled"}
  try:
    jax.profiler.save_device_memory_profile(str(case_output / "memory.prof"))
    row["memory_profile"] = str(case_output / "memory.prof")
  except Exception as error:
    row["memory_profile"] = {"unavailable": str(error)}
  return row


def write_report(report, directory):
  (directory / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
  label = "GPU profile" if report["is_gpu_profile"] else "CPU functional smoke (not GPU profiling)"
  lines = [f"# 3D JAX {label}", "", f"JAX {report['jax_version']}; {report['devices']}", "",
      "Compiled executables, uploaded inputs and all output leaves are synchronized.",
      "Cached timings exclude compilation, NumPy conversion and profiler overhead.",
      "Small NumPy comparisons validate each mode separately from benchmark shapes.", "",
      "| Case | Shape | Cached median s | Mcell/s | Compile s | Output MiB | GPU trace events |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
  for row in report["cases"]:
    lines.append(f"| {row['case']} | {row['shape']} | {row['median_seconds']:.6f} | "
        f"{row['mcell_updates_per_second']:.3f} | {row['compile_seconds']:.3f} | "
        f"{row['output_bytes']/2**20:.3f} | {row['trace'].get('gpu_kernel_count', 'unavailable')} |")
  lines += ["", "## Interpretation", "",
      "Compare ade-final/ade-planes for plane-output overhead; ade-snapshots and",
      "ade-frequency for optional monitor overhead. Compare vacuum/ade for material",
      "cost, and ade-planes/bloch-planes for complex-field and boundary cost.",
      "Argument/output/temp sizes are compiler estimates, not measured bandwidth.",
      "The memory.prof file is a snapshot of all live JAX buffers, not a peak or",
      "the incremental allocation of one kernel. Trace runs are additional untimed runs.", ""]
  if not report["is_gpu_profile"]:
    lines += ["**No GPU performance conclusions can be drawn from this CPU smoke run.**", ""]
  lines += ["A dedicated dispersive CUDA kernel or systolic ABI change requires GPU",
      "trace evidence and a validated end-to-end gain. This script makes no ABI changes.", ""]
  (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--allow-cpu", action="store_true")
  parser.add_argument("--shape", type=int, nargs=3, default=(32, 32, 96))
  parser.add_argument("--steps", type=int, default=160)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--cases", choices=CASES, nargs="+", default=CASES)
  parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite")
  parser.add_argument("--no-trace", action="store_true", help="Disable additional profiler runs")
  parser.add_argument("--cuda-profiler-range", action="store_true",
                      help="Mark an extra cached run for Nsight --capture-range=cudaProfilerApi")
  args = parser.parse_args()
  if min(args.shape[:2]) < 2 or args.shape[2] < 32 or args.steps < 8 or args.repeats < 2:
    parser.error("shape needs nx/ny >= 2 and nz >= 32; steps >= 8; repeats >= 2")
  if len(set(args.cases)) != len(args.cases):
    parser.error("cases must be unique")
  devices = jax.devices()
  is_gpu = jax.default_backend() == "gpu" or all(d.platform == "gpu" for d in devices)
  if not is_gpu and not args.allow_cpu:
    raise SystemExit("GPU profiling stopped: this JAX environment exposes no GPU. "
                     "--allow-cpu is only for a functional smoke run.")
  if args.cuda_profiler_range and not is_gpu:
    parser.error("--cuda-profiler-range requires a GPU")
  args.output.mkdir(parents=True, exist_ok=False)
  report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
      python=platform.python_version(), platform=platform.platform(),
      jax_version=jax.__version__, backend=jax.default_backend(),
      jaxlib_version=jaxlib.__version__,
      runtime_environment={key: os.environ.get(key) for key in
          ("XLA_FLAGS", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "LD_LIBRARY_PATH")},
      devices=[str(d) for d in devices], is_gpu_profile=is_gpu, cases=[])
  try:
    report["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
  except (OSError, subprocess.TimeoutExpired) as error:
    report["nvidia_smi"] = str(error)
  write_report(report, args.output)
  for case in args.cases:
    row = profile_case(case, tuple(args.shape), args.steps, args.repeats,
                        args.output, not args.no_trace, args.cuda_profiler_range)
    report["cases"].append(row)
    write_report(report, args.output)
    print(f"{case}: {row['median_seconds']:.6f} s; "
          f"{row['mcell_updates_per_second']:.3f} Mcell/s", flush=True)
  print(f"{'GPU profile' if is_gpu else 'CPU functional smoke'} saved: {args.output}")


if __name__ == "__main__":
  main()
