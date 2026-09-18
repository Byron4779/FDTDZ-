# Profiling the 3D JAX backend

Run in a Linux NVIDIA environment with a supported JAX/CUDA build:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/profile_jax_gpu_3d.py \
  --shape 32 32 96 --steps 160 --repeats 5 --output outputs/profile-gpu-medium
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/profile_jax_gpu_3d.py \
  --shape 64 64 96 --steps 160 --repeats 5 --output outputs/profile-gpu-large
```

The default requires an actual active JAX GPU. `--allow-cpu` explicitly
permits a **functional smoke run**, which the JSON and Markdown label as
CPU and which supplies no GPU performance conclusions. Output directories
must be new. No existing result is overwritten.

## Workloads and measurements

| Case | Materials | Output |
| --- | --- | --- |
| vacuum-planes | Vacuum | Two Ex/Ey planes at each step |
| ade-final | Two-pole patterned material | Final state only |
| ade-planes | Same two-pole material | Two plane histories |
| ade-snapshots | Same | Planes and two full E/H/D snapshots |
| ade-frequency | Same | Planes and three online complex E/H/D frequencies |
| bloch-planes | Same | Bloch complex fields and two plane histories |

Every case first compares a small nonzero-source 3D NumPy reference against
JAX, including final E/H/D/P and applicable planes/snapshots/frequency fields.
These comparisons use (4,4,48), 128 steps, with a nonzero ADE-polarization
check so that the pulse actually reaches the material; they are not validation of every
larger benchmark shape or a physical convergence study. Timing then uses
the requested benchmark shape/step count. Material maps and compact sources
are identical across ADE cases. The final-only workload uses the same Yee
update with compact plane sources, without allocating a source-volume history.

The script uploads and synchronizes inputs before lowering/compilation,
separates lowering, compilation, first execution and repeated cached runs,
and waits for **all output leaves**. This follows
[JAX benchmarking/profiling guidance](https://docs.jax.dev/en/latest/201/profiling.html).
Cached samples exclude NumPy host conversion, painting geometry, paired
reference/device spectrum analysis, export and trace collection. They measure
the update/monitor scan, not the entire high-level Simulation3D workflow.
There is no universal speedup claim from one laptop or one shape.

The report records device/backend, JAX/Python versions, NVIDIA driver/name/
memory/power limit where available, logical output bytes, compiler argument/
output/temporary/alias memory estimates and cost analysis, allocator memory
statistics and raw timing samples. Compiler byte counts are not measured
GPU bandwidth. `memory.prof` is a snapshot of all live JAX allocations, not
peak device memory or per-case incremental allocation; see
[JAX device-memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html).

Each case also performs a separate untimed trace with a named annotation.
The trace directory holds raw XPlane and Chrome/Perfetto artifacts.
The script lists the top complete **GPU device events** (host CUDA API events
are excluded). Summed kernel durations may overlap and are not wall time.
Inspect the trace to distinguish kernels, copies and synchronization;
`gpu_kernel_count` is a GPU device-event count and can include copies.
Missing CUPTI/TensorFlow hooks or unsupported profiling APIs are reported as
unavailable. A trace containing only host events does not support GPU kernel
conclusions. `--no-trace` disables these additional trace runs.

Inspect `report.md`, `results.json`, and each case's `trace/` and `memory.prof`.
Use the raw trace in a compatible profiler/Perfetto viewer. For pprof:
`pprof --http=: outputs/profile-gpu-medium/ade-planes/memory.prof`.

## How to decide on CUDA work

Measure more than one size. Compare final-only versus planes for output cost,
frequency versus planes for online-monitor cost, vacuum versus ADE for
material cost, and real versus Bloch fields for complex/boundary overhead.
Then inspect launch frequency, kernel durations, memory traffic estimates and
live/temporary allocations. Use Nsight Systems/Compute if memory-bandwidth,
occupancy or hardware-counter evidence is needed; the JAX report alone does
not measure these counters.

For an external Nsight capture, select one case and use the optional CUDA
profiler range. It contains one extra synchronized cached execution, after
validation, compilation and repeated timings:

```sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop -o outputs/ade-cached \
  python examples/profile_jax_gpu_3d.py --cases ade-planes --shape 64 64 96 \
  --steps 160 --repeats 5 --no-trace --cuda-profiler-range \
  --output outputs/ade-cached-profile
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum outputs/ade-cached.nsys-rep
```

The option requires a Linux CUDA runtime exposing `libcudart.so.12`.
Use the WSL helper and flags documented below on this particular machine.

Keep the existing systolic ABI until GPU traces identify a relevant bottleneck
and a prototype demonstrates a numerical-equivalent end-to-end gain. CPU
measurements or a large compiler byte count cannot justify an ABI redesign.

## Local GPU measurements (2026-09-17)

RTX 4060 Laptop GPU, 8188 MiB, Windows driver 577.00, Ubuntu WSL2,
Python 3.10, JAX 0.4.29, jaxlib 0.4.29+cuda12.cudnn91. All six modes
passed their 128-step GPU/NumPy comparisons on both benchmark runs.
The standalone real and Bloch 3D GPU comparisons also passed, using
elementwise `rtol=3e-4, atol=5e-5` to accommodate float32 GPU arithmetic.
The real run's largest absolute D error was 5.2035e-5 (3.1263e-5 of its peak);
the Bloch run's largest error was 1.9233e-5.

The original process returned `CUDA_ERROR_NO_DEVICE`. Explicitly initializing
the WSL system driver exposed the GPU, but the first FDTD execution stalled
with command buffers enabled. This machine successfully ran with
`XLA_FLAGS=--xla_gpu_enable_command_buffer=`. This is an environment-specific
workaround, not a default solver setting or a measured comparison against
working CUDA graphs. See [JAX command-buffer guidance](https://docs.jax.dev/en/latest/gpu_performance_tips.html).
No system drivers, existing environments or CUDA ABI were modified.

Reproduce in this WSL environment from the repository root:

```sh
XLA_FLAGS=--xla_gpu_enable_command_buffer= .venv-wsl/bin/python tools/run_wsl_gpu.py \
  examples/profile_jax_gpu_3d.py --shape 32 32 96 --steps 160 --repeats 5 \
  --output outputs/profile-gpu-medium
```

Repeat with `--shape 64 64 96` and a new output directory. The opt-in helper
initializes `/usr/lib/wsl/lib/libcuda.so.1` before importing JAX and disables
preallocation for this process. It avoids `LD_PRELOAD`, which also stalled
the original setup. The profile JSON records runtime flags and JAX/jaxlib
versions on subsequent runs.

Cached medians for 160 steps, five repeats, all output leaves synchronized:

| Case | 32×32×96, ms | Mcell/s | 64×64×96, ms | Mcell/s |
| --- | ---: | ---: | ---: | ---: |
| vacuum-planes | 22.271 | 706.237 | 82.741 | 760.382 |
| ade-final | 40.277 | 390.509 | 201.264 | 312.597 |
| ade-planes | 41.142 | 382.299 | 202.485 | 310.712 |
| ade-snapshots | 53.825 | 292.218 | 215.429 | 292.043 |
| ade-frequency | 71.449 | 220.137 | 334.728 | 187.957 |
| bloch-planes | 85.030 | 184.977 | 416.465 | 151.068 |

Raw JSON, Markdown, XPlane/Perfetto traces and pprof snapshots are retained in
`tmp/quality/profile-gpu-medium-cupti` and `tmp/quality/profile-gpu-large-cupti`.
These are generated local artifacts, excluded from release archives.
The table uses the final runs with actual GPU-device traces. Their JSON
records flags and library paths; the large run also saves optimized HLO.
Initial runs under `tmp/quality/profile-gpu-medium` and `profile-gpu-large`
contained host events and GPU metadata but no GPU-device events. System
CUPTI and Nsight 2023.4 yielded no kernel activity data; Nsight 2025.6
injection stalled the workload. Loading the existing CUDA 12.9 CUPTI library
for JAX tracing resolved collection without changing system files.
Missing optional TensorFlow Python hooks remained, but all twelve final
case traces contained GPU-device events.

For this machine, copy the installed Nsight Systems `target-linux-x64/libcupti.so.12.9`
into a new local directory as both `libcupti.so.12` and `libcupti.so`, then
prefix the WSL reproduction command with `LD_LIBRARY_PATH=/absolute/path/to/that/directory`.
The measured copies are in `tmp/quality/cupti`; the original comes from the
existing Nsight Systems 2025.6.3 installation. This process-only library
selection is an optional local workaround, not a distribution dependency.

## Findings and ABI decision

The ADE plane workload was 1.85× and 2.45× the vacuum time. Plane output
added about 2.1% over final-only at medium size and 0.6% at large size;
these small differences should not be treated as stable hardware constants.
Two snapshots added 30.8% and 6.4%. Online three-frequency E/H/D
accumulation added 73.7% and 65.3%. Complex Bloch fields added 106.7% and
105.7%. Thus material update and optional frequency/complex-field work are
the primary candidates for further measurement on larger workloads;
removing plane histories alone is not supported as the main optimization.
Sequential cases share a laptop GPU without locked clocks or thermal controls.
The earlier medium vacuum run was 39.745 ms versus 22.271 ms in the final
run, so absolute small-workload timings and cross-run ratios require caution.

The final large ADE plane trace contains 4,336 device events: 3,364 kernels
and 972 D2D copies, about 21 kernel launches and 6 copies per time step.
The following durations are sums from the additional untimed trace,
**not percentages of cached wall time**:

| GPU event | Sum, ms | Share of summed device-event durations |
| --- | ---: | ---: |
| input_add_reduce_fusion | 66.073 | 34.0% |
| loop_divide_select_fusion | 47.923 | 24.7% |
| loop_select_fusion | 23.932 | 12.3% |
| MemcpyD2D | 28.223 | 14.5% |

Optimized HLO links those three fused kernels to ADE pole summation,
electric-field selection and polarization-state selection, respectively
(`jax_yee_ade_3d.py`, lines 213/217/219). Together they account for 71.0%
of summed device-event durations. This establishes ADE material update as
a concrete kernel target in this workload; it does not establish its
bandwidth or occupancy limits. The frequency trace's top two accumulation
fusions add 74.521 and 42.813 ms; the Bloch trace includes 70.151 ms of
D2D copies. Inspect saved HLO alongside traces before rewriting those paths.

The large ADE planes executable reported 119.64 MiB of arguments,
56.50 MiB of outputs and 94.50 MiB of temporary buffers, with zero aliasing.
Frequency monitoring increased logical outputs from 56.50 to 137.50 MiB.
These compiler estimates suggest investigating buffer reuse and monitor
fusion, but do not establish bandwidth saturation or allocation peaks.

**Decision: keep the systolic CUDA ABI unchanged.** Profile monitor fusion,
buffer donation/reuse and the existing JAX ADE update first. Also test a
compatible modern GPU environment with working command buffers to evaluate
launch overhead, since this baseline disables them. Then require a
numerically validated end-to-end prototype gain before considering a
dedicated dispersive CUDA kernel. This GPU baseline completes the requested
profiling step; it is not evidence of a universal bottleneck or speedup.
