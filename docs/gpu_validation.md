# CUDA extension compatibility and GPU validation

## Cleanup validation (2026-09-18)

Removed 14 legacy CUDA test translation units, their three helper headers,
and six duplicate example/notebook wrapper test modules. Core numerical,
material, workflow and CUDA custom-call regression tests remain. All 12
example scripts and five root notebooks remain and are included in the sdist.
The native wrapper no longer includes `testutils.h`; the standalone PTX
build no longer downloads GoogleTest or declares the removed test targets.

The second cleanup removed the standalone NumPy 1D solver, its seven public
exports, its test module and the 2D-to-1D comparison test. No retained example
or notebook imports this solver. The entire paper directory, historical
roadmaps/analysis notes, Sphinx scaffolding and formatter configuration were
also removed (41 source/document files), together with old `build/` and
`.eggs/` artifacts. Unused native aliases, `EmptyKernel`, commented descriptor
code and the obsolete CTest fixture declaration were removed.

Validation after cleanup:

- Windows CPU, Python 3.10 / JAX 0.5.3: 369 passed, 21 GPU-only cases skipped.
- Built the native wheel from the sdist with CUDA 12.4, architecture 89;
  archive checks and isolated installed-wheel smoke passed.
- WSL2 RTX 4060 Laptop, Python 3.10 / JAX 0.6.2: 21 native CUDA cases passed,
  zero skipped; six large memory stress cases were deselected on the 8 GiB GPU.
- Drude-Lorentz 2D, real 3D and Bloch 3D JAX GPU examples passed NumPy
  reference comparisons. The analytic complex permittivity passed a direct
  Drude-plus-Lorentz formula comparison; the physical-unit example also ran.

Final build outputs and CUDA JUnit/device reports are under `tmp/cleanup-round2/`.
Other GPU families were not physically tested during this cleanup.

The dependency test matrix targets JAX/jaxlib 0.4.32, 0.5.3 and 0.6.2
with Python 3.10/3.11. Package requirements are `>=0.4.32,<0.7` for both.
PyPI marks 0.4.32 as yanked; it is retained as an exact-pinned historical
compatibility case, not a default installation choice.
Install matching JAX and jaxlib versions; for GPU use the corresponding
`jax[cuda12]` extra. A pure-Python wheel does not provide the original
systolic CUDA solver, but includes the NumPy/JAX ADE solvers.

## FFI registration

CUDA targets are registered through `jax.ffi.register_ffi_target`, with
`jax.extend.ffi` as the compatibility import on JAX 0.4.32. Explicit
registration `api_version=0` preserves the existing
`(cudaStream_t, void **buffers, const char *opaque, size_t)` ABI. Lowering uses
`jax.ffi.ffi_lowering` on JAX 0.6, and the compatible MLIR builder on older
versions, with explicit `api_version=2` (the previous
`jaxlib.hlo_helpers.custom_call` default) and the existing buffer layouts and
binary descriptor. These two version numbers
refer to different APIs and must not be interchanged.

This migrates the Python registration API. It does not convert native handlers
to typed FFI, which would require a different C++ signature, descriptor
transport and error reporting. The default registration API 1 and lowering
API 4 are for typed handlers and cannot call these existing functions.
See the [official JAX FFI registration documentation](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.register_ffi_target.html).

Automatic launch presets now query the local CUDA device. Cooperative grids
are limited to the physical SM count, and the packaged PTX target is chosen
at or below the physical compute capability. This fixes the former generic
36-block launch on the 24-SM RTX 4060 Laptop and avoids selecting sm75 PTX on
Pascal. Explicit tuple launch parameters remain caller-controlled.

The y-plane source workspace reserves 128 internal float/half2 slots per x
coordinate, including auxiliary PML storage. The old allocation used the
physical z extent and underallocated fp32 workspace when PML was present.
This could corrupt nearby buffers and produce zero x/y fields under outer
JIT. Both precision targets now reserve the full internal source layout;
GPU regressions compare eager, outer JIT and cropped fields in all directions.

## Trusted Linux runner

The manual `.github/workflows/gpu-validation.yml` workflow requires a trusted
runner with NVIDIA drivers, CUDA 12 nvcc, Python development headers and
labels `self-hosted`, `linux`, `x64`, `nvidia-gpu` and `nvidia-<family>`.
Select the matching physical family: pascal, volta, turing, ampere, ada or
hopper. Dispatch separately for each available family. Configure architectures
supported by the toolkit and physical GPU; the default `75;80;86;89` requires
CUDA with Ada support. A Pascal/Volta runner needs an appropriate build target
such as `60`/`70`. Building multiple targets on Ada validates compilation only;
it does not establish execution on the other families.

Each dispatch tests all three JAX versions. The workflow builds an sdist and
native wheel, checks archive contents, installs the wheel and uses isolated
imports. `tools/validate_gpu_environment.py` checks physical compute
capability, JAX GPU devices and both compiled targets. CUDA tests cover fp32
and fp16, x/y/z sources, eager/JIT execution, cropped outputs, wave-equation
residuals. Optional `large_tests=true` additionally runs the six large memory
stress simulations; use a GPU with at least 12 GiB. They are deselected by
default so 8 GiB devices can run the full registration/physics suite without
exhausting memory. A JUnit gate rejects any skipped, failed or
errored custom-call tests. Artifacts retain the physical GPU identity,
driver/JAX/jaxlib versions, installed extension path, test results and ADE
profiling reports/traces. GPU discovery alone is not execution validation.

The workflow additionally compares 2D, real 3D and Bloch 3D ADE updates with
NumPy, then profiles all six 3D modes. WSL environments with driver discovery
stalls can use `tools/run_wsl_gpu.py` and
`XLA_FLAGS=--xla_gpu_enable_command_buffer=` as described in
`docs/jax_gpu_profiling.md`. This workaround is opt-in and process-local.

## Hardware coverage

The local machine exposes only an RTX 4060 Laptop (Ada, compute capability
8.9). Other device families require physical hardware before their execution
can be marked validated. No GitHub publication or workflow dispatch is
performed for the local validation task.

## Local validation record (2026-09-17)

The installed native wheel was tested on Ubuntu WSL2, Python 3.10.20,
RTX 4060 Laptop / 8188 MiB / 24 SMs / compute capability 8.9, driver 577.00.
CUDA 12.4 nvcc compiled the extension for `75;80;86;89`. This is local Linux
execution, not a GitHub Actions runner run. Python 3.11 remains configured in
CPU CI rather than locally validated in this record.

| JAX / jaxlib | Full CPU regression | Installed CUDA core suite | 2D / real 3D / Bloch 3D |
| --- | --- | --- | --- |
| 0.4.32 | 394 passed, 29 skipped | 21 passed, 0 skipped | All passed |
| 0.5.3 | 402 passed, 21 skipped | 21 passed, 0 skipped | All passed |
| 0.6.2 | 402 passed, 21 skipped | 21 passed, 0 skipped | All passed |

The CPU skips are the 21 GPU-only simulations, plus eight optional HDF5
cases in the Windows JAX 0.4.32 environment. GPU runs deselect the six large
memory stress cases on this 8 GiB device; no selected test is skipped. Six
execution cases exercise fp32/fp16 x/y/z sources, nonzero finite fields,
outer JIT equality and cropped-output equality. Nine additional long point
source cases validate wave-equation residuals and cropping.

All three JAX versions passed the small NumPy reference in all six profiling
modes and collected 2413–7056 actual GPU trace events per mode across 18
traces. The optional TensorFlow
Python profiler hook is unavailable; CUDA device traces and profiler reports
were still produced. These timings concern the ADE JAX backend, separately
from the systolic custom-call tests.

Artifacts are retained locally under `tmp/quality/`: `ffi-gpu` (JAX 0.6.2),
`ffi-gpu-jax-0.5.3`, `ffi-gpu-jax-0.4.32`, `ffi-cpu.xml`,
`ffi-release-dist` (native wheel and sdist), and `ffi-pure-final-dist`
(pure-Python wheel and sdist). Each GPU directory contains the installed
extension/device JSON, JUnit report, workflow log and six profiling reports
and traces. Build archive checks, isolated installed-wheel smoke and
`pip check` passed for the native and pure-Python distributions.

To reproduce the installed CUDA core suite after building and installing a
native wheel in a dedicated Linux environment:

```bash
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -I tools/validate_gpu_environment.py --expected-family ada --output outputs/gpu-environment.json
python -I tools/smoke_installed.py
python -m pytest -q -s --tb=short -o pythonpath= --import-mode=importlib \
  -k 'not test_large_sim' --junitxml=outputs/custom-calls.xml tests/test_fdtdz_jax.py
python -I tools/validate_gpu_environment.py --expected-family ada \
  --junit outputs/custom-calls.xml --output outputs/gpu-environment.json
```

For WSL, replace `python -I` in the script commands above with
`python -I tools/run_wsl_gpu.py` and set
`XLA_FLAGS=--xla_gpu_enable_command_buffer=`. To run pytest through that
helper, create a script calling `pytest.main([...])` with the same arguments
and pass its filename. Change the expected family only when the physical
hardware matches it. Pascal, Volta, Turing, Ampere and Hopper remain
unvalidated on physical hardware in this record.
