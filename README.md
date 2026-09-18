# FDTDZ

FDTDZ is a fast, scalable electromagnetic finite-difference time-domain
(FDTD) simulator for photonics research. This repository contains the
CUDA-backed solver and NumPy/JAX implementations for 2D and 3D workflows,
including dispersive materials, Bloch boundaries, diffraction and
metasurface examples.

## Features

- CUDA custom calls for the original high-throughput solver.
- Pure-Python NumPy/JAX ADE solvers for portable CPU and GPU workflows.
- 2D TEz/TMz and 3D simulations, material fitting and physical-unit tools.
- Examples, regression tests, package checks and GitHub Actions CI.

## Requirements

- Python 3.10 or later
- JAX and jaxlib 0.4.32 through 0.6.x
- A C++ compiler, CMake and CUDA toolkit when building the optional CUDA extension

## Install

For a pure-Python installation (no CUDA compilation):

```bash
FDTDZ_BUILD_CUDA=0 python -m pip install .
```

For a CUDA-enabled installation, install a JAX build compatible with your CUDA
environment, then install this package normally:

```bash
python -m pip install "jax[cuda12]"
python -m pip install .
```

Set `FDTDZ_CUDA_ARCHITECTURES` to the CMake architectures appropriate for
your hardware when needed (for example, `75;80;86;89`).

## Quick start

Run an example directly from a source checkout:

```bash
PYTHONPATH=src python examples/metasurface_2d_tez.py
```

More examples are in [`examples/`](examples), and focused guides are in
[`docs/`](docs), including GPU validation, material fitting, oblique incidence
and JAX profiling.

## Testing

```bash
python -m pip install -e ".[test]"
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q
```

CUDA validation requires a compatible NVIDIA GPU. See
[`docs/gpu_validation.md`](docs/gpu_validation.md) for the supported workflow
and hardware caveats.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md),
follow the [Code of Conduct](CODE_OF_CONDUCT.md), and open an issue before
starting substantial changes.

## License

FDTDZ is released under the [MIT License](LICENSE). The original project is
maintained by [SPINS Photonics](https://github.com/spinsphotonics/fdtdz).
