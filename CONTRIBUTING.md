# Contributing to FDTDZ

Thanks for contributing. For bugs and feature proposals, first search existing
issues and open a concise issue describing the problem, environment and a
minimal reproducer where possible.

## Development workflow

1. Create a focused branch from `main`.
2. Install development dependencies: `python -m pip install -e ".[test]"`.
3. Run the CPU suite with `PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q`.
4. Add or update tests and documentation with any behavior change.
5. Open a pull request that explains the motivation, implementation and test
   results. Keep unrelated formatting or refactoring separate.

CUDA changes should identify the CUDA toolkit, GPU family and JAX/jaxlib
versions used. Do not claim GPU compatibility without execution on the stated
hardware; see `docs/gpu_validation.md`.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
