"""Check registration without requiring a compiled extension or GPU."""

import os
from pathlib import Path
import subprocess
import sys


def test_cuda_registration_preserves_legacy_abi():
  script = """
import importlib
import sys
import types
try:
  from jax import ffi
except ImportError:
  from jax.extend import ffi

calls = []
ffi.register_ffi_target = lambda *args, **kwargs: calls.append((args, kwargs))
extension = types.ModuleType('fdtdz_jax.gpu_ops')
targets = {'kernel_f32': object(), 'kernel_f16': object()}
extension.registrations = lambda: targets
sys.modules[extension.__name__] = extension
module = importlib.import_module('fdtdz_jax.fdtdz_jax')
assert module.gpu_ops is extension
assert calls == [((name, capsule), {'platform': 'CUDA', 'api_version': 0})
                 for name, capsule in targets.items()], calls
"""
  env = dict(os.environ)
  env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
  result = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=60)
  assert result.returncode == 0, result.stdout + result.stderr
