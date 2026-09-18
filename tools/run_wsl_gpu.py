"""Opt-in WSL system-driver initialization before running a Python GPU script.

Usage: python tools/run_wsl_gpu.py examples/profile_jax_gpu_3d.py [arguments]
This process-only workaround neither replaces driver files nor changes JAX.
It avoids LD_PRELOAD, which stalled this machine's first GPU transfer.
"""

import argparse
import ctypes
import faulthandler
import os
from pathlib import Path
import runpy
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stack-timeout", type=float, default=0,
                      help="Dump Python stacks after this many seconds for diagnostics.")
  parser.add_argument("script", type=Path)
  parser.add_argument("arguments", nargs=argparse.REMAINDER)
  args = parser.parse_args()
  if args.stack_timeout > 0:
    faulthandler.dump_traceback_later(args.stack_timeout)
  library = Path("/usr/lib/wsl/lib/libcuda.so.1")
  if sys.platform != "linux" or not library.is_file():
    parser.error("This helper requires WSL with its system NVIDIA libcuda.so.1.")
  if not args.script.is_file():
    parser.error(f"Script does not exist: {args.script}")
  driver = ctypes.CDLL(str(library))
  driver.cuInit.argtypes = [ctypes.c_uint]
  driver.cuInit.restype = ctypes.c_int
  status = driver.cuInit(0)
  if status != 0:
    raise SystemExit(f"WSL CUDA initialization failed with driver status {status}.")
  count = ctypes.c_int()
  driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
  driver.cuDeviceGetCount.restype = ctypes.c_int
  status = driver.cuDeviceGetCount(ctypes.byref(count))
  if status != 0 or count.value < 1:
    raise SystemExit(f"WSL CUDA device enumeration failed: status={status}, count={count.value}.")
  os.environ.setdefault("JAX_PLATFORMS", "cuda")
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  print(f"WSL system CUDA driver initialized; devices={count.value}", flush=True)
  sys.argv = [str(args.script), *args.arguments]
  try:
    runpy.run_path(str(args.script), run_name="__main__")
  finally:
    if args.stack_timeout > 0:
      faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
  main()
