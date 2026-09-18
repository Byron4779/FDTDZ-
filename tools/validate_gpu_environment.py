"""Require a physical NVIDIA GPU and record installed-extension CI evidence."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def device_family(capability):
  major, minor = map(int, capability.split("."))
  if major == 6:
    return "pascal"
  if (major, minor) == (7, 0):
    return "volta"
  if (major, minor) == (7, 5):
    return "turing"
  if major == 8:
    return "ada" if minor == 9 else "ampere"
  if major == 9:
    return "hopper"
  raise ValueError(f"Unvalidated NVIDIA compute capability: {capability}")


def execution_summary(path):
  cases = ET.parse(path).getroot().findall(".//testcase")
  if not cases:
    raise ValueError("CUDA test report contains no tests")
  if any(case.find(tag) is not None for case in cases
         for tag in ("skipped", "failure", "error")):
    raise ValueError("CUDA test report contains skipped or unsuccessful tests")
  execution_cases = [case for case in cases
                     if case.attrib["name"].startswith("test_custom_call_precision_and_jit[")]
  if len(execution_cases) != 6:
    raise ValueError("Require all six fp32/fp16, x/y/z execution cases")
  return {"passed": len(cases), "skipped": 0,
          "precision_and_jit_cases": len(execution_cases)}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--expected-family", required=True,
                      choices=["pascal", "volta", "turing", "ampere", "ada", "hopper"])
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--junit", type=Path)
  args = parser.parse_args()

  import jax
  import jaxlib
  from fdtdz_jax import gpu_ops

  devices = jax.devices()
  if not devices or any(d.platform != "gpu" for d in devices):
    raise RuntimeError(f"Require GPU execution, found {devices}")
  if set(gpu_ops.registrations()) != {"kernel_f32", "kernel_f16"}:
    raise RuntimeError("Installed CUDA extension lacks the two kernel targets")
  native_device = gpu_ops.device_info(getattr(devices[0], "local_hardware_id", devices[0].id))
  result = subprocess.run(
      ["nvidia-smi", "--query-gpu=uuid,name,compute_cap,driver_version", "--format=csv,noheader"],
      check=True, capture_output=True, text=True)
  physical = []
  for line in result.stdout.strip().splitlines():
    uuid, name, capability, driver = (part.strip() for part in line.split(","))
    family = device_family(capability)
    if family != args.expected_family:
      raise RuntimeError(f"Expected {args.expected_family}, found {family}: {name}")
    physical.append(dict(uuid=uuid, name=name, compute_capability=capability,
                         family=family, driver=driver))
  if not physical:
    raise RuntimeError("nvidia-smi reported no physical GPUs")
  report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                python=sys.version, jax=jax.__version__, jaxlib=jaxlib.__version__,
                devices=[str(d) for d in devices], physical_gpus=physical,
                native_device=native_device,
                extension=gpu_ops.__file__, registration_api_version=0,
                lowering_api_version=2,
                custom_call_tests=execution_summary(args.junit) if args.junit else None)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
