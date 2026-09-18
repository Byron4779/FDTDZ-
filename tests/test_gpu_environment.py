import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "validate_gpu_environment", Path(__file__).resolve().parents[1]
    / "tools" / "validate_gpu_environment.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


@pytest.mark.parametrize("capability,family", [
    ("6.0", "pascal"), ("7.0", "volta"), ("7.5", "turing"),
    ("8.0", "ampere"), ("8.6", "ampere"), ("8.9", "ada"), ("9.0", "hopper")])
def test_physical_device_family(capability, family):
  assert validation.device_family(capability) == family


@pytest.mark.parametrize("tag", ["skipped", "failure", "error"])
def test_reject_unsuccessful_gpu_reports(tmp_path, tag):
  report = tmp_path / "report.xml"
  report.write_text(f'<testsuite><testcase name="gpu"><{tag}/></testcase></testsuite>')
  with pytest.raises(ValueError, match="skipped or unsuccessful"):
    validation.execution_summary(report)


def test_require_precision_execution_cases(tmp_path):
  report = tmp_path / "report.xml"
  report.write_text('<testsuite><testcase name="test_top_level_api_exports"/></testsuite>')
  with pytest.raises(ValueError, match="all six"):
    validation.execution_summary(report)
