"""Tests for Azure ML pipeline helper scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_script(script_path: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a repository script as a subprocess."""
    return subprocess.run(
        [sys.executable, script_path, *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_prepare_sample_data_script_creates_expected_files(tmp_path):
    output_data = tmp_path / "sample_data"

    result = run_script(
        "mlops/azureml/scripts/prepare_sample_data.py",
        "--output_data",
        str(output_data),
    )

    assert "Prepared deterministic Azure ML sample data." in result.stdout
    assert (output_data / "sample_agents.csv").exists()
    assert (output_data / "sample.db").exists()
    assert (output_data / "eval_questions.jsonl").exists()
    assert (output_data / "manifest.json").exists()

    manifest = json.loads((output_data / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["row_count"] == 20
    assert manifest["column_count"] == 7


def test_evaluate_agent_tools_script_writes_reports(tmp_path):
    input_data = tmp_path / "sample_data"
    output_report = tmp_path / "evaluation_report"

    run_script(
        "mlops/azureml/scripts/prepare_sample_data.py",
        "--output_data",
        str(input_data),
    )

    result = run_script(
        "mlops/azureml/scripts/evaluate_agent_tools.py",
        "--input_data",
        str(input_data),
        "--output_report",
        str(output_report),
    )

    assert "DSA agent evaluation completed." in result.stdout
    assert "passed=True" in result.stdout

    report_json_path = output_report / "evaluation_report.json"
    report_md_path = output_report / "evaluation_report.md"

    assert report_json_path.exists()
    assert report_md_path.exists()
    assert (output_report / "plots" / "histogram_score_by_target.png").exists()
    assert (output_report / "plots" / "kde_score_by_target.png").exists()
    assert (output_report / "plots" / "correlation_heatmap.png").exists()

    report = json.loads(report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["checks"]["overview_has_20_rows"] is True
    assert report["checks"]["sql_employee_count_is_4"] is True