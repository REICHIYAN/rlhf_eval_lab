from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from rlhf_eval_lab.eval.runner import evaluate_artifacts
from rlhf_eval_lab.reporting.artifacts import read_artifacts


def _rm_dir(path: str) -> None:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)


@pytest.mark.integration
def test_offline_harmbench_small_e2e_run_report_validate_table2c_numeric() -> None:
    _rm_dir("artifacts")
    _rm_dir("reports")

    subprocess.run(
        [
            "rlhf-lab",
            "run",
            "--backend",
            "fallback",
            "--preset",
            "offline_harmbench_small",
            "--seed",
            "0",
        ],
        check=True,
    )
    subprocess.run(["rlhf-lab", "report"], check=True)
    subprocess.run(["rlhf-lab", "validate", "--report", "reports"], check=True)

    assert Path("reports/report.md").exists()

    art = read_artifacts("artifacts/ppo_standard/seed_0.json")
    ex = art.extra or {}
    assert ex.get("dataset_key") == "harmbench:eval:local"
    assert isinstance(ex.get("dataset_hash"), str) and len(ex.get("dataset_hash")) >= 8

    metrics = evaluate_artifacts(art)

    # Table 2-C must be numeric (not N/A) for PPO-family.
    assert isinstance(metrics.get("prompt_injection"), float)
    assert 0.0 <= float(metrics["prompt_injection"]) <= 1.0

    assert isinstance(metrics.get("ood_stability"), float)
    assert 0.0 <= float(metrics["ood_stability"]) <= 1.0
