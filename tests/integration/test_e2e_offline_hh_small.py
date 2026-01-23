from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _rm_dir(path: str) -> None:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)


@pytest.mark.integration
def test_offline_hh_small_e2e_run_report_validate() -> None:
    _rm_dir("artifacts")
    _rm_dir("reports")

    subprocess.run(
        ["rlhf-lab", "run", "--backend", "fallback", "--preset", "offline_hh_small", "--seed", "0"],
        check=True,
    )
    subprocess.run(["rlhf-lab", "report"], check=True)
    subprocess.run(["rlhf-lab", "validate"], check=True)

    assert Path("reports/report.md").exists()

    ex = json.load(open("artifacts/ppo_standard/seed_0.json", encoding="utf-8")).get("extra", {})
    assert ex.get("dataset_key") == "hh_rlhf:train:local"
    assert isinstance(ex.get("dataset_hash"), str) and len(ex.get("dataset_hash")) >= 8
