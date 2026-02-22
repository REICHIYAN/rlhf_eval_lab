# tests/integration/test_e2e_fallback.py
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    # tests/integration/test_e2e_fallback.py -> repo root
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def test_e2e_fallback_run_validate_report() -> None:
    root = _repo_root()

    # Ensure clean
    for d in ["artifacts", "reports"]:
        p = root / d
        if p.exists():
            shutil.rmtree(p)

    rlhf_lab = shutil.which("rlhf-lab")
    assert rlhf_lab is not None, "rlhf-lab command not found. Did you run `pip install -e .`?"

    # E2E
    _run([rlhf_lab, "run", "--backend", "fallback", "--seed", "0"], cwd=root)
    _run([rlhf_lab, "report"], cwd=root)

    # Strict validation is the DoD gate (artifacts + report.md)
    _run([rlhf_lab, "validate", "--report", "reports"], cwd=root)

    report_path = root / "reports" / "report.md"
    assert report_path.exists(), "reports/report.md not generated"
