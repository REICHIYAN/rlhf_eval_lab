from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _rm(path: str) -> None:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)


@pytest.mark.integration
def test_hf_e2e_optional_run_report_validate() -> None:
    """
    HF optional E2E: run -> report -> validate
    - Skips by default (so fallback-only CI stays fast and offline-safe)
    - Runs only when:
        (a) transformers is installed, AND
        (b) RUN_HF_TESTS=1 is set
    """
    # (a) skip if transformers not installed
    pytest.importorskip("transformers")

    # (b) skip unless explicitly enabled
    if os.environ.get("RUN_HF_TESTS", "").strip() != "1":
        pytest.skip("HF optional test disabled (set RUN_HF_TESTS=1 to enable).")

    # Clean outputs to avoid cross-test contamination
    _rm("artifacts")
    _rm("reports")

    # Run HF preset (paper_hh) end-to-end
    subprocess.run(
        ["rlhf-lab", "run", "--backend", "hf", "--preset", "paper_hh", "--seed", "0"],
        check=True,
    )
    subprocess.run(["rlhf-lab", "report"], check=True)
    subprocess.run(["rlhf-lab", "validate"], check=True)

    # Assert report exists (basic sanity)
    assert Path("reports/report.md").exists()
