from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest


def _rm(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.integration
def test_hf_e2e_optional_run_report_validate() -> None:
    """
    HF optional E2E: run -> report -> validate

    - Skips by default (so fallback-only CI stays fast and offline-safe)
    - Runs only when:
        (a) transformers is installed, AND
        (b) RUN_HF_TESTS=1 is set

    Additional invariants (C1.8):
    - PPO diagnostics must be *post-update* and auditable.
    - KL proxy diagnostics must be nonnegative (kl_ref_abs, kl_ref_sq).
    - run pipeline must prefer kl_ref_abs as "kl" for HF PPO.
    """
    # (a) skip if transformers not installed
    pytest.importorskip("transformers")

    # (b) skip unless explicitly enabled
    if os.environ.get("RUN_HF_TESTS", "").strip() != "1":
        pytest.skip("HF optional test disabled (set RUN_HF_TESTS=1 to enable).")

    # Clean outputs to avoid cross-test contamination
    _rm("artifacts")
    _rm("reports")

    # Run HF preset (paper_hh_ppo) end-to-end
    subprocess.run(
        ["rlhf-lab", "run", "--backend", "hf", "--preset", "paper_hh_ppo", "--seed", "0"],
        check=True,
    )
    subprocess.run(["rlhf-lab", "report"], check=True)
    subprocess.run(["rlhf-lab", "validate"], check=True)

    # Assert report exists (basic sanity)
    report_path = Path("reports/report.md")
    assert report_path.exists()

    # ---- Audit invariants for HF PPO diagnostics ----
    art_path = Path("artifacts/ppo_standard/seed_0.json")
    assert art_path.exists(), f"Expected artifacts missing: {art_path}"

    art = _load_json(art_path)
    extra = (art.get("extra") or {})  # type: ignore[assignment]
    assert isinstance(extra, dict), "Artifacts extra must be a dict"

    # PPO should be executed (paper_hh_ppo enables hf_ppo_steps)
    assert extra.get("skipped") is False
    assert extra.get("steps") is not None and int(extra["steps"]) > 0  # type: ignore[index]

    # Required keys for post-update diagnostics
    required = [
        "ratio_mean_pre",
        "ratio_mean",
        "kl_ref_pre",
        "kl_ref",
        "kl_ref_abs",
        "kl_ref_sq",
        "clipfrac",
        "ppo_loss",
        "kl",
    ]
    missing = [k for k in required if k not in extra]
    assert not missing, f"Missing required PPO diagnostics keys: {missing}"

    ratio_pre = float(extra["ratio_mean_pre"])
    ratio_post = float(extra["ratio_mean"])
    kl_abs = float(extra["kl_ref_abs"])
    kl_sq = float(extra["kl_ref_sq"])
    kl = float(extra["kl"])

    # Pre-update ratio should be ~1 (same params, dropout disabled)
    assert ratio_pre == pytest.approx(1.0, abs=1e-6)

    # Nonnegative KL proxies (no sign confusion in report interpretation)
    assert kl_abs >= 0.0
    assert kl_sq >= 0.0

    # Run pipeline policy: prefer abs proxy as "kl" for HF PPO
    assert kl == pytest.approx(kl_abs, abs=1e-12)

    # Sanity: ratio_post is a diagnostic, should be finite
    assert ratio_post == ratio_post  # not NaN
    assert ratio_post != float("inf")
    assert ratio_post != float("-inf")

def test_hf_kl_ppo_fixed_minimal_training_optional(tmp_path, monkeypatch):
    import os
    import json
    import subprocess
    import sys

    if os.environ.get("RUN_HF_TESTS") != "1":
        import pytest
        pytest.skip("HF optional test is disabled (set RUN_HF_TESTS=1)")

    # Ensure we run from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(repo_root)

    # Clean outputs
    subprocess.check_call(["rm", "-rf", "artifacts", "reports"])

    # Run HF preset (minimal training for kl_ppo_fixed)
    subprocess.check_call(
        ["rlhf-lab", "run", "--backend", "hf", "--preset", "hf_offline_klppo_fixed", "--seed", "0"],
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )
    subprocess.check_call(["rlhf-lab", "validate"], env=dict(os.environ, PYTHONUNBUFFERED="1"))
    subprocess.check_call(["rlhf-lab", "report"], env=dict(os.environ, PYTHONUNBUFFERED="1"))

    # Verify auditability: training actually executed
    p = os.path.join("artifacts", "kl_ppo_fixed", "seed_0.json")
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    extra = d.get("extra", {})
    assert extra.get("skipped") is False
    assert int(extra.get("steps", 0)) >= 1
