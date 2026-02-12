# tests/integration/test_e2e_level_c_hf.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _hf_tests_enabled() -> bool:
    return os.environ.get("RUN_HF_TESTS", "").strip() == "1"


@pytest.mark.integration
def test_e2e_level_c_hf_preset_produces_phase_a_artifacts(tmp_path: Path) -> None:
    """
    Level-C (research) smoke test:
    - Run CLI SSOT entrypoint with HF preset
    - Ensure Phase-A artifacts exist (dpo/ipo/rrhf/orpo/rlaif)
    - Sanity-check provenance fields exist in one artifact

    Notes:
    - This test assumes requirements-hf.txt is installed.
    - It may download HF weights on first run if not cached.
    """
    if not _has_transformers():
        pytest.skip("transformers is not installed; skip HF integration test.")

    # Keep default CI / local runs fast. Enable explicitly.
    if not _hf_tests_enabled():
        pytest.skip("HF integration test disabled (set RUN_HF_TESTS=1 to enable).")

    out_dir = tmp_path / "artifacts_hf"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Keep CI logs clean / deterministic-ish
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        sys.executable,
        "-m",
        "rlhf_eval_lab.cli.main",
        "run",
        "--preset",
        "hf_gpt2",
        "--seed",
        "0",
        "--out",
        str(out_dir),
    ]

    subprocess.run(cmd, check=True, env=env, cwd=str(Path.cwd()))

    # Phase A must exist
    required = ["dpo", "ipo", "rrhf", "orpo", "rlaif"]
    for m in required:
        p = out_dir / m / "seed_0.json"
        assert p.exists(), f"missing artifact: {p}"

    # Minimal schema sanity: provenance + config hash exist
    sample_path = out_dir / "dpo" / "seed_0.json"
    with sample_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    assert "provenance" in obj, "artifact missing 'provenance'"
    prov = obj["provenance"]
    assert isinstance(prov, dict), "provenance must be a dict"
    assert "config_hash" in prov and prov["config_hash"], "provenance.config_hash missing/empty"
