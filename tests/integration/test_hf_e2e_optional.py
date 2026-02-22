from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

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
        d = json.load(f)
    if not isinstance(d, dict):
        raise AssertionError(f"Expected JSON object at {path}, got {type(d)}")
    return d


def _repo_root() -> Path:
    # tests/integration/test_hf_e2e_optional.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


@contextmanager
def _chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _hf_tests_enabled() -> bool:
    return os.environ.get("RUN_HF_TESTS", "").strip() == "1"


def _run(cmd: list[str], *, env: Mapping[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _env_unbuffered() -> Dict[str, str]:
    e = dict(os.environ)
    e["PYTHONUNBUFFERED"] = "1"
    # optional: reduce noisy tokenizer parallel warnings
    e.setdefault("TOKENIZERS_PARALLELISM", "false")
    return e


def _require_hf_optional() -> None:
    # (a) skip if transformers not installed
    pytest.importorskip("transformers")
    # (b) skip unless explicitly enabled
    if not _hf_tests_enabled():
        pytest.skip("HF optional test disabled (set RUN_HF_TESTS=1 to enable).")


def _clean_outputs() -> None:
    # Avoid cross-test contamination from prior runs.
    _rm("artifacts")
    _rm("reports")


def _run_hf_offline_e2e(preset: str) -> None:
    """Run HF offline preset E2E and validate report.md (optional test).

    We keep this *strict* but *light*:
      - `run` with tiny HF model + bundled prompts
      - `report`
      - `validate --report reports` (artifacts + report.md)
    """
    env = _env_unbuffered()

    _run(["rlhf-lab", "run", "--backend", "hf", "--preset", preset, "--seed", "0"], env=env)
    _run(["rlhf-lab", "report"], env=env)
    _run(["rlhf-lab", "validate", "--report", "reports"], env=env)

    report_path = Path("reports/report.md")
    assert report_path.exists(), "Expected reports/report.md to be generated"


@pytest.mark.integration
@pytest.mark.parametrize(
    "preset,artifact_path",
    [
        ("hf_offline_klppo_fixed", "artifacts/kl_ppo_fixed/seed_0.json"),
        ("hf_offline_klppo_adaptive", "artifacts/kl_ppo_adaptive/seed_0.json"),
    ],
)
def test_hf_offline_klppo_fixed_and_adaptive_e2e_optional(preset: str, artifact_path: str) -> None:
    """HF optional E2E (offline/bundled) for KL-PPO (fixed + adaptive).

    This is the C2t gate: when explicitly enabled, we ensure both presets
    execute PPO (not skipped) and stamp auditable PPO diagnostics.

    Enabled when:
      - transformers is installed, AND
      - RUN_HF_TESTS=1
    """
    _require_hf_optional()

    root = _repo_root()
    with _chdir(root):
        _clean_outputs()
        _run_hf_offline_e2e(preset)

        art_path = Path(artifact_path)
        assert art_path.exists(), f"Expected artifacts missing: {art_path}"

        art = _load_json(art_path)
        extra_any = art.get("extra") or {}
        assert isinstance(extra_any, dict), "Artifacts extra must be a dict"
        extra: Dict[str, Any] = extra_any

        # PPO should be executed
        assert extra.get("skipped") is False
        assert int(extra.get("steps", 0)) >= 1

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
        clipfrac = float(extra["clipfrac"])
        ppo_loss = float(extra["ppo_loss"])
        kl = float(extra["kl"])

        # Pre-update ratio should be ~1 (same params).
        # Keep tolerance slightly loose for numerical stability across platforms.
        assert ratio_pre == pytest.approx(1.0, abs=1e-4)

        # Nonnegative KL proxies
        assert kl_abs >= 0.0
        assert kl_sq >= 0.0

        # clipfrac must be a probability-like value
        assert 0.0 <= clipfrac <= 1.0

        # Reporting policy: prefer abs proxy as "kl"
        assert kl == pytest.approx(kl_abs, abs=1e-9)

        # Sanity: key scalars must be finite (not NaN/Inf)
        for v in (ratio_post, ppo_loss, kl_abs, kl_sq):
            assert v == v  # not NaN
            assert v not in (float("inf"), float("-inf"))


@pytest.mark.integration
def test_hf_offline_kl_ppo_adaptive_beta_updates_optional() -> None:
    """
    HF optional E2E (offline/bundled): KL-PPO adaptive must update beta and stamp it into extra.

    DoD (for this test):
    - skipped=False, steps>=1
    - beta_pre/beta_post/kl_target/kl_measured/beta_lr/beta_clip exist
    - beta_pre != beta_post (update happened)
    """
    _require_hf_optional()

    root = _repo_root()
    with _chdir(root):
        _clean_outputs()
        _run_hf_offline_e2e("hf_offline_klppo_adaptive")

        art_path = Path("artifacts/kl_ppo_adaptive/seed_0.json")
        assert art_path.exists(), f"Expected artifacts missing: {art_path}"

        art = _load_json(art_path)
        extra_any = art.get("extra") or {}
        assert isinstance(extra_any, dict), "Artifacts extra must be a dict"
        extra: Dict[str, Any] = extra_any

        assert extra.get("skipped") is False
        assert int(extra.get("steps", 0)) >= 1

        for k in [
            "beta_init",
            "beta_pre",
            "beta_post",
            "kl_target",
            "kl_measured",
            "beta_lr",
            "beta_clip",
            "beta_min",
            "beta_max",
            "beta_ratio",
            "beta_adj",
        ]:
            assert k in extra, f"Missing adaptive audit key: {k}"

        beta_pre = float(extra["beta_pre"])
        beta_post = float(extra["beta_post"])

        beta_min = float(extra["beta_min"])
        beta_max = float(extra["beta_max"])

        assert beta_min > 0.0
        assert beta_max >= beta_min
        assert beta_min <= beta_pre <= beta_max
        assert beta_min <= beta_post <= beta_max

        # Update should occur (allow extremely tiny updates, but not identical).
        assert abs(beta_post - beta_pre) >= 1e-12, f"beta did not change: pre={beta_pre} post={beta_post}"
