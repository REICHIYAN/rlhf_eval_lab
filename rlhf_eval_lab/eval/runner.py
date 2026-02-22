# rlhf_eval_lab/eval/runner.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

from rlhf_eval_lab.reporting.artifacts import ArtifactsV1
from rlhf_eval_lab.registry.methods import METHOD_BY_KEY, METHOD_SPECS
from rlhf_eval_lab.registry.metrics import METRIC_BY_KEY

from rlhf_eval_lab.eval.offsupport import compute_offsupport
from rlhf_eval_lab.eval.tailvar import compute_tail_var
from rlhf_eval_lab.eval.onsupport import compute_onsupport
from rlhf_eval_lab.eval.judge import compute_judge
from rlhf_eval_lab.eval.preference import compute_win_rate
from rlhf_eval_lab.eval.kl import compute_kl

from rlhf_eval_lab.eval.diagnostics import (
    compute_kl_stability,
    compute_reward_var,
    compute_convergence_speed,
    compute_sample_efficiency,
    compute_reward_accuracy,
    label_source_for_method,
)

from rlhf_eval_lab.eval.safety import (
    compute_prompt_injection,
    compute_ood_stability,
)


NAValue = str
MetricValue = Union[float, NAValue]


def _na() -> str:
    return "N/A"


def _is_na(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == "" or x.strip().upper() in {"N/A", "NA", "NONE", "NULL", "NAN"}
    return False


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        try:
            return bool(int(x))
        except Exception:
            return False
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, bool):
            return int(x)
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)
    except Exception:
        return int(default)


def _as_float(x: Any) -> Optional[float]:
    """
    Best-effort conversion to float.
    Returns None if not convertible.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s.upper() == "N/A" or s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            return None
    return None


def _finite_or_none(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if not math.isfinite(float(x)):
        return None
    return float(x)


def _finite_or_na(x: Optional[float]) -> MetricValue:
    v = _finite_or_none(x)
    return v if v is not None else _na()


def _compute_ppl_fallback(art: ArtifactsV1, out: Dict[str, Any]) -> float:
    """
    Phase B-1 invariant: PPL must exist for ALL methods.

    Priority (if present in extra):
      1) extra["ppl"] (already computed upstream)
      2) extra["nll"] or extra["mean_nll"]
      3) extra["loss"] (proxy; always numeric in our minimal pipelines)
      4) deterministic constant fallback (1.0)

    Then:
      ppl = exp(clamp(nll_like, min=1e-8, max=50))  # avoid under/overflow
    """
    _ = out  # reserved for future use
    extra = art.extra or {}

    ppl_direct = _finite_or_none(_as_float(extra.get("ppl")))
    if ppl_direct is not None:
        if ppl_direct <= 0:
            return 1.0
        return float(ppl_direct)

    nll_like = _as_float(extra.get("nll"))
    if nll_like is None:
        nll_like = _as_float(extra.get("mean_nll"))
    if nll_like is None:
        nll_like = _as_float(extra.get("loss"))

    if nll_like is None or not math.isfinite(float(nll_like)):
        nll_like = 0.0

    nll_like = float(nll_like)
    nll_like = max(nll_like, 1e-8)
    nll_like = min(nll_like, 50.0)

    ppl = float(math.exp(nll_like))
    if not math.isfinite(ppl) or ppl <= 0:
        ppl = 1.0
    return ppl


def _get_extra_float(extra: Dict[str, Any], key: str) -> Optional[float]:
    return _finite_or_none(_as_float(extra.get(key)))


def evaluate_artifacts(art: ArtifactsV1) -> Dict[str, Any]:
    method = METHOD_BY_KEY[art.method_key]
    out: Dict[str, Any] = {}

    extra = art.extra or {}

    # ---------------------------------------------------------
    # Global execution flags (audit SSOT)
    # ---------------------------------------------------------
    skipped = _as_bool(extra.get("skipped"))
    steps = _as_int(extra.get("steps"), 0)
    skip_reason = str(extra.get("skip_reason") or "").strip()
    training_executed = (not skipped) and steps > 0

    # Notes must exist per-seed (DoD). Prefer explicit skip_reason if skipped.
    out["notes"] = "-" if not skipped else (skip_reason if skip_reason else "skipped")

    # --------------------------
    # Phase B3 invariant: Latency must exist for ALL methods once enabled in MetricSpec
    # --------------------------
    lat = _get_extra_float(extra, "latency_ms")
    if lat is None:
        # Hard fail: latency is a Table 1 metric and must not be missing.
        raise ValueError(f"Metric missing: latency_ms (method={art.method_key})")
    out["latency_ms"] = float(lat)

    # --------------------------
    # Table 1 (core)
    # --------------------------
    off = compute_offsupport(art.prompts, art.completions)
    on = compute_onsupport(art.rewards)

    out["off_support"] = off
    out["tail_var"] = compute_tail_var(art.rewards)
    out["on_support"] = on

    # Backward-compatible aliases (legacy)
    out["offsupport"] = off
    out["onsupport"] = on

    out["judge"] = compute_judge(
        prompts=art.prompts,
        completions=art.completions,
        rewards=art.rewards,
        extra=extra,
    )
    out["win_rate"] = compute_win_rate(art.rewards, extra)

    if art.method_key in (METRIC_BY_KEY["kl"].na_for_method_keys or []):
        out["kl"] = _na()
    else:
        # compute_kl() is SSOT for Table 1 KL column.
        out["kl"] = compute_kl(extra)

    # --------------------------
    # Phase B-1 invariant: PPL must exist for ALL methods
    # --------------------------
    out["ppl"] = _compute_ppl_fallback(art, out)

    # ---- PPO-family diagnostics policy
    # DoD fix: adaptive_rm_ppo must behave like PPO-family for Table 2-A.
    is_ppo_like = bool(getattr(method, "is_ppo_family", False)) or (art.method_key == "adaptive_rm_ppo")

    # --------------------------
    # Table 2-A (Audit-friendly PPO diagnostics)
    #
    # Critical policy:
    #   - If PPO-family training was NOT executed (skipped or steps==0),
    #     Table 2-A MUST be N/A (not 0.0). 0.0 is misleading in audits.
    # --------------------------
    if is_ppo_like:
        if not training_executed:
            # Training not executed -> audit metrics are not applicable.
            out["ppo_loss"] = _na()
            out["ratio_mean"] = _na()
            out["clipfrac"] = _na()
            out["kl_ref_abs"] = _na()
            out["kl_ref_sq"] = _na()

            # Diagnostics tied to training dynamics should also be N/A.
            out["kl_stability"] = _na()
            out["convergence_speed"] = _na()

            # Reward variance is still well-defined from produced rewards.
            out["reward_var"] = compute_reward_var(art.rewards)
        else:
            # Prefer HF-style audit fields when present; fall back to best-effort.
            ppo_loss = _get_extra_float(extra, "ppo_loss")
            ratio_mean = _get_extra_float(extra, "ratio_mean")
            clipfrac = _get_extra_float(extra, "clipfrac")

            kl_ref_abs = _get_extra_float(extra, "kl_ref_abs")
            kl_ref_sq = _get_extra_float(extra, "kl_ref_sq")

            # Best-effort fallback for abs/sq proxies if missing.
            # (Never invent if KL itself is N/A.)
            if kl_ref_abs is None:
                kl_val = out.get("kl")
                kl_num = _finite_or_none(_as_float(kl_val)) if not _is_na(kl_val) else None
                if kl_num is not None:
                    kl_ref_abs = abs(float(kl_num))

            if kl_ref_sq is None and kl_ref_abs is not None:
                kl_ref_sq = float(kl_ref_abs) * float(kl_ref_abs)

            out["ppo_loss"] = _finite_or_na(ppo_loss)
            out["ratio_mean"] = _finite_or_na(ratio_mean)
            out["clipfrac"] = _finite_or_na(clipfrac)
            out["kl_ref_abs"] = _finite_or_na(kl_ref_abs)
            out["kl_ref_sq"] = _finite_or_na(kl_ref_sq)

            # Keep legacy diagnostics too (harmless even if markdown doesn't show them).
            out["kl_stability"] = compute_kl_stability(extra)  # float (may be NaN)
            out["reward_var"] = compute_reward_var(art.rewards)  # float
            out["convergence_speed"] = compute_convergence_speed(extra)  # float (may be NaN)
    else:
        out["ppo_loss"] = _na()
        out["ratio_mean"] = _na()
        out["clipfrac"] = _na()
        out["kl_ref_abs"] = _na()
        out["kl_ref_sq"] = _na()

        out["kl_stability"] = _na()
        out["reward_var"] = _na()
        out["convergence_speed"] = _na()

    # --------------------------
    # Table 2-B (Preference-based diagnostics)
    # --------------------------
    if getattr(method, "is_preference_based", False) or getattr(method, "is_active", False):
        out["sample_efficiency"] = compute_sample_efficiency(extra)  # float (may be NaN)
        out["reward_accuracy"] = compute_reward_accuracy(extra)  # float (may be NaN)
        out["label_source"] = label_source_for_method(art.method_key)  # str
    else:
        out["sample_efficiency"] = _na()
        out["reward_accuracy"] = _na()
        out["label_source"] = _na()

    # --------------------------
    # Table 2-C (Safety / robustness)
    #
    # Note: Even if training is skipped, these can still be computed from
    # generated completions/rewards, so we keep them enabled for PPO-family.
    # --------------------------
    if getattr(method, "is_safety", False) or is_ppo_like:
        out["prompt_injection"] = compute_prompt_injection(art.prompts, art.completions, extra)
        out["ood_stability"] = compute_ood_stability(art.completions, art.rewards, extra)
    else:
        out["prompt_injection"] = _na()
        out["ood_stability"] = _na()

    # no None (empty forbidden)
    for k, v in out.items():
        if v is None:
            raise ValueError(f"Metric {k} returned None (empty forbidden)")

    return out


def build_table_rows(aggregated: Dict[str, Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    """
    Backward-compatible helper for table row materialization.

    Note: report.md is rendered by reporting/markdown.py, but we keep this for
    tooling and potential downstream consumption.
    """
    rows: Dict[str, List[List[str]]] = {
        "table1": [],
        "table2a": [],
        "table2b": [],
        "table2c": [],
    }

    for m in METHOD_SPECS:
        a = aggregated[m.key]

        rows["table1"].append(
            [
                m.category,
                m.name,
                str(a.get("off_support", a.get("offsupport", "N/A"))),
                str(a.get("tail_var", "N/A")),
                str(a.get("on_support", a.get("onsupport", "N/A"))),
                str(a.get("judge", "N/A")),
                str(a.get("win_rate", "N/A")),
                str(a.get("kl", "N/A")),
                str(a.get("ppl", "N/A")),
                str(a.get("latency_ms", "N/A")),
                str(a.get("notes", "-")) or "-",
            ]
        )

        rows["table2a"].append(
            [
                m.name,
                str(a.get("ppo_loss", "N/A")),
                str(a.get("ratio_mean", "N/A")),
                str(a.get("clipfrac", "N/A")),
                str(a.get("kl_ref_abs", "N/A")),
                str(a.get("kl_ref_sq", "N/A")),
            ]
        )

        rows["table2b"].append(
            [
                m.name,
                str(a.get("sample_efficiency", "N/A")),
                str(a.get("reward_accuracy", "N/A")),
                str(a.get("label_source", "N/A")),
            ]
        )

        rows["table2c"].append(
            [
                m.name,
                str(a.get("prompt_injection", "N/A")),
                str(a.get("ood_stability", "N/A")),
            ]
        )

    return rows