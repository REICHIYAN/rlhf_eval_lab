# rlhf_eval_lab/eval/diagnostics.py
# sanity tier diagnostics
# - numeric metrics MUST return float (never "N/A")
# - missing information -> NaN (validate should treat as numeric value; report renders as N/A)

from __future__ import annotations

from typing import Any, Dict, List
import math


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_kl_stability(extra: Dict[str, Any]) -> float:
    """Lower is better.
    Sanity tier proxy:
      - If extra contains 'kl_values' list -> return stddev
      - Else if extra contains 'kl' scalar -> return abs(kl)
      - Else -> NaN
    """
    if not isinstance(extra, dict):
        return float("nan")

    vals = extra.get("kl_values")
    if isinstance(vals, list) and vals:
        xs = [_to_float(v) for v in vals]
        xs = [v for v in xs if not math.isnan(v) and not math.isinf(v)]
        if len(xs) <= 1:
            return 0.0
        m = sum(xs) / len(xs)
        var = sum((v - m) ** 2 for v in xs) / (len(xs) - 1)
        return float(math.sqrt(max(0.0, var)))

    kl = extra.get("kl")
    if isinstance(kl, (int, float)):
        return float(abs(float(kl)))

    return float("nan")


def compute_reward_var(rewards: List[float]) -> float:
    """Lower is better. Variance of rewards."""
    if not rewards:
        return float("nan")
    xs = [float(r) for r in rewards]
    m = sum(xs) / len(xs)
    var = sum((v - m) ** 2 for v in xs) / float(max(1, len(xs) - 1))
    return float(var)


def compute_convergence_speed(extra: Dict[str, Any]) -> float:
    """Higher is better.
    Sanity tier proxy:
      - If extra contains 'steps' -> use 1/steps
      - Else if extra contains 'updates' -> use 1/updates
      - Else -> NaN
    """
    if not isinstance(extra, dict):
        return float("nan")

    steps = extra.get("steps")
    if isinstance(steps, (int, float)) and float(steps) > 0:
        return float(1.0 / float(steps))

    updates = extra.get("updates")
    if isinstance(updates, (int, float)) and float(updates) > 0:
        return float(1.0 / float(updates))

    return float("nan")


def compute_sample_efficiency(extra: Dict[str, Any]) -> float:
    """Higher is better.
    Sanity tier proxy:
      - If extra contains 'pairs_used' and 'win_rate' -> win_rate / pairs_used
      - Else -> NaN
    """
    if not isinstance(extra, dict):
        return float("nan")

    pairs = extra.get("pairs_used")
    win = extra.get("win_rate")
    if isinstance(pairs, (int, float)) and float(pairs) > 0 and isinstance(win, (int, float)):
        return float(float(win) / float(pairs))

    return float("nan")


def compute_reward_accuracy(extra: Dict[str, Any]) -> float:
    """Higher is better.
    Sanity tier proxy:
      - If extra contains 'reward_accuracy' -> return it
      - Else -> NaN
    """
    if not isinstance(extra, dict):
        return float("nan")

    ra = extra.get("reward_accuracy")
    if isinstance(ra, (int, float)):
        return float(ra)

    return float("nan")


def label_source_for_method(method_key: str) -> str:
    """String metric. Keep deterministic."""
    # sanity tier: methods that use AI-labeled preferences
    if method_key == "rlaif":
        return "ai"
    # preference-based defaults to human-ish (even if synthetic in sanity tier)
    if method_key in {"dpo", "ipo", "rrhf", "orpo", "active_pref"}:
        return "pref"
    # PPO family / SFT
    return "-"
