# rlhf_eval_lab/eval/kl.py
# KL divergence（低いほど良い）
# DoD 方針：
# - numeric metric は必ず float を返す
# - 欠損・未計測は NaN（"N/A" は validate で禁止）

from __future__ import annotations

from typing import Any, Dict
import math


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def compute_kl(extra: Dict[str, Any]) -> float:
    """
    Sanity tier KL:
    - extra["kl_values"] があれば平均（NaN/inf は除外）
    - extra["kl"] があれば abs(kl)
    - 後方互換：extra["kl_mean"] も拾う
    - なければ NaN
    """
    if not isinstance(extra, dict):
        return float("nan")

    kv = extra.get("kl_values")
    if isinstance(kv, list) and len(kv) > 0:
        vals = []
        for x in kv:
            v = _safe_float(x)
            if v is not None:
                vals.append(v)
        if vals:
            return float(sum(vals)) / float(len(vals))
        return float("nan")

    # SSOT: "kl"
    kl = extra.get("kl")
    v = _safe_float(kl)
    if v is not None:
        return float(abs(v))

    # Backward compatibility: "kl_mean"
    klm = extra.get("kl_mean")
    v2 = _safe_float(klm)
    if v2 is not None:
        return float(abs(v2))

    return float("nan")
