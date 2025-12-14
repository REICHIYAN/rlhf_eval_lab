# rlhf_eval_lab/eval/kl.py
# KL divergence（低いほど良い）
# DoD 方針：
# - numeric metric は必ず float を返す
# - 欠損・未計測は NaN（"N/A" は validate で禁止）

from __future__ import annotations

from typing import Any, Dict
import math


def compute_kl(extra: Dict[str, Any]) -> float:
    """
    Sanity tier KL:
    - extra["kl_values"] があれば平均
    - extra["kl"] があれば abs(kl)
    - なければ NaN
    """
    if not isinstance(extra, dict):
        return float("nan")

    kv = extra.get("kl_values")
    if isinstance(kv, list) and len(kv) > 0:
        vals = []
        for x in kv:
            try:
                v = float(x)
                if not math.isnan(v) and not math.isinf(v):
                    vals.append(v)
            except Exception:
                continue
        if vals:
            return float(sum(vals)) / float(len(vals))
        return float("nan")

    kl = extra.get("kl")
    if isinstance(kl, (int, float)):
        v = float(kl)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return float(abs(v))

    return float("nan")
