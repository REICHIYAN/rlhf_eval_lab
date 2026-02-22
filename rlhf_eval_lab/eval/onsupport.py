# rlhf_eval_lab/eval/onsupport.py
# On-support reward（高いほど良い）
# sanity tier 定義：
# - rewards の単純平均
# - rewards が空なら NaN（後段の N/A ポリシー）

from __future__ import annotations

from typing import List


def compute_onsupport(rewards: List[float]) -> float:
    if not rewards:
        return float("nan")
    xs = [float(r) for r in rewards]
    return float(sum(xs)) / float(len(xs))
