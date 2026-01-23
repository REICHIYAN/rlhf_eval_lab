# rlhf_eval_lab/eval/tailvar.py
# Tail reward variance（上位1%の分散、低いほど良い）
# - n が小さい fallback では最低1サンプルは取る
# - rewards の上位 k = max(1, ceil(0.01*n)) を取り分散

from __future__ import annotations

from typing import List
import math


def _var(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = float(sum(xs)) / float(len(xs))
    return float(sum((x - m) ** 2 for x in xs)) / float(len(xs))


def compute_tail_var(rewards: List[float], tail_frac: float = 0.01) -> float:
    if not rewards:
        return 0.0
    xs = sorted([float(x) for x in rewards], reverse=True)
    n = len(xs)
    k = int(math.ceil(float(tail_frac) * float(n)))
    k = max(1, min(n, k))
    tail = xs[:k]
    return _var(tail)
