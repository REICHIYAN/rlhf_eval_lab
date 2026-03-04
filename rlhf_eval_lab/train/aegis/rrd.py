# rlhf_eval_lab/train/aegis/rrd.py

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from rlhf_eval_lab.eval.judge import score_completion


def _normalize_prompt(p: str) -> str:
    return " ".join((p or "").strip().lower().split())


def simhash64(text: str) -> int:
    """Deterministic 64-bit simhash (token-based)."""
    toks = _normalize_prompt(text).split()
    if not toks:
        return 0
    v = [0] * 64
    for t in toks:
        h = int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(64):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= (1 << i)
    return out


def default_transform_pool(prompt: str) -> List[str]:
    """Small, deterministic prompt-variant pool.

    Intentionally lightweight and dependency-free. The goal is not perfect
    semantic preservation; it's to elicit judge/RM prompt sensitivity.
    """
    p = (prompt or "").strip()
    if not p:
        return [""]

    variants: List[str] = [
        p,
        f"Please answer concisely.\n\n{p}",
        f"Please answer step-by-step.\n\n{p}",
        f"You are a careful assistant. Provide a direct answer first, then brief rationale.\n\n{p}",
        f"Return the answer in bullet points if possible.\n\n{p}",
        f"Be formal and polite.\n\n{p}",
        f"Be terse and to the point.\n\n{p}",
        f"Focus on factual accuracy. If unsure, say so.\n\n{p}",
        f"Ignore style and focus on correctness only.\n\n{p}",
        f"Answer as if writing for a technical audience.\n\n{p}",
        f"Answer as if writing for a general audience.\n\n{p}",
        f"IMPORTANT: Follow the user's request exactly.\n\n{p}",
    ]

    # de-dup while preserving order
    out: List[str] = []
    seen = set()
    for x in variants:
        k = _normalize_prompt(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(statistics.median([float(x) for x in xs]))


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Deterministic linear-interpolated quantile for sorted values."""
    if not sorted_vals:
        return float("nan")
    if q <= 0.0:
        return float(sorted_vals[0])
    if q >= 1.0:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float((1.0 - w) * sorted_vals[lo] + w * sorted_vals[hi])


def _iqr(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    vals = sorted(float(x) for x in xs)
    if len(vals) < 2:
        return 0.0
    q25 = _quantile(vals, 0.25)
    q75 = _quantile(vals, 0.75)
    return float(q75 - q25)


def score_margin(prompt: str, chosen: str, rejected: str, *, scorer: Callable[[str, str], float]) -> float:
    return float(scorer(prompt, chosen) - scorer(prompt, rejected))


@dataclass(frozen=True)
class AegisStats:
    value_margin: float
    uncertainty: float
    pool_size: int


def estimate_reliability(
    prompt: str,
    chosen: str,
    rejected: str,
    *,
    scorer: Callable[[str, str], float] = score_completion,
    k: int | None = None,
) -> AegisStats:
    """Estimate (value_margin, uncertainty) under prompt transforms.

    - value_margin := median of margins
    - uncertainty  := IQR of margins

    If k is provided, only first k transforms are used (deterministic).
    """
    pool = default_transform_pool(prompt)
    if k is not None:
        kk = max(1, min(int(k), len(pool)))
        pool = pool[:kk]

    margins = [score_margin(px, chosen, rejected, scorer=scorer) for px in pool]
    v = _median(margins)
    u = _iqr(margins) if len(margins) >= 2 else 0.0
    return AegisStats(value_margin=float(v), uncertainty=float(u), pool_size=int(len(pool)))
