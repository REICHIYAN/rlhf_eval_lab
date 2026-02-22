# rlhf_eval_lab/eval/offsupport.py
# Off-support visitation rate（低いほど良い）
# 定義（fallback向けの堅牢版）：
# - completion のトークン集合のうち prompt と重ならない割合を off-support とする
# - offsupport = mean(1 - overlap_ratio(prompt, completion))
# - overlap_ratio = |intersection| / max(1, |completion_tokens|)

from __future__ import annotations

from typing import List


def _tokens(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    # 空白区切り（最低限・堅牢）
    return [t for t in s.split() if t]


def overlap_ratio(prompt: str, completion: str) -> float:
    pt = set(_tokens(prompt))
    ct = _tokens(completion)
    if len(ct) == 0:
        return 0.0
    inter = 0
    for t in ct:
        if t in pt:
            inter += 1
    return float(inter) / float(max(1, len(ct)))


def compute_offsupport(prompts: List[str], completions: List[str]) -> float:
    n = min(len(prompts), len(completions))
    if n <= 0:
        return 0.0
    vals = []
    for i in range(n):
        ov = overlap_ratio(prompts[i], completions[i])
        vals.append(1.0 - ov)
    return float(sum(vals)) / float(len(vals))
