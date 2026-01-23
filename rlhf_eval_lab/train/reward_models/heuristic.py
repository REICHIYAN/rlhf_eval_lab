# rlhf_eval_lab/train/reward_models/heuristic.py
# fallback 用の決定論的 heuristic reward
# 目的：空欄ゼロ＆E2E健全性検証（研究用RMではない）
#
# 方針（堅牢性優先）：
# - completion が短すぎる/長すぎる → ペナルティ
# - 繰り返しが多い → ペナルティ
# - 句読点がある → 微ボーナス（説明文っぽさの proxy）
# - prompt との単語重複が極端に低い → 少しペナルティ（暴走の抑制）
#
# 注意：ダミー定数ではなく、入力テキストから計算する

from __future__ import annotations
from typing import List
import math


def _tokens(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    toks = [t for t in s.split() if t]
    if toks:
        return toks
    return list(s)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _uniq_ratio(toks: List[str]) -> float:
    if not toks:
        return 0.0
    return float(len(set(toks))) / float(max(1, len(toks)))


def _overlap_ratio(prompt: str, completion: str) -> float:
    pt = set(_tokens(prompt))
    ct = _tokens(completion)
    if not ct:
        return 0.0
    inter = sum(1 for t in ct if t in pt)
    return float(inter) / float(max(1, len(ct)))


class HeuristicRewardModel:
    def __init__(self):
        pass

    def score(self, prompts: List[str], completions: List[str]) -> List[float]:
        n = min(len(prompts), len(completions))
        out: List[float] = []
        for i in range(n):
            p = prompts[i] or ""
            c = completions[i] or ""
            toks = _tokens(c)
            m = len(toks)

            # 長さ項（目標 24 tokens 前後）
            length_term = -abs(float(m) - 24.0) / 24.0  # 0 に近いほど良い

            # 繰り返し項（ユニーク率）
            uniq = _uniq_ratio(toks)  # 0..1

            # 句読点項
            punct = 1.0 if any(ch in c for ch in (".", "。", "、", ",", "!", "！", "?", "？")) else 0.0

            # prompt との重複（極端に低いと off-support 暴走の proxy）
            ov = _overlap_ratio(p, c)

            # 総合（スケール調整）
            x = (
                1.2 * length_term +
                1.0 * (uniq - 0.5) +
                0.2 * (punct - 0.5) +
                0.4 * (ov - 0.2)
            )
            # 0..1 に圧縮し、平均0付近に寄せるため 0.5 を引く
            r = _sigmoid(3.0 * x) - 0.5
            out.append(float(r))
        return out
