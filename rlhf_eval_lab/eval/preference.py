# rlhf_eval_lab/eval/preference.py
# Preference 指標：Win-rate（高いほど良い）
# - Artifacts.extra に preference 結果があればそれを使う
# - 無ければ rewards の符号（>0）比率で決定論的に算出（空欄禁止）

from __future__ import annotations

from typing import Any, Dict, List


def compute_win_rate(rewards: List[float], extra: Dict[str, Any]) -> float:
    # 優先：明示的な win/loss ログ（例: [{"win":1}, ...] や [0/1]）
    if isinstance(extra.get("wins"), list) and extra["wins"]:
        w = extra["wins"]
        # 0/1 または True/False を想定
        vals = [1.0 if bool(x) else 0.0 for x in w]
        return float(sum(vals)) / float(len(vals))

    # 次点：pairwise の winner（例: [{"winner": 0/1}, ...]）
    if isinstance(extra.get("pairwise"), list) and extra["pairwise"]:
        pw = extra["pairwise"]
        vals: List[float] = []
        for item in pw:
            if isinstance(item, dict) and "winner" in item:
                vals.append(1.0 if int(item["winner"]) == 1 else 0.0)
        if vals:
            return float(sum(vals)) / float(len(vals))

    # fallback：reward > 0 の比率
    if not rewards:
        return 0.0
    vals = [1.0 if float(r) > 0.0 else 0.0 for r in rewards]
    return float(sum(vals)) / float(len(vals))
