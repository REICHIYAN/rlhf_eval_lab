# rlhf_eval_lab/train/reward_models/base.py
# RewardModel 抽象（fallback は決定論的な heuristic を使う）
# ダミー値禁止：入力（prompt/completion）から計算すること

from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod


class RewardModel(ABC):
    @abstractmethod
    def score(self, prompts: List[str], completions: List[str]) -> List[float]:
        """各 (prompt, completion) に対して reward を返す（必ず数値）。"""
        raise NotImplementedError
