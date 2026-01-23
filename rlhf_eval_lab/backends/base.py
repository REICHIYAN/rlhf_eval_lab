# rlhf_eval_lab/backends/base.py
# Backend 抽象：Runner / Train / Eval / Reporting から独立させる SSOT
# - fallback は torch のみで必ず step が回る
# - HF backend は同じインターフェイスを実装する

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, prompts: List[str], max_new_tokens: int = 16) -> List[str]:
        """prompt -> completion を生成（必ず str を返す）"""
        raise NotImplementedError

    @abstractmethod
    def logprobs(self, prompts: List[str], completions: List[str]) -> List[float]:
        """(prompt, completion) の総 logprob を返す（必ず数値）"""
        raise NotImplementedError

    @abstractmethod
    def sft_step(self, texts: List[str]) -> float:
        """SFT 1 step（必ず loss 数値）"""
        raise NotImplementedError

    @abstractmethod
    def ppo_step(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
        kl_beta: float = 0.1,
    ) -> Dict[str, float]:
        """PPO 系 1 step（必ず backward -> step が回り、数値を返す）"""
        raise NotImplementedError

    @abstractmethod
    def preference_step(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        beta: float = 0.1,
    ) -> float:
        """Preference 系 1 step（必ず backward -> step が回り、loss を返す）"""
        raise NotImplementedError
