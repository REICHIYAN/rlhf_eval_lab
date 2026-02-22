# rlhf_eval_lab/utils/time.py
# タイマー（計測・診断）

from __future__ import annotations
from dataclasses import dataclass
import time


@dataclass
class Timer:
    t0: float

    @classmethod
    def start(cls) -> "Timer":
        return cls(t0=time.time())

    def elapsed(self) -> float:
        return float(time.time() - self.t0)
