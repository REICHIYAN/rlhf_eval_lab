# rlhf_eval_lab/data/splits.py
# train/eval split（今は最小）

from __future__ import annotations
from typing import List, Tuple
import random


def train_eval_split(items: List, eval_ratio: float = 0.2, seed: int = 0) -> Tuple[List, List]:
    r = random.Random(seed)
    xs = list(items)
    r.shuffle(xs)
    n_eval = int(len(xs) * eval_ratio)
    return xs[n_eval:], xs[:n_eval]
