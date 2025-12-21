# rlhf_eval_lab/train/preference/trainers/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from ..types import Batch


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


class PreferenceTrainerBase:
    """
    Minimal but correct trainer interface:
      - given a Batch, compute loss and scalar metrics
      - caller handles optimizer step, logging, saving artifacts, etc.
    """
    def __init__(self, *, beta: float) -> None:
        if beta <= 0:
            raise ValueError("beta must be > 0")
        self.beta = float(beta)

    def compute_loss(
        self,
        *,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        batch: Batch,
    ) -> LossOutput:
        raise NotImplementedError
