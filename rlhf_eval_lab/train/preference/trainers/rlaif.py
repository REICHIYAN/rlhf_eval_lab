# rlhf_eval_lab/train/preference/trainers/rlaif.py
from __future__ import annotations

"""
RLAIF-min trainer.

RLAIF here is defined as:
  1) Pseudo-labeling chosen/rejected by an AI signal (heuristic RM in this repo)
  2) Then applying a standard preference objective on the relabeled pairs

This file makes RLAIF explicit as a method (row SSOT),
while keeping the loss "minimal but correct" by reusing RRHF loss form.
"""

import torch

from ..types import Batch
from .base import LossOutput
from .rrhf import RRHFTrainer


class RLAIFTrainer(RRHFTrainer):
    """
    Minimal RLAIF trainer.

    NOTE:
      - Expects pseudo-labeling to be done upstream (run_pref_min.py).
      - Therefore, objective can reuse RRHF (pairwise margin -> -log sigmoid).
    """

    def compute_loss(
        self,
        *,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        batch: Batch,
    ) -> LossOutput:
        return super().compute_loss(model=model, ref_model=ref_model, batch=batch)
