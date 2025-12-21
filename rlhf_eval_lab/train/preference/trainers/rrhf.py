# rlhf_eval_lab/train/preference/trainers/rrhf.py
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ..logprob import completion_logprobs_from_joint
from ..types import Batch
from .base import LossOutput, PreferenceTrainerBase


class RRHFTrainer(PreferenceTrainerBase):
    """
    RRHF (minimal, preference-based):
      margin = pi_c - pi_r
      loss = -log sigmoid(beta * margin)

    - Uses policy logprobs only (no ref_model).
    - completion-only summed logprob from joint encoding.
    """

    def compute_loss(
        self,
        *,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,  # kept for interface compatibility; unused
        batch: Batch,
    ) -> LossOutput:
        pi_c = completion_logprobs_from_joint(
            model,
            batch.input_ids_chosen,
            batch.attn_mask_chosen,
            batch.prompt_lens_chosen,
        ).sum
        pi_r = completion_logprobs_from_joint(
            model,
            batch.input_ids_rejected,
            batch.attn_mask_rejected,
            batch.prompt_lens_rejected,
        ).sum

        margin = pi_c - pi_r  # (B,)
        loss_vec = -F.logsigmoid(self.beta * margin)
        loss = loss_vec.mean()

        metrics: Dict[str, float] = {
            "loss": float(loss.detach().cpu().item()),
            "reward_margin_mean": float(margin.detach().mean().cpu().item()),
            "reward_chosen_mean": float(pi_c.detach().mean().cpu().item()),
            "reward_rejected_mean": float(pi_r.detach().mean().cpu().item()),
        }
        return LossOutput(loss=loss, metrics=metrics)
