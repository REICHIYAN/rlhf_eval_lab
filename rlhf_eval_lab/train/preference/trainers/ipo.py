# rlhf_eval_lab/train/preference/trainers/ipo.py
from __future__ import annotations

from typing import Dict

import torch

from ..logprob import completion_logprobs_from_joint
from ..types import Batch
from .base import LossOutput, PreferenceTrainerBase


class IPOTrainer(PreferenceTrainerBase):
    """
    IPO (squared loss variant):
      h = (pi_c - pi_r) - (ref_c - ref_r)
      loss = (h - 1/(2*beta))^2
    """
    def compute_loss(
        self,
        *,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
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

        with torch.no_grad():
            ref_c = completion_logprobs_from_joint(
                ref_model,
                batch.input_ids_chosen,
                batch.attn_mask_chosen,
                batch.prompt_lens_chosen,
            ).sum
            ref_r = completion_logprobs_from_joint(
                ref_model,
                batch.input_ids_rejected,
                batch.attn_mask_rejected,
                batch.prompt_lens_rejected,
            ).sum

        h = (pi_c - pi_r) - (ref_c - ref_r)  # (B,)
        target = 1.0 / (2.0 * self.beta)
        loss_vec = (h - target) ** 2
        loss = loss_vec.mean()

        metrics: Dict[str, float] = {
            "loss": float(loss.detach().cpu().item()),
            "h_mean": float(h.detach().mean().cpu().item()),
            "h_target": float(target),
            "reward_margin_mean": float((pi_c - pi_r).detach().mean().cpu().item()),
            "ref_margin_mean": float((ref_c - ref_r).detach().mean().cpu().item()),
        }
        return LossOutput(loss=loss, metrics=metrics)
