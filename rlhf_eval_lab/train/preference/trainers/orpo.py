# rlhf_eval_lab/train/preference/trainers/orpo.py
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..logprob import completion_logprobs_from_joint
from ..types import Batch
from .base import LossOutput, PreferenceTrainerBase


def _mean_logprob_per_sample(lp_obj: object) -> torch.Tensor:
    """
    Robust mean logprob per sample for chosen completion tokens.

    Supports multiple possible return shapes from completion_logprobs_from_joint:
      - has attribute 'mean'  : (B,) mean logprob over completion tokens
      - else has 'sum' and 'n_tokens' (or 'count') : mean = sum / max(n,1)
    Raises a clear error if neither exists.
    """
    if hasattr(lp_obj, "mean"):
        mean = getattr(lp_obj, "mean")
        if not isinstance(mean, torch.Tensor):
            raise TypeError("completion_logprobs_from_joint(...).mean must be a torch.Tensor")
        return mean

    if hasattr(lp_obj, "sum"):
        s = getattr(lp_obj, "sum")
        if not isinstance(s, torch.Tensor):
            raise TypeError("completion_logprobs_from_joint(...).sum must be a torch.Tensor")

        n: Optional[torch.Tensor] = None
        if hasattr(lp_obj, "n_tokens"):
            n = getattr(lp_obj, "n_tokens")
        elif hasattr(lp_obj, "count"):
            n = getattr(lp_obj, "count")

        if n is None:
            raise AttributeError(
                "completion_logprobs_from_joint output must provide .mean or (.sum and .n_tokens/.count)."
            )
        if not isinstance(n, torch.Tensor):
            raise TypeError("completion_logprobs_from_joint(...).n_tokens/.count must be a torch.Tensor")

        n_safe = torch.clamp(n, min=1.0)
        return s / n_safe

    raise AttributeError(
        "completion_logprobs_from_joint output must provide .mean or .sum (plus token count)."
    )


class ORPOTrainer(PreferenceTrainerBase):
    """
    ORPO (minimal, hybrid RRHF + SFT auxiliary):

      margin = pi_c_sum - pi_r_sum
      rrhf_loss = -log sigmoid(beta * margin)

      sft_loss = - mean_logprob(chosen_completion)   (per-sample mean over completion tokens)

      total_loss = rrhf_loss + alpha * sft_loss

    Notes:
      - ref_model is unused (kept for interface compatibility).
      - Uses the same completion-only logprob backbone as DPO/IPO/RRHF.
      - Uses mean for the SFT term to reduce length bias.
    """

    def __init__(self, *, beta: float, alpha: float = 0.1) -> None:
        super().__init__(beta=beta)
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)

    def compute_loss(
        self,
        *,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,  # unused
        batch: Batch,
    ) -> LossOutput:
        chosen_lp = completion_logprobs_from_joint(
            model,
            batch.input_ids_chosen,
            batch.attn_mask_chosen,
            batch.prompt_lens_chosen,
        )
        rejected_lp = completion_logprobs_from_joint(
            model,
            batch.input_ids_rejected,
            batch.attn_mask_rejected,
            batch.prompt_lens_rejected,
        )

        pi_c_sum = chosen_lp.sum
        pi_r_sum = rejected_lp.sum

        margin = pi_c_sum - pi_r_sum  # (B,)
        rrhf_loss = (-F.logsigmoid(self.beta * margin)).mean()

        chosen_mean_lp = _mean_logprob_per_sample(chosen_lp)  # (B,)
        sft_loss = (-chosen_mean_lp).mean()

        loss = rrhf_loss + self.alpha * sft_loss

        metrics: Dict[str, float] = {
            "loss": float(loss.detach().cpu().item()),
            "rrhf_loss": float(rrhf_loss.detach().cpu().item()),
            "sft_loss": float(sft_loss.detach().cpu().item()),
            "alpha": float(self.alpha),
            "reward_margin_mean": float(margin.detach().mean().cpu().item()),
            "reward_chosen_mean": float(pi_c_sum.detach().mean().cpu().item()),
            "reward_rejected_mean": float(pi_r_sum.detach().mean().cpu().item()),
        }
        return LossOutput(loss=loss, metrics=metrics)
