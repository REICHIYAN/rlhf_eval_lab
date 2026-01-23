# rlhf_eval_lab/train/preference/trainers/base.py
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch

from ..types import Batch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


@dataclass(frozen=True)
class GradClipMetrics:
    grad_norm_pre: float
    grad_norm_post: float
    did_clip: bool


def global_grad_norm_l2(parameters: Iterable[torch.nn.Parameter]) -> float:
    """
    Global L2 norm of gradients across parameters.
    This matches the notion of "global norm" used by clip_grad_norm_.
    """
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.is_sparse:
            g = g.coalesce().values()
        # g.norm(2) is L2 norm for each tensor
        n = float(g.norm(2).item())
        total_sq += n * n
    return math.sqrt(total_sq)


def clip_gradients_(
    *,
    model: torch.nn.Module,
    max_grad_norm: float = 1.0,
    enabled: bool = True,
    log: bool = True,
) -> GradClipMetrics:
    """
    Apply global-norm gradient clipping to `model` parameters.
    Intended to be called right after `loss.backward()` and before `optimizer.step()`.

    Returns:
      GradClipMetrics with pre/post norms and did_clip flag.
    """
    if not enabled:
        pre = global_grad_norm_l2(model.parameters())
        if log:
            logger.info("[grad_clip] enabled=False pre=%.6g post=%.6g did_clip=%s", pre, pre, False)
        return GradClipMetrics(grad_norm_pre=pre, grad_norm_post=pre, did_clip=False)

    if max_grad_norm is None or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be > 0 when enabled=True")

    pre = global_grad_norm_l2(model.parameters())

    # clip_grad_norm_ returns pre-clip norm, but we already computed `pre` explicitly.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))

    post = global_grad_norm_l2(model.parameters())
    did_clip = pre > float(max_grad_norm)

    if log:
        logger.info(
            "[grad_clip] enabled=True max_grad_norm=%.6g pre=%.6g post=%.6g did_clip=%s",
            float(max_grad_norm),
            float(pre),
            float(post),
            bool(did_clip),
        )
    return GradClipMetrics(grad_norm_pre=pre, grad_norm_post=post, did_clip=did_clip)


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
