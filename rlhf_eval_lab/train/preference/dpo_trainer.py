from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .backend_adapter import BackendAdapter
from .hh_adapter import HHPreferenceExample


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 1e-4
    batch_size: int = 2
    max_steps: int = 20
    grad_clip: float = 1.0
    seed: int = 0


@dataclass
class DPOMetrics:
    step: int
    loss: float
    reward_margin_mean: float
    chosen_logp_mean: float
    rejected_logp_mean: float


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    # If CUDA exists in dev env, keep deterministic-ish
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DPOTrainer:
    """
    Minimal DPO trainer:
      - policy backend: trainable
      - reference backend: frozen
      - dataset: (prompt, chosen, rejected)
    """

    def __init__(
        self,
        policy_backend: Any,
        reference_backend: Any,
        cfg: DPOConfig,
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.device = device or "cpu"
        self.policy = BackendAdapter(policy_backend, device=self.device)
        self.ref = BackendAdapter(reference_backend, device=self.device)

        # Freeze reference params if any
        for p in self.ref.parameters():
            p.requires_grad_(False)

        params = list(self.policy.parameters())
        if len(params) == 0:
            raise ValueError(
                "Policy backend has no trainable parameters. "
                "Expected backend to be torch.nn.Module or have .model as torch.nn.Module."
            )

        self.opt = torch.optim.AdamW(params, lr=float(cfg.lr))

    def _batch(self, data: List[HHPreferenceExample], step: int) -> List[HHPreferenceExample]:
        # deterministic batching by step index
        bs = int(self.cfg.batch_size)
        start = (step * bs) % max(len(data), 1)
        batch = []
        for i in range(bs):
            batch.append(data[(start + i) % len(data)])
        return batch

    def _dpo_loss(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        beta = float(self.cfg.beta)

        # policy logprobs
        pi_c = self.policy.sum_logprobs(prompts, chosen)      # [B]
        pi_r = self.policy.sum_logprobs(prompts, rejected)    # [B]

        # reference logprobs
        with torch.no_grad():
            ref_c = self.ref.sum_logprobs(prompts, chosen)    # [B]
            ref_r = self.ref.sum_logprobs(prompts, rejected)  # [B]

        # DPO objective
        # logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
        logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
        loss = -F.logsigmoid(logits).mean()

        aux = {
            "pi_c": pi_c.detach(),
            "pi_r": pi_r.detach(),
            "ref_c": ref_c.detach(),
            "ref_r": ref_r.detach(),
            "logits": logits.detach(),
        }
        return loss, aux

    def train_steps(self, train_data: List[HHPreferenceExample]) -> List[DPOMetrics]:
        if len(train_data) == 0:
            raise ValueError("train_data is empty.")

        self.policy.train()
        self.ref.eval()

        metrics: List[DPOMetrics] = []

        for step in range(int(self.cfg.max_steps)):
            batch = self._batch(train_data, step)
            prompts = [ex.prompt for ex in batch]
            chosen = [ex.chosen for ex in batch]
            rejected = [ex.rejected for ex in batch]

            self.opt.zero_grad(set_to_none=True)
            loss, aux = self._dpo_loss(prompts, chosen, rejected)
            loss.backward()

            # grad clip
            if float(self.cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()), float(self.cfg.grad_clip))

            self.opt.step()

            # "meaningful numbers"
            reward_margin = (aux["pi_c"] - aux["pi_r"]).mean().item()
            m = DPOMetrics(
                step=step,
                loss=float(loss.item()),
                reward_margin_mean=float(reward_margin),
                chosen_logp_mean=float(aux["pi_c"].mean().item()),
                rejected_logp_mean=float(aux["pi_r"].mean().item()),
            )
            metrics.append(m)

        return metrics
