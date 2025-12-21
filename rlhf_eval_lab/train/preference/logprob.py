# rlhf_eval_lab/train/preference/logprob.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CompletionLogProbs:
    """
    Sum/mean of log-probabilities over completion tokens only.
    """
    sum: torch.Tensor   # (B,)
    mean: torch.Tensor  # (B,)
    n_tokens: torch.Tensor  # (B,) number of completion tokens counted (>=1)


def _build_suffix_pred_mask(
    attention_mask: torch.Tensor,  # (B, T)
    prompt_lens: torch.Tensor,      # (B,)
) -> torch.Tensor:
    """
    Build a boolean mask over positions t in [0, T-2] corresponding to predicting token t+1,
    where the predicted token belongs to the completion (suffix).

    We treat a token position i as "completion token" if i >= prompt_len.
    Prediction index t predicts token (t+1), so we want (t+1) >= prompt_len.
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be 2D (B,T), got {tuple(attention_mask.shape)}")
    if prompt_lens.dim() != 1:
        raise ValueError(f"prompt_lens must be 1D (B,), got {tuple(prompt_lens.shape)}")

    B, T = attention_mask.shape
    if T < 2:
        raise ValueError("Sequence length must be >= 2 to compute next-token logprobs.")

    # prediction positions: t = 0..T-2
    pos = torch.arange(T - 1, device=attention_mask.device).unsqueeze(0).expand(B, -1)  # (B, T-1)
    # token index being predicted is (t+1)
    predicted_token_idx = pos + 1  # (B, T-1)

    # completion tokens: token_idx >= prompt_len
    prompt_lens_exp = prompt_lens.unsqueeze(1)  # (B,1)
    suffix_token_mask = predicted_token_idx >= prompt_lens_exp  # (B, T-1)

    # also require both current token and predicted token to be "real" tokens under attention_mask.
    # attention_mask at t and t+1 should be 1.
    am_t = attention_mask[:, :-1].bool()
    am_t1 = attention_mask[:, 1:].bool()
    valid_mask = am_t & am_t1

    return suffix_token_mask & valid_mask  # (B, T-1)


@torch.no_grad()
def debug_check_masks(
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    *,
    name: str,
    max_print: int = 2,
) -> None:
    """
    Lightweight sanity check for mask shapes and a few examples.
    Safe to call in step==0 debug mode.
    """
    m = _build_suffix_pred_mask(attention_mask, prompt_lens)
    B, T = attention_mask.shape
    # n_tokens counted equals sum(mask)
    n = m.sum(dim=1)
    print(f"[debug:{name}] B={B} T={T} n_completion_pred_positions(min/mean/max)={int(n.min())}/{float(n.float().mean()):.2f}/{int(n.max())}")
    for i in range(min(B, max_print)):
        pl = int(prompt_lens[i].item())
        # show first/last few indices
        on = torch.nonzero(m[i]).squeeze(-1)
        head = on[:5].tolist()
        tail = on[-5:].tolist() if on.numel() > 5 else []
        print(f"[debug:{name}] sample={i} prompt_len={pl} mask_on_head={head} mask_on_tail={tail}")


def completion_logprobs_from_joint(
    model: torch.nn.Module,
    input_ids: torch.Tensor,        # (B, T)
    attention_mask: torch.Tensor,   # (B, T)
    prompt_lens: torch.Tensor,      # (B,)
) -> CompletionLogProbs:
    """
    Compute completion-only log-prob sum/mean for a batch, from joint (prompt+completion) tokens.

    Returns:
      sum:  (B,) sum log p(y_t | y_<t) over completion tokens only
      mean: (B,) mean over those tokens
      n_tokens: (B,) number of tokens counted (>=1 if guard ensured)
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D (B,T), got {tuple(input_ids.shape)}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have same shape as input_ids")
    if prompt_lens.dim() != 1 or prompt_lens.shape[0] != input_ids.shape[0]:
        raise ValueError("prompt_lens must be (B,) aligned with input_ids")

    # forward (with grads; caller decides)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (B, T, V)
    if logits.dim() != 3:
        raise RuntimeError(f"model logits must be 3D (B,T,V), got {tuple(logits.shape)}")

    # shift for next-token prediction
    logits_t = logits[:, :-1, :]           # (B, T-1, V)
    target_t1 = input_ids[:, 1:]           # (B, T-1)

    logp = F.log_softmax(logits_t, dim=-1)  # (B, T-1, V)
    token_logp = logp.gather(dim=-1, index=target_t1.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    mask = _build_suffix_pred_mask(attention_mask, prompt_lens)  # (B, T-1) bool
    token_logp = token_logp * mask.to(token_logp.dtype)

    n_tokens = mask.sum(dim=1).to(token_logp.dtype)  # (B,)
    # Avoid division by zero: caller must ensure guards; still keep safe.
    n_safe = torch.clamp(n_tokens, min=1.0)

    sum_lp = token_logp.sum(dim=1)          # (B,)
    mean_lp = sum_lp / n_safe              # (B,)

    return CompletionLogProbs(sum=sum_lp, mean=mean_lp, n_tokens=n_tokens)
