# /mnt/c/Users/s5aba/Projects/rlhf_eval_lab/rlhf_eval_lab/train/preference/debug_adapter.py
from __future__ import annotations

from typing import Dict

import torch


def debug_mask_sanity(enc: Dict[str, torch.Tensor], completion_mask: torch.Tensor, *, name: str) -> None:
    """
    Verify completion_mask is a suffix mask *within the non-padding region*.

    Important:
      We must ignore padding positions (attention_mask==0). If we multiply a suffix mask by attention_mask,
      the padded tail becomes 0 and will create a 1->0 fall, which is expected and should NOT fail.

    enc:
      tokenizer output dict with input_ids/attention_mask (B, T)
    completion_mask:
      (B, T) binary 0/1, already aligned to enc grid
    """
    assert "input_ids" in enc, f"{name}: enc missing input_ids"
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))

    if completion_mask.dim() != 2:
        raise AssertionError(f"{name}: completion_mask must be 2D (B,T). got {tuple(completion_mask.shape)}")
    if input_ids.dim() != 2:
        raise AssertionError(f"{name}: input_ids must be 2D (B,T). got {tuple(input_ids.shape)}")
    if completion_mask.shape != input_ids.shape:
        raise AssertionError(
            f"{name}: shape mismatch. input_ids={tuple(input_ids.shape)} completion_mask={tuple(completion_mask.shape)}"
        )

    B, T = input_ids.shape

    attn_sum = int(attn.sum().item())
    cm_sum = int(completion_mask.sum().item())

    rises_total = 0
    falls_total = 0

    # check suffix property only on valid token span [0, L) where L = attention_mask.sum()
    for i in range(B):
        L = int(attn[i].sum().item())
        if L <= 0:
            continue

        cm_valid = completion_mask[i, :L].to(dtype=torch.long)  # (L,)
        # count transitions in valid region only
        diffs = cm_valid[1:] - cm_valid[:-1]
        rises_total += int((diffs == 1).sum().item())
        falls_total += int((diffs == -1).sum().item())

        # In valid region, suffix mask means: no 1->0 transitions.
        if int((diffs == -1).sum().item()) != 0:
            # Print a small diagnostic (first offending position)
            idxs = (diffs == -1).nonzero(as_tuple=False)
            first = int(idxs[0].item()) if idxs.numel() > 0 else -1
            raise AssertionError(
                f"{name}: completion mask fell back to 0 within non-pad region at pos={first} "
                f"(L={L}, T={T}). This indicates boundary/mask construction bug."
            )

    # Print a compact summary for step==0 debugging.
    print(
        f"[mask:{name}] B={B} T={T} attn_sum={attn_sum} completion_mask_sum={cm_sum} rises={rises_total} falls={falls_total}"
    )


def debug_logp_sanity(*, name: str, logp_sum: torch.Tensor) -> None:
    """
    Print mean/min/max for (B,) logp_sum tensor.
    """
    if logp_sum.dim() != 1:
        raise AssertionError(f"{name}: expected 1D (B,) logp_sum. got {tuple(logp_sum.shape)}")
    mean = float(logp_sum.mean().item())
    minv = float(logp_sum.min().item())
    maxv = float(logp_sum.max().item())
    print(f"[logp:{name}] mean={mean:.3f} min={minv:.3f} max={maxv:.3f}")
