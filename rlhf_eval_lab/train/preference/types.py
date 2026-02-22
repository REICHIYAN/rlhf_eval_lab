# rlhf_eval_lab/train/preference/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass(frozen=True)
class PrefPair:
    """A single preference pair for one prompt."""
    uid: str
    prompt: str
    chosen: str
    rejected: str


@dataclass
class EncodedPair:
    """
    Tokenized prompt+completion pair for chosen and rejected.
    prompt_len_* is token length of prompt-only encoding (per sample) used to mask suffix.
    """
    uid: str

    chosen_enc: Dict[str, torch.Tensor]     # keys: input_ids, attention_mask; shape (1, T)
    rejected_enc: Dict[str, torch.Tensor]   # shape (1, T)

    prompt_len_chosen: int
    prompt_len_rejected: int


@dataclass
class Batch:
    """
    Batched tensors for chosen/rejected prompt+completion encodings.
    Shapes:
      input_ids_*: (B, T)
      attention_mask_*: (B, T)
      prompt_lens_*: (B,)
    """
    uids: List[str]

    input_ids_chosen: torch.Tensor
    attn_mask_chosen: torch.Tensor
    prompt_lens_chosen: torch.Tensor

    input_ids_rejected: torch.Tensor
    attn_mask_rejected: torch.Tensor
    prompt_lens_rejected: torch.Tensor
