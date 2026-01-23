from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LogProbsResult:
    sum_logprobs: torch.Tensor  # shape [B]


def _require(obj: Any, names: Sequence[str]) -> Tuple[str, Any]:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return n, v
    raise AttributeError(
        f"Backend missing required attribute/method. Tried: {list(names)}. "
        f"Available: {sorted([k for k in dir(obj) if not k.startswith('_')])[:120]}..."
    )


def _is_torch_module(x: Any) -> bool:
    return isinstance(x, torch.nn.Module)


def _extract_sum_logprobs(out: Any, device: torch.device) -> Optional[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        t = out.to(device)
        return t.view(-1) if t.ndim != 1 else t

    if isinstance(out, dict):
        for k in ("sum_logprobs", "logprobs", "scores"):
            v = out.get(k, None)
            if isinstance(v, torch.Tensor):
                t = v.to(device)
                return t.view(-1) if t.ndim != 1 else t

    v = getattr(out, "sum_logprobs", None)
    if isinstance(v, torch.Tensor):
        t = v.to(device)
        return t.view(-1) if t.ndim != 1 else t

    return None


def _ensure_padding_token(tok: Any, model: Optional[torch.nn.Module]) -> None:
    pad = getattr(tok, "pad_token", None)
    if pad is not None:
        return

    eos = getattr(tok, "eos_token", None)
    if eos is not None:
        try:
            tok.pad_token = eos
            return
        except Exception:
            pass

    add_special = getattr(tok, "add_special_tokens", None)
    if callable(add_special):
        add_special({"pad_token": "[PAD]"})
        if model is not None:
            resize = getattr(model, "resize_token_embeddings", None)
            if callable(resize):
                try:
                    resize(len(tok))
                except Exception:
                    pass
        return

    raise ValueError(
        "Tokenizer does not have pad_token and cannot be configured automatically. "
        "Please set tokenizer.pad_token or provide a tokenizer with padding support."
    )


class BackendAdapter:
    """
    Adapter to compute sum log-probabilities of a completion conditioned on a prompt.

    Preferred order:
      1) backend.logprobs(prompts, completions)
      2) backend.logprobs(texts) where texts are prompt+completion
      3) backend.score(prompts, completions)
      4) backend.score(texts)
      5) model+tokenizer: compute completion-only CE using robust prompt/completion boundary.

    Important: For DPO training, the returned score MUST be differentiable wrt policy parameters.
    """

    def __init__(self, backend: Any, device: Optional[str] = None):
        self.backend = backend
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        if _is_torch_module(self.backend):
            self.backend.to(self.device)

        m = getattr(self.backend, "model", None)
        if _is_torch_module(m):
            m.to(self.device)

    def parameters(self):
        if _is_torch_module(self.backend):
            return self.backend.parameters()
        m = getattr(self.backend, "model", None)
        if _is_torch_module(m):
            return m.parameters()
        return []

    def train(self):
        if _is_torch_module(self.backend):
            self.backend.train()
        m = getattr(self.backend, "model", None)
        if _is_torch_module(m):
            m.train()

    def eval(self):
        if _is_torch_module(self.backend):
            self.backend.eval()
        m = getattr(self.backend, "model", None)
        if _is_torch_module(m):
            m.eval()

    @torch.no_grad()
    def _call_logprobs(self, prompts, completions) -> Optional[torch.Tensor]:
        lp = getattr(self.backend, "logprobs", None)
        if not callable(lp):
            return None

        try:
            out = lp(prompts, completions)
            t = _extract_sum_logprobs(out, self.device)
            if t is not None:
                return t
        except TypeError:
            pass
        except Exception:
            pass

        texts = [p + c for p, c in zip(prompts, completions)]
        try:
            out = lp(texts)
            t = _extract_sum_logprobs(out, self.device)
            if t is not None:
                return t
        except Exception:
            return None

        return None

    @torch.no_grad()
    def _call_score(self, prompts, completions) -> Optional[torch.Tensor]:
        sc = getattr(self.backend, "score", None)
        if not callable(sc):
            return None

        try:
            out = sc(prompts, completions)
            t = _extract_sum_logprobs(out, self.device)
            if t is not None:
                return t
        except TypeError:
            pass
        except Exception:
            pass

        texts = [p + c for p, c in zip(prompts, completions)]
        try:
            out = sc(texts)
            t = _extract_sum_logprobs(out, self.device)
            if t is not None:
                return t
        except Exception:
            return None

        return None

    def sum_logprobs(self, prompts, completions) -> torch.Tensor:
        # fast paths
        t = self._call_logprobs(prompts, completions)
        if t is not None:
            return t

        t = self._call_score(prompts, completions)
        if t is not None:
            return t

        # fallback via model+tokenizer (trainable)
        _, tok = _require(self.backend, ["tokenizer", "tok", "tokenizer_obj"])
        _, model = _require(self.backend, ["model", "lm", "language_model"])

        if not _is_torch_module(model):
            raise TypeError(f"Expected backend.model to be torch.nn.Module, got {type(model)}")
        if not callable(tok):
            raise TypeError(f"Expected backend.tokenizer to be callable tokenizer, got {type(tok)}")

        _ensure_padding_token(tok, model)

        # Robust boundary: prefer text_pair encoding if supported.
        # This makes "which tokens belong to completion" much more stable than (prompt) and (prompt+completion) length diff.
        try:
            enc = tok(
                list(prompts),
                list(completions),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            pair_mode = True
        except TypeError:
            # tokenizer doesn't support text_pair in batch call (rare)
            pair_mode = False
            joint_texts = [p + c for p, c in zip(prompts, completions)]
            enc = tok(joint_texts, return_tensors="pt", padding=True, truncation=True)

        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.device)

        out = model(input_ids=input_ids, attention_mask=attn)  # type: ignore
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, (tuple, list)) and len(out) > 0:
            logits = out[0]
        if logits is None or not isinstance(logits, torch.Tensor):
            raise RuntimeError("Model forward did not return logits tensor.")

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        logp_all = F.log_softmax(shift_logits, dim=-1)
        token_logp = logp_all.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        # Build completion-only mask.
        B, Tm1 = token_logp.shape
        pos = torch.arange(Tm1, device=self.device).unsqueeze(0).expand(B, Tm1)

        if pair_mode:
            # If we have sequence_ids (fast tokenizer), use it.
            # sequence_ids are aligned to original tokens, but for simplicity we reconstruct token-level mask:
            # token_type_ids often exist for pair encodings; if so, tokens with token_type_id==1 belong to completion.
            tti = enc.get("token_type_ids", None)
            if tti is None:
                # If token_type_ids is absent (e.g., GPT2), fall back to "special_tokens_mask" + heuristic:
                # We compute prompt lengths by encoding prompts alone WITH SAME SETTINGS and align by min length.
                prm = tok(list(prompts), return_tensors="pt", padding=True, truncation=True)
                prm_ids = prm["input_ids"].to(self.device)
                prm_attn = prm.get("attention_mask", None)
                if prm_attn is None:
                    prm_lens = (prm_ids != 0).sum(dim=1)
                else:
                    prm_lens = prm_attn.sum(dim=1).to(self.device)
                comp_mask = (pos + 1) >= prm_lens.unsqueeze(1)
            else:
                tti = tti.to(self.device)
                # shift to align with token_logp positions (predict token at position i+1)
                comp_mask = (tti[:, 1:] == 1)
        else:
            # non-pair fallback: prompt length heuristic
            prm = tok(list(prompts), return_tensors="pt", padding=True, truncation=True)
            prm_ids = prm["input_ids"].to(self.device)
            prm_attn = prm.get("attention_mask", None)
            if prm_attn is None:
                prm_lens = (prm_ids != 0).sum(dim=1)
            else:
                prm_lens = prm_attn.sum(dim=1).to(self.device)
            comp_mask = (pos + 1) >= prm_lens.unsqueeze(1)

        if attn is not None:
            shift_attn = attn[:, 1:]
            comp_mask = comp_mask & shift_attn.bool()

        sum_logp = (token_logp * comp_mask.float()).sum(dim=1)
        return sum_logp
