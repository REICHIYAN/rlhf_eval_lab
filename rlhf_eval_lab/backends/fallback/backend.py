# rlhf_eval_lab/backends/fallback/backend.py
# FallbackBackend：torch のみ、GRU 固定
# - transformers 完全禁止
# - CPU-only / sanity tier: 数値が出る・配管が壊れないことが最優先

from __future__ import annotations

from typing import Any, Dict, List
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from rlhf_eval_lab.backends.base import ModelBackend
from .tokenizer import SimpleTokenizer
from .tiny_lm import GRULM


class FallbackBackend(ModelBackend):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cpu")

        # Fixed-size vocab: guaranteed in-range ids
        vocab_size = int(self.config.get("tiny_lm", {}).get("vocab_size", 4096))
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)

        self.model = GRULM(
            vocab_size=vocab_size,
            emb_dim=int(self.config.get("tiny_lm", {}).get("emb_dim", 64)),
            hidden_dim=int(self.config.get("tiny_lm", {}).get("hidden_dim", 128)),
            num_layers=int(self.config.get("tiny_lm", {}).get("num_layers", 1)),
        ).to(self.device)

        # reference model (frozen) for KL
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ref_model.parameters():
            p.requires_grad = False

        lr = float(self.config.get("train", {}).get("lr", 1e-3))
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    def _encode_batch(self, texts: List[str], max_len: int = 128) -> torch.Tensor:
        ids = [self.tokenizer.encode(t)[:max_len] for t in texts]
        maxL = max(1, max(len(x) for x in ids))
        pad = 0
        out = []
        for x in ids:
            if len(x) < maxL:
                x = x + [pad] * (maxL - len(x))
            out.append(x)
        return torch.tensor(out, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def generate(self, prompts: List[str], max_new_tokens: int = 16) -> List[str]:
        max_len = int(self.config.get("tiny_lm", {}).get("max_seq_len", 128))
        inp = self._encode_batch(prompts, max_len=max_len)
        out = self.model.generate(inp, max_new_tokens=max_new_tokens)
        res: List[str] = []
        for i in range(out.size(0)):
            res.append(self.tokenizer.decode(out[i].tolist(), skip_special=True))
        return res

    def logprobs(self, prompts: List[str], completions: List[str]) -> List[float]:
        max_len = int(self.config.get("tiny_lm", {}).get("max_seq_len", 128))
        texts = [(p or "") + " " + (c or "") for p, c in zip(prompts, completions)]
        x = self._encode_batch(texts, max_len=max_len)
        lp = self.model.logprob_of_sequence(x)
        return [float(v) for v in lp.detach().cpu().tolist()]

    def sft_step(self, texts: List[str]) -> float:
        max_len = int(self.config.get("tiny_lm", {}).get("max_seq_len", 128))
        x = self._encode_batch(texts, max_len=max_len)
        loss = self.model.nll_loss(x)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_grad()
        self.opt.step()
        return float(loss.detach().cpu().item())

    def ppo_step(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
        kl_beta: float = 0.1,
    ) -> Dict[str, float]:
        max_len = int(self.config.get("tiny_lm", {}).get("max_seq_len", 128))
        texts = [(p or "") + " " + (c or "") for p, c in zip(prompts, completions)]
        x = self._encode_batch(texts, max_len=max_len)

        lp_new = self.model.logprob_of_sequence(x)
        with torch.no_grad():
            lp_ref = self.ref_model.logprob_of_sequence(x)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        kl = (lp_new - lp_ref)
        obj = (r - kl_beta * kl).detach() * lp_new
        loss = -obj.mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_grad()
        self.opt.step()

        # sanity tier: hard update ref snapshot
        self.ref_model.load_state_dict(self.model.state_dict())

        return {
            "loss": float(loss.detach().cpu().item()),
            "kl_mean": float(kl.detach().mean().cpu().item()),
        }

    def preference_step(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        beta: float = 0.1,
    ) -> float:
        max_len = int(self.config.get("tiny_lm", {}).get("max_seq_len", 128))
        x_c = self._encode_batch([(prompt or "") + " " + (chosen or "")], max_len=max_len)
        x_r = self._encode_batch([(prompt or "") + " " + (rejected or "")], max_len=max_len)
        lp_c = self.model.logprob_of_sequence(x_c)
        lp_r = self.model.logprob_of_sequence(x_r)
        diff = lp_c - lp_r
        loss = -F.logsigmoid(beta * diff).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_grad()
        self.opt.step()
        return float(loss.detach().cpu().item())

    def _clip_grad(self) -> None:
        g = float(self.config.get("train", {}).get("grad_clip", 1.0))
        if g and g > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=g)
