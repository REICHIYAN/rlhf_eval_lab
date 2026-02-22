# rlhf_eval_lab/backends/fallback/backend.py
# FallbackBackend：torch のみ、GRU 固定
# - transformers 完全禁止
# - CPU-only / sanity tier: 数値が出る・配管が壊れないことが最優先

from __future__ import annotations

from typing import Any, Dict, List, Optional
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

        vocab_size = int(self.config.get("tiny_lm", {}).get("vocab_size", 4096))
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)

        self.model = GRULM(
            vocab_size=vocab_size,
            emb_dim=int(self.config.get("tiny_lm", {}).get("emb_dim", 64)),
            hidden_dim=int(self.config.get("tiny_lm", {}).get("hidden_dim", 128)),
            num_layers=int(self.config.get("tiny_lm", {}).get("num_layers", 1)),
        ).to(self.device)

        # ref model (frozen) for KL
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ref_model.parameters():
            p.requires_grad = False

        lr = float(self.config.get("train", {}).get("lr", 1e-3))
        self.opt = optim.SGD(self.model.parameters(), lr=lr)

        self._pad_id = 0

    def _encode_batch(self, texts: List[str], max_len: int = 128) -> torch.Tensor:
        ids = [self.tokenizer.encode(t)[:max_len] for t in texts]
        maxL = max(1, max(len(x) for x in ids))
        out: List[List[int]] = []
        for x in ids:
            if len(x) < maxL:
                x = x + [self._pad_id] * (maxL - len(x))
            out.append(x)
        return torch.tensor(out, dtype=torch.long, device=self.device)

    def _token_count(self, x: torch.Tensor) -> torch.Tensor:
        return (x != self._pad_id).sum(dim=1).to(dtype=torch.float32)

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
        *,
        ref_state: Optional[Dict[str, torch.Tensor]] = None,
        update_ref: bool = True,
    ) -> Dict[str, float]:
        """
        Sanity PPO (fallback SSOT):

        - logprob_of_sequence() は「系列合計(sum logprob)」を返す前提。
        - 真の KL は >= 0 なので、sanity tier では penalty を必ず非負にする。
        - さらに “手法差” を太く観測するため、学習の KL 罰は sum ベースを採用。
        """
        max_len = int(self.config.get("tiny_lm", {}).get("max_seq_len", 128))
        texts = [(p or "") + " " + (c or "") for p, c in zip(prompts, completions)]
        x = self._encode_batch(texts, max_len=max_len)

        tok_cnt = self._token_count(x)  # (B,)
        tok_cnt_clamped = torch.clamp(tok_cnt, min=1.0)

        saved_ref_state: Optional[Dict[str, torch.Tensor]] = None
        if ref_state is not None:
            saved_ref_state = copy.deepcopy(self.ref_model.state_dict())
            self.ref_model.load_state_dict(ref_state)

        # snapshot params for param_delta
        with torch.no_grad():
            pre_params = [p.detach().clone() for p in self.model.parameters()]

        # --- pre-step logprobs (sequence-sum) ---
        lp_new_pre = self.model.logprob_of_sequence(x)  # (B,)
        with torch.no_grad():
            lp_ref_pre = self.ref_model.logprob_of_sequence(x)  # (B,)

        # reward normalization (DoD stability)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        r = r - r.mean()
        r = r / (r.std(unbiased=False) + 1e-6)

        # signed delta (diagnostics)
        dlp_pre_sum = (lp_new_pre - lp_ref_pre)                  # (B,)
        dlp_pre_mean = dlp_pre_sum / tok_cnt_clamped             # (B,)

        # non-negative "KL-like" penalty (SSOT)
        kl_abs_pre_sum = dlp_pre_sum.abs()                       # (B,)
        kl_abs_pre_mean = kl_abs_pre_sum / tok_cnt_clamped       # (B,)

        # reward term
        loss_reward = -(r.detach() * lp_new_pre).mean()

        # ★ここが変更点：KL罰を sum ベースにして “差” を太くする
        loss_kl = kl_beta * kl_abs_pre_sum.mean()

        loss = loss_reward + loss_kl

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_l2 = self._grad_l2()
        self._clip_grad()
        self.opt.step()

        # param delta
        with torch.no_grad():
            delta_sq = 0.0
            for p, pre in zip(self.model.parameters(), pre_params):
                d = (p.detach() - pre)
                delta_sq += float(d.pow(2).sum().cpu().item())
            param_delta_l2 = float(delta_sq ** 0.5)

        # --- post-step measurements vs ref_pre ---
        with torch.no_grad():
            lp_new_post = self.model.logprob_of_sequence(x)       # (B,)
            dlp_post_sum = (lp_new_post - lp_ref_pre)             # (B,)
            dlp_post_mean = dlp_post_sum / tok_cnt_clamped        # (B,)

            kl_abs_post_sum = dlp_post_sum.abs()                  # (B,)
            kl_abs_post_mean = kl_abs_post_sum / tok_cnt_clamped  # (B,)

        if saved_ref_state is not None:
            self.ref_model.load_state_dict(saved_ref_state)

        if update_ref:
            self.ref_model.load_state_dict(self.model.state_dict())

        tok_mean = float(tok_cnt.detach().mean().cpu().item())
        kl_sum_mean = float(kl_abs_post_sum.detach().mean().cpu().item())
        kl_mean_mean = float(kl_abs_post_mean.detach().mean().cpu().item())

        return {
            "loss": float(loss.detach().cpu().item()),
            "loss_reward": float(loss_reward.detach().cpu().item()),
            "loss_kl": float(loss_kl.detach().cpu().item()),

            # SSOT (>=0)
            "kl": kl_sum_mean,
            "kl_sum": kl_sum_mean,
            "kl_mean": kl_mean_mean,

            # diagnostics: signed
            "kl_signed_sum": float(dlp_post_sum.detach().mean().cpu().item()),
            "kl_signed_mean": float(dlp_post_mean.detach().mean().cpu().item()),
            "kl_pre_signed_mean": float(dlp_pre_mean.detach().mean().cpu().item()),
            "kl_pre_signed_sum": float(dlp_pre_sum.detach().mean().cpu().item()),

            # diagnostics: abs pre
            "kl_pre_sum": float(kl_abs_pre_sum.detach().mean().cpu().item()),
            "kl_pre_mean": float(kl_abs_pre_mean.detach().mean().cpu().item()),

            "token_count": tok_mean,
            "grad_l2": float(grad_l2),
            "param_delta_l2": float(param_delta_l2),
            "adv_scale": float(r.detach().abs().mean().cpu().item()),
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

    def _grad_l2(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            total += float(p.grad.detach().pow(2).sum().cpu().item())
        return float(total ** 0.5)
