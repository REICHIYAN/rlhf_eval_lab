# rlhf_eval_lab/backends/hf/backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import copy

import torch


@dataclass
class _PPOBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lens: torch.Tensor
    pad_id: int


class HFBackend:
    """
    HF backend (optional, transformers required).

    Goals:
      - Lazy import transformers (do not break fallback CI).
      - Provide a PPO step compatible with fallback interface used by run.py.
      - Keep runs auditable and deterministic by default.

    Notes:
      - Decoder-only generation should use left padding.
      - For PPO math we disable dropout (eval mode) to reduce ratio noise.
      - `logprobs()` is exposed for auditability (policy-shift checks).
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device("cpu")

        hf_cfg = cfg.get("hf", {}) or {}
        train_cfg = cfg.get("train", {}) or {}
        eval_cfg = cfg.get("eval", {}) or {}

        self.model_name = str(hf_cfg.get("model_name", "gpt2"))
        self.temperature = float(hf_cfg.get("temperature", 1.0))

        # SFT optimizer lr (legacy key: train.lr)
        self.train_lr = float(train_cfg.get("lr", 1e-3))

        # PPO knobs
        self.ppo_lr = float(train_cfg.get("ppo_lr", 1e-6))
        self.ppo_clip = float(train_cfg.get("ppo_clip", 0.2))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))

        self.hf_max_seq_len = int(train_cfg.get("hf_max_seq_len", 256))
        self.max_new_tokens = int(eval_cfg.get("max_new_tokens", 16))

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "HFBackend requires transformers. Install extras or run with --backend fallback."
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # GPT2-like tokenizers often have no pad token; set to eos for batching
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT for decoder-only generation: left padding
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.train()

        # Reference model for KL (frozen)
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self._sft_optim = torch.optim.AdamW(self.model.parameters(), lr=self.train_lr)
        self._ppo_optim = torch.optim.AdamW(self.model.parameters(), lr=self.ppo_lr)

    # -------------------------
    # Public API
    # -------------------------
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 16) -> List[str]:
        max_new = int(max_new_tokens)

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.hf_max_seq_len,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # Deterministic by default + silence pad warnings
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=int(self.tokenizer.pad_token_id),
                eos_token_id=int(self.tokenizer.eos_token_id),
            )
        self.model.train(was_training)

        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        res: List[str] = []
        for p, t in zip(prompts, texts):
            if t.startswith(p):
                res.append(t[len(p) :].strip())
            else:
                res.append(t.strip())
        return res

    def logprobs(self, prompts: Sequence[str], completions: Sequence[str]) -> List[float]:
        """
        Public wrapper for auditability:
        returns sum logprob over completion tokens for each (prompt, completion).

        - Dropout OFF (eval)
        - No grads (torch.no_grad)
        - Restores original training mode
        """
        if len(prompts) != len(completions):
            raise ValueError("HFBackend.logprobs expects prompts and completions with the same length")

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            lp_sum = self._logprob_completion_sum(self.model, prompts, completions)
        self.model.train(was_training)

        return [float(x) for x in lp_sum.detach().cpu().tolist()]

    def sft_step(self, texts: Sequence[str]) -> float:
        enc = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.hf_max_seq_len,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        was_training = self.model.training
        self.model.train()
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = out.loss

        self._sft_optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self._sft_optim.step()

        self.model.train(was_training)
        return float(loss.detach().cpu().item())

    def preference_step(self, prompt: str, chosen: str, rejected: str, beta: float = 0.1) -> float:
        beta_f = float(beta)

        # Disable dropout for logprob (stability) but keep gradients
        was_training = self.model.training
        self.model.eval()
        lp_c = self._logprob_completion_sum(self.model, [prompt], [chosen])[0]
        lp_r = self._logprob_completion_sum(self.model, [prompt], [rejected])[0]
        self.model.train(was_training)

        x = beta_f * (lp_c - lp_r)
        loss = -torch.nn.functional.logsigmoid(x)

        self._sft_optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self._sft_optim.step()

        return float(loss.detach().cpu().item())

    def ppo_step(
        self,
        prompts: Sequence[str],
        completions: Optional[Sequence[str]] = None,
        rewards: Optional[Sequence[float]] = None,
        kl_beta: float = 0.0,
        ref_state: Any = None,  # interface parity; ignored
        update_ref: bool = False,
        **_: Any,  # swallow unknown kwargs safely
    ) -> Dict[str, float]:
        """
        PPO step with fallback-compatible interface.

        We return *post-update* diagnostics for auditability:
          - ratio_mean: exp(logp_post - logp_old) mean
          - kl_ref: (logp_post - logp_ref) mean   (signed; not a true KL)
          - clipfrac: computed from post-update ratio

        Additionally:
          - ratio_mean_pre, kl_ref_pre: pre-update values
          - kl_ref_abs, kl_ref_sq: nonnegative KL-proxy diagnostics (post-update)
        """
        _ = ref_state  # intentionally unused

        if rewards is None:
            raise ValueError("HFBackend.ppo_step requires rewards")

        if completions is None:
            completions = self.generate(prompts, max_new_tokens=self.max_new_tokens)

        rewards_t = torch.tensor(list(rewards), dtype=torch.float32, device=self.device)
        adv = rewards_t - rewards_t.mean()
        if float(adv.abs().sum().detach().cpu().item()) < 1e-12:
            adv = rewards_t

        clip = float(self.ppo_clip)
        klb = float(kl_beta)

        # Disable dropout during PPO math (including the forward used for loss)
        was_training = self.model.training
        self.model.eval()

        # Pre-update (old/ref) logprobs
        with torch.no_grad():
            logp_old_sum = self._logprob_completion_sum(self.model, prompts, completions)
            logp_ref_sum = self._logprob_completion_sum(self.ref_model, prompts, completions)

        # Pre-update "new" logprob with grad enabled (same weights as old at this point)
        logp_new_sum = self._logprob_completion_sum(self.model, prompts, completions)

        tok_cnt = self._completion_token_counts(prompts, completions).to(self.device)
        tok_cnt = torch.clamp(tok_cnt, min=1)

        logp_old = logp_old_sum / tok_cnt
        logp_new = logp_new_sum / tok_cnt
        logp_ref = logp_ref_sum / tok_cnt

        ratio_pre = torch.exp(logp_new - logp_old)
        clipped_pre = torch.clamp(ratio_pre, 1.0 - clip, 1.0 + clip)

        surr1 = ratio_pre * adv
        surr2 = clipped_pre * adv
        surr = torch.minimum(surr1, surr2)

        kl_pre = (logp_new - logp_ref)  # per-token mean (signed)
        loss = -(surr.mean()) + klb * kl_pre.mean()

        self._ppo_optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self._ppo_optim.step()

        # Post-update diagnostics: recompute logprob with updated parameters
        with torch.no_grad():
            logp_post_sum = self._logprob_completion_sum(self.model, prompts, completions)

        logp_post = logp_post_sum / tok_cnt
        ratio_post = torch.exp(logp_post - logp_old)

        # "KL proxy" relative to frozen reference (signed + nonnegative proxies)
        kl_post = (logp_post - logp_ref)  # per-token mean (signed)
        kl_ref_abs = kl_post.abs().mean()
        kl_ref_sq = (kl_post * kl_post).mean()

        clipfrac_post = ((ratio_post > (1.0 + clip)) | (ratio_post < (1.0 - clip))).float().mean()

        if update_ref:
            self.ref_model.load_state_dict(self.model.state_dict())

        # restore training mode
        self.model.train(was_training)

        return {
            "ppo_loss": float(loss.detach().cpu().item()),
            "ratio_mean": float(ratio_post.detach().mean().cpu().item()),
            "clipfrac": float(clipfrac_post.detach().cpu().item()),
            "kl_ref": float(kl_post.detach().mean().cpu().item()),
            # audit helpers (pre)
            "ratio_mean_pre": float(ratio_pre.detach().mean().cpu().item()),
            "kl_ref_pre": float(kl_pre.detach().mean().cpu().item()),
            # nonnegative KL proxy (post)
            "kl_ref_abs": float(kl_ref_abs.detach().cpu().item()),
            "kl_ref_sq": float(kl_ref_sq.detach().cpu().item()),
        }

    # -------------------------
    # Internals
    # -------------------------
    def _make_ppo_batch(self, prompts: Sequence[str], completions: Sequence[str]) -> _PPOBatch:
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids_list: List[List[int]] = []
        prompt_lens: List[int] = []

        for p, c in zip(prompts, completions):
            p_ids = self.tokenizer(p, add_special_tokens=False).input_ids
            full_ids = self.tokenizer((p + " " + c).strip(), add_special_tokens=False).input_ids

            if len(full_ids) > self.hf_max_seq_len:
                full_ids = full_ids[-self.hf_max_seq_len :]
                pl = min(len(p_ids), len(full_ids))
            else:
                pl = len(p_ids)

            input_ids_list.append(list(full_ids))
            prompt_lens.append(int(pl))

        max_len = max(len(x) for x in input_ids_list) if input_ids_list else 1

        attn_list: List[List[int]] = []
        for i, ids in enumerate(input_ids_list):
            pad_n = max_len - len(ids)
            padded = ids + [pad_id] * pad_n
            attn = [1] * len(ids) + [0] * pad_n
            input_ids_list[i] = padded
            attn_list.append(attn)

        input_ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        attn_t = torch.tensor(attn_list, dtype=torch.long, device=self.device)
        pl_t = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)
        return _PPOBatch(input_ids=input_ids_t, attention_mask=attn_t, prompt_lens=pl_t, pad_id=pad_id)

    def _completion_token_counts(self, prompts: Sequence[str], completions: Sequence[str]) -> torch.Tensor:
        counts: List[int] = []
        for p, c in zip(prompts, completions):
            p_ids = self.tokenizer(p, add_special_tokens=False).input_ids
            full_ids = self.tokenizer((p + " " + c).strip(), add_special_tokens=False).input_ids
            if len(full_ids) > self.hf_max_seq_len:
                full_ids = full_ids[-self.hf_max_seq_len :]
                pl = min(len(p_ids), len(full_ids))
            else:
                pl = len(p_ids)
            counts.append(max(1, len(full_ids) - pl))
        return torch.tensor(counts, dtype=torch.long, device=self.device)

    def _logprob_completion_sum(
        self,
        model: torch.nn.Module,
        prompts: Sequence[str],
        completions: Sequence[str],
    ) -> torch.Tensor:
        batch = self._make_ppo_batch(prompts, completions)
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        prompt_lens = batch.prompt_lens
        pad_id = batch.pad_id

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        logp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_logp = logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        bsz, t1 = token_logp.shape
        pos = torch.arange(t1, device=self.device).unsqueeze(0).expand(bsz, t1)

        start = torch.clamp(prompt_lens - 1, min=0).unsqueeze(1)
        mask = (pos >= start) & (shift_labels != pad_id)

        summed = (token_logp * mask.float()).sum(dim=1)
        return summed
