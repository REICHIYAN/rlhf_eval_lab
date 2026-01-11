# rlhf_eval_lab/backends/hf/backend.py
# HFBackend：論文用（optional）
# CI では fallback-only を前提にするため、ここは最小限の骨格。
# ただし HF Step2 では SFT を「最小で実際に学習」させる（監査可能）。

from __future__ import annotations

from typing import Any, Dict, List, Optional
import copy
import logging

import torch

from rlhf_eval_lab.backends.base import ModelBackend
from rlhf_eval_lab.utils.exceptions import DependencyMissingError
from .utils import lazy_import_transformers


class HFBackend(ModelBackend):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        try:
            tf = lazy_import_transformers()
        except Exception as e:
            raise DependencyMissingError(str(e)) from e

        # Transformers のログは実験/CIの監査性のため安定化（必要なら ERROR 以上へ）
        try:
            tf.utils.logging.set_verbosity_error()
            tf.utils.logging.disable_progress_bar()
        except Exception:
            pass

        try:
            logging.getLogger("transformers").setLevel(logging.ERROR)
        except Exception:
            pass

        model_name = str(config.get("hf", {}).get("model_name", "gpt2"))

        # ------------------------------
        # ★ warning の発生源を潰す：config を先に作って loss_type を埋めてから model を作る
        # ------------------------------
        cfg = tf.AutoConfig.from_pretrained(model_name)
        # Transformers の版差で「loss_type=None が config に入っている」と warning が出るケースがある。
        # ここで “確実に” 文字列へ上書きしてから model をロードする。
        try:
            setattr(cfg, "loss_type", "ForCausalLMLoss")
        except Exception:
            try:
                cfg.__dict__["loss_type"] = "ForCausalLMLoss"
            except Exception:
                pass

        self.tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
        self.model = tf.AutoModelForCausalLM.from_pretrained(model_name, config=cfg)

        # GPT2など pad_token が無いモデル向けの安全策
        if self.tokenizer.pad_token_id is None:
            # eos を pad として扱う（generateの警告も減る）
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        self.ref_model = None

    def clone_reference(self) -> None:
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.to(self.device)
        self.ref_model.eval()

    @torch.no_grad()
    def generate(self, prompts: List[str], max_new_tokens: int = 16) -> List[str]:
        out: List[str] = []
        temperature = float(self.config.get("hf", {}).get("temperature", 1.0))
        for p in prompts:
            inp = self.tokenizer(p, return_tensors="pt").to(self.device)
            gen = self.model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=int(self.tokenizer.pad_token_id),
            )
            out.append(self.tokenizer.decode(gen[0], skip_special_tokens=True))
        return out

    @torch.no_grad()
    def logprobs(self, texts: List[str]) -> List[float]:
        vals: List[float] = []
        for t in texts:
            enc = self.tokenizer(t, return_tensors="pt").to(self.device)
            ids = enc["input_ids"]
            logits = self.model(ids).logits[:, :-1, :]
            tgt = ids[:, 1:]
            logp = torch.log_softmax(logits, dim=-1)
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()
            vals.append(float(lp.detach().cpu().item()))
        return vals

    def sft_step(self, texts: List[str]) -> float:
        """
        HF Step2: Minimal SFT that *actually trains* for a few steps.

        Config (train):
          - hf_sft_steps: int (default 0)
          - lr: float (default 1e-4)
          - grad_clip: float (default 1.0)
          - hf_max_seq_len: int (default 256)
        """
        train_cfg = self.config.get("train", {}) or {}
        steps = int(train_cfg.get("hf_sft_steps", 0))
        if steps <= 0:
            return 0.0

        lr = float(train_cfg.get("lr", 1e-4))
        grad_clip = float(train_cfg.get("grad_clip", 1.0))
        max_len = int(train_cfg.get("hf_max_seq_len", 256))

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.model.train()

        last_loss = 0.0
        for _ in range(steps):
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(self.device)

            input_ids = enc["input_ids"]
            attn = enc.get("attention_mask", None)

            # causal LM: pad は loss から除外（-100）
            labels = input_ids.clone()
            if attn is not None:
                labels = labels.masked_fill(attn == 0, -100)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels,
            )
            loss = out.loss
            if loss is None:
                raise RuntimeError("HFBackend.sft_step: model did not return loss")

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            opt.step()
            last_loss = float(loss.detach().cpu().item())

        self.model.eval()
        return last_loss

    def get_ref_kl(self, texts: List[str]) -> List[float]:
        if self.ref_model is None:
            return [0.0 for _ in texts]
        vals: List[float] = []
        for t in texts:
            enc = self.tokenizer(t, return_tensors="pt").to(self.device)
            ids = enc["input_ids"]
            with torch.no_grad():
                lp = self._seq_logprob(self.model, ids)
                lp_ref = self._seq_logprob(self.ref_model, ids)
            vals.append(float(abs(lp - lp_ref)))
        return vals

    @staticmethod
    @torch.no_grad()
    def _seq_logprob(model, ids: torch.Tensor) -> float:
        logits = model(ids).logits[:, :-1, :]
        tgt = ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()
        return float(lp.detach().cpu().item())

    # NOTE: HF PPO / preference は Step2 以降で拡張。
    # ここは run.py が HF で呼ばない前提だが、IFの整合性は崩さない。

    def ppo_step(
        self,
        prompts: List[str],
        completions: Optional[List[str]] = None,
        rewards: Optional[List[float]] = None,
        kl_beta: float = 0.0,
        ref_state: Optional[Dict[str, torch.Tensor]] = None,
        update_ref: bool = False,
    ) -> Dict[str, Any]:
        _ = (completions, rewards, kl_beta, ref_state, update_ref)
        return {"loss": 0.0, "kl": 0.0, "steps": 0}

    def preference_step(self, prompt: str, chosen: str, rejected: str, beta: float = 0.1) -> float:
        _ = (prompt, chosen, rejected, beta)
        return 0.0
