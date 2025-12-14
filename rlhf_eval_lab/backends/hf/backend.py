# rlhf_eval_lab/backends/hf/backend.py
# HFBackend：論文用（optional）
# CI では fallback-only を前提にするため、ここは最小限の骨格

from __future__ import annotations
from typing import Any, Dict, List
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

        model_name = config.get("hf", {}).get("model_name", "gpt2")
        self.tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
        self.model = tf.AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.ref_model = None

    def clone_reference(self) -> None:
        # 参照固定（簡易）
        import copy
        self.ref_model = copy.deepcopy(self.model)

    @torch.no_grad()
    def generate(self, prompts: List[str], max_new_tokens: int = 16) -> List[str]:
        out = []
        for p in prompts:
            inp = self.tokenizer(p, return_tensors="pt").to(self.device)
            gen = self.model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=float(self.config.get("hf", {}).get("temperature", 1.0)),
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
        # 論文用は別途 trainer で行う想定。ここでは未対応でも良いが、数値は返す。
        return 0.0

    def get_ref_kl(self, texts: List[str]) -> List[float]:
        if self.ref_model is None:
            return [0.0 for _ in texts]
        # 簡易 proxy（logprob差）
        vals = []
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

    def ppo_step(self, prompts: List[str], rewards: List[float]) -> Dict[str, Any]:
        # HF backend の PPO は本スコープ外（論文実装で拡張）
        return {"loss": 0.0, "kl_values": [0.0 for _ in prompts], "steps": 0}

    def preference_step(self, prompts: List[str], rewards: List[float]) -> Dict[str, Any]:
        # HF backend の preference は本スコープ外（論文実装で拡張）
        return {"loss": 0.0, "wins": [0 for _ in prompts], "steps": 0}
