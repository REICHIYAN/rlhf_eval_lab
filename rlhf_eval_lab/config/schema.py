# rlhf_eval_lab/config/schema.py
# Config スキーマ（型＆検証）
# tiny_lm.arch は gru 固定（将来拡張余地は残すがスコープ外）

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rlhf_eval_lab.utils.exceptions import ConfigError


@dataclass(frozen=True)
class TinyLMConfig:
    arch: str = "gru"
    emb_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    max_seq_len: int = 128


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    grad_clip: float = 1.0


@dataclass(frozen=True)
class EvalConfig:
    max_new_tokens: int = 16


@dataclass(frozen=True)
class PPOConfig:
    kl_beta: float = 0.1


@dataclass(frozen=True)
class HFConfig:
    model_name: str = "gpt2"
    temperature: float = 1.0


@dataclass(frozen=True)
class Config:
    tiny_lm: TinyLMConfig = TinyLMConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    ppo: PPOConfig = PPOConfig()
    hf: HFConfig = HFConfig()


def _get(d: Dict[str, Any], k: str, default: Any) -> Any:
    v = d.get(k, default)
    return default if v is None else v


def validate_config_dict(cfg: Dict[str, Any]) -> None:
    tiny = cfg.get("tiny_lm", {}) or {}
    arch = str(_get(tiny, "arch", "gru"))
    if arch != "gru":
        raise ConfigError(f"tiny_lm.arch must be 'gru' (got {arch})")


def build_config(cfg: Dict[str, Any]) -> Config:
    validate_config_dict(cfg)

    tiny = cfg.get("tiny_lm", {}) or {}
    train = cfg.get("train", {}) or {}
    ev = cfg.get("eval", {}) or {}
    ppo = cfg.get("ppo", {}) or {}
    hf = cfg.get("hf", {}) or {}

    return Config(
        tiny_lm=TinyLMConfig(
            arch=str(_get(tiny, "arch", "gru")),
            emb_dim=int(_get(tiny, "emb_dim", 64)),
            hidden_dim=int(_get(tiny, "hidden_dim", 128)),
            num_layers=int(_get(tiny, "num_layers", 1)),
            max_seq_len=int(_get(tiny, "max_seq_len", 128)),
        ),
        train=TrainConfig(
            lr=float(_get(train, "lr", 1e-3)),
            grad_clip=float(_get(train, "grad_clip", 1.0)),
        ),
        eval=EvalConfig(
            max_new_tokens=int(_get(ev, "max_new_tokens", 16)),
        ),
        ppo=PPOConfig(
            kl_beta=float(_get(ppo, "kl_beta", 0.1)),
        ),
        hf=HFConfig(
            model_name=str(_get(hf, "model_name", "gpt2")),
            temperature=float(_get(hf, "temperature", 1.0)),
        ),
    )
