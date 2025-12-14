# rlhf_eval_lab/config/io.py
# defaults.yaml + preset + user overrides を deep merge して dict を返す
# 依存増やさず PyYAML のみ

from __future__ import annotations
from typing import Any, Dict, Optional
import os

import yaml

from rlhf_eval_lab.utils.exceptions import ConfigError
from rlhf_eval_lab.config.schema import validate_config_dict


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _pkg_dir() -> str:
    return os.path.dirname(__file__)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ConfigError(f"Config YAML must be a dict: {path}")
    return obj


def load_config(preset: Optional[str] = "fallback_tiny", user_path: Optional[str] = None) -> Dict[str, Any]:
    base = _read_yaml(os.path.join(_pkg_dir(), "defaults.yaml"))

    # CLI から preset=None が来ても fallback_tiny に落とす（DoD: run が必ず完走）
    if not preset:
        preset = "fallback_tiny"

    preset_path = os.path.join(_pkg_dir(), "presets", f"{preset}.yaml")
    if not os.path.exists(preset_path):
        raise ConfigError(f"Unknown preset: {preset} ({preset_path} not found)")
    p = _read_yaml(preset_path)

    cfg = _deep_merge(base, p)

    if user_path:
        user_path = os.path.abspath(user_path)
        u = _read_yaml(user_path)
        cfg = _deep_merge(cfg, u)

    validate_config_dict(cfg)
    return cfg
