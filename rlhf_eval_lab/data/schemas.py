# rlhf_eval_lab/data/schemas.py
# jsonl の最小スキーマ（今回は薄く、後で拡張）

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class PromptItem:
    prompt: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PromptItem":
        return PromptItem(prompt=str(d.get("prompt", "")))
