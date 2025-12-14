# rlhf_eval_lab/data/loaders.py
# jsonl ローダー（依存最小）

from __future__ import annotations
from typing import Any, Dict, List
import json


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
