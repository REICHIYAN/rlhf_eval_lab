# rlhf_eval_lab/utils/hashing.py
# 設定・入出力のハッシュ（provenance 用）

from __future__ import annotations
from typing import Any
import hashlib
import json


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def hash_config(config: Any) -> str:
    return sha256_hex(stable_json_dumps(config))
