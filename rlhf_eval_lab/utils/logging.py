# rlhf_eval_lab/utils/logging.py
# 最低限の構造化ログ（依存を増やさない）
# - CI/Colab で必ず動くことを優先

from __future__ import annotations
from typing import Any, Dict
import json
import time as _time


def log_event(event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"ts": int(_time.time()), "event": event}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False))
