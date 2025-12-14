# rlhf_eval_lab/reporting/provenance.py
# provenance（来歴）を artifacts / report に刻む SSOT
# - git commit は取得できる場合のみ（失敗しても止めない）

from __future__ import annotations

import subprocess
from typing import Any, Dict, Optional

from rlhf_eval_lab.reporting.artifacts import ProvenanceV1
from rlhf_eval_lab.utils.hashing import hash_config


def detect_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8").strip()
        return s if s else None
    except Exception:
        return None


def build_provenance(
    cfg: Dict[str, Any],
    *,
    backend: str,
    model_id: str,
    tokenizer: str,
    seed: int,
) -> ProvenanceV1:
    return ProvenanceV1(
        backend=str(backend),
        model_id=str(model_id),
        tokenizer=str(tokenizer),
        config_hash=str(hash_config(cfg)),
        git_commit=detect_git_commit(),
        seed=int(seed),
    )
