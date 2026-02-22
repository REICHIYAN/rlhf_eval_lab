# rlhf_eval_lab/reporting/provenance.py
# provenance（来歴）を artifacts / report に刻む SSOT
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict

# NOTE:
# - hashing.py の関数名揺れに依存しない（DoD: 壊れない）
# - 使える関数があればそれを使い、無ければここで sha256 を作る


def _stable_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # 最悪でも repr で落とさない
        return repr(obj)


def _sha256_hex(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="replace"))
    return h.hexdigest()


def hash_config(cfg: Dict[str, Any]) -> str:
    """
    config のハッシュ（provenance 用）。
    可能なら rlhf_eval_lab.utils.hashing の既存関数を使う。
    無ければこの関数で安定sha256を返す。
    """
    try:
        from rlhf_eval_lab.utils import hashing as H  # type: ignore

        # よくある命名揺れを全部吸収
        for name in (
            "hash_config_dict",
            "hash_dict",
            "stable_hash_dict",
            "hash_any",
            "hash_json",
        ):
            fn = getattr(H, name, None)
            if callable(fn):
                v = fn(cfg)  # type: ignore[misc]
                s = str(v).strip()
                if s:
                    return s
    except Exception:
        pass

    # fallback: ここで確定
    return _sha256_hex(_stable_json(cfg))


@dataclass(frozen=True)
class ProvenanceV1:
    backend: str
    model_id: str
    tokenizer: str
    config_hash: str
    git_commit: str
    seed: int


def detect_git_commit() -> str:
    """
    git が無い / リポジトリ外 / 失敗でも落とさない（DoD: 壊れない）。
    """
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="replace").strip()
        return s if s else "N/A"
    except Exception:
        return "N/A"


def build_provenance(
    cfg: Dict[str, Any],
    *,
    backend: str,
    model_id: str,
    tokenizer: str,
    seed: int,
) -> ProvenanceV1:
    config_hash = hash_config(cfg)
    git_commit = detect_git_commit()
    return ProvenanceV1(
        backend=str(backend),
        model_id=str(model_id),
        tokenizer=str(tokenizer),
        config_hash=str(config_hash),
        git_commit=str(git_commit),
        seed=int(seed),
    )
