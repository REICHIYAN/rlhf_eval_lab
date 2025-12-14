# rlhf_eval_lab/reporting/artifacts.py
# ArtifactsV1: すべての指標計算の単一情報源（SSOT）
# - 「空欄ゼロ」を保証するため、validate_strict を厳格にする
# - run は必ず artifacts.json を吐き、report はそれだけを読む

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


class ArtifactsValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ProvenanceV1:
    backend: str                       # "fallback" / "hf"
    model_id: str                      # "tiny-gru" / "gpt2" etc.
    tokenizer: str                     # "simple" / "hf"
    config_hash: str                   # hashing.py で計算
    git_commit: Optional[str]          # 取得できるなら
    seed: int


@dataclass(frozen=True)
class ArtifactsV1:
    # 必須：識別子
    method_key: str
    dataset_key: str
    provenance: ProvenanceV1

    # 必須：生成ログ（評価の原材料）
    prompts: List[str]
    completions: List[str]

    # 必須：学習/評価の数値ログ（runが“実測で”埋める）
    rewards: List[float]               # completion単位
    # 追加情報（将来拡張）。辞書の中身は run 側で埋める
    extra: Dict[str, Any]

    def validate_strict(self) -> None:
        # 型・存在の厳格チェック（欠損は即失敗）
        if not isinstance(self.method_key, str) or not self.method_key:
            raise ArtifactsValidationError("method_key is missing/invalid")
        if not isinstance(self.dataset_key, str) or not self.dataset_key:
            raise ArtifactsValidationError("dataset_key is missing/invalid")
        if not isinstance(self.provenance, ProvenanceV1):
            raise ArtifactsValidationError("provenance is missing/invalid")

        if not isinstance(self.prompts, list) or not all(isinstance(x, str) for x in self.prompts):
            raise ArtifactsValidationError("prompts must be List[str]")
        if not isinstance(self.completions, list) or not all(isinstance(x, str) for x in self.completions):
            raise ArtifactsValidationError("completions must be List[str]")
        if len(self.prompts) != len(self.completions):
            raise ArtifactsValidationError("prompts and completions length mismatch")

        if not isinstance(self.rewards, list) or not all(isinstance(x, (int, float)) for x in self.rewards):
            raise ArtifactsValidationError("rewards must be List[float]")
        if len(self.rewards) != len(self.completions):
            raise ArtifactsValidationError("rewards and completions length mismatch")

        if not isinstance(self.extra, dict):
            raise ArtifactsValidationError("extra must be dict")

        # provenance の必須チェック
        p = self.provenance
        if p.backend not in ("fallback", "hf"):
            raise ArtifactsValidationError(f"provenance.backend invalid: {p.backend}")
        if not isinstance(p.model_id, str) or not p.model_id:
            raise ArtifactsValidationError("provenance.model_id missing")
        if not isinstance(p.tokenizer, str) or not p.tokenizer:
            raise ArtifactsValidationError("provenance.tokenizer missing")
        if not isinstance(p.config_hash, str) or not p.config_hash:
            raise ArtifactsValidationError("provenance.config_hash missing")
        if not isinstance(p.seed, int):
            raise ArtifactsValidationError("provenance.seed invalid")

        # “必ず数値が出る”の最低条件：少なくとも1件はある
        if len(self.prompts) < 1:
            raise ArtifactsValidationError("prompts must contain at least 1 item")
        if len(self.rewards) < 1:
            raise ArtifactsValidationError("rewards must contain at least 1 item")


def _to_jsonable(a: ArtifactsV1) -> Dict[str, Any]:
    d = asdict(a)
    # dataclass nested の Provenance を辞書として保持（asdictでOK）
    return d


def write_artifacts(path: str, a: ArtifactsV1) -> None:
    a.validate_strict()
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(a), f, ensure_ascii=False, indent=2)


def _parse_provenance(d: Dict[str, Any]) -> ProvenanceV1:
    return ProvenanceV1(
        backend=str(d["backend"]),
        model_id=str(d["model_id"]),
        tokenizer=str(d["tokenizer"]),
        config_hash=str(d["config_hash"]),
        git_commit=(None if d.get("git_commit") in (None, "") else str(d.get("git_commit"))),
        seed=int(d["seed"]),
    )


def read_artifacts(path: str) -> ArtifactsV1:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)

    a = ArtifactsV1(
        method_key=str(d["method_key"]),
        dataset_key=str(d["dataset_key"]),
        provenance=_parse_provenance(d["provenance"]),
        prompts=list(d["prompts"]),
        completions=list(d["completions"]),
        rewards=[float(x) for x in d["rewards"]],
        extra=dict(d.get("extra", {})),
    )
    a.validate_strict()
    return a
