# rlhf_eval_lab/reporting/artifacts.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from rlhf_eval_lab.reporting.provenance import ProvenanceV1


class ArtifactsValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactsV1:
    method_key: str
    dataset_key: str
    provenance: ProvenanceV1
    prompts: List[str]
    completions: List[str]
    rewards: List[float]
    extra: Dict[str, Any]

    def validate(self) -> None:
        # 基本
        if not isinstance(self.method_key, str) or not self.method_key.strip():
            raise ArtifactsValidationError("method_key missing/invalid")
        if not isinstance(self.dataset_key, str) or not self.dataset_key.strip():
            raise ArtifactsValidationError("dataset_key missing/invalid")
        if not isinstance(self.prompts, list):
            raise ArtifactsValidationError("prompts missing/invalid")
        if not isinstance(self.completions, list):
            raise ArtifactsValidationError("completions missing/invalid")
        if not isinstance(self.rewards, list):
            raise ArtifactsValidationError("rewards missing/invalid")
        if len(self.prompts) != len(self.completions) or len(self.prompts) != len(self.rewards):
            raise ArtifactsValidationError("prompts/completions/rewards length mismatch")
        if not isinstance(self.extra, dict):
            raise ArtifactsValidationError("extra missing/invalid")

        # provenance（必須）
        if not isinstance(self.provenance, ProvenanceV1):
            raise ArtifactsValidationError("provenance is missing/invalid")

        p = self.provenance
        if p.backend not in {"fallback", "hf"}:
            raise ArtifactsValidationError(f"provenance.backend invalid: {p.backend}")
        if not p.model_id:
            raise ArtifactsValidationError("provenance.model_id missing")
        if not p.tokenizer:
            raise ArtifactsValidationError("provenance.tokenizer missing")
        if not p.config_hash:
            raise ArtifactsValidationError("provenance.config_hash missing")
        if not isinstance(p.seed, int):
            raise ArtifactsValidationError("provenance.seed invalid")


def _parse_provenance(d: Dict[str, Any]) -> ProvenanceV1:
    # ProvenanceV1 は dataclass の想定
    return ProvenanceV1(
        backend=str(d.get("backend", "")),
        model_id=str(d.get("model_id", "")),
        tokenizer=str(d.get("tokenizer", "")),
        config_hash=str(d.get("config_hash", "")),
        git_commit=str(d.get("git_commit", "")),
        seed=int(d.get("seed", 0)),
    )


def read_artifacts(path: str) -> ArtifactsV1:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # 互換：トップレベル provenance が dict
    prov = _parse_provenance(d.get("provenance", {}))

    art = ArtifactsV1(
        method_key=str(d.get("method_key", "")),
        dataset_key=str(d.get("dataset_key", "")),
        provenance=prov,
        prompts=list(d.get("prompts", [])),
        completions=list(d.get("completions", [])),
        rewards=[float(x) for x in d.get("rewards", [])],
        extra=dict(d.get("extra", {})),
    )
    art.validate()
    return art


def write_artifacts(path: str, art: ArtifactsV1) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "method_key": art.method_key,
        "dataset_key": art.dataset_key,
        "provenance": {
            "backend": art.provenance.backend,
            "model_id": art.provenance.model_id,
            "tokenizer": art.provenance.tokenizer,
            "config_hash": art.provenance.config_hash,
            "git_commit": art.provenance.git_commit,
            "seed": art.provenance.seed,
        },
        "prompts": art.prompts,
        "completions": art.completions,
        "rewards": art.rewards,
        "extra": art.extra,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_artifacts_tree(root_dir: str) -> List[ArtifactsV1]:
    """
    artifacts_root 以下を再帰的に走査して ArtifactsV1 を収集する。
    - 1ファイル壊れても全体が死ぬのは DoD 的に危険なので、ここは fail-fast にする
      （壊れているなら validate で落として原因を即特定）
    """
    root = os.path.abspath(root_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"artifacts root not found: {root}")

    arts: List[ArtifactsV1] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".json"):
                continue
            p = os.path.join(dirpath, fn)
            arts.append(read_artifacts(p))

    # 安定順（method_key, seed, path っぽい順）
    arts.sort(key=lambda a: (a.method_key, a.provenance.seed))
    return arts
