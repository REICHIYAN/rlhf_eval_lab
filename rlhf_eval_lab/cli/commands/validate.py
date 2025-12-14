# rlhf_eval_lab/cli/commands/validate.py
# 厳格検証：
# - ArtifactsV1 が全手法ぶん存在
# - eval/runner が全 metric を返し、None/空欄がない
# - N/A 規約（列単位）を満たす（例：preference は KL = N/A）
# - 数値列は float に変換できる
# - seed/provenance 必須

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import glob
import os

from rlhf_eval_lab.registry.methods import METHOD_KEYS, METHOD_BY_KEY
from rlhf_eval_lab.registry.metrics import METRIC_SPECS, METRIC_BY_KEY
from rlhf_eval_lab.reporting.artifacts import read_artifacts, ArtifactsV1
from rlhf_eval_lab.eval.runner import evaluate_artifacts


def _na() -> str:
    return "N/A"


def _get_artifacts_root(args) -> str:
    # run.py は args.out を使う。validate/report は args.in / args.out どちらでも動くようにする。
    cand = [
        getattr(args, "in_dir", None),
        getattr(args, "input", None),
        getattr(args, "in_path", None),
        getattr(args, "out", None),
        getattr(args, "artifacts", None),
    ]
    for c in cand:
        if c:
            return os.path.abspath(str(c))
    return os.path.abspath("artifacts")


def _collect_artifact_paths(root: str) -> List[str]:
    # {root}/{method}/seed_*.json
    pats = [os.path.join(root, "*", "seed_*.json")]
    out: List[str] = []
    for p in pats:
        out.extend(glob.glob(p))
    return sorted(set(out))


def _is_na(v: Any) -> bool:
    return isinstance(v, str) and v.strip().upper() == "N/A"


def _validate_provenance(art: ArtifactsV1) -> None:
    p = art.provenance
    missing = []
    for k in ["backend", "model_id", "tokenizer", "config_hash", "seed"]:
        if getattr(p, k, None) in (None, ""):
            missing.append(k)
    if missing:
        raise ValueError(f"Missing provenance fields: {missing} (method={art.method_key})")


def _validate_artifacts_shape(art: ArtifactsV1) -> None:
    if art.method_key not in METHOD_BY_KEY:
        raise ValueError(f"Unknown method_key: {art.method_key}")
    if not isinstance(art.prompts, list) or not art.prompts:
        raise ValueError(f"prompts missing/empty (method={art.method_key})")
    if not isinstance(art.completions, list) or len(art.completions) != len(art.prompts):
        raise ValueError(f"completions shape mismatch (method={art.method_key})")
    if not isinstance(art.rewards, list) or len(art.rewards) != len(art.prompts):
        raise ValueError(f"rewards shape mismatch (method={art.method_key})")
    _validate_provenance(art)


def _validate_metrics(
    method_key: str,
    metrics: Dict[str, Any],
) -> None:
    # 1) 欠損禁止
    for ms in METRIC_SPECS:
        if ms.key not in metrics:
            raise ValueError(f"Metric missing: {ms.key} (method={method_key})")
        if metrics[ms.key] is None:
            raise ValueError(f"Metric None forbidden: {ms.key} (method={method_key})")

    # 2) N/A 規約（列単位）
    for ms in METRIC_SPECS:
        v = metrics[ms.key]
        na_list = ms.na_for_method_keys or []
        if method_key in na_list:
            if not _is_na(v):
                raise ValueError(
                    f"N/A rule violated: metric={ms.key} must be N/A for method={method_key}, got={v}"
                )
        else:
            # dtype=str の列以外は数値である必要（N/A 不可）
            if ms.dtype != "str":
                if _is_na(v):
                    raise ValueError(
                        f"Unexpected N/A: metric={ms.key} is numeric but got N/A for method={method_key}"
                    )
                try:
                    float(v)
                except Exception as e:
                    raise ValueError(
                        f"Metric not numeric: metric={ms.key} method={method_key} value={v}"
                    ) from e
            else:
                # str 列は空文字禁止（表の空欄禁止）
                if not isinstance(v, str) or v.strip() == "":
                    raise ValueError(
                        f"String metric empty forbidden: metric={ms.key} method={method_key} value={v}"
                    )


def validate_cmd(args) -> int:
    root = _get_artifacts_root(args)
    paths = _collect_artifact_paths(root)
    if not paths:
        raise FileNotFoundError(f"No artifacts found under: {root}")

    # method -> list of (seed, path)
    by_method: Dict[str, List[str]] = {}
    for p in paths:
        # expect .../{method}/seed_{seed}.json
        method_key = os.path.basename(os.path.dirname(p))
        by_method.setdefault(method_key, []).append(p)

    # 全手法存在チェック（最低1 seed）
    missing_methods = [m for m in METHOD_KEYS if m not in by_method]
    if missing_methods:
        raise ValueError(f"Missing method artifacts: {missing_methods}")

    # 各ファイル検証 + eval 実行
    for method_key, ps in by_method.items():
        if method_key not in METHOD_BY_KEY:
            raise ValueError(f"Unknown method dir: {method_key}")

        for p in ps:
            art = read_artifacts(p)
            _validate_artifacts_shape(art)

            # SSOT = evaluate_artifacts が返す全metricが正
            metrics = evaluate_artifacts(art)
            _validate_metrics(method_key, metrics)

    return 0
