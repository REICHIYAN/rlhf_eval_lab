# rlhf_eval_lab/cli/commands/validate.py

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

from rlhf_eval_lab.eval.runner import evaluate_artifacts
from rlhf_eval_lab.registry.methods import METHOD_BY_KEY, METHOD_KEYS
from rlhf_eval_lab.registry.metrics import METRIC_SPECS
from rlhf_eval_lab.reporting.artifacts import ArtifactsV1, read_artifacts
from rlhf_eval_lab.reporting.paths import report_md_path
from rlhf_eval_lab.reporting.validate_report_md import validate_report_md_or_raise


def _na() -> str:
    return "N/A"


def _is_na(v: Any) -> bool:
    return isinstance(v, str) and v.strip().upper() == "N/A"


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        try:
            return bool(int(x))
        except Exception:
            return False
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, bool):
            return int(x)
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)
    except Exception:
        return int(default)


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


def _validate_provenance(art: ArtifactsV1) -> None:
    p = art.provenance
    missing: List[str] = []
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


# runner.py と同じ PPO-like 判定（DoD fix: adaptive_rm_ppo は PPO-family 扱い）
def _is_ppo_like(method_key: str) -> bool:
    m = METHOD_BY_KEY.get(method_key)
    if m is None:
        return False
    return bool(getattr(m, "is_ppo_family", False)) or (method_key == "adaptive_rm_ppo")


# runner.py の挙動に合わせ、training 未実行のときだけ N/A を許可する監査系メトリクス
_PPO_AUDIT_METRIC_KEYS = {
    "ppo_loss",
    "ratio_mean",
    "clipfrac",
    "kl_ref_abs",
    "kl_ref_sq",
    "kl_stability",
    "convergence_speed",
}


def _training_executed(extra: Dict[str, Any]) -> bool:
    skipped = _as_bool(extra.get("skipped"))
    steps = _as_int(extra.get("steps"), 0)
    return (not skipped) and steps > 0


def _dynamic_na_allowed(method_key: str, metric_key: str, extra: Dict[str, Any]) -> bool:
    """
    Dynamic N/A policy:
      - PPO-like methods
      - AND training not executed (HF Step1 / explicitly skipped)
      - AND metric is PPO audit diagnostic
    """
    if not _is_ppo_like(method_key):
        return False
    if metric_key not in _PPO_AUDIT_METRIC_KEYS:
        return False
    if _training_executed(extra):
        return False
    return True


def _validate_metrics(method_key: str, metrics: Dict[str, Any], extra: Dict[str, Any]) -> None:
    # 1) 欠損禁止（evaluate_artifacts が SSOT）
    for ms in METRIC_SPECS:
        if ms.key not in metrics:
            raise ValueError(f"Metric missing: {ms.key} (method={method_key})")
        if metrics[ms.key] is None:
            raise ValueError(f"Metric None forbidden: {ms.key} (method={method_key})")

    # 2) N/A 規約（列単位） + 動的例外（PPO監査列）
    for ms in METRIC_SPECS:
        v = metrics[ms.key]
        na_list = ms.na_for_method_keys or []

        # (A) 列ポリシーで N/A 指定のメソッド
        if method_key in na_list:
            if not _is_na(v):
                raise ValueError(
                    f"N/A rule violated: metric={ms.key} must be N/A for method={method_key}, got={v}"
                )
            continue

        # (B) それ以外：基本は dtype に従う
        if ms.dtype != "str":
            # 数値列は N/A 不可。ただし動的例外を許可する。
            if _is_na(v):
                if _dynamic_na_allowed(method_key, ms.key, extra):
                    continue
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

    # method -> list of artifact paths
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
            _validate_metrics(method_key, metrics, art.extra or {})

    # Optional: validate rendered report.md (DoD invariants).
    report_md: Optional[str] = getattr(args, "report_md", None)
    report_dir: Optional[str] = getattr(args, "report_dir", None)
    if report_md or report_dir:
        if not report_md:
            report_md = report_md_path(os.path.abspath(str(report_dir)))
        report_md = os.path.abspath(str(report_md))
        if not os.path.exists(report_md):
            raise FileNotFoundError(f"report.md not found: {report_md}")
        with open(report_md, "r", encoding="utf-8") as f:
            md = f.read()
        validate_report_md_or_raise(md)

    return 0
