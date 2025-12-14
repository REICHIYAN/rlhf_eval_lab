# rlhf_eval_lab/reporting/aggregate.py
# 集計：
# - method ごとに seed_* を読み
# - evaluate_artifacts() を回して metric dict を得る
# - numeric は mean を返す
# - "N/A" は列規約で method に固定される前提（混在したら validate で落ちる）
# - Notes は空欄禁止なので "-" を入れる（将来拡張で extra から注入可）

from __future__ import annotations

from typing import Any, Dict, List
import glob
import os
import math

from rlhf_eval_lab.registry.methods import METHOD_SPECS
from rlhf_eval_lab.registry.metrics import METRIC_SPECS
from rlhf_eval_lab.reporting.artifacts import read_artifacts
from rlhf_eval_lab.eval.runner import evaluate_artifacts


def _is_na(v: Any) -> bool:
    return isinstance(v, str) and v.strip().upper() == "N/A"


def _collect_method_seed_paths(root: str, method_key: str) -> List[str]:
    pat = os.path.join(root, method_key, "seed_*.json")
    return sorted(glob.glob(pat))


def _mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / float(len(xs)))


def aggregate(root: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      method_key -> metric_key -> aggregated value (float or "N/A" or str)
    """
    root = os.path.abspath(root)
    out: Dict[str, Dict[str, Any]] = {}

    for m in METHOD_SPECS:
        paths = _collect_method_seed_paths(root, m.key)
        if not paths:
            raise FileNotFoundError(f"Missing artifacts for method={m.key} under {root}")

        metric_rows: List[Dict[str, Any]] = []
        for p in paths:
            art = read_artifacts(p)
            metric_rows.append(evaluate_artifacts(art))

        # aggregate per metric
        agg: Dict[str, Any] = {}
        for ms in METRIC_SPECS:
            vals = [row[ms.key] for row in metric_rows]

            if ms.dtype == "str":
                # all strings should be identical per method (e.g., label_source)
                s0 = str(vals[0])
                for v in vals[1:]:
                    if str(v) != s0:
                        raise ValueError(
                            f"String metric inconsistent across seeds: method={m.key} metric={ms.key} {s0} vs {v}"
                        )
                agg[ms.key] = s0
                continue

            # numeric or "N/A"
            if _is_na(vals[0]):
                # should all be N/A
                for v in vals[1:]:
                    if not _is_na(v):
                        raise ValueError(
                            f"N/A mixed with numeric: method={m.key} metric={ms.key}"
                        )
                agg[ms.key] = "N/A"
            else:
                xs: List[float] = []
                for v in vals:
                    if _is_na(v):
                        raise ValueError(
                            f"Unexpected N/A in numeric metric: method={m.key} metric={ms.key}"
                        )
                    xs.append(float(v))
                agg[ms.key] = _mean(xs)

        # Notes: 空欄禁止（表セル全埋め）
        agg["notes"] = "-"  # 表では Notes 列にこれを使う
        out[m.key] = agg

    return out


# Backward-compatible alias (older modules import this name)
def aggregate_seed_means(root: str) -> Dict[str, Dict[str, Any]]:
    return aggregate(root)
