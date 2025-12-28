# rlhf_eval_lab/reporting/aggregate.py
# 集計：
# - method ごとに seed_* を読み
# - evaluate_artifacts() を回して metric dict を得る
# - numeric は mean を返す
# - "N/A" は列規約で method に固定される前提（混在したら validate で落ちる）
# - Notes は空欄禁止なので "-" を入れる（将来拡張で extra から注入可）
#
# SSOT整合（重要）：
# - Table 1 の描画で期待されるキーを必ず埋める
#   - category / method_name
#   - off_support / on_support（旧 offsupport/onsupport 互換も維持）
# - report.md に "nan"/"inf" を絶対に出さない（float nan は markdown 側で "nan" になるため）

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


def _is_bad_float(v: Any) -> bool:
    return isinstance(v, float) and (math.isnan(v) or math.isinf(v))


def _sanitize_value(v: Any) -> Any:
    # report に nan/inf を出さない（最終的に markdown 側で N/A になる）
    if _is_bad_float(v):
        return None
    return v


def _collect_method_seed_paths(root: str, method_key: str) -> List[str]:
    pat = os.path.join(root, method_key, "seed_*.json")
    return sorted(glob.glob(pat))


def _mean(xs: List[float]) -> Any:
    # xs が空 or 全部 bad の場合は None（= markdownで N/A）
    if not xs:
        return None
    m = float(sum(xs) / float(len(xs)))
    if math.isnan(m) or math.isinf(m):
        return None
    return m


def _inject_ssot_fields(method_spec: Any, agg: Dict[str, Any]) -> Dict[str, Any]:
    """
    SSOT整合のため、Table1 で必須の列を必ず埋める。
    互換性のため旧キーも残しつつ、新キーも用意する。
    """
    # category / method_name（Table1）
    if "category" not in agg:
        agg["category"] = getattr(method_spec, "category", None)
    if "method_name" not in agg:
        # methods.py の spec に name がある想定
        agg["method_name"] = getattr(method_spec, "name", None)

    # offsupport/onsupport -> off_support/on_support（alias）
    # ※evaluate/metrics 側が offsupport/onsupport のままでも report が壊れないようにする
    if "offsupport" in agg and "off_support" not in agg:
        agg["off_support"] = agg["offsupport"]
    if "onsupport" in agg and "on_support" not in agg:
        agg["on_support"] = agg["onsupport"]

    # 念のため sanitize（ここで nan/inf を完全に潰す）
    for k, v in list(agg.items()):
        agg[k] = _sanitize_value(v)

    return agg


def aggregate(root: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      method_key -> metric_key -> aggregated value (float or "N/A" or str or None)
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
            row = evaluate_artifacts(art)
            # runner が返す値にも nan/inf が紛れ得るので sanitize
            metric_rows.append({k: _sanitize_value(v) for k, v in row.items()})

        # aggregate per metric
        agg: Dict[str, Any] = {}
        for ms in METRIC_SPECS:
            vals = [row.get(ms.key) for row in metric_rows]

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

            # numeric or "N/A" or None
            if vals and _is_na(vals[0]):
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
                    v = _sanitize_value(v)
                    if v is None:
                        # nan/inf は None に落として平均から除外
                        continue
                    xs.append(float(v))
                agg[ms.key] = _mean(xs)

        # Notes: 空欄禁止（表セル全埋め）
        agg["notes"] = "-"  # 表では Notes 列にこれを使う

        # SSOT fields (category/method_name + key aliases + sanitize)
        agg = _inject_ssot_fields(m, agg)

        out[m.key] = agg

    return out


# Backward-compatible alias (older modules import this name)
def aggregate_seed_means(root: str) -> Dict[str, Dict[str, Any]]:
    return aggregate(root)
