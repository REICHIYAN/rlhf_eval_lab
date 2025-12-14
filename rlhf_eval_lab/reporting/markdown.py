# rlhf_eval_lab/reporting/markdown.py
# Markdown 出力：
# - Table 1 / 2A / 2B / 2C を必ず出す
# - 全セルを数値 or "N/A" で埋める（Notes も "-"）
# - method 順序は registry.methods の SSOT
# - 列順は registry.metrics の SSOT

from __future__ import annotations

from typing import Any, Dict, List
import math

from rlhf_eval_lab.registry.methods import METHOD_SPECS


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        x = float(v)
    except Exception:
        return str(v)
    if math.isnan(x) or math.isinf(x):
        # 空欄禁止なので明示
        return "N/A"
    # sanity tier: 小数 4 桁固定（論文時に変えるならここで一括）
    return f"{x:.4f}"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        if len(r) != len(headers):
            raise ValueError("Row length mismatch in markdown table")
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def render_report(
    aggregated: Dict[str, Dict[str, Any]],
) -> str:
    """
    aggregated: method_key -> metric_key -> value
    """
    parts: List[str] = []

    # -------------------------
    # Table 1
    # -------------------------
    t1_headers = [
        "Category",
        "Method",
        "Off-support ↓",
        "Tail Var ↓",
        "On-support ↑",
        "Judge ↑",
        "Win-rate ↑",
        "KL ↓",
        "Notes",
    ]
    t1_rows: List[List[str]] = []
    for m in METHOD_SPECS:
        a = aggregated[m.key]
        t1_rows.append(
            [
                m.category,
                m.name,
                _fmt(a["offsupport"]),
                _fmt(a["tail_var"]),
                _fmt(a["onsupport"]),
                _fmt(a["judge"]),
                _fmt(a["win_rate"]),
                _fmt(a["kl"]),
                str(a.get("notes", "-")) or "-",  # 空欄禁止
            ]
        )

    parts.append("## 🟦 Table 1：Unified Comparison (Main Results)")
    parts.append(_md_table(t1_headers, t1_rows))
    parts.append("")

    # -------------------------
    # Table 2-A (PPO-family)
    # -------------------------
    t2a_headers = ["Method", "KL Stability", "Reward Var", "Convergence Speed"]
    t2a_rows: List[List[str]] = []
    for m in METHOD_SPECS:
        a = aggregated[m.key]
        t2a_rows.append(
            [
                m.name,
                _fmt(a["kl_stability"]),
                _fmt(a["reward_var"]),
                _fmt(a["convergence_speed"]),
            ]
        )
    parts.append("## 🟩 Table 2-A：PPO-family Diagnostics")
    parts.append(_md_table(t2a_headers, t2a_rows))
    parts.append("")

    # -------------------------
    # Table 2-B (Preference-based)
    # -------------------------
    t2b_headers = ["Method", "Sample Efficiency", "Reward Accuracy", "Label Source"]
    t2b_rows: List[List[str]] = []
    for m in METHOD_SPECS:
        a = aggregated[m.key]
        t2b_rows.append(
            [
                m.name,
                _fmt(a["sample_efficiency"]),
                _fmt(a["reward_accuracy"]),
                str(a["label_source"]),
            ]
        )
    parts.append("## 🟨 Table 2-B：Preference-based Diagnostics")
    parts.append(_md_table(t2b_headers, t2b_rows))
    parts.append("")

    # -------------------------
    # Table 2-C (Safety / Robustness)
    # -------------------------
    t2c_headers = ["Method", "Prompt Injection", "OOD Stability"]
    t2c_rows: List[List[str]] = []
    for m in METHOD_SPECS:
        a = aggregated[m.key]
        t2c_rows.append(
            [
                m.name,
                _fmt(a["prompt_injection"]),
                _fmt(a["ood_stability"]),
            ]
        )
    parts.append("## 🟥 Table 2-C：Safety / Robustness")
    parts.append(_md_table(t2c_headers, t2c_rows))

    parts.append("")
    return "\n".join(parts)


# Backward-compatible alias (older modules import this name)
def render_report_markdown(aggregated: Dict[str, Dict[str, Any]]) -> str:
    return render_report(aggregated)
