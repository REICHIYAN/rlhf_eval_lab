# rlhf_eval_lab/reporting/markdown.py
# 目的：
# - Table 1 / 2A / 2B / 2C を必ず出す（全セル埋め / N/A規約）
# - provenance（backend/model/tokenizer/config_hash/git_commit/seed）を report に刻む
# 注意：
# - fallback sanity tier を最優先（見た目より “壊れない”）
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

from rlhf_eval_lab.reporting.artifacts import ArtifactsV1


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    # Markdown table builder（空欄ゼロを強制：None/"" は "N/A"）
    def _cell(x: object) -> str:
        if x is None:
            return "N/A"
        s = str(x)
        if s.strip() == "":
            return "N/A"
        return s

    out: List[str] = []
    out.append("| " + " | ".join([_cell(h) for h in headers]) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join([_cell(c) for c in r]) + " |")
    return "\n".join(out)


def _fmt_float(x: object, nd: int = 4) -> str:
    try:
        if x is None:
            return "N/A"

        # string normalizer
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return "N/A"
            if s.upper() in {"N/A", "NA", "NONE", "NULL", "NAN"}:
                return "N/A"
            x = float(s)

        v = float(x)  # type: ignore[arg-type]

        # NaN / inf guard
        # (NaN is the only float where v != v is True)
        if v != v:
            return "N/A"
        if v == float("inf") or v == float("-inf"):
            return "N/A"

        return f"{v:.{nd}f}"
    except Exception:
        return "N/A"


def _as_str(x: object) -> str:
    if x is None:
        return "N/A"
    s = str(x)
    return s if s.strip() else "N/A"


def _short_hash(s: str, n: int = 12) -> str:
    if not s or s.strip() == "":
        return "N/A"
    s = s.strip()
    return s[:n]


def _collect_provenance(arts: Sequence[ArtifactsV1]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for a in arts:
        pd = asdict(a.provenance)
        rows.append(
            {
                "method_key": _as_str(a.method_key),
                "backend": _as_str(pd.get("backend")),
                "model_id": _as_str(pd.get("model_id")),
                "tokenizer": _as_str(pd.get("tokenizer")),
                "config_hash": _as_str(pd.get("config_hash")),
                "git_commit": _as_str(pd.get("git_commit")),
                "seed": _as_str(pd.get("seed")),
            }
        )
    return rows


def _provenance_summary(prows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    keys = ["backend", "model_id", "tokenizer", "config_hash", "git_commit", "seed"]

    def uniq(k: str) -> List[str]:
        vs: List[str] = []
        for r in prows:
            v = _as_str(r.get(k, "N/A"))
            if v not in vs:
                vs.append(v)
        return vs

    out: Dict[str, str] = {}
    for k in keys:
        u = uniq(k)
        if len(u) == 1:
            out[k] = u[0]
        else:
            u_non = [x for x in u if x != "N/A"]
            out[k] = "N/A" if len(u_non) == 0 else "MIXED"
    return out


def _render_provenance_section(arts: Sequence[ArtifactsV1]) -> str:
    prows = _collect_provenance(arts)
    summ = _provenance_summary(prows)

    summary_rows = [
        ["backend", _as_str(summ.get("backend"))],
        ["model_id", _as_str(summ.get("model_id"))],
        ["tokenizer", _as_str(summ.get("tokenizer"))],
        ["config_hash", _short_hash(_as_str(summ.get("config_hash")), 12)],
        ["git_commit", _short_hash(_as_str(summ.get("git_commit")), 12)],
        ["seed", _as_str(summ.get("seed"))],
    ]
    summary_md = _md_table(["Field", "Value"], summary_rows)

    # 安定表示（method_keyでソート）
    per_rows: List[List[str]] = []
    for r in sorted(prows, key=lambda d: d.get("method_key", "")):
        per_rows.append(
            [
                _as_str(r.get("method_key")),
                _as_str(r.get("backend")),
                _as_str(r.get("model_id")),
                _as_str(r.get("tokenizer")),
                _short_hash(_as_str(r.get("config_hash")), 12),
                _short_hash(_as_str(r.get("git_commit")), 12),
                _as_str(r.get("seed")),
            ]
        )
    per_md = _md_table(
        ["Method", "Backend", "Model", "Tokenizer", "Config", "Git", "Seed"],
        per_rows,
    )

    parts: List[str] = []
    parts.append("## 🧾 Provenance")
    parts.append("")
    parts.append("- 目的：report 単体で再現性（backend/model/tokenizer/config/git/seed）を監査できるようにする")
    parts.append("- `MIXED` が出たら、run/aggregate/report のどこかで条件が揺れています")
    parts.append("")
    parts.append("### Summary")
    parts.append(summary_md)
    parts.append("")
    parts.append("### Per-method")
    parts.append(per_md)
    return "\n".join(parts)


def _method_label(method_key: str, metrics: Dict[str, Any]) -> str:
    # method_name が無い/壊れてる時でも method_key で必ず埋める
    label = _as_str(metrics.get("method_name"))
    if label == "N/A":
        return _as_str(method_key)
    return label


def _stable_items(aggregated: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    # 出力の安定性（壊れない）を優先し、method_key でソートして固定
    return sorted(aggregated.items(), key=lambda kv: kv[0])


# ===== Public API (SSOT) =====
def render_report_markdown(
    aggregated: Dict[str, Dict[str, Any]],
    artifacts: Sequence[ArtifactsV1],
) -> str:
    """
    report.py から呼ばれる “公開API名” はこれ。
    aggregated: method_key -> metric_key -> value
    artifacts: ArtifactsV1 の生データ（provenance を report に刻むために必要）
    """
    parts: List[str] = []

    # Table 1
    parts.append("## 🟦 Table 1：Unified Comparison (Main Results)")
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
    for method_key, metrics in _stable_items(aggregated):
        t1_rows.append(
            [
                _as_str(metrics.get("category")),
                _method_label(method_key, metrics),
                _fmt_float(metrics.get("off_support"), 4),
                _fmt_float(metrics.get("tail_var"), 4),
                _fmt_float(metrics.get("on_support"), 4),
                _fmt_float(metrics.get("judge"), 4),
                _fmt_float(metrics.get("win_rate"), 4),
                _fmt_float(metrics.get("kl"), 4),
                _as_str(metrics.get("notes")),
            ]
        )
    parts.append(_md_table(t1_headers, t1_rows))
    parts.append("")

    # Table 2-A
    parts.append("## 🟩 Table 2-A：PPO-family Diagnostics")
    t2a_headers = ["Method", "KL Stability", "Reward Var", "Convergence Speed"]
    t2a_rows: List[List[str]] = []
    for method_key, metrics in _stable_items(aggregated):
        t2a_rows.append(
            [
                _method_label(method_key, metrics),
                _fmt_float(metrics.get("kl_stability"), 4),
                _fmt_float(metrics.get("reward_var"), 4),
                _fmt_float(metrics.get("convergence_speed"), 4),
            ]
        )
    parts.append(_md_table(t2a_headers, t2a_rows))
    parts.append("")

    # Table 2-B
    parts.append("## 🟨 Table 2-B：Preference-based Diagnostics")
    t2b_headers = ["Method", "Sample Efficiency", "Reward Accuracy", "Label Source"]
    t2b_rows: List[List[str]] = []
    for method_key, metrics in _stable_items(aggregated):
        t2b_rows.append(
            [
                _method_label(method_key, metrics),
                _fmt_float(metrics.get("sample_efficiency"), 4),
                _fmt_float(metrics.get("reward_accuracy"), 4),
                _as_str(metrics.get("label_source")),
            ]
        )
    parts.append(_md_table(t2b_headers, t2b_rows))
    parts.append("")

    # Table 2-C
    parts.append("## 🟥 Table 2-C：Safety / Robustness")
    t2c_headers = ["Method", "Prompt Injection", "OOD Stability"]
    t2c_rows: List[List[str]] = []
    for method_key, metrics in _stable_items(aggregated):
        t2c_rows.append(
            [
                _method_label(method_key, metrics),
                _fmt_float(metrics.get("prompt_injection"), 4),
                _fmt_float(metrics.get("ood_stability"), 4),
            ]
        )
    parts.append(_md_table(t2c_headers, t2c_rows))
    parts.append("")

    # Provenance
    parts.append(_render_provenance_section(artifacts))
    parts.append("")

    return "\n".join(parts)


# ===== Backward-compatible alias =====
# 古い呼び出し側が `render_report` を import しても壊れないように残す
def render_report(
    aggregated: Dict[str, Dict[str, Any]],
    artifacts: Sequence[ArtifactsV1],
) -> str:
    return render_report_markdown(aggregated, artifacts)
