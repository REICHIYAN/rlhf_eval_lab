# rlhf_eval_lab/reporting/markdown.py
# Purpose:
# - Always render Table 1 / 2A / 2B / 2C (no empty cells; enforce N/A policy)
# - Stamp provenance (backend/model/tokenizer/config_hash/git_commit/seed) into report.md
# Notes:
# - Prioritize the fallback sanity tier (robustness over cosmetics)

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

from rlhf_eval_lab.reporting.artifacts import ArtifactsV1


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    # Markdown table builder (enforce no empty cells: None/"" -> "N/A")
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

        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return "N/A"
            if s.upper() in {"N/A", "NA", "NONE", "NULL", "NAN"}:
                return "N/A"
            x = float(s)

        v = float(x)  # type: ignore[arg-type]

        # Guard against NaN / inf to avoid leaking "nan"/"inf" into markdown.
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


def _provenance_summary_strict(prows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    """
    Strict provenance summarizer.

    Policy:
      - Each field must be either:
          (a) exactly 1 unique non-N/A value across all methods, OR
          (b) all N/A (unknown everywhere).
      - Any mixing (including some N/A + some non-N/A) is a hard error.

    Rationale:
      - The report must be self-auditable; partial provenance is not allowed.
    """
    keys = ["backend", "model_id", "tokenizer", "config_hash", "git_commit", "seed"]
    out: Dict[str, str] = {}

    for k in keys:
        vals = [_as_str(r.get(k, "N/A")) for r in prows]
        uniq_all: List[str] = []
        for v in vals:
            if v not in uniq_all:
                uniq_all.append(v)

        uniq_non = [v for v in uniq_all if v != "N/A"]

        if len(uniq_non) == 0:
            # All N/A
            out[k] = "N/A"
            continue

        if len(uniq_non) == 1 and len(uniq_all) == 1:
            # Single value, no N/A
            out[k] = uniq_non[0]
            continue

        # Mixed conditions OR partial provenance -> hard fail
        raise ValueError(
            "Provenance is inconsistent across methods. "
            f"field={k} unique_values={uniq_all}. "
            "This indicates run/aggregate/report conditions are not fixed."
        )

    return out


def _render_provenance_section(arts: Sequence[ArtifactsV1]) -> str:
    prows = _collect_provenance(arts)
    summ = _provenance_summary_strict(prows)

    summary_rows = [
        ["backend", _as_str(summ.get("backend"))],
        ["model_id", _as_str(summ.get("model_id"))],
        ["tokenizer", _as_str(summ.get("tokenizer"))],
        ["config_hash", _short_hash(_as_str(summ.get("config_hash")), 12)],
        ["git_commit", _short_hash(_as_str(summ.get("git_commit")), 12)],
        ["seed", _as_str(summ.get("seed"))],
    ]
    summary_md = _md_table(["Field", "Value"], summary_rows)

    # Stable rendering order (sort by method_key)
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
    parts.append("- Purpose: make the report self-auditable (backend/model/tokenizer/config/git/seed).")
    parts.append("- If this section is not stable, report generation must raise (no mixed conditions allowed).")
    parts.append("")
    parts.append("### Summary")
    parts.append(summary_md)
    parts.append("")
    parts.append("### Per-method")
    parts.append(per_md)
    return "\n".join(parts)


def _render_interpretation_section() -> str:
    """
    Phase C: Embed fixed semantics (HF + dataset SSOT + metric semantics) into report.md.

    Notes:
    - This is not a "result"; it is a fixed spec for interpretation.
    - Keep wording minimal and stable (report must remain self-auditable).
    """
    parts: List[str] = []
    parts.append("## 📌 Interpretation (Phase C / HF)")
    parts.append("")
    parts.append("This report is **self-auditable**: every cell is either a numeric value or column-policy `N/A` (never missing).")
    parts.append("`N/A` means **not applicable by design**, not missing data.")
    parts.append("")
    parts.append("### Execution modes (two layers)")
    parts.append("")
    parts.append("- **fallback (DoD / OSS reliability):** torch-only deterministic backend (no `transformers`).")
    parts.append("- **hf (Level-C research, optional):** Hugging Face backend (`transformers` required).")
    parts.append("  - **HF Step1 (generation-only):** `generate → evaluate → artifacts`; training is explicitly marked as skipped.")
    parts.append("  - **HF Step2 (minimal SFT):** when `train.hf_sft_steps > 0`, SFT runs a minimal, auditable training loop.")
    parts.append("  - PPO-family training on HF is enabled only when explicitly implemented.")
    parts.append("")
    parts.append("Artifacts enforce auditability via `extra` fields:")
    parts.append("")
    parts.append("- `extra.skipped`: `true|false`")
    parts.append("- `extra.skip_reason`: e.g. `hf_step1_generation_only` or `\"\"`")
    parts.append("- `extra.steps`: number of executed update steps (0 if no training)")
    parts.append("")
    parts.append("### Dataset SSOT & reproducibility")
    parts.append("")
    parts.append("Each run fixes dataset identity as:")
    parts.append("")
    parts.append("- `dataset_key`: human-readable dataset identifier (e.g. `hh_rlhf:train:local`)")
    parts.append("- `dataset_hash`: stable hash of the effective prompt set used for this run")
    parts.append("")
    parts.append("Reproducibility relies on fixed `seed` and preset-controlled split/subsample policy.")
    parts.append("If `dataset_key` or `dataset_hash` differ across methods, comparisons are not meaningful.")
    parts.append("")
    parts.append("### Metric semantics (direction & applicability)")
    parts.append("")
    parts.append("- `↓` lower is better; `↑` higher is better.")
    parts.append("- Applicability is enforced by registry policy (column-level rules); non-applicable metrics become `N/A`.")
    parts.append("")
    parts.append("Core metrics (Table 1):")
    parts.append("")
    parts.append("- **Off-support ↓**: policy drift outside the support region (proxy definitions must match the implementation in `eval/`).")
    parts.append("- **Tail Var ↓**: variance of the reward tail (e.g. top 1%); lower implies fewer extreme spikes.")
    parts.append("- **On-support ↑**: average reward within supported region.")
    parts.append("- **Win-rate ↑ / Judge ↑**: comparison/judge signals when available (otherwise `N/A`).")
    parts.append("- **KL ↓**: divergence from the reference policy (policy drift).")
    parts.append("")
    parts.append("### Table 2 blocks")
    parts.append("")
    parts.append("- **Table 2-A**: diagnostics meaningful only for PPO-family methods (others are `N/A`).")
    parts.append("- **Table 2-B**: diagnostics meaningful only for Preference/Active methods (label source is `pref` / `ai`).")
    parts.append("- **Table 2-C**: safety/robustness diagnostics (may be `N/A` depending on dataset/method).")
    parts.append("")
    return "\n".join(parts)


def _method_label(method_key: str, metrics: Dict[str, Any]) -> str:
    label = _as_str(metrics.get("method_name"))
    if label == "N/A":
        return _as_str(method_key)
    return label


def _stable_items(aggregated: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    return sorted(aggregated.items(), key=lambda kv: kv[0])


# ===== Public API (SSOT) =====
def render_report_markdown(
    aggregated: Dict[str, Dict[str, Any]],
    artifacts: Sequence[ArtifactsV1],
) -> str:
    """
    Public API called by report.py.

    aggregated: method_key -> metric_key -> value
    artifacts: raw ArtifactsV1 (required to stamp provenance into the report)
    """
    parts: List[str] = []

    # Phase B-2: Interpretation (fixed semantics)
    parts.append(_render_interpretation_section())

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
def render_report(
    aggregated: Dict[str, Dict[str, Any]],
    artifacts: Sequence[ArtifactsV1],
) -> str:
    return render_report_markdown(aggregated, artifacts)
