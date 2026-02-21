# rlhf_eval_lab/reporting/markdown.py

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

from rlhf_eval_lab.reporting.artifacts import ArtifactsV1


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Markdown table builder (enforces no empty cells: None/"" -> "N/A")."""

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
    """Format float with nd decimals; robustly returns N/A for None/NaN/inf/non-numeric."""
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
      - For fields other than `seed`:
          Each field must be either:
            (a) exactly 1 unique non-N/A value across all methods, OR
            (b) all N/A (unknown everywhere).
          Any mixing (including some N/A + some non-N/A) is a hard error.
      - For `seed`:
          Multiple non-N/A values are allowed (seed aggregation).
          But mixing N/A and non-N/A is still a hard error.

    Rationale:
      - The report must be self-auditable; environment/provenance drift is not allowed.
      - Seed is the only supported multi-valued field to enable multi-seed aggregation reports.
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
            out[k] = "N/A"
            continue

        # Reject partial provenance: some N/A + some non-N/A
        if "N/A" in uniq_all and len(uniq_non) > 0:
            raise ValueError(
                "Provenance is partially missing across methods. "
                f"field={k} unique_values={uniq_all}. "
                "This indicates run/aggregate/report conditions are not fixed."
            )

        if k == "seed":
            # Allow multi-seed aggregation: show sorted unique seeds as a comma list.
            try:
                uniq_sorted = sorted(uniq_non, key=lambda s: int(str(s)))
            except Exception:
                uniq_sorted = sorted(uniq_non)
            out[k] = ",".join(uniq_sorted)
            continue

        # Non-seed fields must be single-valued
        if len(uniq_non) == 1 and len(uniq_all) == 1:
            out[k] = uniq_non[0]
            continue

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
    parts.append(
        "This report is **self-auditable**: every cell is either a numeric value or column-policy `N/A` (never missing)."
    )
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
    parts.append("- Latency (ms): wall-clock runtime per method (excludes dataset loading and one-time setup).")
    parts.append("")

    # --- C1.7: Clarify HF KL proxy & PPO ratio diagnostics (pre/post) ---
    parts.append("### KL & PPO diagnostics (HF backend)")
    parts.append("")
    parts.append(
        "For `backend=hf`, KL-related values shown in the tables are **audit-oriented proxies** derived from sampled "
        "trajectory log-probabilities (token-mean), not the full-distribution KL."
    )
    parts.append("")
    parts.append("- `kl`: preferred **non-negative proxy** (stable for reporting)")
    parts.append("  - `kl = E[ | logp_post - logp_ref | ]` (token-mean)")
    parts.append("- `kl_ref_sq`: squared-difference proxy for drift magnitude")
    parts.append("  - `kl_ref_sq = E[ (logp_post - logp_ref)^2 ]` (token-mean)")
    parts.append("- `kl_ref` / `kl_ref_pre`: signed mean differences kept for debugging (can be negative)")
    parts.append("  - `kl_ref = E[ logp_post - logp_ref ]`, `kl_ref_pre = E[ logp_pre - logp_ref ]`")
    parts.append("")
    parts.append("PPO ratio diagnostics are computed on per-token mean logprobs:")
    parts.append("")
    parts.append("- `ratio_mean_pre`: pre-update `E[ exp(logp_pre - logp_old) ]` (typically ≈ 1)")
    parts.append("- `ratio_mean`: post-update `E[ exp(logp_post - logp_old) ]`")
    parts.append("- `clipfrac`: fraction of samples where post-update ratio is outside `[1-clip, 1+clip]`")
    parts.append("")

    parts.append("### Table 2 blocks")
    parts.append("")
    parts.append(
        "- **Table 2-A**: PPO-family audit diagnostics. Non-PPO methods are `N/A` by policy "
        "(and PPO-like may be `N/A` when training is skipped)."
    )
    parts.append("- **Table 2-B**: diagnostics meaningful only for Preference/Active methods.")
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


def _table_flag_attr(table_id: str) -> str:
    """Map a table id to MetricSpec boolean membership attribute name."""
    t = table_id.strip().lower()
    if t in {"table1", "t1", "1", "table_1", "table-1"}:
        return "in_table1"
    if t in {"table2a", "t2a", "2a", "table_2a", "table-2a"}:
        return "in_table2a"
    if t in {"table2b", "t2b", "2b", "table_2b", "table-2b"}:
        return "in_table2b"
    if t in {"table2c", "t2c", "2c", "table_2c", "table-2c"}:
        return "in_table2c"
    raise ValueError(f"Unknown table id: {table_id}")


def _metric_specs_for_table(table_id: str) -> List[Any]:
    """Return MetricSpec list for the given table id, preserving registry order."""
    from rlhf_eval_lab.registry import metrics as reg  # local import to avoid import-time cycles

    if not hasattr(reg, "METRIC_SPECS"):
        raise RuntimeError("registry.metrics has no METRIC_SPECS")

    specs = getattr(reg, "METRIC_SPECS")
    if not isinstance(specs, list):
        # In this project, METRIC_SPECS is expected to be a list of MetricSpec.
        raise RuntimeError(f"METRIC_SPECS must be a list, got: {type(specs)}")

    attr = _table_flag_attr(table_id)
    out: List[Any] = []
    for s in specs:
        if hasattr(s, attr) and bool(getattr(s, attr)):
            out.append(s)
    return out


def _spec_key(spec: Any) -> str:
    if hasattr(spec, "key"):
        return str(getattr(spec, "key"))
    raise RuntimeError(f"MetricSpec has no .key: type={type(spec)}")


def _spec_name(spec: Any) -> str:
    # MetricSpec uses `.name` as a human label.
    if hasattr(spec, "name"):
        return str(getattr(spec, "name"))
    return _spec_key(spec)


def _spec_dtype(spec: Any) -> str:
    if hasattr(spec, "dtype"):
        return str(getattr(spec, "dtype"))
    return "float"


def _spec_decimals(spec: Any) -> int:
    # Prefer explicit decimals-like fields if they exist on MetricSpec.
    for attr in ("decimals", "precision", "digits", "ndigits"):
        if hasattr(spec, attr):
            v = getattr(spec, attr)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass

    # Otherwise infer from dtype (must match README generator's default policy).
    dt = _spec_dtype(spec).lower()
    if "int" in dt:
        return 0
    if "float" in dt:
        return 4
    return 4


def _fmt_by_spec(spec: Any, value: object) -> str:
    """Format a metric value using MetricSpec dtype + decimals policy."""
    dt = _spec_dtype(spec).lower()
    if dt in {"str", "string", "text"}:
        return _as_str(value)
    if "int" in dt:
        # Use float formatter with 0 decimals to avoid empty cells and keep a single policy.
        return _fmt_float(value, 0)
    # Default: float-like.
    return _fmt_float(value, _spec_decimals(spec))


def _render_table_1(aggregated: Dict[str, Dict[str, Any]]) -> str:
    specs = _metric_specs_for_table("Table1")
    headers = ["Category", "Method"] + [_spec_name(s) for s in specs] + ["Notes"]

    rows: List[List[str]] = []
    for method_key, metrics in _stable_items(aggregated):
        metric_cells = [_fmt_by_spec(s, metrics.get(_spec_key(s))) for s in specs]
        rows.append(
            [
                _as_str(metrics.get("category")),
                _method_label(method_key, metrics),
                *metric_cells,
                _as_str(metrics.get("notes")),
            ]
        )
    return "\n".join(
        [
            "## 🟦 Table 1：Unified Comparison (Main Results)",
            _md_table(headers, rows),
            "",
        ]
    )


def _render_table_2a(aggregated: Dict[str, Dict[str, Any]]) -> str:
    specs = _metric_specs_for_table("Table2A")
    headers = ["Method"] + [_spec_name(s) for s in specs]

    rows: List[List[str]] = []
    for method_key, metrics in _stable_items(aggregated):
        metric_cells = [_fmt_by_spec(s, metrics.get(_spec_key(s))) for s in specs]
        rows.append([_method_label(method_key, metrics), *metric_cells])

    return "\n".join(
        [
            "## 🟩 Table 2-A：PPO-family Diagnostics (Audit)",
            _md_table(headers, rows),
            "",
        ]
    )


def _render_table_2b(aggregated: Dict[str, Dict[str, Any]]) -> str:
    specs = _metric_specs_for_table("Table2B")
    headers = ["Method"] + [_spec_name(s) for s in specs]

    rows: List[List[str]] = []
    for method_key, metrics in _stable_items(aggregated):
        metric_cells = [_fmt_by_spec(s, metrics.get(_spec_key(s))) for s in specs]
        rows.append([_method_label(method_key, metrics), *metric_cells])

    return "\n".join(
        [
            "## 🟨 Table 2-B：Preference-based Diagnostics",
            _md_table(headers, rows),
            "",
        ]
    )


def _render_table_2c(aggregated: Dict[str, Dict[str, Any]]) -> str:
    specs = _metric_specs_for_table("Table2C")
    headers = ["Method"] + [_spec_name(s) for s in specs]

    rows: List[List[str]] = []
    for method_key, metrics in _stable_items(aggregated):
        metric_cells = [_fmt_by_spec(s, metrics.get(_spec_key(s))) for s in specs]
        rows.append([_method_label(method_key, metrics), *metric_cells])

    return "\n".join(
        [
            "## 🟥 Table 2-C：Safety / Robustness",
            _md_table(headers, rows),
            "",
        ]
    )


# ===== Public API =====
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

    # Fixed interpretation section (stable semantics).
    parts.append(_render_interpretation_section())

    # Tables are SSOT-driven via registry.metrics.METRIC_SPECS.
    parts.append(_render_table_1(aggregated))
    parts.append(_render_table_2a(aggregated))
    parts.append(_render_table_2b(aggregated))
    parts.append(_render_table_2c(aggregated))

    # Provenance (strict / self-auditable).
    parts.append(_render_provenance_section(artifacts))
    parts.append("")

    return "\n".join(parts)


# ===== Backward-compatible alias =====
def render_report(
    aggregated: Dict[str, Dict[str, Any]],
    artifacts: Sequence[ArtifactsV1],
) -> str:
    return render_report_markdown(aggregated, artifacts)