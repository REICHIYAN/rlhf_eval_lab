# rlhf_eval_lab/cli/commands/report.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rlhf_eval_lab.reporting.aggregate import aggregate
from rlhf_eval_lab.reporting.artifacts import read_artifacts_tree
from rlhf_eval_lab.reporting.markdown import render_report
from rlhf_eval_lab.reporting.paths import report_md_path

_AEGIS_APPENDIX_MARKER = "## Appendix: Aegis audit metrics"
_APPENDIX_METHOD_FILTER = {"aegis", "dpo"}  # <--- Aegis + DPO only


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        x = float(v)
        if abs(x) >= 1000:
            return f"{x:.2f}"
        return f"{x:.6g}"
    s = str(v)
    return s if s.strip() else "-"


def _collect_seed_files(artifacts_root: Path) -> List[Tuple[str, str, Path]]:
    """
    Return list of (method_key, seed_name, path) for seed_*.json under artifacts_root/*/,
    filtered to appendix-relevant methods only.
    """
    out: List[Tuple[str, str, Path]] = []
    if not artifacts_root.exists():
        return out

    for method_dir in sorted([p for p in artifacts_root.iterdir() if p.is_dir()]):
        method_key = method_dir.name
        if method_key not in _APPENDIX_METHOD_FILTER:
            continue

        for p in sorted(method_dir.glob("seed_*.json")):
            out.append((method_key, p.name, p))

    return out


def _build_aegis_audit_table(artifacts_root: Path) -> str:
    """
    Build markdown appendix table reading artifacts extra fields.
    Only includes Aegis + DPO rows (seed-aware).
    """
    cols: List[Tuple[str, str]] = [
        ("pref_loss", "pref_loss"),
        ("steps", "steps"),
        ("pairs_used", "pairs_used"),
        ("aegis_alpha", "aegis_alpha"),
        ("aegis_value_margin", "aegis_value_margin"),
        ("aegis_uncertainty", "aegis_uncertainty"),
        ("aegis_weight", "aegis_weight"),
        ("aegis_beta_eff", "aegis_beta_eff"),
    ]

    rows: List[List[str]] = []
    for method_key, seed_name, p in _collect_seed_files(artifacts_root):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        extra = obj.get("extra", {}) if isinstance(obj, dict) else {}
        if not isinstance(extra, dict):
            extra = {}

        row = [method_key, seed_name] + [_fmt_cell(extra.get(k)) for k, _ in cols]
        rows.append(row)

    if not rows:
        return ""

    header = ["method", "seed"] + [label for _, label in cols]

    md: List[str] = []
    md.append("\n\n---\n\n")
    md.append(f"{_AEGIS_APPENDIX_MARKER}\n")
    md.append("> This appendix is filtered to method ∈ {aegis, dpo}. Aegis-only keys show '-' for DPO by design.\n")
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")

    for r in rows:
        md.append("| " + " | ".join(r) + " |")

    md.append("")
    return "\n".join(md)


def _append_aegis_appendix(out_path: str, artifacts_root: str) -> None:
    """
    Idempotently append (or replace) Aegis audit appendix at end of report.md.
    Fail-open: never raises.
    """
    try:
        report_p = Path(out_path)
        art_p = Path(artifacts_root)

        appendix = _build_aegis_audit_table(art_p)
        if not appendix:
            return

        txt = report_p.read_text(encoding="utf-8") if report_p.exists() else ""

        # Remove previous appendix if exists (marker-based)
        if _AEGIS_APPENDIX_MARKER in txt:
            txt = txt.split(_AEGIS_APPENDIX_MARKER)[0].rstrip() + "\n"

        report_p.write_text(txt + appendix, encoding="utf-8")
    except Exception as e:
        # Do not break report generation
        print(f"[rlhf-lab][warn] failed to append aegis appendix: {e}")


def report_cmd(args) -> int:
    artifacts_root = os.path.abspath(args.in_dir)
    reports_root = os.path.abspath(args.out)
    os.makedirs(reports_root, exist_ok=True)

    artifacts = read_artifacts_tree(artifacts_root)
    aggregated = aggregate(artifacts_root)

    md = render_report(aggregated, artifacts)

    out_path = report_md_path(reports_root)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    # Append Aegis appendix (fail-open)
    _append_aegis_appendix(out_path, artifacts_root)

    print(f"[rlhf-lab] Report written to: {out_path}")
    return 0
