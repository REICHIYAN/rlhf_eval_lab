# rlhf_eval_lab/reporting/paths.py
from __future__ import annotations

import os


def default_artifacts_dir() -> str:
    return os.path.abspath("artifacts")


def default_reports_dir() -> str:
    return os.path.abspath("reports")


def report_md_path(out_dir: str) -> str:
    return os.path.join(os.path.abspath(out_dir), "report.md")


def resolve_output_paths(
    artifacts_dir: str | None = None,
    reports_dir: str | None = None,
) -> dict:
    """Resolve default output paths used by CLI/reporting.

    This is a small helper to keep paths logic centralized (SSOT).
    """
    a_dir = os.path.abspath(artifacts_dir) if artifacts_dir else default_artifacts_dir()
    r_dir = os.path.abspath(reports_dir) if reports_dir else default_reports_dir()
    return {
        "artifacts_dir": a_dir,
        "reports_dir": r_dir,
        "report_md": report_md_path(r_dir),
    }
