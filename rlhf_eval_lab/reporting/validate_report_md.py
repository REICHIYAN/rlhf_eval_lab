# rlhf_eval_lab/reporting/validate_report_md.py
"""Strict validation for rendered report.md.

DoD invariants (must hold for any report.md produced by this OSS):

- Markdown tables must have no empty cells (after stripping).
- Forbidden tokens must not appear anywhere (nan/inf/None/null/etc.).
- Provenance must not contain "MIXED" (report must be self-auditable).

This module is intentionally dependency-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


_FORBIDDEN_TOKEN_PATTERNS: list[re.Pattern[str]] = [
    # Common poison values that must never leak into report.md
    re.compile(r"\bnan\b", re.IGNORECASE),
    re.compile(r"\b-inf\b", re.IGNORECASE),
    re.compile(r"\binf\b", re.IGNORECASE),
    re.compile(r"\binfinity\b", re.IGNORECASE),
    re.compile(r"\bnone\b", re.IGNORECASE),
    re.compile(r"\bnull\b", re.IGNORECASE),
]


_MIXED_PATTERN = re.compile(r"\bMIXED\b")


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and "|" in s[1:-1]


def _split_table_cells(line: str) -> list[str]:
    # Strip leading/trailing '|' and split.
    return [c.strip() for c in line.strip().strip("|").split("|")]


_SEP_CELL_RE = re.compile(r"^:?-{3,}:?$")


def _is_separator_row(cells: list[str]) -> bool:
    # Markdown header separator row like:
    # | --- | --- | or alignment variants | :--- | ---: |
    return all(_SEP_CELL_RE.fullmatch(c) is not None for c in cells)


@dataclass(frozen=True)
class ReportValidationError:
    code: str
    message: str


def validate_report_markdown(md: str) -> list[ReportValidationError]:
    """Validate report.md content.

    Returns a list of errors. Empty list means OK.
    """
    errors: list[ReportValidationError] = []

    # 1) Forbidden tokens
    for pat in _FORBIDDEN_TOKEN_PATTERNS:
        if pat.search(md) is not None:
            errors.append(
                ReportValidationError(
                    code="forbidden_token",
                    message=f"Found forbidden token pattern: {pat.pattern}",
                )
            )

    # 2) MIXED provenance forbidden
    if _MIXED_PATTERN.search(md) is not None:
        errors.append(
            ReportValidationError(
                code="mixed_provenance",
                message="Found 'MIXED' in report.md provenance (must not mix conditions).",
            )
        )

    # 3) Markdown tables: no empty cells
    lines = md.splitlines()

    # Identify contiguous table blocks to enforce consistent column counts per table.
    i = 0
    while i < len(lines):
        if not _is_table_line(lines[i]):
            i += 1
            continue

        # Start table block
        block: list[str] = []
        while i < len(lines) and _is_table_line(lines[i]):
            block.append(lines[i].rstrip("\n"))
            i += 1

        # Validate this block
        expected_cols: int | None = None
        for ln in block:
            cells = _split_table_cells(ln)
            if _is_separator_row(cells):
                continue

            if expected_cols is None:
                expected_cols = len(cells)
            elif len(cells) != expected_cols:
                errors.append(
                    ReportValidationError(
                        code="table_column_mismatch",
                        message=(
                            f"Inconsistent column count in markdown table: got {len(cells)} expected {expected_cols}. "
                            f"Row: {ln}"
                        ),
                    )
                )
                # Keep going to report more issues.

            for c in cells:
                if c == "":
                    errors.append(
                        ReportValidationError(
                            code="empty_cell",
                            message=f"Empty cell detected in markdown table row: {ln}",
                        )
                    )

    return errors


def assert_report_markdown_ok(md: str) -> None:
    """Raise ValueError if report markdown violates DoD invariants."""
    errs = validate_report_markdown(md)
    if not errs:
        return
    msg = "\n".join([f"[{e.code}] {e.message}" for e in errs])
    raise ValueError(f"report.md validation failed:\n{msg}")


# Backward/CLI-facing aliases (preferred public API)
def validate_report_md(md: str) -> None:
    """Validate report.md content; raise ValueError on any violation."""
    assert_report_markdown_ok(md)


def validate_report_md_or_raise(md: str) -> None:
    """Alias of validate_report_md for call sites that read clearer."""
    validate_report_md(md)
