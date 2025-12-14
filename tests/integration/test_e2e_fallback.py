# tests/integration/test_e2e_fallback.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    # tests/integration/test_e2e_fallback.py -> repo root
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _assert_markdown_tables_no_empty_cells(md: str) -> None:
    # Markdown table row pattern: | a | b | c |
    # We ignore header separator lines like | --- | --- |
    lines = [ln.rstrip("\n") for ln in md.splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.strip().startswith("|") and ln.strip().endswith("|")]

    assert table_lines, "No markdown tables found in report.md"

    def is_separator_row(s: str) -> bool:
        # e.g. | --- | --- | --- |
        cells = [c.strip() for c in s.strip("|").split("|")]
        return all(re.fullmatch(r"-{3,}", c) is not None for c in cells)

    # For each table, enforce:
    # - no empty cell (after stripping)
    # - consistent column count within contiguous table block
    i = 0
    while i < len(table_lines):
        # Start of a table block
        block = []
        while i < len(table_lines):
            block.append(table_lines[i])
            # peek next original line adjacency is lost; but we can treat all table_lines as one stream.
            # We break a block when a non-table line existed, but here we filtered. So we instead split blocks
            # by detecting a header row followed by separator row pattern.
            # Simpler: process per-line with "current expected columns" reset on separator row.
            i += 1
            # continue; we'll handle with state below
            # (no explicit block splitting needed)
        break  # we process all at once below

    expected_cols: int | None = None
    for ln in table_lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if is_separator_row(ln):
            # keep expected_cols
            continue
        if expected_cols is None:
            expected_cols = len(cells)
        else:
            # Allow new table header to reset expected cols:
            # Detect a header row by checking if the next separator row exists in the original md is hard,
            # so instead we reset whenever column count changes AND line looks like a header row.
            if len(cells) != expected_cols:
                # Heuristic reset: if this line contains "Category" or "Method" or "Table"
                headerish = any(tok in ln for tok in ["Category", "Method", "Off-support", "Tail Var", "Prompt Injection"])
                if headerish:
                    expected_cols = len(cells)
                else:
                    raise AssertionError(f"Inconsistent column count: got {len(cells)} expected {expected_cols}: {ln}")

        for c in cells:
            assert c != "", f"Empty cell detected in table row: {ln}"


def test_e2e_fallback_run_validate_report() -> None:
    root = _repo_root()

    # Ensure clean
    for d in ["artifacts", "reports"]:
        p = root / d
        if p.exists():
            shutil.rmtree(p)

    rlhf_lab = shutil.which("rlhf-lab")
    assert rlhf_lab is not None, "rlhf-lab command not found. Did you run `pip install -e .`?"

    # E2E
    _run([rlhf_lab, "run", "--backend", "fallback", "--seed", "0"], cwd=root)
    _run([rlhf_lab, "validate"], cwd=root)
    _run([rlhf_lab, "report"], cwd=root)

    report_path = root / "reports" / "report.md"
    assert report_path.exists(), "reports/report.md not generated"

    md = report_path.read_text(encoding="utf-8")
    _assert_markdown_tables_no_empty_cells(md)
