# tests/unit/test_report_md_validate.py
from __future__ import annotations

import pytest

from rlhf_eval_lab.reporting.validate_report_md import validate_report_md_or_raise


def test_validate_report_md_ok_minimal_table() -> None:
    md = """
# Report

| A | B |
| --- | --- |
| 1 | 2 |
"""
    validate_report_md_or_raise(md)


@pytest.mark.parametrize(
    "bad_md, token",
    [
        ("| A | B |\n| --- | --- |\n| 1 | |\n", "empty cell"),
        ("| A | B |\n| --- | --- |\n| NaN | 2 |\n", "nan"),
        ("| A | B |\n| --- | --- |\n| INF | 2 |\n", "inf"),
        ("| A | B |\n| --- | --- |\n| None | 2 |\n", "none"),
        ("| A | B |\n| --- | --- |\n| null | 2 |\n", "null"),
        ("Provenance: MIXED\n| A | B |\n| --- | --- |\n| 1 | 2 |\n", "MIXED"),
    ],
)
def test_validate_report_md_rejects_bad_content(bad_md: str, token: str) -> None:
    with pytest.raises(ValueError):
        validate_report_md_or_raise(bad_md)
