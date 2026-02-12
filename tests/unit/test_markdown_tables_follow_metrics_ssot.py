from __future__ import annotations

import re
from typing import Any, Dict, List

from rlhf_eval_lab.registry import metrics as reg
from rlhf_eval_lab.reporting import markdown as md


def _specs_for_table(table_attr: str) -> List[Any]:
    specs = reg.METRIC_SPECS
    assert isinstance(specs, list)
    return [s for s in specs if hasattr(s, table_attr) and bool(getattr(s, table_attr))]


def _dummy_value_for_spec(spec: Any) -> Any:
    dt = str(getattr(spec, "dtype", "float")).lower()
    if dt in {"str", "string", "text"}:
        return "x"
    if "int" in dt:
        return 0
    return 0.0


def _extract_headers(section_md: str) -> List[str]:
    # Find the first markdown table header line: "| a | b | c |"
    lines = section_md.splitlines()
    header_line = None
    for ln in lines:
        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            header_line = ln.strip()
            break
    assert header_line is not None, "No markdown table header line found"

    # Split by pipes; drop empty edges.
    parts = [p.strip() for p in header_line.split("|")]
    parts = [p for p in parts if p != ""]
    return parts


def _make_dummy_aggregated() -> Dict[str, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for method_key in ["m1", "m2"]:
        d: Dict[str, Any] = {
            "category": "test",
            "method_name": method_key,
            "notes": "-",
        }
        # Fill every metric key so formatting never depends on missing keys.
        for s in reg.METRIC_SPECS:
            k = str(getattr(s, "key"))
            d[k] = _dummy_value_for_spec(s)
        aggregated[method_key] = d
    return aggregated


def test_table1_headers_follow_metrics_ssot() -> None:
    aggregated = _make_dummy_aggregated()
    section = md._render_table_1(aggregated)

    # Expect: Category, Method, <MetricSpec.name...>, Notes
    specs = _specs_for_table("in_table1")
    expected = ["Category", "Method"] + [str(getattr(s, "name")) for s in specs] + ["Notes"]

    got = _extract_headers(section)
    assert got == expected


def test_table2a_headers_follow_metrics_ssot() -> None:
    aggregated = _make_dummy_aggregated()
    section = md._render_table_2a(aggregated)

    specs = _specs_for_table("in_table2a")
    expected = ["Method"] + [str(getattr(s, "name")) for s in specs]

    got = _extract_headers(section)
    assert got == expected


def test_table2b_headers_follow_metrics_ssot() -> None:
    aggregated = _make_dummy_aggregated()
    section = md._render_table_2b(aggregated)

    specs = _specs_for_table("in_table2b")
    expected = ["Method"] + [str(getattr(s, "name")) for s in specs]

    got = _extract_headers(section)
    assert got == expected


def test_table2c_headers_follow_metrics_ssot() -> None:
    aggregated = _make_dummy_aggregated()
    section = md._render_table_2c(aggregated)

    specs = _specs_for_table("in_table2c")
    expected = ["Method"] + [str(getattr(s, "name")) for s in specs]

    got = _extract_headers(section)
    assert got == expected
