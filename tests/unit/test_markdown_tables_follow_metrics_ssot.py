from __future__ import annotations

import re
from typing import Any, Dict, List

from rlhf_eval_lab.registry import metrics as reg
from rlhf_eval_lab.reporting import markdown as md


T1_PREFIX = "## 🟦 Table 1："
T2A_PREFIX = "## 🟩 Table 2-A："
T2B_PREFIX = "## 🟨 Table 2-B："
T2C_PREFIX = "## 🟥 Table 2-C："


def _dummy_value_for_spec(spec: Any) -> Any:
    dt = str(getattr(spec, "dtype", "float")).lower()
    if dt in {"str", "string", "text"}:
        return "x"
    if "int" in dt:
        return 0
    return 0.0


def _make_dummy_aggregated() -> Dict[str, Dict[str, Any]]:
    """
    Create a minimal aggregated dict that is safe to render.
    Values can be arbitrary; these tests only validate header alignment.
    """
    aggregated: Dict[str, Dict[str, Any]] = {}
    for method_key in ["m1", "m2"]:
        d: Dict[str, Any] = {"category": "test", "method_name": method_key, "notes": "-"}
        for s in reg.METRIC_SPECS:
            d[str(getattr(s, "key"))] = _dummy_value_for_spec(s)
        aggregated[method_key] = d
    return aggregated


def _extract_section(report_md: str, heading_prefix: str) -> str:
    # Match a section starting with the given "## ..." prefix until the next "## " heading.
    m = re.search(rf"{re.escape(heading_prefix)}.*?(?=\n## |\Z)", report_md, flags=re.S)
    assert m, f"Section not found: {heading_prefix}"
    return m.group(0)


def _extract_headers(section_md: str) -> List[str]:
    """
    Extract the first markdown table header row within a section.
    Expected format: "| a | b | c |"
    """
    header_line = None
    for ln in section_md.splitlines():
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            header_line = s
            break
    assert header_line is not None, "No markdown table header line found"

    parts = [p.strip() for p in header_line.split("|")]
    parts = [p for p in parts if p != ""]
    return parts


def _specs_for_table(flag_attr: str) -> List[Any]:
    return [s for s in reg.METRIC_SPECS if bool(getattr(s, flag_attr, False))]


def test_table1_headers_follow_metrics_ssot() -> None:
    aggregated = _make_dummy_aggregated()
    report = md.render_report_markdown(aggregated=aggregated, artifacts=[])

    section = _extract_section(report, T1_PREFIX)
    got = _extract_headers(section)

    specs = _specs_for_table("in_table1")
    expected = ["Category", "Method"] + [str(getattr(s, "name")) for s in specs] + ["Notes"]
    assert got == expected


def test_table2a_headers_follow_metrics_ssot() -> None:
    aggregated = _make_dummy_aggregated()
    report = md.render_report_markdown(aggregated=aggregated, artifacts=[])

    section = _extract_section(report, T2A_PREFIX)
    got = _extract_headers(section)

    specs = _specs_for_table("in_table2a")
    expected = ["Method"] + [str(getattr(s, "name")) for s in specs]
    assert got == expected


def test_table2b_headers_follow_metrics_ssot() -> None:
    aggregated = _make_dummy_aggregated()
    report = md.render_report_markdown(aggregated=aggregated, artifacts=[])

    section = _extract_section(report, T2B_PREFIX)
    got = _extract_headers(section)

    specs = _specs_for_table("in_table2b")
    expected = ["Method"] + [str(getattr(s, "name")) for s in specs]
    assert got == expected


def test_table2c_headers_follow_metrics_ssot() -> None:
    aggregated = _make_dummy_aggregated()
    report = md.render_report_markdown(aggregated=aggregated, artifacts=[])

    section = _extract_section(report, T2C_PREFIX)
    got = _extract_headers(section)

    specs = _specs_for_table("in_table2c")
    expected = ["Method"] + [str(getattr(s, "name")) for s in specs]
    assert got == expected
