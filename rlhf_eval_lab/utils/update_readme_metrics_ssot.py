from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple

README_PATH = Path("README.md")

START_MARKER = "<!-- METRICS_SSOT:START -->"
END_MARKER = "<!-- METRICS_SSOT:END -->"

# Prefer stable, human-friendly order if these table ids exist.
TABLE_ORDER = ["table1", "table2a", "table2b", "table2c", "Table1", "Table2A", "Table2B", "Table2C"]


def _get_field(spec: Any, names: List[str], default: Any = None) -> Any:
    """Best-effort getter for both dict-like specs and object specs."""
    if isinstance(spec, Mapping):
        for n in names:
            if n in spec:
                return spec[n]
        return default
    for n in names:
        if hasattr(spec, n):
            return getattr(spec, n)
    return default


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _key_to_str(k: Any) -> str:
    # Enum-like: prefer `.value` if it is a string.
    if hasattr(k, "value") and isinstance(getattr(k, "value"), str):
        return str(getattr(k, "value"))
    return str(k)


def _iter_metric_specs() -> List[Tuple[str, Any]]:
    """
    Normalize registry.metrics.METRIC_SPECS into list[(key:str, spec:any)].

    Supported METRIC_SPECS forms:
      - dict-like mapping: {key: spec}
      - iterable of (key, spec)
      - iterable of spec objects/dicts that contain key/name/id fields
    """
    from rlhf_eval_lab.registry import metrics as m  # type: ignore

    if not hasattr(m, "METRIC_SPECS"):
        raise RuntimeError("registry.metrics has no METRIC_SPECS")

    specs = getattr(m, "METRIC_SPECS")

    # dict-like
    if isinstance(specs, Mapping):
        return [(_key_to_str(k), v) for k, v in specs.items()]

    # iterable (list/tuple/etc.)
    if isinstance(specs, Iterable) and not isinstance(specs, (str, bytes)):
        out: List[Tuple[str, Any]] = []
        for i, item in enumerate(specs):
            # (key, spec) tuple/list
            if isinstance(item, (tuple, list)) and len(item) == 2:
                k, v = item[0], item[1]
                out.append((_key_to_str(k), v))
                continue

            spec = item
            key = _get_field(
                spec,
                ["key", "name", "id", "metric", "metric_key", "metric_name", "col", "column"],
                default=None,
            )
            if key is None:
                raise RuntimeError(
                    "METRIC_SPECS is a list, but an element has no key/name/id field and is not (key, spec). "
                    f"index={i}, type={type(spec)}"
                )
            out.append((_key_to_str(key), spec))
        return out

    raise RuntimeError(f"Unsupported METRIC_SPECS type: {type(specs)}")


def _table_ids_for_spec(spec: Any) -> List[str]:
    """
    Extract table membership for a metric spec.

    Supports:
      - explicit fields: tables/table_id/table_ids
      - MetricSpec-style booleans: in_table1/in_table2a/in_table2b/in_table2c
    """
    out: List[str] = []

    # New-style explicit table fields.
    t = _get_field(spec, ["tables", "table_ids", "table_id", "table"], default=None)
    for v in _as_list(t):
        if v is None:
            continue
        out.append(str(v))

    # MetricSpec-style boolean flags.
    flag_map = [
        ("Table1", "in_table1"),
        ("Table2A", "in_table2a"),
        ("Table2B", "in_table2b"),
        ("Table2C", "in_table2c"),
    ]
    for table_name, attr in flag_map:
        if hasattr(spec, attr) and bool(getattr(spec, attr)):
            out.append(table_name)

    # De-dup while preserving order.
    seen = set()
    dedup: List[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup


def _label_for_spec(key: str, spec: Any) -> str:
    # MetricSpec uses `.name` for a human label.
    return str(_get_field(spec, ["name", "label", "display_name", "display", "title"], default=key))


def _decimals_for_spec(spec: Any) -> str:
    # Prefer explicit decimals/precision if defined.
    v = _get_field(spec, ["decimals", "precision", "digits", "ndigits"], default=None)
    if v is not None:
        return str(v)

    # Fallback heuristic from dtype.
    dtype = _get_field(spec, ["dtype", "type"], default=None)
    if dtype is None:
        return "-"

    dt = str(dtype).lower()
    if "float" in dt:
        return "4"
    if "int" in dt:
        return "0"
    return "-"


def _na_policy_for_spec(spec: Any) -> str:
    """
    Render N/A policy in a compact, human-readable way.

    Supports:
      - explicit policy fields (na/na_policy/etc.)
      - MetricSpec-style `na_for_method_keys`
    """
    v = _get_field(spec, ["na", "na_policy", "na_rule", "allow_na", "allow_na_when"], default=None)
    if v is not None:
        if isinstance(v, bool):
            return "allowed" if v else "disallowed"
        return str(v)

    mk = _get_field(spec, ["na_for_method_keys"], default=None)
    if mk is None:
        return "-"

    if isinstance(mk, (list, tuple, set)):
        xs = sorted([str(x) for x in mk])
        if len(xs) <= 6:
            return "methods: " + ", ".join(xs)
        return f"methods: {len(xs)}"
    return str(mk)


def _notes_for_spec(spec: Any) -> str:
    # MetricSpec uses `.direction` (↑/↓) and `.dtype` which are useful in README.
    desc = _get_field(spec, ["description", "desc", "notes", "note", "meaning"], default=None)
    direction = _get_field(spec, ["direction"], default=None)
    dtype = _get_field(spec, ["dtype"], default=None)

    parts: List[str] = []
    if direction is not None:
        parts.append(f"dir={direction}")
    if dtype is not None:
        parts.append(f"dtype={dtype}")
    if desc:
        parts.append(str(desc).replace("\n", " ").strip())

    return " | ".join(parts).strip()


def _sort_tables(tables: List[str]) -> List[str]:
    order = {name: i for i, name in enumerate(TABLE_ORDER)}
    return sorted(tables, key=lambda x: (order.get(x, 10_000), x))


def render_metrics_ssot_block() -> str:
    metric_items = _iter_metric_specs()

    grouped: Dict[str, List[Tuple[str, Any]]] = {}
    for key, spec in metric_items:
        tables = _table_ids_for_spec(spec)
        if not tables:
            continue
        for t in tables:
            grouped.setdefault(t, []).append((key, spec))

    tables_sorted = _sort_tables(list(grouped.keys()))

    lines: List[str] = []
    lines.append("## Metrics SSOT (auto-generated from registry/metrics.py)")
    lines.append("")
    lines.append("This block is updated by `python -m rlhf_eval_lab.utils.update_readme_metrics_ssot`.")
    lines.append("It synchronizes metric names, N/A policies, and formatting rules with the code SSOT.")
    lines.append("")

    for t in tables_sorted:
        lines.append(f"### {t}")
        lines.append("")
        lines.append("| key | label | decimals | N/A | notes |")
        lines.append("|---|---|---:|---|---|")
        for key, spec in grouped[t]:
            label = _label_for_spec(key, spec)
            dec = _decimals_for_spec(spec)
            na = _na_policy_for_spec(spec)
            notes = _notes_for_spec(spec)
            if len(notes) > 140:
                notes = notes[:137] + "..."
            lines.append(f"| `{key}` | {label} | {dec} | {na} | {notes} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def update_readme(path: Path = README_PATH) -> None:
    readme = path.read_text(encoding="utf-8")

    if START_MARKER not in readme or END_MARKER not in readme:
        raise SystemExit(
            "README.md needs SSOT markers. Please add the following lines somewhere in README.md:\n\n"
            f"{START_MARKER}\n{END_MARKER}\n"
        )

    block = render_metrics_ssot_block()

    new = re.sub(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        f"{START_MARKER}\n\n{block}\n{END_MARKER}",
        readme,
        flags=re.S,
    )

    path.write_text(new, encoding="utf-8")
    print("updated:", path)


def main() -> None:
    update_readme()


if __name__ == "__main__":
    main()
