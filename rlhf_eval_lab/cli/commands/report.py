# rlhf_eval_lab/cli/commands/report.py
from __future__ import annotations

import os

from rlhf_eval_lab.reporting.aggregate import aggregate
from rlhf_eval_lab.reporting.artifacts import read_artifacts_tree
from rlhf_eval_lab.reporting.markdown import render_report
from rlhf_eval_lab.reporting.paths import report_md_path


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

    print(f"[rlhf-lab] Report written to: {out_path}")
    return 0
