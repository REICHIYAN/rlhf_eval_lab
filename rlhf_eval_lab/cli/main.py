# rlhf_eval_lab/cli/main.py
from __future__ import annotations

import argparse
import sys

from rlhf_eval_lab.cli.commands.run import run_cmd
from rlhf_eval_lab.cli.commands.validate import validate_cmd
from rlhf_eval_lab.cli.commands.report import report_cmd
from rlhf_eval_lab.reporting.paths import (
    default_artifacts_dir,
    default_reports_dir,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rlhf-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    # -----------------
    # run
    # -----------------
    p_run = sub.add_parser("run", help="Run methods and generate artifacts")
    p_run.add_argument("--backend", choices=["fallback", "hf"], default="fallback")
    p_run.add_argument("--preset", default=None, help="Config preset (overrides backend)")
    p_run.add_argument("--config", default=None, help="User config YAML")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument(
        "--out",
        default=default_artifacts_dir(),
        help="Artifacts output dir (default: artifacts/)",
    )

    # -----------------
    # validate
    # -----------------
    p_val = sub.add_parser("validate", help="Validate artifacts and metrics (strict)")
    p_val.add_argument(
        "--in",
        dest="in_dir",
        default=default_artifacts_dir(),
        help="Artifacts input dir (default: artifacts/)",
    )

    # -----------------
    # report
    # -----------------
    p_rep = sub.add_parser("report", help="Aggregate artifacts and render report")
    p_rep.add_argument(
        "--in",
        dest="in_dir",
        default=default_artifacts_dir(),
        help="Artifacts input dir (default: artifacts/)",
    )
    p_rep.add_argument(
        "--out",
        default=default_reports_dir(),
        help="Reports output dir (default: reports/)",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        return run_cmd(args)
    if args.command == "validate":
        return validate_cmd(args)
    if args.command == "report":
        return report_cmd(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
