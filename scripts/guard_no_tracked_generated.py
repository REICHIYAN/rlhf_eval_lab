#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List


TARGETS = [
    "artifacts",
    "reports",
    "outputs",
    "data",  # root-level generated cache dir (NOT rlhf_eval_lab/data)
    "report.md",
]


def _git(*args: str) -> str:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = e.output.decode("utf-8", errors="replace")
        raise RuntimeError(f"git {' '.join(args)} failed:\n{msg}") from e


def _tracked_under(path: str) -> List[str]:
    # Use -z for safe parsing
    out = subprocess.check_output(["git", "ls-files", "-z", "--", path])
    if not out:
        return []
    items = out.decode("utf-8", errors="replace").split("\x00")
    return [x for x in items if x]


def main() -> int:
    try:
        _ = _git("rev-parse", "--is-inside-work-tree")
    except Exception as e:
        print(f"[guard] ERROR: not a git repo? {e}", file=sys.stderr)
        return 2

    offenders: List[str] = []
    for t in TARGETS:
        offenders.extend(_tracked_under(t))

    if offenders:
        print("[guard] ERROR: generated outputs are tracked by git.", file=sys.stderr)
        print("These paths must NOT be committed. Remove them from index:", file=sys.stderr)
        for p in sorted(set(offenders)):
            print(f"  - {p}", file=sys.stderr)

        print("\nSuggested fix:", file=sys.stderr)
        print("  git rm -r --cached artifacts reports outputs data report.md || true", file=sys.stderr)
        print("  git commit -m \"Stop tracking generated outputs\"", file=sys.stderr)
        return 1

    print("[guard] OK: no tracked generated outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
