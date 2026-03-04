# rlhf_eval_lab/train/aegis/__init__.py

"""Aegis: reliability-weighted preference update.

Design goals (mirrors governor philosophy):
- Keep algorithm logic isolated under rlhf_eval_lab/train/aegis/
- Keep fallback tier dependency-free (uses eval/judge heuristic by default)
- Fail-open: if anything goes wrong, fall back to a plain preference_step

Aegis is intended to reduce prompt/judge bias by estimating the reliability of a
(preference) pair under lightweight prompt transforms, then down-weighting the
update when uncertainty is high.
"""

from .driver import run_aegis  # noqa: F401
