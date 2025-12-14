# rlhf_eval_lab/backends/hf/utils.py
# transformers は任意依存：遅延 import で fallback に混ざらないようにする

from __future__ import annotations
from typing import Any


def lazy_import_transformers() -> Any:
    try:
        import transformers  # type: ignore
        return transformers
    except Exception as e:
        raise ImportError(
            "transformers is not installed. Install with requirements-hf.txt to use HF backend."
        ) from e
