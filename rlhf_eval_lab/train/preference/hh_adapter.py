from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class HHPreferenceExample:
    uid: str
    prompt: str
    chosen: str
    rejected: str


def _read_jsonl(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl not found: {path}")
    rows: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid JSONL object (not dict) at {path}:{ln}: {type(obj)}")
            rows.append(obj)
    return rows


def _build_prompt_map(prompts_rows: List[dict]) -> Dict[str, str]:
    """
    prompts.jsonl is expected to have:
      - id (or prompt_id) -> prompt text
      - prompt (or text)  -> prompt text
    """
    m: Dict[str, str] = {}
    for i, r in enumerate(prompts_rows):
        pid = r.get("id", r.get("prompt_id", None))
        if pid is None:
            raise ValueError(f"prompts.jsonl row missing id/prompt_id at index={i}: keys={list(r.keys())}")
        prompt = r.get("prompt", r.get("text", None))
        if prompt is None:
            raise ValueError(f"prompts.jsonl row missing prompt/text at index={i}: keys={list(r.keys())}")
        m[str(pid)] = str(prompt)
    return m


def _join_pairs(
    prefs_rows: List[dict],
    prompt_map: Dict[str, str],
    limit: int,
) -> List[HHPreferenceExample]:
    """
    prefs.jsonl is expected to have:
      - prompt_id (or id) : join key to prompts.jsonl
      - chosen
      - rejected
      - optional uid
    """
    out: List[HHPreferenceExample] = []
    for i, r in enumerate(prefs_rows):
        if limit is not None and limit > 0 and len(out) >= int(limit):
            break

        pid = r.get("prompt_id", r.get("id", None))
        if pid is None:
            raise ValueError(f"prefs.jsonl row missing prompt_id/id at index={i}: keys={list(r.keys())}")
        pid = str(pid)

        prompt = prompt_map.get(pid)
        if prompt is None:
            # Strict join: if prompt_id can't be resolved, that's a data error in this pipeline
            raise ValueError(f"prompt_id '{pid}' not found in prompts.jsonl (prefs index={i})")

        chosen = r.get("chosen", None)
        rejected = r.get("rejected", None)
        if chosen is None or rejected is None:
            raise ValueError(f"prefs.jsonl row missing chosen/rejected at index={i}: keys={list(r.keys())}")

        uid = r.get("uid", None)
        if uid is None:
            uid = f"{pid}:{i}"

        out.append(
            HHPreferenceExample(
                uid=str(uid),
                prompt=str(prompt),
                chosen=str(chosen),
                rejected=str(rejected),
            )
        )
    return out


def _deterministic_split(
    examples: List[HHPreferenceExample],
    seed: int,
    val_ratio: float,
) -> Tuple[List[HHPreferenceExample], List[HHPreferenceExample]]:
    if not (0.0 <= float(val_ratio) <= 1.0):
        raise ValueError(f"val_ratio must be in [0,1], got {val_ratio}")

    n = len(examples)
    if n == 0:
        return [], []

    rng = random.Random(int(seed))
    idx = list(range(n))
    rng.shuffle(idx)

    n_val = int(round(n * float(val_ratio)))

    # safety: if n>=2 and ratio>0, ensure val has at least 1
    if n >= 2 and float(val_ratio) > 0.0 and n_val == 0:
        n_val = 1

    n_val = max(0, min(n_val, n))

    val_set = set(idx[:n_val])
    train = [examples[i] for i in range(n) if i not in val_set]
    val = [examples[i] for i in range(n) if i in val_set]

    # safety: keep both non-empty if possible
    if n >= 2 and len(val) == 0:
        val = [train.pop()]
    if n >= 2 and len(train) == 0:
        train = [val.pop()]

    return train, val


def load_hh_rlhf_split(
    prefs_path: str,
    prompts_path: str,
    seed: int,
    val_ratio: float,
    limit: int,
) -> Tuple[List[HHPreferenceExample], List[HHPreferenceExample]]:
    """
    Loader used by Level-C training. Intentionally independent of rlhf_eval_lab.data.hh_rlhf
    to avoid dependency on internal function names.

    Inputs:
      - prefs.jsonl: prompt_id + chosen/rejected
      - prompts.jsonl: id -> prompt text
    """
    prefs_rows = _read_jsonl(prefs_path)
    prompts_rows = _read_jsonl(prompts_path)
    prompt_map = _build_prompt_map(prompts_rows)
    examples = _join_pairs(prefs_rows, prompt_map, limit=int(limit))
    train, val = _deterministic_split(examples, seed=int(seed), val_ratio=float(val_ratio))
    return train, val
