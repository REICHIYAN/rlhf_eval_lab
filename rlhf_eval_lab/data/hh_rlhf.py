# rlhf_eval_lab/data/hh_rlhf.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PreferencePair:
    """Unified internal schema for pairwise preference learning."""
    uid: str
    prompt: str
    chosen: str
    rejected: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class PreferenceSplit:
    """Deterministic split result."""
    train: List[PreferencePair]
    val: List[PreferencePair]


class HHRLHFFormatError(ValueError):
    """Raised when input jsonl does not contain required fields or cannot be joined."""


def _first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _make_uid(prompt: str, chosen: str, rejected: str, i: int) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(chosen.encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(rejected.encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(str(i).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def _deterministic_split_indices(
    n: int, seed: int, val_ratio: float
) -> Tuple[List[int], List[int]]:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio}")
    if n == 0:
        return [], []

    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    val_n = int(round(n * val_ratio))

    # Safety: avoid empty splits when 0<val_ratio<1 and n is small.
    if 0.0 < val_ratio < 1.0:
        val_n = max(1, val_n)
        val_n = min(n - 1, val_n) if n >= 2 else 0

    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    return train_idx, val_idx



def _parse_pair_record(
    rec: Dict[str, Any],
    i: int,
    *,
    prompt_lookup: Optional[Dict[str, str]] = None,
) -> PreferencePair:
    """
    Parse a preference pair record. Supports two formats:

    (1) Inline prompt:
        {prompt|instruction|..., chosen, rejected}

    (2) Joined prompt by ID:
        {prompt_id, chosen, rejected} + prompts.jsonl provides id->prompt_text
    """
    chosen = _first_present(
        rec,
        keys=("chosen", "accept", "accepted", "response_chosen", "completion_chosen", "winner"),
    )
    rejected = _first_present(
        rec,
        keys=("rejected", "reject", "rejected_response", "response_rejected", "completion_rejected", "loser"),
    )

    chosen_s = _as_text(chosen).strip()
    rejected_s = _as_text(rejected).strip()
    if not chosen_s or not rejected_s:
        present_keys = sorted(list(rec.keys()))
        raise HHRLHFFormatError(
            "HH-RLHF record missing required fields (chosen/rejected) "
            f"or empty after stripping at line={i}. present_keys={present_keys}"
        )

    # Prompt can be inline OR via lookup by prompt_id
    prompt = _first_present(
        rec,
        keys=("prompt", "instruction", "input", "question", "query", "context"),
    )
    prompt_s = _as_text(prompt).strip()

    if not prompt_s:
        prompt_id = _first_present(rec, keys=("prompt_id", "promptId", "pid", "id_prompt"))
        pid_s = _as_text(prompt_id).strip()
        if pid_s and prompt_lookup is not None and pid_s in prompt_lookup:
            prompt_s = prompt_lookup[pid_s].strip()

    if not prompt_s:
        present_keys = sorted(list(rec.keys()))
        raise HHRLHFFormatError(
            "HH-RLHF record missing prompt text. Provide inline 'prompt' "
            "or use joined loader with prompts.jsonl (prompt_id -> prompt). "
            f"line={i}. present_keys={present_keys}"
        )

    uid = _make_uid(prompt_s, chosen_s, rejected_s, i)

    meta: Dict[str, Any] = {}
    # Keep minimal metadata
    for k in ("source", "id", "task", "category", "split", "prompt_id"):
        if k in rec:
            meta[k] = rec[k]

    return PreferencePair(
        uid=uid,
        prompt=prompt_s,
        chosen=chosen_s,
        rejected=rejected_s,
        meta=meta,
    )


def load_prompts_jsonl(path: str | Path) -> Dict[str, str]:
    """
    Load prompts mapping from jsonl.

    Expected each line to contain:
      - an identifier key: id|prompt_id|promptId|pid
      - prompt text key: prompt|instruction|text|input|question|query|context
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompts jsonl not found: {p}")

    lookup: Dict[str, str] = {}
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise HHRLHFFormatError(f"prompts.jsonl line must be an object at line={i}")

            pid = _first_present(rec, keys=("id", "prompt_id", "promptId", "pid"))
            prompt = _first_present(
                rec,
                keys=("prompt", "instruction", "text", "input", "question", "query", "context"),
            )
            pid_s = _as_text(pid).strip()
            prompt_s = _as_text(prompt).strip()

            if not pid_s or not prompt_s:
                present_keys = sorted(list(rec.keys()))
                raise HHRLHFFormatError(
                    "prompts.jsonl missing required fields (id/prompt). "
                    f"line={i}. present_keys={present_keys}"
                )
            lookup[pid_s] = prompt_s

    if not lookup:
        raise HHRLHFFormatError(f"no valid prompts loaded from: {p}")
    return lookup


def load_hh_rlhf_pairs_jsonl(
    path: str | Path,
    *,
    seed: int = 0,
    val_ratio: float = 0.1,
    max_samples: Optional[int] = None,
) -> PreferenceSplit:
    """
    Load HH-RLHF-style jsonl file into unified PreferencePair schema with
    deterministic train/val split.

    This expects inline prompt text in each record.
    If your file contains only prompt_id, use load_hh_rlhf_pairs_jsonl_joined().
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl not found: {p}")

    pairs: List[PreferencePair] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise HHRLHFFormatError(f"jsonl line must be an object at line={i}")
            pair = _parse_pair_record(rec, i, prompt_lookup=None)
            pairs.append(pair)
            if max_samples is not None and len(pairs) >= max_samples:
                break

    if len(pairs) == 0:
        raise HHRLHFFormatError(f"no valid records loaded from: {p}")

    train_idx, val_idx = _deterministic_split_indices(len(pairs), seed=seed, val_ratio=val_ratio)
    train = [pairs[j] for j in train_idx]
    val = [pairs[j] for j in val_idx]
    return PreferenceSplit(train=train, val=val)


def load_hh_rlhf_pairs_jsonl_joined(
    prefs_path: str | Path,
    prompts_path: str | Path,
    *,
    seed: int = 0,
    val_ratio: float = 0.1,
    max_samples: Optional[int] = None,
) -> PreferenceSplit:
    """
    Load preference pairs from prefs.jsonl and join prompt text by prompt_id
    using prompts.jsonl.
    """
    prompt_lookup = load_prompts_jsonl(prompts_path)

    p = Path(prefs_path)
    if not p.exists():
        raise FileNotFoundError(f"prefs jsonl not found: {p}")

    pairs: List[PreferencePair] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise HHRLHFFormatError(f"prefs.jsonl line must be an object at line={i}")
            pair = _parse_pair_record(rec, i, prompt_lookup=prompt_lookup)
            pairs.append(pair)
            if max_samples is not None and len(pairs) >= max_samples:
                break

    if len(pairs) == 0:
        raise HHRLHFFormatError(f"no valid joined records loaded from: {p}")

    train_idx, val_idx = _deterministic_split_indices(len(pairs), seed=seed, val_ratio=val_ratio)
    train = [pairs[j] for j in train_idx]
    val = [pairs[j] for j in val_idx]
    return PreferenceSplit(train=train, val=val)


def _smoke_print(split: PreferenceSplit) -> None:
    print(f"train={len(split.train)} val={len(split.val)}")
    ex = split.train[0] if split.train else split.val[0]
    print(f"example.uid={ex.uid}")
    print(f"example.prompt[0:80]={ex.prompt[:80]!r}")
    print(f"example.chosen[0:80]={ex.chosen[:80]!r}")
    print(f"example.rejected[0:80]={ex.rejected[:80]!r}")


if __name__ == "__main__":
    # Minimal smoke runner (no CLI changes).
    # Examples:
    #   python -m rlhf_eval_lab.data.hh_rlhf test_data/prefs.jsonl
    #   python -m rlhf_eval_lab.data.hh_rlhf test_data/prefs.jsonl 0 0.1 50 test_data/prompts.jsonl
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "  python -m rlhf_eval_lab.data.hh_rlhf <prefs_jsonl> [seed] [val_ratio] [max_samples] [prompts_jsonl]\n"
            "Notes:\n"
            "  - If prompts_jsonl is provided, prompt_id will be joined to prompt text.\n"
        )

    prefs_path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    val_ratio = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.1
    max_samples = int(sys.argv[4]) if len(sys.argv) >= 5 else None
    prompts_path = sys.argv[5] if len(sys.argv) >= 6 else None

    if prompts_path is None:
        s = load_hh_rlhf_pairs_jsonl(prefs_path, seed=seed, val_ratio=val_ratio, max_samples=max_samples)
    else:
        s = load_hh_rlhf_pairs_jsonl_joined(
            prefs_path,
            prompts_path,
            seed=seed,
            val_ratio=val_ratio,
            max_samples=max_samples,
        )
    _smoke_print(s)
