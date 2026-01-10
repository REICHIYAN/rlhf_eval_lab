# rlhf_eval_lab/data/loaders.py
# jsonl ローダー（依存最小） + dataset wiring helpers

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import hashlib
import json
import os
import random


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dataset_enabled(ds: Dict[str, Any]) -> bool:
    name = str(ds.get("name", "") or "").strip()
    path = str(ds.get("path", "") or "").strip()
    return bool(name) or bool(path)


def build_dataset_hash(ds: Dict[str, Any]) -> str:
    """
    Dataset hash for provenance without bundling data.

    Policy:
      - include file sha256 (for local source)
      - include dataset config fields that affect sampling/splits
    """
    source = str(ds.get("source", "local") or "local").strip()
    payload: Dict[str, Any] = {
        "name": str(ds.get("name", "") or "").strip(),
        "source": source,
        "split": str(ds.get("split", "train") or "train").strip(),
        "subsample_n": int(ds.get("subsample_n", 0) or 0),
        "seed": int(ds.get("seed", 0) or 0),
    }

    if source == "local":
        path = str(ds.get("path", "") or "").strip()
        payload["path_basename"] = os.path.basename(path) if path else ""
        payload["file_sha256"] = _sha256_file(path) if path else ""
    else:
        payload["hf_id"] = str(ds.get("hf_id", "") or "").strip()

    h = hashlib.sha256()
    h.update(_stable_json(payload).encode("utf-8"))
    return h.hexdigest()


def _extract_prompt_from_record(r: Dict[str, Any]) -> str:
    """
    Flexible prompt extraction for local jsonl.

    Supported minimal formats:
      - {"prompt": "..."}
      - {"instruction": "..."} / {"question": "..."} / {"input": "..."} / {"text": "..."}
      - {"chosen": "...", "rejected": "...", "prompt": "..."} (prompt preferred)
    """
    for k in ("prompt", "instruction", "question", "input", "text"):
        v = r.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    msgs = r.get("messages", None)
    if isinstance(msgs, list) and msgs:
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                c = m["content"].strip()
                if c:
                    return c

    raise ValueError("No prompt-like field found in jsonl record")


def load_prompts_from_dataset_config(ds: Dict[str, Any]) -> Tuple[List[str], str, str]:
    """
    Returns:
      prompts: list[str]
      dataset_base_key: str (stable identifier)
      dataset_hash: str (sha256)
    """
    if not _dataset_enabled(ds):
        return ([], "builtin_prompts", "")

    name = str(ds.get("name", "") or "").strip()
    source = str(ds.get("source", "local") or "local").strip()
    split = str(ds.get("split", "train") or "train").strip()
    subsample_n = int(ds.get("subsample_n", 0) or 0)
    seed = int(ds.get("seed", 0) or 0)

    if source != "local":
        raise NotImplementedError("dataset.source='hf' is not implemented yet (use local jsonl for now).")

    path = str(ds.get("path", "") or "").strip()
    if not path:
        raise ValueError("dataset.path is required for local datasets")

    rows = load_jsonl(path)
    prompts_all: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            p = _extract_prompt_from_record(r)
        except ValueError:
            continue
        prompts_all.append(p)

    if not prompts_all:
        raise ValueError(f"No prompts extracted from dataset jsonl: {path}")

    rng = random.Random(seed)
    idxs = list(range(len(prompts_all)))
    rng.shuffle(idxs)

    if subsample_n and subsample_n > 0:
        idxs = idxs[: min(subsample_n, len(idxs))]

    prompts = [prompts_all[i] for i in idxs]

    dataset_base_key = f"{name}:{split}:{source}"
    dataset_hash = build_dataset_hash(ds)
    return (prompts, dataset_base_key, dataset_hash)
