# rlhf_eval_lab/data/loaders.py
# jsonl loader (minimal deps) + dataset wiring helpers

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
    """Dataset hash for provenance without bundling data.

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
        payload["hf_config"] = str(ds.get("hf_config", "") or "").strip()
        payload["data_dir"] = str(ds.get("data_dir", "") or "").strip()

    h = hashlib.sha256()
    h.update(_stable_json(payload).encode("utf-8"))
    return h.hexdigest()


def _extract_prompt_from_record(r: Dict[str, Any]) -> str:
    """Flexible prompt extraction for local jsonl.

    Supported minimal formats:
      - {"prompt": "..."}
      - {"instruction": "..."} / {"question": "..."} / {"input": "..."} / {"text": "..."}
      - {"chosen": "...", "rejected": "...", "prompt": "..."} (prompt preferred)
      - {"messages":[{"role":"user","content":"..."}...]}
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


def _hh_prompt_from_chosen(chosen: str) -> str:
    s = chosen or ""
    marker = "\n\nAssistant:"
    if marker in s:
        idx = s.rfind(marker)
        return s[: idx + len(marker)]
    return s.strip() + marker


def _harmbench_prompt(r: Dict[str, Any]) -> str:
    behavior = (r.get("Behavior") or "").strip() if isinstance(r.get("Behavior"), str) else ""
    ctx = (r.get("ContextString") or "").strip() if isinstance(r.get("ContextString"), str) else ""
    if ctx:
        p = ctx + "\n\n" + behavior
    else:
        p = behavior
    return p.strip() + "\n\nAssistant:"


def _extract_prompt_from_hf_record(r: Dict[str, Any]) -> str:
    # 1) standard prompt-like fields
    try:
        return _extract_prompt_from_record(r)
    except ValueError:
        pass

    # 2) HH-RLHF style: chosen/rejected
    chosen = r.get("chosen", None)
    if isinstance(chosen, str) and chosen.strip():
        return _hh_prompt_from_chosen(chosen)

    # 3) HarmBench style: Behavior / ContextString
    if "Behavior" in r:
        return _harmbench_prompt(r)

    raise ValueError("No supported prompt-like field found in HF dataset record")


def load_prompts_from_dataset_config(ds: Dict[str, Any]) -> Tuple[List[str], str, str]:
    """Returns:
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

    if source == "local":
        path = str(ds.get("path", "") or "").strip()
        if not path:
            raise ValueError("dataset.path is required for local datasets")
        rows = load_jsonl(path)
    elif source == "hf":
        hf_id = str(ds.get("hf_id", "") or "").strip()
        if not hf_id:
            raise ValueError("dataset.hf_id is required for hf datasets")
        hf_config = ds.get("hf_config", None)
        data_dir = ds.get("data_dir", None)

        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("dataset.source='hf' requires `datasets`. Install with: pip install -e '.[hf]'") from e

        kwargs: Dict[str, Any] = {}
        if data_dir:
            kwargs["data_dir"] = str(data_dir)
        if hf_config:
            rows_ds = load_dataset(hf_id, str(hf_config), split=split, **kwargs)
        else:
            rows_ds = load_dataset(hf_id, split=split, **kwargs)

        # Convert to list[dict] without printing contents
        rows = [dict(x) for x in rows_ds]
    else:
        raise ValueError(f"Unknown dataset.source: {source}")

    prompts_all: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            p = _extract_prompt_from_hf_record(r) if source == "hf" else _extract_prompt_from_record(r)
        except ValueError:
            continue
        if isinstance(p, str) and p.strip():
            prompts_all.append(p.strip())

    if not prompts_all:
        raise ValueError(f"No prompts extracted from dataset ({source}): name={name}")

    rng = random.Random(seed)
    idxs = list(range(len(prompts_all)))
    rng.shuffle(idxs)
    if subsample_n and subsample_n > 0:
        idxs = idxs[: min(subsample_n, len(idxs))]

    prompts = [prompts_all[i] for i in idxs]
    dataset_base_key = f"{name}:{split}:{source}"
    dataset_hash = build_dataset_hash(ds)
    return (prompts, dataset_base_key, dataset_hash)
