from __future__ import annotations

from typing import Dict, Any

from rlhf_eval_lab.data.loaders import load_prompts_from_dataset_config


def _write_jsonl(path: str, rows) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_load_prompts_from_local_jsonl_hh(tmp_path) -> None:
    p = tmp_path / "hh.jsonl"
    _write_jsonl(
        str(p),
        [
            {"prompt": "Hello", "chosen": "A", "rejected": "B"},
            {"prompt": "World", "chosen": "C", "rejected": "D"},
        ],
    )

    ds: Dict[str, Any] = {
        "name": "hh_rlhf",
        "source": "local",
        "path": str(p),
        "split": "train",
        "subsample_n": 1,
        "seed": 0,
    }

    prompts, dkey, dhash = load_prompts_from_dataset_config(ds)
    assert len(prompts) == 1
    assert dkey.startswith("hh_rlhf:")
    assert isinstance(dhash, str) and len(dhash) == 64
