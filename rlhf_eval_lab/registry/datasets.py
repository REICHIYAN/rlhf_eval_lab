# rlhf_eval_lab/registry/datasets.py
# DatasetSpec は「何のデータが必要か」を宣言する薄いSSOT
# 現時点では最小限。後で "metrics_requires" のような宣言を足せる設計。

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    # jsonl ファイル名（test_data/ 配下を想定）
    filename: str


DATASET_SPECS: List[DatasetSpec] = [
    DatasetSpec(key="prompts", name="Prompts", filename="prompts.jsonl"),
    DatasetSpec(key="comparisons", name="Comparisons", filename="comparisons.jsonl"),
    DatasetSpec(key="sft_train", name="SFT Train", filename="sft_train.jsonl"),
    DatasetSpec(key="pref_train", name="Preference Train", filename="pref_train.jsonl"),
    DatasetSpec(key="ood_prompts", name="OOD Prompts", filename="ood_prompts.jsonl"),
    DatasetSpec(
        key="injection_base_prompts",
        name="Injection Base Prompts",
        filename="injection_base_prompts.jsonl",
    ),
]

DATASET_BY_KEY = {d.key: d for d in DATASET_SPECS}
