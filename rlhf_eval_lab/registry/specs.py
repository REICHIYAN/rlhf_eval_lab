# rlhf_eval_lab/registry/specs.py
# SSOT: 表のスキーマ（どの表にどの列を出すか）を定義する
# - methods.py が「行」
# - metrics.py が「列」
# - datasets.py は将来拡張用（評価に必要な入力データ宣言）

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TableSpec:
    # 例: "table1", "table2a" ...
    key: str
    # 表の人間向けタイトル
    title: str
    # その表に含める列キー（metrics.py の MetricSpec.key を参照）
    metric_keys: List[str]
