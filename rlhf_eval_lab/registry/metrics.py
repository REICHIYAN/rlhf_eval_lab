# rlhf_eval_lab/registry/metrics.py
# 表の「列」を定義する SSOT
# - Table 1: main results（全手法共通）
# - Table 2-A/B/C: family別 diagnostics
# - N/A 規約は「列単位」で固定（ここが唯一の根拠）

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class MetricSpec:
    # 内部キー（Artifacts → eval → aggregate → markdown で使用）
    key: str
    # 表示名（Markdown 列名）
    name: str
    # 期待する最終型（基本は float）
    dtype: str = "float"
    # 良い方向（↓ or ↑）。表示用。None は表示なし
    direction: Optional[str] = None
    # Table 1 に含めるか
    in_table1: bool = False
    # Table 2-A/B/C に含めるか（それぞれ）
    in_table2a: bool = False
    in_table2b: bool = False
    in_table2c: bool = False
    # 列単位 N/A 規約（例：Preference系は KL を N/A）
    # ここは「空欄禁止」設計の要
    na_for_method_keys: Optional[List[str]] = None


# =========================
# SSOT: Method keys（registry/methods.py と一致させる）
# =========================

# PPO-family（2Aが数値になり得る）
PPO_FAMILY = [
    "ppo_standard",
    "kl_ppo_fixed",
    "kl_ppo_adaptive",
    "safe_ppo",
    "adaptive_rm_ppo",
]

# Preference / reward-free / active（2Bが主戦場、KLはN/Aになり得る）
PREF_FAMILY = [
    "dpo",
    "ipo",
    "rrhf",
    "orpo",
    "rlaif",
    "active_pref",
]

# 非PPO（= 2A は N/A）
NON_PPO = ["sft"] + PREF_FAMILY

# 非Preference（= 2B は N/A）
NON_PREF = ["sft"] + PPO_FAMILY

# Safety/Robustness を計測する側（2Cを数値にする）
SAFETY_FAMILY = [
    "ppo_standard",
    "kl_ppo_fixed",
    "kl_ppo_adaptive",
    "safe_ppo",
    "adaptive_rm_ppo",
]

# 2C を N/A にする側
NON_SAFETY = ["sft"] + PREF_FAMILY


# =========================
# SSOT: Metric list
# =========================

# Table 1 columns（固定順）
TABLE1_METRICS: List[MetricSpec] = [
    MetricSpec(key="offsupport", name="Off-support ↓", direction="↓", in_table1=True),
    MetricSpec(key="tail_var", name="Tail Var ↓", direction="↓", in_table1=True),
    MetricSpec(key="onsupport", name="On-support ↑", direction="↑", in_table1=True),
    MetricSpec(key="judge", name="Judge ↑", direction="↑", in_table1=True),
    MetricSpec(key="win_rate", name="Win-rate ↑", direction="↑", in_table1=True),
    # PPL は B-1 で追加（まずは N/A を許さず必ず数値）
    MetricSpec(key="ppl", name="PPL ↓", direction="↓", in_table1=True),
    # KL は preference 系 / active 系では列規約で N/A を許可（runnerがN/Aを返す設計）
    MetricSpec(
        key="kl",
        name="KL ↓",
        direction="↓",
        in_table1=True,
        na_for_method_keys=PREF_FAMILY,
    ),
    MetricSpec(key="notes", name="Notes", dtype="str", in_table1=True),
]

# Table 2-A（PPO-family diagnostics）
TABLE2A_METRICS: List[MetricSpec] = [
    MetricSpec(
        key="kl_stability",
        name="KL Stability ↓",
        direction="↓",
        in_table2a=True,
        na_for_method_keys=NON_PPO,  # PPO以外はN/A
    ),
    MetricSpec(
        key="reward_var",
        name="Reward Var ↓",
        direction="↓",
        in_table2a=True,
        na_for_method_keys=NON_PPO,  # PPO以外はN/A
    ),
    MetricSpec(
        key="convergence_speed",
        name="Convergence Speed ↑",
        direction="↑",
        in_table2a=True,
        na_for_method_keys=NON_PPO,  # PPO以外はN/A
    ),
]

# Table 2-B（Preference-based diagnostics）
TABLE2B_METRICS: List[MetricSpec] = [
    MetricSpec(
        key="sample_efficiency",
        name="Sample Efficiency ↑",
        direction="↑",
        in_table2b=True,
        na_for_method_keys=NON_PREF,  # preference以外はN/A
    ),
    MetricSpec(
        key="reward_accuracy",
        name="Reward Accuracy ↑",
        direction="↑",
        in_table2b=True,
        na_for_method_keys=NON_PREF,  # preference以外はN/A
    ),
    # Label Source は str 列（runnerがN/A or "pref"/"ai"/"-" を返す）
    MetricSpec(key="label_source", name="Label Source", dtype="str", in_table2b=True),
]

# Table 2-C（Safety / Robustness）
TABLE2C_METRICS: List[MetricSpec] = [
    MetricSpec(
        key="prompt_injection",
        name="Prompt Injection ↓",
        direction="↓",
        in_table2c=True,
        na_for_method_keys=NON_SAFETY,  # Safety側以外はN/A
    ),
    MetricSpec(
        key="ood_stability",
        name="OOD Stability ↓",
        direction="↓",
        in_table2c=True,
        na_for_method_keys=NON_SAFETY,  # Safety側以外はN/A
    ),
]

# 全列のSSOT（順序は Table1 → 2A → 2B → 2C）
METRIC_SPECS: List[MetricSpec] = (
    TABLE1_METRICS + TABLE2A_METRICS + TABLE2B_METRICS + TABLE2C_METRICS
)

METRIC_BY_KEY = {m.key: m for m in METRIC_SPECS}
