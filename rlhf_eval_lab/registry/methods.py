# rlhf_eval_lab/registry/methods.py
# 表の「行」を定義する SSOT
# - 順序は論文・OSSともに固定
# - ここを変えない限り、Markdown / 集計 / テストの行順は不変

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MethodSpec:
    # 内部キー（Artifacts / 集計 / CLI で使用）
    key: str
    # 表示名（Markdown / 論文）
    name: str
    # Table 1 の Category 列
    category: str

    # family flags（Table 2 系の対象判定に使用）
    is_ppo_family: bool = False
    is_preference_based: bool = False
    is_active: bool = False
    is_safety: bool = False
    is_adaptive_reward: bool = False


# =========================
# SSOT: Method list (ORDER FIXED)
# =========================

METHOD_SPECS: List[MethodSpec] = [
    # ---- Baseline ----
    MethodSpec(
        key="sft",
        name="SFT",
        category="Baseline",
    ),
    MethodSpec(
        key="ppo_standard",
        name="PPO-RLHF (Standard)",
        category="Baseline",
        is_ppo_family=True,
    ),

    # ---- Policy Regularization ----
    MethodSpec(
        key="kl_ppo_fixed",
        name="KL-PPO (Fixed β)",
        category="Policy Regularization",
        is_ppo_family=True,
    ),
    MethodSpec(
        key="kl_ppo_adaptive",
        name="KL-PPO (Adaptive / Target-KL)",
        category="Policy Regularization",
        is_ppo_family=True,
    ),

    # ---- Reward-free / Preference-based ----
    MethodSpec(
        key="dpo",
        name="DPO",
        category="Reward-free / Preference-based",
        is_preference_based=True,
    ),
    MethodSpec(
        key="ipo",
        name="IPO",
        category="Reward-free / Preference-based",
        is_preference_based=True,
    ),
    MethodSpec(
        key="rrhf",
        name="RRHF",
        category="Reward-free / Preference-based",
        is_preference_based=True,
    ),
    MethodSpec(
        key="orpo",
        name="ORPO",
        category="Reward-free / Preference-based",
        is_preference_based=True,
    ),
    MethodSpec(
        key="rlaif",
        name="RLAIF",
        category="Reward-free / Preference-based",
        is_preference_based=True,
    ),

    # ---- Active Preference ----
    MethodSpec(
        key="active_pref",
        name="Active Preference Learning",
        category="Active Preference",
        is_preference_based=True,
        is_active=True,
    ),

    # ---- Safety ----
    MethodSpec(
        key="safe_ppo",
        name="Safety-RLHF (Safe PPO)",
        category="Safety",
        is_ppo_family=True,
        is_safety=True,
    ),

    # ---- Adaptive Reward ----
    MethodSpec(
        key="adaptive_rm_ppo",
        name="Adaptive Reward Model + PPO",
        category="Adaptive Reward",
        is_ppo_family=True,
        is_adaptive_reward=True,
    ),
]


# 便利参照（テスト用）
METHOD_BY_KEY = {m.key: m for m in METHOD_SPECS}
METHOD_KEYS = [m.key for m in METHOD_SPECS]

# Backward-compatible alias (CLI/older modules may import this)
METHODS = METHOD_SPECS
