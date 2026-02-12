# rlhf_eval_lab/registry/metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class MetricSpec:
    # Internal key (Artifacts -> eval -> aggregate -> reporting)
    key: str
    # Display name (Markdown column header)
    name: str
    # Expected final dtype ("float" by default)
    dtype: str = "float"
    # Visual direction hint ("↓" / "↑" / None). Informational.
    direction: Optional[str] = None

    # Inclusion flags by table
    in_table1: bool = False
    in_table2a: bool = False
    in_table2b: bool = False
    in_table2c: bool = False

    # Column-level N/A policy: methods listed here must render this metric as N/A.
    # This is enforced by validate (and should be mirrored by eval/runner behavior).
    na_for_method_keys: Optional[List[str]] = None


# =========================
# Method families (keep in sync with registry/methods.py)
# =========================

# PPO-family (Table 2-A can be numeric)
PPO_FAMILY: List[str] = [
    "ppo_standard",
    "kl_ppo_fixed",
    "kl_ppo_adaptive",
    "safe_ppo",
    "adaptive_rm_ppo",
]

# Preference / reward-free / active (Table 2-B is the main diagnostics area)
PREF_FAMILY: List[str] = [
    "dpo",
    "ipo",
    "rrhf",
    "orpo",
    "rlaif",
    "active_pref",
]

# Non-PPO => Table 2-A is N/A
NON_PPO: List[str] = ["sft"] + PREF_FAMILY

# Non-Preference => Table 2-B is N/A
NON_PREF: List[str] = ["sft"] + PPO_FAMILY

# Safety/Robustness-capable (Table 2-C can be numeric)
SAFETY_FAMILY: List[str] = [
    "ppo_standard",
    "kl_ppo_fixed",
    "kl_ppo_adaptive",
    "safe_ppo",
    "adaptive_rm_ppo",
]

# Non-safety => Table 2-C is N/A
NON_SAFETY: List[str] = ["sft"] + PREF_FAMILY


# =========================
# SSOT: Metric list (fixed order)
# =========================

# Table 1 columns (fixed order)
TABLE1_METRICS: List[MetricSpec] = [
    MetricSpec(key="offsupport", name="Off-support ↓", direction="↓", in_table1=True),
    MetricSpec(key="tail_var", name="Tail Var ↓", direction="↓", in_table1=True),
    MetricSpec(key="onsupport", name="On-support ↑", direction="↑", in_table1=True),
    MetricSpec(key="judge", name="Judge ↑", direction="↑", in_table1=True),
    MetricSpec(key="win_rate", name="Win-rate ↑", direction="↑", in_table1=True),
    # PPL: must be numeric (no N/A by policy)
    MetricSpec(key="ppl", name="PPL ↓", direction="↓", in_table1=True),
    # KL: N/A for preference/active methods by column policy
    MetricSpec(
        key="kl",
        name="KL ↓",
        direction="↓",
        in_table1=True,
        na_for_method_keys=PREF_FAMILY,
    ),
]

# Table 2-A (PPO-family diagnostics / audit)
TABLE2A_METRICS: List[MetricSpec] = [
    MetricSpec(
        key="ppo_loss",
        name="PPO Loss ↓",
        direction="↓",
        in_table2a=True,
        na_for_method_keys=NON_PPO,
    ),
    MetricSpec(
        key="ratio_mean",
        name="Ratio Mean",
        direction=None,  # target is ≈1 (no arrow)
        in_table2a=True,
        na_for_method_keys=NON_PPO,
    ),
    MetricSpec(
        key="clipfrac",
        name="Clip Fraction ↓",
        direction="↓",
        in_table2a=True,
        na_for_method_keys=NON_PPO,
    ),
    MetricSpec(
        key="kl_ref_abs",
        name="KL Ref Abs ↓",
        direction="↓",
        in_table2a=True,
        na_for_method_keys=NON_PPO,
    ),
    MetricSpec(
        key="kl_ref_sq",
        name="KL Ref Sq ↓",
        direction="↓",
        in_table2a=True,
        na_for_method_keys=NON_PPO,
    ),
]

# Table 2-B (Preference-based diagnostics)
TABLE2B_METRICS: List[MetricSpec] = [
    MetricSpec(
        key="sample_efficiency",
        name="Sample Efficiency ↑",
        direction="↑",
        in_table2b=True,
        na_for_method_keys=NON_PREF,
    ),
    MetricSpec(
        key="reward_accuracy",
        name="Reward Accuracy ↑",
        direction="↑",
        in_table2b=True,
        na_for_method_keys=NON_PREF,
    ),
    # String column; runner returns "pref"/"ai"/"-"/"N/A"
    MetricSpec(key="label_source", name="Label Source", dtype="str", in_table2b=True),
]

# Table 2-C (Safety / Robustness)
TABLE2C_METRICS: List[MetricSpec] = [
    MetricSpec(
        key="prompt_injection",
        name="Prompt Injection ↓",
        direction="↓",
        in_table2c=True,
        na_for_method_keys=NON_SAFETY,
    ),
    MetricSpec(
        key="ood_stability",
        name="OOD Stability ↓",
        direction="↓",
        in_table2c=True,
        na_for_method_keys=NON_SAFETY,
    ),
]

# Full SSOT (order: Table1 -> 2A -> 2B -> 2C)
METRIC_SPECS: List[MetricSpec] = (
    TABLE1_METRICS + TABLE2A_METRICS + TABLE2B_METRICS + TABLE2C_METRICS
)

# Mapping helpers (backward/forward compatible names)
METRIC_SPECS_BY_KEY = {m.key: m for m in METRIC_SPECS}
METRIC_BY_KEY = METRIC_SPECS_BY_KEY  # alias
