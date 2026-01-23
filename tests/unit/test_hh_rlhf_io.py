# tests/unit/test_hh_rlhf_io.py
from __future__ import annotations

from rlhf_eval_lab.data.hh_rlhf import load_hh_rlhf_pairs_jsonl_joined


def test_hh_rlhf_joined_loader_smoke() -> None:
    """
    Regression test for HH-RLHF I/O wiring:
    - prefs.jsonl contains prompt_id + chosen/rejected
    - prompts.jsonl provides prompt text by id
    - loader joins them and produces non-empty splits
    """
    split = load_hh_rlhf_pairs_jsonl_joined(
        prefs_path="test_data/prefs.jsonl",
        prompts_path="test_data/prompts.jsonl",
        seed=0,
        val_ratio=0.1,
        max_samples=50,
    )

    assert len(split.train) > 0
    assert len(split.val) > 0  # ensured by split safety when 0<val_ratio<1

    ex = split.train[0]
    assert ex.uid
    assert ex.prompt.strip()
    assert ex.chosen.strip()
    assert ex.rejected.strip()


def test_hh_rlhf_split_is_deterministic() -> None:
    """
    Same seed -> same split membership (by uid).
    """
    s1 = load_hh_rlhf_pairs_jsonl_joined(
        prefs_path="test_data/prefs.jsonl",
        prompts_path="test_data/prompts.jsonl",
        seed=0,
        val_ratio=0.1,
        max_samples=50,
    )
    s2 = load_hh_rlhf_pairs_jsonl_joined(
        prefs_path="test_data/prefs.jsonl",
        prompts_path="test_data/prompts.jsonl",
        seed=0,
        val_ratio=0.1,
        max_samples=50,
    )

    uids1_train = [p.uid for p in s1.train]
    uids2_train = [p.uid for p in s2.train]
    uids1_val = [p.uid for p in s1.val]
    uids2_val = [p.uid for p in s2.val]

    assert uids1_train == uids2_train
    assert uids1_val == uids2_val


def test_hh_rlhf_different_seed_changes_split() -> None:
    """
    Different seed should usually change the split membership.
    With very small test_data, this may occasionally collide, so we check
    for "not exactly identical" rather than enforcing full difference.
    """
    s1 = load_hh_rlhf_pairs_jsonl_joined(
        prefs_path="test_data/prefs.jsonl",
        prompts_path="test_data/prompts.jsonl",
        seed=0,
        val_ratio=0.1,
        max_samples=50,
    )
    s2 = load_hh_rlhf_pairs_jsonl_joined(
        prefs_path="test_data/prefs.jsonl",
        prompts_path="test_data/prompts.jsonl",
        seed=1,
        val_ratio=0.1,
        max_samples=50,
    )

    uids1 = ([p.uid for p in s1.train], [p.uid for p in s1.val])
    uids2 = ([p.uid for p in s2.train], [p.uid for p in s2.val])

    assert uids1 != uids2
