# rlhf_eval_lab/cli/commands/run.py
# 目的：
# - 全手法を最低 1 step 回す（実測で backward->step）
# - ArtifactsV1 を必ず吐く（空欄ゼロ設計の入力SSOT）
# 注意：
# - 研究最適化ではなく sanity tier（fallback）を最優先

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os
import random

import torch

from rlhf_eval_lab.config.io import load_config
from rlhf_eval_lab.registry.methods import METHOD_SPECS
from rlhf_eval_lab.reporting.artifacts import ArtifactsV1, write_artifacts
from rlhf_eval_lab.reporting.provenance import build_provenance
from rlhf_eval_lab.backends.fallback.backend import FallbackBackend
from rlhf_eval_lab.train.reward_models.heuristic import HeuristicRewardModel


_PPO_METHOD_KEYS = {
    "ppo_standard",
    "kl_ppo_fixed",
    "kl_ppo_adaptive",
    "safe_ppo",
    "adaptive_rm_ppo",
}


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _truncate_text(s: str) -> str:
    toks = (s or "").split()
    if len(toks) <= 2:
        return s
    return " ".join(toks[: max(2, len(toks) // 2)])


def _make_pref_pair(prompt: str, completion: str, reward: float) -> Tuple[str, str]:
    """
    決定論的に (chosen, rejected) を作る。
    """
    alt = _truncate_text(completion)
    if alt.strip() == (completion or "").strip():
        alt = (completion or "") + " ."
    return (completion, alt) if reward >= 0.0 else (alt, completion)


def run_cmd(args) -> int:
    cfg = load_config(preset=args.preset, user_path=args.config)
    seed = int(args.seed)

    out_dir = os.path.abspath(args.out)
    _ensure_dir(out_dir)

    rm = HeuristicRewardModel()

    prompts: List[str] = [
        "Explain what reinforcement learning is.",
        "What is PPO in simple terms?",
        "Define reward in machine learning.",
    ]

    max_new = int(cfg.get("eval", {}).get("max_new_tokens", 16))
    kl_beta_base = float(cfg.get("train", {}).get("kl_beta", 0.1))
    pref_beta = float(cfg.get("train", {}).get("pref_beta", 0.1))

    kl_beta_map: Dict[str, float] = {
        "ppo_standard": 0.0,
        "kl_ppo_fixed": kl_beta_base,
        "kl_ppo_adaptive": kl_beta_base * 2.0,
        "safe_ppo": kl_beta_base * 5.0,
        "adaptive_rm_ppo": kl_beta_base * 1.0,
    }

    ppo_steps = int(cfg.get("train", {}).get("sanity_ppo_steps", 20))
    if ppo_steps < 2:
        ppo_steps = 2

    # ==========================================================
    # ★ Run-level SSOT（ここが今回の本質）
    # ==========================================================
    _set_seed(seed)
    base = FallbackBackend(cfg)

    ppo_ref_state = {k: v.detach().clone() for k, v in base.ref_model.state_dict().items()}
    ppo_policy_init_state = {k: v.detach().clone() for k, v in base.model.state_dict().items()}

    ppo_prompts = list(prompts)
    ppo_base_completions = base.generate(ppo_prompts, max_new_tokens=max_new)
    ppo_base_rewards = rm.score(ppo_prompts, ppo_base_completions)

    # ★ provenance は run 開始時に一度だけ確定（SSOT）
    run_provenance = build_provenance(
        cfg,
        backend="fallback",
        model_id="tiny-gru",
        tokenizer="simple",
        seed=seed,
    )

    # ==========================================================
    # method loop
    # ==========================================================
    for m in METHOD_SPECS:
        _set_seed(seed)
        backend = FallbackBackend(cfg)

        method_dir = os.path.join(out_dir, m.key)
        _ensure_dir(method_dir)

        extra: Dict[str, Any] = {}
        dataset_key = "prompts"

        if m.key == "sft":
            completions = list(ppo_base_completions)
            rewards = rm.score(prompts, completions)

            texts = [f"{p} {c}".strip() for p, c in zip(prompts, completions)]
            loss = backend.sft_step(texts)

            extra["sft_loss"] = float(loss)
            extra["kl"] = 0.0
            extra["steps"] = 1
            dataset_key = "sft_train"

        elif m.key in _PPO_METHOD_KEYS:
            backend.model.load_state_dict(ppo_policy_init_state)

            completions = list(ppo_base_completions)
            rewards = list(ppo_base_rewards)

            kl_beta_eff = float(kl_beta_map.get(m.key, kl_beta_base))

            last_out: Dict[str, float] = {}
            for _ in range(ppo_steps):
                out = backend.ppo_step(
                    prompts=ppo_prompts,
                    completions=completions,
                    rewards=rewards,
                    kl_beta=kl_beta_eff,
                    ref_state=ppo_ref_state,
                    update_ref=False,
                )
                last_out = {k: float(v) for k, v in out.items()}

            extra.update(last_out)

            if "kl_sum" in extra:
                extra["kl"] = float(extra["kl_sum"])
            elif "kl_mean" in extra:
                extra["kl"] = float(extra["kl_mean"])
            else:
                extra["kl"] = 0.0

            extra["steps"] = int(ppo_steps)
            extra["kl_beta"] = float(kl_beta_eff)

            print(
                f"[run] method={m.key} kl_beta={kl_beta_eff} steps={ppo_steps} "
                f"ppo_out_keys={sorted(list(last_out.keys()))}"
            )

        else:
            completions = backend.generate(prompts, max_new_tokens=max_new)
            rewards = rm.score(prompts, completions)

            dataset_key = "pref_train" if m.is_preference_based else "comparisons"
            chosen, rejected = _make_pref_pair(prompts[0], completions[0], rewards[0])

            loss = backend.preference_step(
                prompt=prompts[0],
                chosen=chosen,
                rejected=rejected,
                beta=pref_beta,
            )

            extra["pref_loss"] = float(loss)
            extra["steps"] = 1
            extra["pair_prompt"] = prompts[0]
            extra["pair_chosen"] = chosen
            extra["pair_rejected"] = rejected

        # ★ provenance は必ず run_provenance を使う
        art = ArtifactsV1(
            method_key=m.key,
            dataset_key=dataset_key,
            provenance=run_provenance,
            prompts=prompts,
            completions=completions,
            rewards=rewards,
            extra=extra,
        )

        path = os.path.join(method_dir, f"seed_{seed}.json")
        write_artifacts(path, art)

    return 0
