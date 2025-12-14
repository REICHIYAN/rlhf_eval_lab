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


def _make_pref_pair(
    prompt: str,
    completion: str,
    reward: float,
) -> Tuple[str, str]:
    """
    決定論的に (chosen, rejected) を作る。
    - completion をベースに「短縮版」を作り、heuristic reward の大小で chosen を決める。
    """
    alt = _truncate_text(completion)
    # alt が同一になった場合は、確実に違う文字列にする（決定論的）
    if alt.strip() == (completion or "").strip():
        alt = (completion or "") + " ."

    return (completion, alt) if reward >= 0.0 else (alt, completion)


def run_cmd(args) -> int:
    cfg = load_config(preset=args.preset, user_path=args.config)
    seed = int(args.seed)
    _set_seed(seed)

    out_dir = os.path.abspath(args.out)
    _ensure_dir(out_dir)

    # backend: fallback only（CLI arg は将来 hf を足す）
    backend = FallbackBackend(cfg)
    rm = HeuristicRewardModel()

    # 最小プロンプト（fallback sanity）
    prompts: List[str] = [
        "Explain what reinforcement learning is.",
        "What is PPO in simple terms?",
        "Define reward in machine learning.",
    ]

    max_new = int(cfg.get("eval", {}).get("max_new_tokens", 16))
    kl_beta = float(cfg.get("train", {}).get("kl_beta", 0.1))
    pref_beta = float(cfg.get("train", {}).get("pref_beta", 0.1))

    for m in METHOD_SPECS:
        method_dir = os.path.join(out_dir, m.key)
        _ensure_dir(method_dir)

        completions = backend.generate(prompts, max_new_tokens=max_new)
        rewards = rm.score(prompts, completions)

        extra: Dict[str, Any] = {}
        dataset_key = "prompts"

        # ---- 1 step 学習（手法別：sanity tier）----
        if m.key == "sft":
            # SFT は prompt+completion を “テキスト列” として 1 step
            texts = [f"{p} {c}".strip() for p, c in zip(prompts, completions)]
            loss = backend.sft_step(texts)
            extra["sft_loss"] = float(loss)
            extra["steps"] = 1
            dataset_key = "sft_train"

        elif m.is_ppo_family:
            out = backend.ppo_step(
                prompts=prompts,
                completions=completions,
                rewards=rewards,
                kl_beta=kl_beta,
            )
            extra.update({k: float(v) for k, v in out.items()})
            extra["steps"] = 1
            dataset_key = "prompts"

        else:
            # Preference / Active は、最小の 1 pair で 1 step
            dataset_key = "pref_train" if m.is_preference_based else "comparisons"
            # 先頭データで確実に1回は backward->step を回す
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

        prov = build_provenance(
            cfg,
            backend="fallback",
            model_id="tiny-gru",
            tokenizer="simple",
            seed=seed,
        )

        art = ArtifactsV1(
            method_key=m.key,
            dataset_key=dataset_key,
            provenance=prov,
            prompts=prompts,
            completions=completions,
            rewards=rewards,
            extra=extra,
        )

        path = os.path.join(method_dir, f"seed_{seed}.json")
        write_artifacts(path, art)

    return 0
