# rlhf_eval_lab/cli/commands/run.py
# 目的：
# - 全手法を最低 1 step 回す（fallbackでは実測で backward->step）
# - ArtifactsV1 を必ず吐く（空欄ゼロ設計の入力SSOT）
# 注意：
# - 研究最適化ではなく sanity tier（fallback）を最優先
# - HF backend は optional なので遅延 import で守る（transformers未導入でも落とさない）
# - Step1ではHFは「生成→評価→artifacts」まで（学習は次コミット）

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
from rlhf_eval_lab.data.loaders import load_prompts_from_dataset_config


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
    Deterministically construct (chosen, rejected) from a completion + scalar reward.
    """
    alt = _truncate_text(completion)
    if alt.strip() == (completion or "").strip():
        alt = (completion or "") + " ."
    return (completion, alt) if reward >= 0.0 else (alt, completion)


def _infer_backend_name(args_backend: str | None, preset: str | None) -> str:
    """
    Backend selection policy (must not break fallback CI):
      - If args.backend is explicitly provided: honor it.
      - Else infer from preset name:
          hf_* or paper_* -> hf
          otherwise -> fallback
    """
    b = (args_backend or "").strip().lower()
    if b:
        return b
    p = (preset or "").strip().lower()
    if p.startswith("hf_") or p.startswith("paper_"):
        return "hf"
    return "fallback"


def _make_backend(backend_name: str, cfg: Dict[str, Any]):
    """
    Create backend instance + metadata needed for provenance.
    IMPORTANT: HFBackend must be lazily imported.
    """
    if backend_name == "hf":
        # lazy import to avoid import error when transformers is not installed
        from rlhf_eval_lab.backends.hf.backend import HFBackend  # pylint: disable=import-error

        model_name = str(cfg.get("hf", {}).get("model_name", "gpt2"))
        backend = HFBackend(cfg)
        model_id = model_name
        tokenizer = f"hf:{model_name}"
        return backend, "hf", model_id, tokenizer

    # default: fallback
    backend = FallbackBackend(cfg)
    model_id = "tiny-gru"
    tokenizer = "simple"
    return backend, "fallback", model_id, tokenizer


def run_cmd(args) -> int:
    cfg = load_config(preset=args.preset, user_path=args.config)
    seed = int(args.seed)

    out_dir = os.path.abspath(args.out)
    _ensure_dir(out_dir)

    rm = HeuristicRewardModel()

    # prompts SSOT: dataset-aware (paper presets rely on this)
    ds_cfg = cfg.get("dataset", {}) or {}
    prompts, dataset_base_key, dataset_hash = (
        load_prompts_from_dataset_config(ds_cfg)
        if isinstance(ds_cfg, dict) and (ds_cfg.get("name") or ds_cfg.get("path"))
        else (
            [
                "Explain what reinforcement learning is.",
                "What is PPO in simple terms?",
                "Define reward in machine learning.",
            ],
            "builtin_prompts",
            "",
        )
    )

    if dataset_hash:
        print(f"[run] dataset={dataset_base_key} prompts={len(prompts)} hash={dataset_hash[:8]}")
    else:
        print(f"[run] dataset={dataset_base_key} prompts={len(prompts)}")

    max_new = int(cfg.get("eval", {}).get("max_new_tokens", 16))

    backend_name = _infer_backend_name(getattr(args, "backend", None), getattr(args, "preset", None))
    base_backend, prov_backend, prov_model_id, prov_tokenizer = _make_backend(backend_name, cfg)

    # provenance is SSOT: fixed once per run
    _set_seed(seed)
    run_provenance = build_provenance(
        cfg,
        backend=prov_backend,
        model_id=prov_model_id,
        tokenizer=prov_tokenizer,
        seed=seed,
    )

    # ==========================================================
    # fallback-only PPO setup (HF step1 does NOT train)
    # ==========================================================
    use_fallback_ppo = prov_backend == "fallback"

    kl_beta_base = float(cfg.get("ppo", {}).get("kl_beta", 0.1))
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

    if use_fallback_ppo:
        # baseline completions/rewards for fallback PPO family
        ppo_prompts = list(prompts)
        ppo_base_completions = base_backend.generate(ppo_prompts, max_new_tokens=max_new)
        ppo_base_rewards = rm.score(ppo_prompts, ppo_base_completions)

        # reference snapshots
        ppo_ref_state = {k: v.detach().clone() for k, v in base_backend.ref_model.state_dict().items()}
        ppo_policy_init_state = {k: v.detach().clone() for k, v in base_backend.model.state_dict().items()}
    else:
        # HF Step1: no PPO training; still define variables to keep code simple
        ppo_prompts = list(prompts)
        ppo_base_completions = base_backend.generate(ppo_prompts, max_new_tokens=max_new)
        ppo_base_rewards = rm.score(ppo_prompts, ppo_base_completions)
        ppo_ref_state = {}
        ppo_policy_init_state = {}

    # ==========================================================
    # method loop
    # ==========================================================
    for m in METHOD_SPECS:
        _set_seed(seed)

        # For HF Step1, reuse the already-loaded model to avoid re-downloading per method.
        # For fallback, keep per-method fresh backend to keep "step actually runs" invariant.
        backend = base_backend if prov_backend == "hf" else FallbackBackend(cfg)

        method_dir = os.path.join(out_dir, m.key)
        _ensure_dir(method_dir)

        extra: Dict[str, Any] = {}

        # Step A) auditability flags (L8)
        extra["skipped"] = False
        extra["skip_reason"] = ""

        if dataset_hash:
            extra["dataset_key"] = dataset_base_key
            extra["dataset_hash"] = dataset_hash

        dataset_key = dataset_base_key

        # ----------------------------------------------------------
        # SFT
        # ----------------------------------------------------------
        if m.key == "sft":
            completions = list(ppo_base_completions)
            rewards = rm.score(prompts, completions)

            # fallback: actually step; hf(step1): keep 0.0 to avoid training
            if prov_backend == "fallback":
                texts = [f"{p} {c}".strip() for p, c in zip(prompts, completions)]
                loss = backend.sft_step(texts)
                extra["sft_loss"] = float(loss)
                extra["steps"] = 1
            else:
                extra["sft_loss"] = 0.0
                extra["steps"] = 0
                extra["skipped"] = True
                extra["skip_reason"] = "hf_step1_generation_only"

            extra["kl"] = 0.0
            dataset_key = "sft_train"

        # ----------------------------------------------------------
        # PPO family (fallback-only training in this step)
        # ----------------------------------------------------------
        elif m.key in _PPO_METHOD_KEYS:
            completions = list(ppo_base_completions)
            rewards = list(ppo_base_rewards)

            if use_fallback_ppo:
                backend.model.load_state_dict(ppo_policy_init_state)

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
                # HF Step1: no PPO training yet; emit sane numeric placeholders (never empty)
                extra["kl"] = 0.0
                extra["steps"] = 0
                extra["kl_beta"] = float(kl_beta_map.get(m.key, kl_beta_base))
                extra["skipped"] = True
                extra["skip_reason"] = "hf_step1_generation_only"

        # ----------------------------------------------------------
        # preference-based + others
        # ----------------------------------------------------------
        else:
            completions = backend.generate(prompts, max_new_tokens=max_new)
            rewards = rm.score(prompts, completions)

            dataset_key = "pref_train" if m.is_preference_based else "comparisons"

            # fallback: actually step; hf(step1): placeholders only
            if prov_backend == "fallback":
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
            else:
                extra["pref_loss"] = 0.0
                extra["steps"] = 0
                extra["skipped"] = True
                extra["skip_reason"] = "hf_step1_generation_only"

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
