# rlhf_eval_lab/cli/commands/run.py

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, Iterable, Tuple

import torch

from rlhf_eval_lab.backends.fallback.backend import FallbackBackend
from rlhf_eval_lab.config.io import load_config
from rlhf_eval_lab.data.loaders import load_prompts_from_dataset_config
from rlhf_eval_lab.registry.methods import METHOD_SPECS
from rlhf_eval_lab.reporting.artifacts import ArtifactsV1, write_artifacts
from rlhf_eval_lab.reporting.provenance import build_provenance
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


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return float(sum(float(x) for x in xs) / len(xs))


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except Exception:
        return float(default)


def _model_checksum(model: torch.nn.Module) -> Tuple[float, float]:
    """
    Cheap global checksum for "did parameters change at all?" auditing.
    Returns (abs_sum, sq_sum). If both deltas are ~0, update likely did not apply.
    """
    abs_sum = 0.0
    sq_sum = 0.0
    with torch.no_grad():
        for p in model.parameters():
            t = p.detach()
            abs_sum += float(t.abs().sum().item())
            sq_sum += float((t * t).sum().item())
    return abs_sum, sq_sum


def _truncate_text(s: str) -> str:
    toks = (s or "").split()
    if len(toks) <= 2:
        return s
    return " ".join(toks[: max(2, len(toks) // 2)])


def _make_pref_pair(prompt: str, completion: str, reward: float) -> Tuple[str, str]:
    """
    Deterministically construct (chosen, rejected) from a completion + scalar reward.
    (prompt is unused but kept for future extensibility)
    """
    _ = prompt
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

        model_name = str((cfg.get("hf", {}) or {}).get("model_name", "gpt2"))
        backend = HFBackend(cfg)
        model_id = model_name
        tokenizer = f"hf:{model_name}"
        return backend, "hf", model_id, tokenizer

    # default: fallback
    backend = FallbackBackend(cfg)
    model_id = "tiny-gru"
    tokenizer = "simple"
    return backend, "fallback", model_id, tokenizer


def _choose_nonneg_kl_proxy(extra: Dict[str, Any]) -> float:
    """
    HF PPO "KL" display policy (C1.6+):
      - Prefer nonnegative proxies if available.
      - Fall back to abs(kl_ref) / abs(kl_est) as a last resort.
    """
    for k in ("kl_ref_abs", "kl_est_abs"):
        v = extra.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass

    for k in ("kl_ref", "kl_est"):
        v = extra.get(k)
        if v is not None:
            try:
                return float(abs(float(v)))
            except Exception:
                pass

    return 0.0


def _stamp_ppo_audit_from_fallback(extra: Dict[str, Any], ppo_out: Dict[str, Any]) -> None:
    """
    Map fallback PPO outputs to standardized audit keys (Table 2-A SSOT):
      - ppo_loss
      - ratio_mean
      - clipfrac
      - kl_ref_abs
      - kl_ref_sq

    Fallback does not implement logprob-ratio based diagnostics, so we provide
    explicit, auditable defaults:
      ratio_mean = 1.0, clipfrac = 0.0, ratio_mean_pre = 1.0
    """
    # ppo_loss: fallback typically returns "loss"
    extra["ppo_loss"] = float(_as_float(ppo_out.get("ppo_loss", ppo_out.get("loss", 0.0)), 0.0))

    # ratio_mean / clipfrac: not available in fallback -> stable defaults
    extra["ratio_mean"] = float(_as_float(ppo_out.get("ratio_mean", 1.0), 1.0))
    extra.setdefault("ratio_mean_pre", 1.0)
    extra["clipfrac"] = float(_as_float(ppo_out.get("clipfrac", 0.0), 0.0))

    # KL proxy: prefer kl_mean, else kl, else kl_sum/token_count
    kl_mean = ppo_out.get("kl_mean", None)
    if kl_mean is not None:
        kl_signed = float(_as_float(kl_mean, 0.0))
    else:
        kl_v = ppo_out.get("kl", None)
        if kl_v is not None:
            kl_signed = float(_as_float(kl_v, 0.0))
        else:
            kl_sum = float(_as_float(ppo_out.get("kl_sum", 0.0), 0.0))
            token_count = float(_as_float(ppo_out.get("token_count", 0.0), 0.0))
            kl_signed = kl_sum / token_count if token_count > 0.0 else kl_sum

    kl_abs = abs(float(kl_signed))
    extra["kl_ref_abs"] = float(kl_abs)
    extra["kl_ref_sq"] = float(kl_abs * kl_abs)


def _kl_beta_adapt_update(
    beta_pre: float,
    ppo_out: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Minimal, safe KL-beta adaptation rule (audit-first).

    Update uses a bounded multiplicative rule:
      ratio = kl_measured / kl_target
      adj   = clip(ratio - 1, [-beta_clip, +beta_clip])
      beta_post = clamp(beta_pre * exp(beta_lr * adj), [beta_min, beta_max])

    Returns:
      (beta_post, audit_fields)
    """
    beta_pre = _as_float(beta_pre, 0.0)

    kl_target = _as_float(train_cfg.get("kl_target", 0.1), 0.1)
    beta_lr = _as_float(train_cfg.get("kl_beta_lr", 0.1), 0.1)
    beta_min = _as_float(train_cfg.get("kl_beta_min", 1e-5), 1e-5)
    beta_max = _as_float(train_cfg.get("kl_beta_max", 100.0), 100.0)
    beta_clip = abs(_as_float(train_cfg.get("kl_beta_clip", 0.2), 0.2))

    # stable KL proxy preference: abs proxy > abs(signed) > sqrt(sq)
    kl_measured = ppo_out.get("kl_ref_abs", None)
    if kl_measured is None:
        kl_measured = abs(_as_float(ppo_out.get("kl_ref", 0.0), 0.0))
    else:
        kl_measured = _as_float(kl_measured, 0.0)

    if kl_measured == 0.0:
        kl_measured = math.sqrt(max(_as_float(ppo_out.get("kl_ref_sq", 0.0), 0.0), 0.0))

    # sanitize
    eps = 1e-12
    if (not math.isfinite(kl_target)) or (kl_target <= 0.0):
        kl_target = 0.1
    if (not math.isfinite(kl_measured)) or (kl_measured < 0.0):
        kl_measured = 0.0

    ratio = kl_measured / max(kl_target, eps)
    if not math.isfinite(ratio):
        ratio = 0.0

    adj = ratio - 1.0
    if beta_clip > 0.0:
        adj = max(-beta_clip, min(beta_clip, adj))

    beta_post = beta_pre * math.exp(beta_lr * adj)
    if not math.isfinite(beta_post):
        beta_post = beta_pre

    beta_post = max(beta_min, min(beta_max, beta_post))

    audit = {
        "beta_pre": float(beta_pre),
        "beta_post": float(beta_post),
        "kl_target": float(kl_target),
        "kl_measured": float(kl_measured),
        "beta_lr": float(beta_lr),
        "beta_clip": float(beta_clip),
        "beta_min": float(beta_min),
        "beta_max": float(beta_max),
        "beta_ratio": float(ratio),
        "beta_adj": float(adj),
    }
    return float(beta_post), audit


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

    max_new = int((cfg.get("eval", {}) or {}).get("max_new_tokens", 16))
    backend_name = _infer_backend_name(getattr(args, "backend", None), getattr(args, "preset", None))

    # IMPORTANT: seed must be fixed BEFORE backend init (reproducibility)
    _set_seed(seed)

    base_backend, prov_backend, prov_model_id, prov_tokenizer = _make_backend(backend_name, cfg)

    # provenance is SSOT: fixed once per run
    run_provenance = build_provenance(
        cfg,
        backend=prov_backend,
        model_id=prov_model_id,
        tokenizer=prov_tokenizer,
        seed=seed,
    )

    # ==========================================================
    # training knobs
    # ==========================================================
    use_fallback_ppo = prov_backend == "fallback"

    kl_beta_base = float((cfg.get("ppo", {}) or {}).get("kl_beta", 0.1))
    pref_beta = float((cfg.get("train", {}) or {}).get("pref_beta", 0.1))

    # HF knobs (optional)
    train_cfg = cfg.get("train", {}) or {}
    hf_sft_steps = int(train_cfg.get("hf_sft_steps", 0))
    hf_ppo_steps = int(train_cfg.get("hf_ppo_steps", 0))

    hf_ppo_method_keys_raw = train_cfg.get("hf_ppo_method_keys", None)
    if hf_ppo_method_keys_raw is None:
        hf_ppo_method_keys = {"ppo_standard"}
    elif isinstance(hf_ppo_method_keys_raw, (list, tuple, set)):
        hf_ppo_method_keys = {str(x) for x in hf_ppo_method_keys_raw}
    else:
        hf_ppo_method_keys = {str(hf_ppo_method_keys_raw)}

    ppo_clip = float(train_cfg.get("ppo_clip", 0.2))
    ppo_lr = float(train_cfg.get("ppo_lr", 1e-6))
    sft_lr = float(train_cfg.get("lr", 1e-3))

    kl_beta_map: Dict[str, float] = {
        "ppo_standard": 0.0,
        "kl_ppo_fixed": kl_beta_base,
        "kl_ppo_adaptive": kl_beta_base * 2.0,
        "safe_ppo": kl_beta_base * 5.0,
        "adaptive_rm_ppo": kl_beta_base * 1.0,
    }

    # fallback PPO steps (sanity tier)
    ppo_steps = int(train_cfg.get("sanity_ppo_steps", 20))
    if ppo_steps < 2:
        ppo_steps = 2

    # Baseline completions/rewards (used for SFT + PPO-family baseline)
    _set_seed(seed)
    ppo_prompts = list(prompts)
    ppo_base_completions = base_backend.generate(ppo_prompts, max_new_tokens=max_new)
    ppo_base_rewards = rm.score(ppo_prompts, ppo_base_completions)

    # reference snapshots (fallback only)
    if use_fallback_ppo:
        # NOTE: rely on base_backend having model/ref_model in fallback layer
        ppo_ref_state = {k: v.detach().clone() for k, v in base_backend.ref_model.state_dict().items()}
        ppo_policy_init_state = {k: v.detach().clone() for k, v in base_backend.model.state_dict().items()}
    else:
        ppo_ref_state = {}
        ppo_policy_init_state = {}

    # HF per-method isolation: keep initial weights snapshot (avoid cross-method contamination)
    hf_policy_init_state: Dict[str, torch.Tensor] = {}
    if prov_backend == "hf":
        hf_policy_init_state = {k: v.detach().clone() for k, v in base_backend.model.state_dict().items()}

    # ==========================================================
    # method loop
    # ==========================================================
    for m in METHOD_SPECS:
        _set_seed(seed)

        # For HF, reuse already-loaded model (avoid reload per method).
        # For fallback, keep per-method fresh backend for "step actually runs" invariant.
        backend = base_backend if prov_backend == "hf" else FallbackBackend(cfg)

        # HF: reset weights every method (strict isolation)
        if prov_backend == "hf":
            backend.model.load_state_dict(hf_policy_init_state)

            # Best-effort: reset optimizer state (avoid momentum contamination)
            if hasattr(backend, "_ppo_optim"):
                backend._ppo_optim = torch.optim.AdamW(backend.model.parameters(), lr=ppo_lr)
            if hasattr(backend, "_sft_optim"):
                backend._sft_optim = torch.optim.AdamW(backend.model.parameters(), lr=sft_lr)

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

            if prov_backend == "fallback":
                # fallback: actually step
                texts = [f"{p} {c}".strip() for p, c in zip(prompts, completions)]
                loss = backend.sft_step(texts)
                extra["sft_loss"] = float(loss)
                extra["steps"] = 1
                extra["skipped"] = False
                extra["skip_reason"] = ""
            else:
                # HF Step2: run SFT only when enabled
                if hf_sft_steps > 0:
                    texts = [f"{p} {c}".strip() for p, c in zip(prompts, completions)]
                    last_loss = 0.0
                    for _ in range(int(hf_sft_steps)):
                        loss = backend.sft_step(texts)
                        last_loss = float(loss)

                    extra["sft_loss"] = float(last_loss)
                    extra["steps"] = int(hf_sft_steps)
                    extra["skipped"] = False
                    extra["skip_reason"] = ""

                    # Store "after SFT" behavior in artifacts body (audit-friendly)
                    completions = backend.generate(prompts, max_new_tokens=max_new)
                    rewards = rm.score(prompts, completions)
                else:
                    extra["sft_loss"] = 0.0
                    extra["steps"] = 0
                    extra["skipped"] = True
                    extra["skip_reason"] = "hf_step1_generation_only"

            extra["kl"] = 0.0
            dataset_key = "sft_train"

        # ----------------------------------------------------------
        # PPO family
        # ----------------------------------------------------------
        elif m.key in _PPO_METHOD_KEYS:
            completions = list(ppo_base_completions)
            rewards = list(ppo_base_rewards)

            kl_beta_eff = float(kl_beta_map.get(m.key, kl_beta_base))
            if m.key == "kl_ppo_adaptive":
                # allow presets to override initial beta (HF-focused, safe default)
                kl_beta_eff = float(train_cfg.get("kl_beta_init", kl_beta_eff))

            if use_fallback_ppo:
                backend.model.load_state_dict(ppo_policy_init_state)

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

                # Raw dump + standardized audit stamping (Table 2-A SSOT)
                extra.update(last_out)
                _stamp_ppo_audit_from_fallback(extra, last_out)

                # Table 1 KL policy: prefer nonnegative proxy for stability (align with HF)
                extra["kl"] = float(extra.get("kl_ref_abs", 0.0))

                extra["steps"] = int(ppo_steps)
                extra["kl_beta"] = float(kl_beta_eff)
                extra["skipped"] = False
                extra["skip_reason"] = ""

                print(
                    f"[run] method={m.key} kl_beta={kl_beta_eff} steps={ppo_steps} "
                    f"ppo_out_keys={sorted(list(last_out.keys()))}"
                )
            else:
                # HF: enable minimal PPO for selected methods when explicitly requested.
                extra["ppo_clip"] = float(ppo_clip)
                extra["ppo_lr"] = float(ppo_lr)

                # Ensure numeric placeholders exist (no empty cells in downstream rendering)
                for k in (
                    "ppo_loss",
                    "ratio_mean",
                    "clipfrac",
                    "kl_ref",
                    "ratio_mean_pre",
                    "kl_ref_pre",
                    "kl_ref_abs",
                    "kl_ref_sq",
                    # adaptive beta audit placeholders
                    "beta_pre",
                    "beta_post",
                    "kl_target",
                    "kl_measured",
                    "beta_lr",
                    "beta_clip",
                    "beta_min",
                    "beta_max",
                    "beta_ratio",
                    "beta_adj",
                ):
                    extra.setdefault(k, 0.0)

                if (m.key in hf_ppo_method_keys) and (hf_ppo_steps > 0):
                    # --- pre snapshot (C8) ---
                    completions_pre = list(completions)
                    rewards_pre = list(rewards)

                    # C8++: distribution shift audit (same completions, pre)
                    try:
                        lp_pre_mean = _mean(backend.logprobs(ppo_prompts, completions_pre))
                    except Exception:
                        lp_pre_mean = 0.0

                    abs_pre, sq_pre = _model_checksum(backend.model)

                    last_out: Dict[str, float] = {}

                    beta_init = float(kl_beta_eff)
                    beta_audit_last: Dict[str, float] = {}

                    for _ in range(int(hf_ppo_steps)):
                        beta_pre = float(kl_beta_eff)

                        out = backend.ppo_step(
                            prompts=ppo_prompts,
                            completions=completions,
                            rewards=rewards,
                            kl_beta=beta_pre,
                            ref_state=None,
                            update_ref=False,
                        )

                        last_out = {k: float(v) for k, v in out.items()}

                        # C2b: adaptive beta update (audit-first)
                        if m.key == "kl_ppo_adaptive":
                            beta_post, beta_audit = _kl_beta_adapt_update(beta_pre, out, train_cfg)
                            kl_beta_eff = float(beta_post)
                            beta_audit_last = beta_audit

                    abs_post, sq_post = _model_checksum(backend.model)

                    # C8++: distribution shift audit (same completions, post)
                    try:
                        lp_post_mean = _mean(backend.logprobs(ppo_prompts, completions_pre))
                    except Exception:
                        lp_post_mean = lp_pre_mean

                    extra.update(last_out)
                    extra["steps"] = int(hf_ppo_steps)
                    extra["kl_beta"] = float(kl_beta_eff)
                    extra["skipped"] = False
                    extra["skip_reason"] = ""

                    # C2b: record beta audit fields (last step) + initial beta
                    if m.key == "kl_ppo_adaptive":
                        extra["beta_init"] = float(beta_init)
                        extra.update({k: float(v) for k, v in beta_audit_last.items()})

                    # Parameter-change audit (C8+)
                    extra["param_abs_sum_pre"] = float(abs_pre)
                    extra["param_abs_sum_post"] = float(abs_post)
                    extra["param_abs_sum_delta"] = float(abs_post - abs_pre)
                    extra["param_sq_sum_pre"] = float(sq_pre)
                    extra["param_sq_sum_post"] = float(sq_post)
                    extra["param_sq_sum_delta"] = float(sq_post - sq_pre)

                    # Logprob-shift audit (C8++)
                    extra["pre_logprob_mean_on_pre"] = float(lp_pre_mean)
                    extra["post_logprob_mean_on_pre"] = float(lp_post_mean)
                    extra["logprob_delta_mean_on_pre"] = float(lp_post_mean - lp_pre_mean)

                    # After update, regenerate completions for artifacts (C8: post snapshot)
                    completions_post = backend.generate(ppo_prompts, max_new_tokens=max_new)
                    rewards_post = rm.score(ppo_prompts, completions_post)

                    extra["pre_reward_mean"] = _mean(rewards_pre)
                    extra["post_reward_mean"] = _mean(rewards_post)
                    extra["reward_delta_mean"] = float(extra["post_reward_mean"] - extra["pre_reward_mean"])

                    n = max(1, len(completions_pre))
                    changed = sum(
                        1
                        for a, b in zip(completions_pre, completions_post)
                        if (a or "").strip() != (b or "").strip()
                    )
                    extra["completion_changed_frac"] = float(changed / n)

                    # Store "after PPO" behavior in artifacts body
                    completions = list(completions_post)
                    rewards = list(rewards_post)

                    # Populate kl metric (C1.6+): prefer nonnegative proxy for report stability
                    extra["kl"] = _choose_nonneg_kl_proxy(extra)

                    if abs(extra.get("param_abs_sum_delta", 0.0)) < 1e-9 and abs(extra.get("param_sq_sum_delta", 0.0)) < 1e-9:
                        print("[run][warn] HF PPO produced ~0 parameter change (check optimizer/lr/step application)")

                    print(
                        f"[run] method={m.key} hf_ppo_steps={hf_ppo_steps} "
                        f"ppo_out_keys={sorted(list(last_out.keys()))}"
                    )
                else:
                    # HF Step1: placeholders only
                    extra["kl"] = 0.0
                    extra["steps"] = 0
                    extra["kl_beta"] = float(kl_beta_eff)
                    extra["skipped"] = True
                    extra["skip_reason"] = "hf_step1_generation_only"

        # ----------------------------------------------------------
        # preference-based + others
        # ----------------------------------------------------------
        else:
            completions = backend.generate(prompts, max_new_tokens=max_new)
            rewards = rm.score(prompts, completions)

            dataset_key = "pref_train" if m.is_preference_based else "comparisons"

            if prov_backend == "fallback":
                # fallback: actually step
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
                extra["skipped"] = False
                extra["skip_reason"] = ""
            else:
                # HF Step1: placeholders only (generation+reward is real; training is skipped)
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
