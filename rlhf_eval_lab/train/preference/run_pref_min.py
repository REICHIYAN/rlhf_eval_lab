# =============================================================================
# RLAIF-min (Heuristic AI-feedback pseudo-labeling)
#
# This implementation follows the core RLAIF idea:
#   - Use the standard preference-pair format (prompt, response_a, response_b),
#   - Replace the chosen/rejected label using AI feedback instead of human labels.
#
# In the RLAIF paper (arXiv:2309.00267), preference labels are produced by an
# off-the-shelf LLM judge. Here, we use a deterministic heuristic reward function
# as a reproducible surrogate for that AI judge.
#
# Note:
#   - Canonical RLAIF trains a reward model and then applies RL.
#   - This module skips RM training and RL, and directly applies
#     preference-loss optimization (DPO / RRHF / ORPO-style) on AI-labeled pairs.
#
# This can be summarized as:
#   "RLAIF-style AI preference labeling + preference-loss training",
# prioritizing reproducibility and minimal-but-correct learning mechanics.
#
# IMPORTANT:
#   This comment block refers to the behavior when running with:
#     --method rlaif
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "HF backend requires transformers. "
        "In level-c-research this is allowed; DoD (fallback) is unaffected."
    ) from e

from .batch import load_prefs_jsonl, load_prompts_jsonl, make_batch, make_pref_pairs
from .logprob import debug_check_masks
from .trainers.base import clip_gradients_, global_grad_norm_l2
from .trainers.dpo import DPOTrainer
from .trainers.ipo import IPOTrainer
from .trainers.orpo import ORPOTrainer
from .trainers.rrhf import RRHFTrainer


# -----------------------------------------------------------------------------
# Heuristic RM import (best-effort) + deterministic fallback implementation
# -----------------------------------------------------------------------------
# We DO NOT assume the module path. We attempt a few plausible locations.
# If none exists, we provide an in-file deterministic implementation that
# matches the required batch API:
#   score(prompts: List[str], completions: List[str]) -> List[float]
#
# This guarantees `--method rlaif` can run end-to-end without external deps.
# -----------------------------------------------------------------------------
_HeuristicRMImportError: Exception | None = None
HeuristicRewardModel: Any

try:  # 1) same package (if you later add it)
    from .heuristic_reward import HeuristicRewardModel as _HRM  # type: ignore
    HeuristicRewardModel = _HRM
except Exception as e1:  # pragma: no cover
    _HeuristicRMImportError = e1
    try:  # 2) train.reward_models.*
        from ..reward_models.heuristic_reward import HeuristicRewardModel as _HRM  # type: ignore
        HeuristicRewardModel = _HRM
        _HeuristicRMImportError = None
    except Exception as e2:  # pragma: no cover
        _HeuristicRMImportError = e2
        try:  # 3) train.selection.* etc (repo may differ)
            from ..reward_models.heuristic import HeuristicRewardModel as _HRM  # type: ignore
            HeuristicRewardModel = _HRM
            _HeuristicRMImportError = None
        except Exception as e3:  # pragma: no cover
            _HeuristicRMImportError = e3

if _HeuristicRMImportError is not None:
    class HeuristicRewardModel:  # type: ignore[no-redef]
        """
        Deterministic heuristic reward model (fallback).

        Purpose:
          - Provide a reproducible "AI feedback" signal to relabel preference pairs
            for RLAIF-min.
          - Must be stable across CPU/Colab/CI.

        API (required):
          score(prompts: List[str], completions: List[str]) -> List[float]

        Heuristic (minimal but sane):
          + reward longer, non-empty completions
          + reward lexical diversity (unique token ratio)
          - penalize excessive repetition (character 4-gram repeats)
          - penalize too many URLs / boilerplate markers
        """
        def __init__(self, seed: int = 0) -> None:
            self.seed = int(seed)

        @staticmethod
        def _tokenize(text: str) -> List[str]:
            return [t for t in re.split(r"\s+", text.strip()) if t]

        @staticmethod
        def _repeat_penalty(text: str) -> float:
            t = text
            if len(t) < 12:
                return 0.0
            grams = [t[i : i + 4] for i in range(0, len(t) - 3)]
            freq: Dict[str, int] = {}
            for g in grams:
                freq[g] = freq.get(g, 0) + 1
            repeats = sum(max(0, c - 1) for c in freq.values())
            return float(repeats)

        def score(self, prompts: List[str], completions: List[str]) -> List[float]:
            if not isinstance(prompts, list) or not isinstance(completions, list):
                raise TypeError("score expects prompts/completions as List[str].")
            if len(prompts) != len(completions):
                raise ValueError("score expects prompts and completions to have the same length.")

            outs: List[float] = []
            for p, c in zip(prompts, completions):
                _ = p  # prompt currently unused (kept for signature compatibility)

                text = c or ""
                toks = self._tokenize(text)
                n_tok = len(toks)

                if n_tok == 0:
                    outs.append(-1e9)
                    continue

                uniq = len(set(toks))
                uniq_ratio = uniq / max(1, n_tok)

                rep_pen = self._repeat_penalty(text)
                url_pen = 1.0 if ("http://" in text or "https://" in text) else 0.0
                boiler_pen = 1.0 if ("As an AI" in text or "I can't" in text) else 0.0

                score = 0.0
                score += 0.02 * float(len(text))
                score += 2.0 * float(uniq_ratio)
                score -= 0.01 * float(rep_pen)
                score -= 0.5 * float(url_pen + boiler_pen)

                outs.append(float(score))

            return outs


def _instantiate_heuristic_rm(seed: int) -> Any:
    """
    Instantiate HeuristicRewardModel robustly.

    We cannot assume ctor signature across implementations.
    We try:
      1) HeuristicRewardModel()
      2) HeuristicRewardModel(seed)
      3) HeuristicRewardModel(seed=seed)
    """
    try:
        return HeuristicRewardModel()
    except TypeError:
        pass

    try:
        return HeuristicRewardModel(seed)
    except TypeError:
        pass

    try:
        return HeuristicRewardModel(seed=seed)
    except TypeError as e:
        raise RuntimeError(
            "Failed to instantiate HeuristicRewardModel. "
            "Tried: (), (seed), (seed=seed). "
            f"Last error: {e}"
        ) from e


def _sha256_of_dict(d: Dict[str, Any]) -> str:
    b = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# RLAIF helpers (batch-API SSOT)
# ---------------------------------------------------------------------
def _rlaif_score_pair_with_heuristic_rm(
    *,
    rm: Any,
    prompt: str,
    a: str,
    b: str,
) -> Tuple[float, float]:
    """
    Score two candidate responses with a heuristic RM.

    REQUIRED interface:
        rm.score(prompts: List[str], completions: List[str]) -> List[float]
    """
    scores = rm.score([prompt, prompt], [a, b])
    if not isinstance(scores, list) or len(scores) != 2:
        raise RuntimeError(
            "HeuristicRewardModel.score must return List[float] of length 2 "
            f"(got {type(scores)} len={getattr(scores, '__len__', lambda: 'N/A')()})"
        )
    return float(scores[0]), float(scores[1])


def _rlaif_pseudo_label_pairs(
    pairs: List[Any],
    rm: Any,
) -> Tuple[List[Any], Dict[str, float]]:
    """
    Given preference pairs (prompt, chosen, rejected), overwrite chosen/rejected
    direction using heuristic RM preference.

    We keep the container schema unchanged; we only flip (chosen, rejected) when
    the RM prefers the other response.

    Returns:
      new_pairs, stats:
        - rlaif_flip_rate
        - rlaif_rm_margin_mean (abs margin after relabel; >= 0)
    """
    flips = 0
    margins: List[float] = []
    out: List[Any] = []

    for it in pairs:
        prompt = getattr(it, "prompt")
        chosen = getattr(it, "chosen")
        rejected = getattr(it, "rejected")

        s_c, s_r = _rlaif_score_pair_with_heuristic_rm(rm=rm, prompt=prompt, a=chosen, b=rejected)

        if s_r > s_c:
            flips += 1
            try:
                it2 = replace(it, chosen=rejected, rejected=chosen)
            except Exception:
                try:
                    it2 = replace(it)
                    setattr(it2, "chosen", rejected)
                    setattr(it2, "rejected", chosen)
                except Exception:
                    it2 = it
                    setattr(it2, "chosen", rejected)
                    setattr(it2, "rejected", chosen)
            margins.append(float(s_r - s_c))
            out.append(it2)
        else:
            margins.append(float(s_c - s_r))
            out.append(it)

    n = max(len(pairs), 1)
    stats = {
        "rlaif_flip_rate": float(flips) / float(n),
        "rlaif_rm_margin_mean": float(sum(margins) / float(len(margins))) if margins else 0.0,
    }
    return out, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefs", type=str, required=True)
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--method", type=str, choices=["dpo", "ipo", "rrhf", "orpo", "rlaif"], required=True)

    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.1)  # kept for compatibility; not used in min runner
    p.add_argument("--limit", type=int, default=50)

    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--orpo_alpha", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--min_comp_tokens", type=int, default=1)
    p.add_argument("--allow_short_completion", action="store_true", default=True)
    p.add_argument("--no_allow_short_completion", dest="allow_short_completion", action="store_false")

    # -------------------------
    # Gradient clipping (stability backbone)
    # -------------------------
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--no_grad_clip", dest="grad_clip", action="store_false")
    p.set_defaults(grad_clip=True)

    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    if args.beta <= 0:
        raise ValueError("beta must be > 0")
    if args.lr <= 0:
        raise ValueError("lr must be > 0")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.max_length <= 0:
        raise ValueError("max_length must be > 0")
    if args.min_comp_tokens < 0:
        raise ValueError("min_comp_tokens must be >= 0")
    if args.grad_clip and args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be > 0 when grad_clip is enabled")
    if args.orpo_alpha < 0:
        raise ValueError("orpo_alpha must be >= 0")

    _set_seed(args.seed)
    device = "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad/eos token; cannot proceed safely.")
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.train()

    ref_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    if args.method == "dpo":
        trainer = DPOTrainer(beta=args.beta)
    elif args.method == "ipo":
        trainer = IPOTrainer(beta=args.beta)
    elif args.method == "rrhf":
        trainer = RRHFTrainer(beta=args.beta)
    elif args.method == "orpo":
        trainer = ORPOTrainer(beta=args.beta, alpha=args.orpo_alpha)
    else:
        trainer = RRHFTrainer(beta=args.beta)

    prompts = load_prompts_jsonl(args.prompts)
    prefs_rows = load_prefs_jsonl(args.prefs)
    pairs = make_pref_pairs(prompts, prefs_rows, limit=args.limit, seed=args.seed)

    if len(pairs) == 0:
        raise RuntimeError("No valid preference pairs after filtering. Check prompts/prefs format.")

    # -------------------------
    # RLAIF-min: overwrite labels by heuristic RM preference
    # -------------------------
    rlaif_stats: Dict[str, float] = {}
    if args.method == "rlaif":
        rm = _instantiate_heuristic_rm(args.seed)
        pairs, rlaif_stats = _rlaif_pseudo_label_pairs(pairs, rm=rm)
        print(
            "[rlaif] "
            f"flip_rate={rlaif_stats.get('rlaif_flip_rate', 0.0):.6f} "
            f"rm_margin_mean={rlaif_stats.get('rlaif_rm_margin_mean', 0.0):.6f}"
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    last_metrics: Dict[str, float] = {}
    for batch in make_batch(
        tok,
        pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        min_comp_tokens=args.min_comp_tokens,
        allow_short_completion=args.allow_short_completion,
        max_attempts_per_sample=50,
        seed=args.seed,
        device=device,
    ):
        if step >= args.max_steps:
            break

        if step == 0:
            debug_check_masks(batch.attn_mask_chosen, batch.prompt_lens_chosen, name="chosen")
            debug_check_masks(batch.attn_mask_rejected, batch.prompt_lens_rejected, name="rejected")

        opt.zero_grad(set_to_none=True)
        out = trainer.compute_loss(model=model, ref_model=ref_model, batch=batch)
        out.loss.backward()

        clip_m = clip_gradients_(
            model=model,
            max_grad_norm=float(args.max_grad_norm),
            enabled=bool(args.grad_clip),
            log=True,
        )

        if step == 0:
            gn = global_grad_norm_l2(model.parameters())
            print(f"[debug:grad] global_grad_norm_l2={gn:.6f}")

        opt.step()

        last_metrics = dict(out.metrics)
        last_metrics["loss"] = float(out.loss.detach().cpu().item())
        last_metrics["grad_norm_pre"] = float(clip_m.grad_norm_pre)
        last_metrics["grad_norm_post"] = float(clip_m.grad_norm_post)
        last_metrics["did_clip"] = 1.0 if clip_m.did_clip else 0.0

        if args.method == "rlaif":
            last_metrics["rlaif_flip_rate"] = float(rlaif_stats.get("rlaif_flip_rate", 0.0))
            last_metrics["rlaif_rm_margin_mean"] = float(rlaif_stats.get("rlaif_rm_margin_mean", 0.0))

        msg = " ".join([f"{k}={v:.6f}" for k, v in last_metrics.items()])
        print(f"[train] step={step} {msg}")

        step += 1

    if step == 0:
        raise RuntimeError("Training loop did not run any steps. Check batch construction constraints.")

    payload: Dict[str, Any] = {
        "provenance": {
            "method": args.method,
            "backend": "hf",
            "model": args.model,
            "seed": args.seed,
            "beta": args.beta,
            "orpo_alpha": float(args.orpo_alpha),
            "lr": args.lr,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "min_comp_tokens": args.min_comp_tokens,
            "allow_short_completion": args.allow_short_completion,
            "grad_clip": bool(args.grad_clip),
            "max_grad_norm": float(args.max_grad_norm),
            "rlaif_flip_rate": float(rlaif_stats.get("rlaif_flip_rate", 0.0)),
            "rlaif_rm_margin_mean": float(rlaif_stats.get("rlaif_rm_margin_mean", 0.0)),
            "rlaif_label_source": "heuristic_rm" if args.method == "rlaif" else "",
        },
        "metrics": last_metrics,
        "config_hash": _sha256_of_dict(
            {
                "method": args.method,
                "model": args.model,
                "seed": args.seed,
                "beta": args.beta,
                "orpo_alpha": float(args.orpo_alpha),
                "lr": args.lr,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "min_comp_tokens": args.min_comp_tokens,
                "allow_short_completion": args.allow_short_completion,
                "grad_clip": bool(args.grad_clip),
                "max_grad_norm": float(args.max_grad_norm),
                "rlaif_label_source": "heuristic_rm" if args.method == "rlaif" else "",
            }
        ),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote: {args.out}")
    print(f"[ok] final_loss={payload['metrics'].get('loss', None)}")


if __name__ == "__main__":
    main()
