# rlhf_eval_lab/train/preference/run_pref_min.py
# =============================================================================
# Preference-min runner (DPO/IPO/RRHF/ORPO/RLAIF-min)
#
# IMPORTANT (DoD / Provenance design):
#   - `config_hash` MUST represent *run-level invariants only*.
#   - Method-specific hyperparameters (beta, lr, etc.) MUST NOT be included
#     in `config_hash`, otherwise report provenance becomes MIXED.
#   - Method-specific settings belong to `provenance` and `metrics`, not to
#     the run-level hash.
#
# This guarantees:
#   - report.md is self-auditable
#   - E2E never produces MIXED provenance
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
from .trainers.dpo import DPOTrainer
from .trainers.ipo import IPOTrainer
from .trainers.orpo import ORPOTrainer
from .trainers.rlaif import RLAIFTrainer
from .trainers.rrhf import RRHFTrainer


# -----------------------------------------------------------------------------
# Heuristic RM (deterministic, dependency-free)
# -----------------------------------------------------------------------------
class HeuristicRewardModel:
    """
    Deterministic heuristic reward model.

    Purpose:
      - Provide reproducible AI-style preference signals (RLAIF-min)
      - Stable across CPU / Colab / CI
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.split(r"\s+", text.strip()) if t]

    @staticmethod
    def _repeat_penalty(text: str) -> float:
        if len(text) < 12:
            return 0.0
        grams = [text[i : i + 4] for i in range(len(text) - 3)]
        freq: Dict[str, int] = {}
        for g in grams:
            freq[g] = freq.get(g, 0) + 1
        return float(sum(max(0, c - 1) for c in freq.values()))

    def score(self, prompts: List[str], completions: List[str]) -> List[float]:
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have same length")

        outs: List[float] = []
        for c in completions:
            text = c or ""
            toks = self._tokenize(text)
            if not toks:
                outs.append(-1e9)
                continue

            uniq_ratio = len(set(toks)) / max(1, len(toks))
            score = 0.02 * len(text)
            score += 2.0 * uniq_ratio
            score -= 0.01 * self._repeat_penalty(text)
            score -= 0.5 * float("http://" in text or "https://" in text)
            score -= 0.5 * float("As an AI" in text or "I can't" in text)

            outs.append(float(score))
        return outs


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _sha256_of_dict(d: Dict[str, Any]) -> str:
    b = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_config_hash(args: argparse.Namespace) -> str:
    """
    Run-level invariant hash.

    MUST NOT include:
      - method
      - beta / lr / batch_size / etc.

    Purpose:
      - guarantee identical execution conditions across methods
      - avoid MIXED provenance in report.md
    """
    return _sha256_of_dict(
        {
            "backend": "hf",
            "model": args.model,
            "seed": args.seed,
            "max_length": args.max_length,
        }
    )


# -----------------------------------------------------------------------------
# RLAIF helpers (pseudo-labeling)
# -----------------------------------------------------------------------------
def _rlaif_score_pair(rm: HeuristicRewardModel, prompt: str, a: str, b: str) -> Tuple[float, float]:
    s = rm.score([prompt, prompt], [a, b])
    return float(s[0]), float(s[1])


def _rlaif_relabel_pairs(pairs: List[Any], rm: HeuristicRewardModel) -> Tuple[List[Any], Dict[str, float]]:
    flips = 0
    margins: List[float] = []
    out: List[Any] = []

    for it in pairs:
        s_c, s_r = _rlaif_score_pair(rm, it.prompt, it.chosen, it.rejected)
        if s_r > s_c:
            flips += 1
            it = replace(it, chosen=it.rejected, rejected=it.chosen)
            margins.append(s_r - s_c)
        else:
            margins.append(s_c - s_r)
        out.append(it)

    return out, {
        "rlaif_flip_rate": flips / max(1, len(pairs)),
        "rlaif_rm_margin_mean": sum(margins) / max(1, len(margins)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefs", type=str, required=True)
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--method", choices=["dpo", "ipo", "rrhf", "orpo", "rlaif"], required=True)

    p.add_argument("--model", default="gpt2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=50)

    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--orpo_alpha", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--min_comp_tokens", type=int, default=1)
    p.add_argument("--allow_short_completion", action="store_true", default=True)

    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    _set_seed(args.seed)
    device = "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).train()
    ref_model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    for p_ in ref_model.parameters():
        p_.requires_grad_(False)

    trainer = {
        "dpo": DPOTrainer,
        "ipo": IPOTrainer,
        "rrhf": RRHFTrainer,
        "orpo": lambda **kw: ORPOTrainer(alpha=args.orpo_alpha, **kw),
        "rlaif": RLAIFTrainer,
    }[args.method](beta=args.beta)

    prompts = load_prompts_jsonl(args.prompts)
    prefs = load_prefs_jsonl(args.prefs)
    pairs = make_pref_pairs(prompts, prefs, limit=args.limit, seed=args.seed)

    rlaif_stats: Dict[str, float] = {}
    if args.method == "rlaif":
        rm = HeuristicRewardModel(seed=args.seed)
        pairs, rlaif_stats = _rlaif_relabel_pairs(pairs, rm)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    last_metrics: Dict[str, float] = {}
    for step, batch in enumerate(
        make_batch(
            tok,
            pairs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            min_comp_tokens=args.min_comp_tokens,
            allow_short_completion=args.allow_short_completion,
            seed=args.seed,
            device=device,
        )
    ):
        if step >= args.max_steps:
            break

        opt.zero_grad(set_to_none=True)
        out = trainer.compute_loss(model=model, ref_model=ref_model, batch=batch)
        out.loss.backward()
        opt.step()

        last_metrics = dict(out.metrics)
        last_metrics["loss"] = float(out.loss.detach().cpu())

    payload = {
        "provenance": {
            "method": args.method,
            "backend": "hf",
            "model": args.model,
            "seed": args.seed,
            "config_hash": _run_config_hash(args),
            "rlaif_label_source": "heuristic_rm" if args.method == "rlaif" else "human",
            **rlaif_stats,
        },
        "metrics": last_metrics,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[ok] wrote: {args.out}")


if __name__ == "__main__":
    main()
