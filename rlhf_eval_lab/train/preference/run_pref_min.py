# rlhf_eval_lab/train/preference/run_pref_min.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Any, Dict

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "HF backend requires transformers. "
        "In level-c-research this is allowed; DoD (fallback) is unaffected."
    ) from e

from .batch import load_prompts_jsonl, load_prefs_jsonl, make_pref_pairs, make_batch
from .logprob import debug_check_masks
from .trainers.base import clip_gradients_, global_grad_norm_l2
from .trainers.dpo import DPOTrainer
from .trainers.ipo import IPOTrainer


def _sha256_of_dict(d: Dict[str, Any]) -> str:
    b = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefs", type=str, required=True)
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--method", type=str, choices=["dpo", "ipo"], required=True)

    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.1)  # kept for compatibility; not used in min runner
    p.add_argument("--limit", type=int, default=50)

    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
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

    _set_seed(args.seed)
    device = "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    # pad handling for GPT2-like
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad/eos token; cannot proceed safely.")
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.train()

    # ref_model fixed snapshot
    ref_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    if args.method == "dpo":
        trainer = DPOTrainer(beta=args.beta)
    else:
        trainer = IPOTrainer(beta=args.beta)

    prompts = load_prompts_jsonl(args.prompts)
    prefs_rows = load_prefs_jsonl(args.prefs)
    pairs = make_pref_pairs(prompts, prefs_rows, limit=args.limit, seed=args.seed)

    if len(pairs) == 0:
        raise RuntimeError("No valid preference pairs after filtering. Check prompts/prefs format.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Minimal training loop
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
            # sanity check masks (chosen/rejected)
            debug_check_masks(batch.attn_mask_chosen, batch.prompt_lens_chosen, name="chosen")
            debug_check_masks(batch.attn_mask_rejected, batch.prompt_lens_rejected, name="rejected")

        opt.zero_grad(set_to_none=True)
        out = trainer.compute_loss(model=model, ref_model=ref_model, batch=batch)
        out.loss.backward()

        # -----------------------------------------
        # Gradient clipping (global norm) + metrics
        # -----------------------------------------
        clip_m = clip_gradients_(
            model=model,
            max_grad_norm=float(args.max_grad_norm),
            enabled=bool(args.grad_clip),
            log=True,
        )

        # Debug probe for step==0 (now consistent with global norm)
        if step == 0:
            gn = global_grad_norm_l2(model.parameters())
            print(f"[debug:grad] global_grad_norm_l2={gn:.6f}")

        opt.step()

        # Merge metrics: trainer metrics + clip metrics
        last_metrics = dict(out.metrics)
        last_metrics["loss"] = float(out.loss.detach().cpu().item())
        last_metrics["grad_norm_pre"] = float(clip_m.grad_norm_pre)
        last_metrics["grad_norm_post"] = float(clip_m.grad_norm_post)
        # metrics is Dict[str,float], store as 0.0/1.0
        last_metrics["did_clip"] = 1.0 if clip_m.did_clip else 0.0

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
            "lr": args.lr,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "min_comp_tokens": args.min_comp_tokens,
            "allow_short_completion": args.allow_short_completion,
            "grad_clip": bool(args.grad_clip),
            "max_grad_norm": float(args.max_grad_norm),
        },
        "metrics": last_metrics,
        "config_hash": _sha256_of_dict(
            {
                "method": args.method,
                "model": args.model,
                "seed": args.seed,
                "beta": args.beta,
                "lr": args.lr,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "min_comp_tokens": args.min_comp_tokens,
                "allow_short_completion": args.allow_short_completion,
                "grad_clip": bool(args.grad_clip),
                "max_grad_norm": float(args.max_grad_norm),
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
