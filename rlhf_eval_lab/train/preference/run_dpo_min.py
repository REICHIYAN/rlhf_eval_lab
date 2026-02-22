# /mnt/c/Users/s5aba/Projects/rlhf_eval_lab/rlhf_eval_lab/train/preference/run_dpo_min.py
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "HF backend requires transformers. "
        "In level-c-research this is allowed; DoD (fallback) is unaffected."
    ) from e

from rlhf_eval_lab.train.preference.debug_adapter import (
    debug_logp_sanity,
    debug_mask_sanity,
)

# =============================================================================
# Design note (CRITICAL / DoD):
#
# - config_hash MUST represent run-level invariants only.
# - Method-specific hyperparameters (beta, lr, batch_size, etc.)
#   MUST NOT be included in config_hash.
# - Violating this causes MIXED provenance and breaks report.md validation.
# =============================================================================


# -------------------------
# Data I/O (jsonl)
# -------------------------
@dataclasses.dataclass(frozen=True)
class PrefExample:
    uid: str
    prompt: str
    chosen: str
    rejected: str


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_hh_like_pairs(prefs_path: str, prompts_path: str) -> List[PrefExample]:
    prefs = _read_jsonl(prefs_path)
    prompts = _read_jsonl(prompts_path)

    prompt_by_id: Dict[str, str] = {}
    bad_prompts = 0
    for r in prompts:
        if "id" not in r or "prompt" not in r:
            bad_prompts += 1
            continue
        pid = str(r["id"])
        ptxt = str(r["prompt"])
        if pid and ptxt:
            prompt_by_id[pid] = ptxt
        else:
            bad_prompts += 1

    out: List[PrefExample] = []
    skipped = 0
    missing_prompt = 0
    bad_rows = 0

    for r in prefs:
        try:
            uid = str(r["id"])
            pid = str(r["prompt_id"])
            chosen = str(r["chosen"])
            rejected = str(r["rejected"])
        except Exception:
            bad_rows += 1
            continue

        prompt = prompt_by_id.get(pid, "")
        if not prompt:
            missing_prompt += 1
            skipped += 1
            continue

        if not uid or not chosen or not rejected:
            skipped += 1
            continue

        out.append(PrefExample(uid=uid, prompt=prompt, chosen=chosen, rejected=rejected))

    if not out:
        raise ValueError(
            "No valid preference pairs loaded. "
            f"prompts_loaded={len(prompt_by_id)} bad_prompts={bad_prompts} "
            f"prefs_rows={len(prefs)} bad_prefs_rows={bad_rows} "
            f"missing_prompt={missing_prompt} skipped={skipped}"
        )

    print(
        f"[data] loaded_pairs={len(out)} prompts_loaded={len(prompt_by_id)} "
        f"bad_prompts={bad_prompts} prefs_rows={len(prefs)} "
        f"bad_prefs_rows={bad_rows} missing_prompt={missing_prompt} skipped={skipped}"
    )
    return out


def deterministic_split(
    examples: List[PrefExample], *, seed: int, val_ratio: float, limit: int
) -> Tuple[List[PrefExample], List[PrefExample]]:
    xs = examples[:]
    rng = random.Random(seed)
    rng.shuffle(xs)
    if limit > 0:
        xs = xs[:limit]
    n = len(xs)
    n_val = int(round(n * val_ratio))
    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if n_val >= n and n > 1:
        n_val = n - 1
    val = xs[:n_val]
    train = xs[n_val:]
    if not train:
        train = val[:1]
        val = val[1:]
    return train, val


# -------------------------
# Tokenization + completion-only logprob
# -------------------------
def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def encode_prompt_completion(
    tokenizer,
    prompts: List[str],
    completions: List[str],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    assert len(prompts) == len(completions)
    ensure_pad_token(tokenizer)

    prompt_basis = [p + "\n" for p in prompts]
    joint_texts = [p + "\n" + c for p, c in zip(prompts, completions)]

    enc_prompt = tokenizer(
        prompt_basis,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    enc = tokenizer(
        joint_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )

    enc = {k: v.to(device) for k, v in enc.items()}
    enc_prompt = {k: v.to(device) for k, v in enc_prompt.items()}

    prompt_lens = enc_prompt["attention_mask"].sum(dim=1)

    attn = enc["attention_mask"]
    B, T = enc["input_ids"].shape
    completion_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    for i in range(B):
        pl = int(prompt_lens[i].item())
        if pl < T:
            completion_mask[i, pl:] = 1
        completion_mask[i] *= attn[i]

    return enc, completion_mask


def logprob_sum_completion_only(
    model, enc: Dict[str, torch.Tensor], completion_mask: torch.Tensor
) -> torch.Tensor:
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    logits = model(input_ids=input_ids, attention_mask=attn).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn = attn[:, 1:]
    shift_mask = completion_mask[:, 1:]

    logp = F.log_softmax(shift_logits, dim=-1)
    token_logp = torch.gather(logp, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    eff = shift_attn * shift_mask
    return (token_logp * eff).sum(dim=1)


def completion_token_count(
    enc: Dict[str, torch.Tensor], completion_mask: torch.Tensor
) -> torch.Tensor:
    shift_attn = enc["attention_mask"][:, 1:]
    shift_mask = completion_mask[:, 1:]
    return (shift_attn * shift_mask).sum(dim=1)


# -------------------------
# DPO core
# -------------------------
def dpo_loss_and_delta(
    *,
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    loss = -F.logsigmoid(beta * delta)
    return loss.mean(), delta


def run_config_hash(args: argparse.Namespace) -> str:
    """
    Run-level invariant hash.
    MUST stay identical across methods to avoid MIXED provenance.
    """
    payload = {
        "backend": "hf",
        "model": "gpt2",
        "seed": args.seed,
        "max_length": 256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# -------------------------
# Main
# -------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefs", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--backend", choices=["hf"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--val_ratio", type=float, required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_pad_token(tokenizer)

    policy = AutoModelForCausalLM.from_pretrained(model_name).to(device).train()
    ref = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    examples = load_hh_like_pairs(args.prefs, args.prompts)
    train, val = deterministic_split(
        examples, seed=args.seed, val_ratio=args.val_ratio, limit=args.limit
    )
    print(f"[data] train={len(train)} val={len(val)} device={device.type}")

    curve: List[Dict[str, float]] = []
    reward_margins: List[float] = []

    rng = random.Random(args.seed)
    min_comp_tokens = 1
    max_attempts_per_step = 50

    for step in range(args.max_steps):
        attempt = 0
        while True:
            attempt += 1
            if attempt > max_attempts_per_step:
                raise RuntimeError("Too many attempts to sample a valid batch")

            batch = (
                rng.sample(train, k=args.batch_size)
                if args.batch_size <= len(train)
                else [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
            )

            prompts = [b.prompt for b in batch]
            chosen = [b.chosen for b in batch]
            rejected = [b.rejected for b in batch]

            enc_c, mask_c = encode_prompt_completion(tokenizer, prompts, chosen, device)
            enc_r, mask_r = encode_prompt_completion(tokenizer, prompts, rejected, device)

            if (
                completion_token_count(enc_c, mask_c).min() < min_comp_tokens
                or completion_token_count(enc_r, mask_r).min() < min_comp_tokens
            ):
                continue
            break

        if step == 0:
            debug_mask_sanity(enc_c, mask_c, "chosen")
            debug_mask_sanity(enc_r, mask_r, "rejected")

        pi_c = logprob_sum_completion_only(policy, enc_c, mask_c)
        pi_r = logprob_sum_completion_only(policy, enc_r, mask_r)
        with torch.no_grad():
            ref_c = logprob_sum_completion_only(ref, enc_c, mask_c)
            ref_r = logprob_sum_completion_only(ref, enc_r, mask_r)

        if step == 0:
            debug_logp_sanity("pi_chosen", pi_c)
            debug_logp_sanity("pi_rejected", pi_r)
            debug_logp_sanity("ref_chosen", ref_c)
            debug_logp_sanity("ref_rejected", ref_r)

        loss, delta = dpo_loss_and_delta(
            pi_chosen=pi_c,
            pi_rejected=pi_r,
            ref_chosen=ref_c,
            ref_rejected=ref_r,
            beta=args.beta,
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        reward_margin = float((pi_c - pi_r).mean().item())
        reward_margins.append(reward_margin)
        curve.append(
            {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "reward_margin": reward_margin,
            }
        )

        if step in (0, args.max_steps - 1):
            print(f"[curve] step={step} loss={loss.item():.6f} reward_margin={reward_margin:.6f}")

    payload = {
        "provenance": {
            "method": "dpo",
            "backend": "hf",
            "model": model_name,
            "tokenizer": tokenizer.name_or_path,
            "seed": args.seed,
            "config_hash": run_config_hash(args),
        },
        "metrics": {
            "final_loss": float(curve[-1]["loss"]),
            "reward_margin_mean": float(sum(reward_margins) / max(1, len(reward_margins))),
        },
        "curve": curve,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote: {args.out}")


if __name__ == "__main__":
    main()
