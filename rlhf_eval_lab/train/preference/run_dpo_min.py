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

from rlhf_eval_lab.train.preference.debug_adapter import debug_logp_sanity, debug_mask_sanity


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
    """
    Loads preference pairs by joining:
      prefs.jsonl:   {id, prompt_id, chosen, rejected}
      prompts.jsonl: {id, prompt}

    This matches the project's test_data schema (Day1 HH-RLHF I/O output).
    """
    prefs = _read_jsonl(prefs_path)
    prompts = _read_jsonl(prompts_path)

    # Build prompt lookup: prompts.id -> prompts.prompt
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
            "No valid preference pairs loaded (prompt_id join failed). "
            f"prompts_loaded={len(prompt_by_id)} bad_prompts={bad_prompts} "
            f"prefs_rows={len(prefs)} bad_prefs_rows={bad_rows} missing_prompt={missing_prompt} skipped={skipped}"
        )

    print(
        f"[data] loaded_pairs={len(out)} prompts_loaded={len(prompt_by_id)} "
        f"bad_prompts={bad_prompts} prefs_rows={len(prefs)} bad_prefs_rows={bad_rows} "
        f"missing_prompt={missing_prompt} skipped={skipped}"
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
    """
    Returns:
      enc: dict(input_ids, attention_mask)
      completion_mask: (B, T) binary mask for completion tokens only (suffix mask)

    Strategy (boundary-safe):
      - Use an explicit separator that is present in BOTH:
          (a) prompt-only texts
          (b) prompt+completion joint texts
        so that prompt_len is measured on the same tokenization basis as the joint.
      - completion tokens are positions >= prompt_len (in joint encoding, excluding padding)
    """
    assert len(prompts) == len(completions)
    ensure_pad_token(tokenizer)

    # Boundary disambiguation:
    # Use the exact same prompt basis for both prompt-only and joint.
    # This prevents tokenization boundary artifacts (prompt_len mismatch).
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

    prompt_attn = enc_prompt.get("attention_mask", torch.ones_like(enc_prompt["input_ids"]))
    prompt_lens = prompt_attn.sum(dim=1)  # (B,)

    attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
    B, T = enc["input_ids"].shape
    completion_mask = torch.zeros((B, T), dtype=torch.long, device=device)

    for i in range(B):
        pl = int(prompt_lens[i].item())
        if pl < T:
            completion_mask[i, pl:] = 1
        completion_mask[i] = completion_mask[i] * attn[i].to(dtype=torch.long)

    return enc, completion_mask


def logprob_sum_completion_only(model, enc: Dict[str, torch.Tensor], completion_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute sum of log p(x_t | x_<t) over completion tokens only.
    Returns: (B,) tensor
    """
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits  # (B, T, V)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn = attn[:, 1:]
    shift_mask = completion_mask[:, 1:]

    logp = F.log_softmax(shift_logits, dim=-1)
    token_logp = torch.gather(logp, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    eff = shift_mask.to(dtype=token_logp.dtype) * shift_attn.to(dtype=token_logp.dtype)  # (B, T-1)
    sums = (token_logp * eff).sum(dim=1)  # (B,)
    return sums


def completion_token_count(enc: Dict[str, torch.Tensor], completion_mask: torch.Tensor) -> torch.Tensor:
    """
    Count completion tokens on the SAME grid used by logprob_sum_completion_only:
      - shift to (B, T-1) via [:, 1:]
      - apply attention_mask
    Returns: (B,) long tensor
    """
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))
    shift_attn = attn[:, 1:]
    shift_mask = completion_mask[:, 1:]
    eff = shift_mask.to(dtype=torch.long) * shift_attn.to(dtype=torch.long)
    return eff.sum(dim=1)  # (B,)


# -------------------------
# DPO
# -------------------------
def dpo_loss_and_delta(
    *,
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    delta = (pi_c - pi_r) - (ref_c - ref_r)
    loss  = -log sigmoid(beta * delta)
    """
    delta = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    loss = -F.logsigmoid(torch.tensor(beta, device=delta.device, dtype=delta.dtype) * delta)
    return loss.mean(), delta


def compute_config_hash(args: argparse.Namespace) -> str:
    payload = {
        "prefs": os.path.basename(args.prefs),
        "prompts": os.path.basename(args.prompts),
        "backend": args.backend,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "limit": args.limit,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "beta": args.beta,
        "lr": args.lr,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefs", type=str, required=True)
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--backend", type=str, required=True, choices=["hf", "fallback"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--val_ratio", type=float, required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    if args.backend != "hf":
        raise RuntimeError(
            "This minimal research runner supports only --backend hf. "
            "DoD fallback is separate and unchanged."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    ensure_pad_token(tokenizer)

    policy = AutoModelForCausalLM.from_pretrained(model_name)
    ref = AutoModelForCausalLM.from_pretrained(model_name)

    policy.to(device)
    ref.to(device)

    ref.eval()
    for p_ in ref.parameters():
        p_.requires_grad = False

    policy.train()
    for p_ in policy.parameters():
        p_.requires_grad = True

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    examples = load_hh_like_pairs(args.prefs, args.prompts)
    train, val = deterministic_split(examples, seed=args.seed, val_ratio=args.val_ratio, limit=args.limit)
    print(f"[data] train={len(train)} val={len(val)} device={device.type}")

    curve: List[Dict[str, float]] = []
    reward_margins: List[float] = []

    rng = random.Random(args.seed)

    # Research safety: skip too-short completion examples (unstable logp).
    min_comp_tokens = 1
    # Avoid infinite loops when train is tiny and many samples are invalid under the threshold.
    max_attempts_per_step = 50

    for step in range(args.max_steps):
        # We may need multiple attempts to assemble a valid batch under min_comp_tokens.
        attempt = 0
        while True:
            attempt += 1
            if attempt > max_attempts_per_step:
                raise RuntimeError(
                    f"Too many attempts ({max_attempts_per_step}) to sample a valid batch. "
                    f"train={len(train)} batch_size={args.batch_size} min_comp_tokens={min_comp_tokens}. "
                    "Lower min_comp_tokens or increase training data."
                )

            # Avoid duplicate examples in a batch when possible (crucial for small train sets).
            if args.batch_size <= len(train):
                batch = rng.sample(train, k=args.batch_size)
            else:
                batch = [train[rng.randrange(0, len(train))] for _ in range(args.batch_size)]

            prompts = [b.prompt for b in batch]
            chosen = [b.chosen for b in batch]
            rejected = [b.rejected for b in batch]

            enc_chosen, mask_chosen = encode_prompt_completion(tokenizer, prompts, chosen, device=device)
            enc_rejected, mask_rejected = encode_prompt_completion(tokenizer, prompts, rejected, device=device)

            chosen_tok = completion_token_count(enc_chosen, mask_chosen)
            rejected_tok = completion_token_count(enc_rejected, mask_rejected)

            if int(chosen_tok.min().item()) < min_comp_tokens or int(rejected_tok.min().item()) < min_comp_tokens:
                if step == 0:
                    print(
                        f"[skip] too_short_completion attempt={attempt} "
                        f"chosen={chosen_tok.tolist()} rejected={rejected_tok.tolist()}"
                    )
                continue

            # Valid batch acquired
            break

        if step == 0:
            debug_mask_sanity(enc_chosen, mask_chosen, name="chosen")
            debug_mask_sanity(enc_rejected, mask_rejected, name="rejected")

            # Fingerprints: detect identical tokenized inputs across batch samples.
            ids_c = enc_chosen["input_ids"]
            ids_r = enc_rejected["input_ids"]
            print(f"[fp:chosen] sample0={int(ids_c[0].sum().item())} sample1={int(ids_c[1].sum().item())}")
            print(f"[fp:rejected] sample0={int(ids_r[0].sum().item())} sample1={int(ids_r[1].sum().item())}")

        pi_logp_chosen_sum = logprob_sum_completion_only(policy, enc_chosen, mask_chosen)
        pi_logp_rejected_sum = logprob_sum_completion_only(policy, enc_rejected, mask_rejected)

        with torch.no_grad():
            ref_logp_chosen_sum = logprob_sum_completion_only(ref, enc_chosen, mask_chosen)
            ref_logp_rejected_sum = logprob_sum_completion_only(ref, enc_rejected, mask_rejected)

        if step == 0:
            debug_logp_sanity(name="pi_chosen", logp_sum=pi_logp_chosen_sum)
            debug_logp_sanity(name="pi_rejected", logp_sum=pi_logp_rejected_sum)
            debug_logp_sanity(name="ref_chosen", logp_sum=ref_logp_chosen_sum)
            debug_logp_sanity(name="ref_rejected", logp_sum=ref_logp_rejected_sum)

            print(
                f"[shape] "
                f"pi_chosen={tuple(pi_logp_chosen_sum.shape)} "
                f"pi_rejected={tuple(pi_logp_rejected_sum.shape)} "
                f"ref_chosen={tuple(ref_logp_chosen_sum.shape)} "
                f"ref_rejected={tuple(ref_logp_rejected_sum.shape)}"
            )

            print(f"[len] chosen_tokens={chosen_tok.tolist()} rejected_tokens={rejected_tok.tolist()}")

            chosen_den = chosen_tok.clamp_min(1).to(dtype=pi_logp_chosen_sum.dtype)
            rejected_den = rejected_tok.clamp_min(1).to(dtype=pi_logp_rejected_sum.dtype)

            pi_chosen_mean = (pi_logp_chosen_sum / chosen_den).detach().float()
            pi_rejected_mean = (pi_logp_rejected_sum / rejected_den).detach().float()
            ref_chosen_mean = (ref_logp_chosen_sum / chosen_den).detach().float()
            ref_rejected_mean = (ref_logp_rejected_sum / rejected_den).detach().float()

            print(f"[logp_mean:pi_chosen] {pi_chosen_mean.tolist()}")
            print(f"[logp_mean:pi_rejected] {pi_rejected_mean.tolist()}")
            print(f"[logp_mean:ref_chosen] {ref_chosen_mean.tolist()}")
            print(f"[logp_mean:ref_rejected] {ref_rejected_mean.tolist()}")

        loss, delta = dpo_loss_and_delta(
            pi_chosen=pi_logp_chosen_sum,
            pi_rejected=pi_logp_rejected_sum,
            ref_chosen=ref_logp_chosen_sum,
            ref_rejected=ref_logp_rejected_sum,
            beta=args.beta,
        )

        if step == 0:
            d = delta.detach().float()
            print(f"[delta] mean={d.mean().item():.3f} min={d.min().item():.3f} max={d.max().item():.3f}")

        optim.zero_grad(set_to_none=True)
        loss.backward()

        if step == 0:
            total_abs = 0.0
            for p_ in policy.parameters():
                if p_.grad is not None:
                    total_abs += float(p_.grad.detach().abs().sum().item())
            print(f"[grad] abs_sum={total_abs:.6f}")
            assert total_abs > 0.0, "No gradients: policy is not learning (no_grad/detach/eval?)"

        optim.step()

        reward_margin = (pi_logp_chosen_sum - pi_logp_rejected_sum).detach().mean().item()
        reward_margins.append(float(reward_margin))

        curve.append(
            {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "reward_margin": float(reward_margin),
            }
        )

        if step in (0, 1, args.max_steps - 2, args.max_steps - 1):
            print(f"[curve] step={step} loss={loss.detach().item():.6f} reward_margin={reward_margin:.6f}")

    final_loss = float(curve[-1]["loss"])
    reward_margin_mean = float(sum(reward_margins) / max(1, len(reward_margins)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "kind": "level_c_dpo_min",
        "backend": args.backend,
        "seed": args.seed,
        "model": model_name,
        "tokenizer": getattr(tokenizer, "name_or_path", model_name),
        "config_hash": compute_config_hash(args),
        "beta": args.beta,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "train_size": len(train),
        "val_size": len(val),
        "min_comp_tokens": min_comp_tokens,
        "final_loss": final_loss,
        "reward_margin_mean": reward_margin_mean,
        "curve": curve,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote: {args.out}")
    print(f"[ok] final_loss={final_loss:.6f} reward_margin_mean={reward_margin_mean:.6f}")


if __name__ == "__main__":
    main()
