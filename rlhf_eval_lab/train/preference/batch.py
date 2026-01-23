# rlhf_eval_lab/train/preference/batch.py
from __future__ import annotations

import json
import random
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import PreTrainedTokenizerBase

from .types import Batch, EncodedPair, PrefPair


# -------------------------
# Robust field extraction
# -------------------------
def _stringify_if_possible(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, (int, float, bool)):
        return str(v)
    return None


def _extract_text_from_obj(obj: Any) -> Optional[str]:
    s = _stringify_if_possible(obj)
    if s is not None:
        return s

    if isinstance(obj, dict):
        for k in [
            "text",
            "content",
            "prompt",
            "completion",
            "answer",
            "response",
            "output",
            "input",
            "query",
            "instruction",
        ]:
            if k in obj:
                s2 = _stringify_if_possible(obj[k])
                if s2 is not None:
                    return s2
                s2 = _extract_text_from_obj(obj[k])
                if s2 is not None:
                    return s2

        if "message" in obj:
            s2 = _extract_text_from_obj(obj["message"])
            if s2 is not None:
                return s2

    return None


def _pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k not in d or d[k] is None:
            continue

        v = d[k]
        s = _stringify_if_possible(v)
        if s is not None:
            return s

        s = _extract_text_from_obj(v)
        if s is not None:
            return s

    return None


def _pick_uid_from_row(r: Dict[str, Any]) -> Optional[str]:
    # flat
    uid = _pick_first(r, ["uid", "id", "prompt_id", "promptId"])
    if uid is not None:
        return uid

    # nested prompt.id
    prompt_obj = r.get("prompt")
    if isinstance(prompt_obj, dict):
        uid = _pick_first(prompt_obj, ["uid", "id", "prompt_id", "promptId"])
        if uid is not None:
            return uid

    # nested meta.id
    meta_obj = r.get("meta")
    if isinstance(meta_obj, dict):
        uid = _pick_first(meta_obj, ["uid", "id", "prompt_id", "promptId"])
        if uid is not None:
            return uid

    return None


# -------------------------
# IO
# -------------------------
def load_prompts_jsonl(path: str) -> Dict[str, str]:
    """
    Load prompts mapping key -> prompt text.

    Keys supported:
      - uid keys in file: uid, id, prompt_id, promptId
      - ALSO add index keys: "0", "1", "2", ... (0-based line index among valid rows)

    Prompt text supported:
      - prompt, text, input, instruction, query
      - nested objects are also supported
    """
    out: Dict[str, str] = {}
    bad = 0
    dup = 0
    valid_idx = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue

            uid = _pick_first(obj, ["uid", "id", "prompt_id", "promptId"])
            prompt = _pick_first(obj, ["prompt", "text", "input", "instruction", "query"])

            if prompt is None:
                bad += 1
                continue

            # 0-based sequential key for robustness against prefs using indices
            idx_key = str(valid_idx)
            if idx_key not in out:
                out[idx_key] = prompt

            if uid is None:
                # still keep idx mapping even if uid missing
                valid_idx += 1
                continue

            if uid in out:
                dup += 1
                # keep the first occurrence for determinism
                valid_idx += 1
                continue

            out[uid] = prompt
            valid_idx += 1

    print(f"[data] prompts_loaded={len(out)} bad_prompts={bad} dup_prompts={dup}")
    return out


def load_prefs_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def make_pref_pairs(
    prompts: Dict[str, str],
    prefs_rows: List[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
    seed: int = 0,
) -> List[PrefPair]:
    """
    Build PrefPair list from prefs rows.

    Prompt resolution order:
      1) prompts[uid] where uid is from uid/id/prompt_id/promptId (flat or nested)
      2) prompts[prompt_id] where prompt_id is an index-like field (0/1/2...) (flat or nested)
      3) inline prompt text in prefs row (prompt/text/input/instruction/query) (flat or nested)

    chosen/rejected resolution:
      - supports flat or nested objects
    """
    rng = random.Random(seed)
    pairs: List[PrefPair] = []
    bad = 0
    missing = 0
    used_inline_prompt = 0
    used_index_prompt = 0

    for r in prefs_rows:
        uid = _pick_uid_from_row(r)
        if uid is None:
            bad += 1
            continue

        prompt_text = prompts.get(uid)

        # (2) Try index-like prompt_id if uid didn't match prompts
        if prompt_text is None:
            pid = _pick_first(r, ["prompt_id", "promptId", "prompt_index", "promptIndex", "index"])
            if pid is None and isinstance(r.get("prompt"), dict):
                pid = _pick_first(r["prompt"], ["id", "prompt_id", "promptId", "index"])
            if pid is not None:
                prompt_text = prompts.get(str(pid))
                if prompt_text is not None:
                    used_index_prompt += 1

        # (3) inline prompt in prefs row
        if prompt_text is None:
            prompt_text = _pick_first(
                r,
                ["prompt", "prompt_text", "promptText", "instruction", "query", "input", "text"],
            )
            if prompt_text is not None:
                used_inline_prompt += 1
            else:
                missing += 1
                continue

        chosen = _pick_first(
            r,
            [
                "chosen",
                "accepted",
                "winner",
                "preferred",
                "chosen_response",
                "chosenResponse",
                "completion_a",
                "completionA",
                "a",
            ],
        )
        rejected = _pick_first(
            r,
            [
                "rejected",
                "rejected_response",
                "rejectedResponse",
                "loser",
                "other",
                "completion_b",
                "completionB",
                "b",
            ],
        )

        # nested "responses" patterns
        if chosen is None:
            resp_obj = r.get("response") or r.get("responses")
            if isinstance(resp_obj, dict):
                chosen = _pick_first(resp_obj, ["chosen", "accepted", "winner", "a"])
        if rejected is None:
            resp_obj = r.get("response") or r.get("responses")
            if isinstance(resp_obj, dict):
                rejected = _pick_first(resp_obj, ["rejected", "loser", "other", "b"])

        if chosen is None or rejected is None:
            bad += 1
            continue

        pairs.append(PrefPair(uid=uid, prompt=prompt_text, chosen=chosen, rejected=rejected))

    rng.shuffle(pairs)
    if limit is not None:
        pairs = pairs[: int(limit)]

    print(
        f"[data] loaded_pairs={len(pairs)} bad_prefs_rows={bad} missing_prompt={missing} "
        f"used_index_prompt={used_index_prompt} used_inline_prompt={used_inline_prompt}"
    )
    return pairs


# -------------------------
# Encoding
# -------------------------
def _encode_joint(
    tok: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    *,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    text = prompt + completion
    enc = tok(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    return enc


def _encode_prompt_only(
    tok: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    enc = tok(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    return enc


def encode_pair(
    tok: PreTrainedTokenizerBase,
    pair: PrefPair,
    *,
    max_length: int,
) -> EncodedPair:
    chosen_enc = _encode_joint(tok, pair.prompt, pair.chosen, max_length=max_length)
    rejected_enc = _encode_joint(tok, pair.prompt, pair.rejected, max_length=max_length)

    prompt_only = _encode_prompt_only(tok, pair.prompt, max_length=max_length)
    prompt_len = int(prompt_only["input_ids"].shape[1])

    return EncodedPair(
        uid=pair.uid,
        chosen_enc=chosen_enc,
        rejected_enc=rejected_enc,
        prompt_len_chosen=prompt_len,
        prompt_len_rejected=prompt_len,
    )


def _completion_token_count(enc: Dict[str, torch.Tensor], prompt_len: int) -> int:
    T = int(enc["input_ids"].shape[1])
    return max(0, T - prompt_len)


# -------------------------
# Batching
# -------------------------
def make_batch(
    tok: PreTrainedTokenizerBase,
    pairs: List[PrefPair],
    *,
    batch_size: int,
    max_length: int,
    min_comp_tokens: int = 1,
    allow_short_completion: bool = True,
    max_attempts_per_sample: int = 20,
    seed: int = 0,
    device: str = "cpu",
) -> Iterable[Batch]:
    rng = random.Random(seed)
    idxs = list(range(len(pairs)))
    rng.shuffle(idxs)

    cursor = 0
    while cursor < len(idxs):
        chosen_list: List[torch.Tensor] = []
        chosen_mask_list: List[torch.Tensor] = []
        chosen_pl_list: List[int] = []

        rej_list: List[torch.Tensor] = []
        rej_mask_list: List[torch.Tensor] = []
        rej_pl_list: List[int] = []

        uids: List[str] = []

        while len(uids) < batch_size and cursor < len(idxs):
            attempts = 0
            accepted = False
            while attempts < max_attempts_per_sample and cursor < len(idxs):
                pair = pairs[idxs[cursor]]
                cursor += 1
                attempts += 1

                ep = encode_pair(tok, pair, max_length=max_length)
                c_comp = _completion_token_count(ep.chosen_enc, ep.prompt_len_chosen)
                r_comp = _completion_token_count(ep.rejected_enc, ep.prompt_len_rejected)
                too_short = (c_comp < min_comp_tokens) or (r_comp < min_comp_tokens)

                if too_short and not allow_short_completion:
                    continue

                uids.append(ep.uid)

                chosen_list.append(ep.chosen_enc["input_ids"].squeeze(0))
                chosen_mask_list.append(ep.chosen_enc["attention_mask"].squeeze(0))
                chosen_pl_list.append(ep.prompt_len_chosen)

                rej_list.append(ep.rejected_enc["input_ids"].squeeze(0))
                rej_mask_list.append(ep.rejected_enc["attention_mask"].squeeze(0))
                rej_pl_list.append(ep.prompt_len_rejected)

                accepted = True
                break

            if not accepted:
                raise RuntimeError(
                    f"[batch] failed to sample a valid pair after {max_attempts_per_sample} attempts. "
                    f"Consider allow_short_completion=True or lowering min_comp_tokens."
                )

        if not uids:
            break

        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id
            if pad_id is None:
                raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id; cannot pad safely.")

        def _pad_stack(seqs: List[torch.Tensor]) -> torch.Tensor:
            maxT = max(int(s.shape[0]) for s in seqs)
            out = torch.full((len(seqs), maxT), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : s.shape[0]] = s
            return out

        def _pad_mask(seqs: List[torch.Tensor]) -> torch.Tensor:
            maxT = max(int(s.shape[0]) for s in seqs)
            out = torch.zeros((len(seqs), maxT), dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : s.shape[0]] = s
            return out

        input_ids_chosen = _pad_stack(chosen_list).to(device)
        attn_mask_chosen = _pad_mask(chosen_mask_list).to(device)
        prompt_lens_chosen = torch.tensor(chosen_pl_list, dtype=torch.long, device=device)

        input_ids_rejected = _pad_stack(rej_list).to(device)
        attn_mask_rejected = _pad_mask(rej_mask_list).to(device)
        prompt_lens_rejected = torch.tensor(rej_pl_list, dtype=torch.long, device=device)

        yield Batch(
            uids=uids,
            input_ids_chosen=input_ids_chosen,
            attn_mask_chosen=attn_mask_chosen,
            prompt_lens_chosen=prompt_lens_chosen,
            input_ids_rejected=input_ids_rejected,
            attn_mask_rejected=attn_mask_rejected,
            prompt_lens_rejected=prompt_lens_rejected,
        )
