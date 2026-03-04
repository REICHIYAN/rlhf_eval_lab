# rlhf_eval_lab/train/aegis/driver.py

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from rlhf_eval_lab.train.aegis.rrd import estimate_reliability


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except Exception:
        return float(default)


def _truncate_text(s: str) -> str:
    toks = (s or "").split()
    if len(toks) <= 2:
        return s
    return " ".join(toks[: max(2, len(toks) // 2)])


def _make_pref_pair(prompt: str, completion: str, reward: float) -> Tuple[str, str]:
    _ = prompt
    alt = _truncate_text(completion)
    if alt.strip() == (completion or "").strip():
        alt = (completion or "") + " ."
    return (completion, alt) if reward >= 0.0 else (alt, completion)


def run_aegis(
    *,
    backend: Any,
    prompts: Sequence[str],
    completions: Sequence[str],
    rewards: Sequence[float],
    rm: Any,
    max_new_tokens: int,
    train_cfg: Dict[str, Any],
    pref_beta: float,
) -> Tuple[List[str], List[float], Dict[str, Any]]:
    """Aegis preference update (sanity tier).

    - Builds a single preference pair from (prompt, completion, reward)
    - Estimates reliability via prompt transforms (median margin, IQR)
    - Down-weights the update when uncertainty is high

    Fail-open behavior:
      If reliability estimation fails, fall back to beta_eff=pref_beta.
    """
    prompts_l = list(prompts)
    completions_l = list(completions)
    rewards_l = [float(x) for x in rewards]

    extra: Dict[str, Any] = {}

    a = (train_cfg.get("aegis", {}) or {}) if isinstance(train_cfg, dict) else {}
    alpha = _as_float(a.get("alpha", 1.0), 1.0)
    k = a.get("k", None)

    # Keep DoD-style: at least one step
    used_prompt = prompts_l[0] if prompts_l else ""
    used_completion = completions_l[0] if completions_l else ""
    used_reward = rewards_l[0] if rewards_l else 0.0

    chosen, rejected = _make_pref_pair(used_prompt, used_completion, used_reward)

    value_margin = float("nan")
    uncertainty = 0.0
    pool_size = 0

    beta_eff = float(pref_beta)
    weight = 1.0

    try:
        st = estimate_reliability(used_prompt, chosen, rejected, k=k)
        value_margin = float(st.value_margin)
        uncertainty = float(st.uncertainty)
        pool_size = int(st.pool_size)

        # w = exp(-alpha * unc)
        if math.isfinite(alpha) and math.isfinite(uncertainty) and uncertainty >= 0.0:
            weight = float(math.exp(-float(alpha) * float(uncertainty)))
            # guard: never exactly zero
            weight = max(1e-8, min(1.0, weight))
            beta_eff = float(pref_beta) * float(weight)
    except Exception as e:
        extra["aegis_failopen"] = True
        extra["aegis_error"] = str(e)

    # Step: backend preference update
    loss = backend.preference_step(
        prompt=used_prompt,
        chosen=chosen,
        rejected=rejected,
        beta=float(beta_eff),
    )

    # Post-update snapshot
    completions_post = backend.generate(prompts_l, max_new_tokens=int(max_new_tokens))
    rewards_post = [float(x) for x in rm.score(prompts_l, completions_post)]

    extra.update(
        {
            "pref_loss": float(_as_float(loss, 0.0)),
            "steps": 1,
            "skipped": False,
            "skip_reason": "",
            "pair_prompt": used_prompt,
            "pair_chosen": chosen,
            "pair_rejected": rejected,
            # Aegis-specific audit fields
            "aegis_alpha": float(alpha),
            "aegis_k": int(k) if isinstance(k, int) else -1,
            "aegis_pool_size": int(pool_size),
            "aegis_value_margin": float(value_margin) if math.isfinite(value_margin) else float("nan"),
            "aegis_uncertainty": float(uncertainty),
            "aegis_weight": float(weight),
            "aegis_beta_eff": float(beta_eff),
            "pairs_used": 1,
        }
    )

    return list(completions_post), list(rewards_post), extra
