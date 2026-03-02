# rlhf_eval_lab/train/governor/driver.py

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rlhf_eval_lab.eval.offsupport import compute_offsupport
from rlhf_eval_lab.train.governor.controller import GovernorConfig, GovernorController


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except Exception:
        return float(default)


def _reward_var(rewards: Sequence[float]) -> float:
    xs = [float(x) for x in rewards]
    if not xs:
        return 0.0
    m = sum(xs) / float(len(xs))
    v = sum((x - m) * (x - m) for x in xs) / float(len(xs))
    if not math.isfinite(v) or v < 0.0:
        return 0.0
    return float(v)


def _stamp_ppo_audit_from_fallback(extra: Dict[str, Any], ppo_out: Dict[str, Any]) -> None:
    extra["ppo_loss"] = float(_as_float(ppo_out.get("ppo_loss", ppo_out.get("loss", 0.0)), 0.0))
    extra["ratio_mean"] = float(_as_float(ppo_out.get("ratio_mean", 1.0), 1.0))
    extra.setdefault("ratio_mean_pre", 1.0)
    extra["clipfrac"] = float(_as_float(ppo_out.get("clipfrac", 0.0), 0.0))

    kl_mean = ppo_out.get("kl_mean", None)
    if kl_mean is not None:
        kl_signed = float(_as_float(kl_mean, 0.0))
    else:
        kl_v = ppo_out.get("kl", None)
        kl_signed = float(_as_float(kl_v, 0.0)) if kl_v is not None else 0.0

    kl_abs = abs(float(kl_signed))
    extra["kl_ref_abs"] = float(kl_abs)
    extra["kl_ref_sq"] = float(kl_abs * kl_abs)


def _choose_nonneg_kl_proxy(extra: Dict[str, Any]) -> float:
    for k in ("kl_ref_abs", "kl_est_abs"):
        v = extra.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    for k in ("kl_ref", "kl_est", "kl", "kl_mean"):
        v = extra.get(k)
        if v is not None:
            try:
                return float(abs(float(v)))
            except Exception:
                pass
    return 0.0


def _set_optimizer_lr(backend: Any, lr: float) -> bool:
    try:
        lr = float(lr)
        if lr <= 0.0 or (not math.isfinite(lr)):
            return False

        if hasattr(backend, "_ppo_optim"):
            optim = getattr(backend, "_ppo_optim")
        elif hasattr(backend, "opt"):
            optim = getattr(backend, "opt")
        else:
            return False

        if hasattr(optim, "param_groups") and optim.param_groups:
            optim.param_groups[0]["lr"] = lr
            return True
        return False
    except Exception:
        return False


def _set_clip(backend: Any, clip: float) -> bool:
    try:
        clip = float(clip)
        if clip <= 0.0 or (not math.isfinite(clip)):
            return False
        if hasattr(backend, "ppo_clip"):
            setattr(backend, "ppo_clip", clip)
            return True
        return False
    except Exception:
        return False


def _load_governor_cfg(train_cfg: Dict[str, Any], *, beta_fallback: float) -> GovernorConfig:
    g = (train_cfg.get("governor", {}) or {}) if isinstance(train_cfg, dict) else {}

    def f(key: str, default: float) -> float:
        return float(_as_float(g.get(key, default), default))

    def s(key: str, default: str) -> str:
        v = g.get(key, default)
        return str(v) if v is not None else default

    return GovernorConfig(
        kl_target=f("kl_target", 0.08),
        kl_max=f("kl_max", 0.50),
        off_max=f("off_max", 0.95),
        risk_crit=f("risk_crit", 1.20),
        w_kl=f("w_kl", 1.00),
        w_off=f("w_off", 0.70),
        w_var=f("w_var", 0.30),
        gk_j1=f("gk_j1", 0.02),
        gk_j2=f("gk_j2", 0.02),
        reward_scale_min=f("reward_scale_min", 0.10),
        lr_scale_min=f("lr_scale_min", 0.05),
        beta_init=f("beta_init", beta_fallback),
        beta_min=f("beta_min", 1e-5),
        beta_max=f("beta_max", 50.0),
        kp=f("kp", 0.60),
        ki=f("ki", 0.08),
        i_min=f("i_min", -5.0),
        i_max=f("i_max", 5.0),
        beta_gate_gain=f("beta_gate_gain", 1.00),
        impulse=s("impulse", "stop"),
    )


def run_governor(
    *,
    backend: Any,
    prompts: Sequence[str],
    completions: Sequence[str],
    rewards: Sequence[float],
    rm: Any,
    max_new_tokens: int,
    train_cfg: Dict[str, Any],
    steps: int,
    ppo_lr_base: float,
    ppo_clip_base: float,
    use_fallback: bool,
    ref_state: Optional[Dict[str, Any]] = None,
    policy_init_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[float], Dict[str, Any]]:
    prompts_l = list(prompts)
    completions_l = list(completions)
    rewards_l = [float(x) for x in rewards]

    if steps < 1:
        steps = 1

    if use_fallback and policy_init_state is not None and hasattr(backend, "model"):
        backend.model.load_state_dict(policy_init_state)

    cfg = _load_governor_cfg(train_cfg, beta_fallback=float(train_cfg.get("kl_beta", 0.1)))
    ctrl = GovernorController(cfg)

    extra: Dict[str, Any] = {}
    gates: List[float] = []
    risks: List[float] = []

    lr_scale = 1.0
    clip_scale = 1.0
    reward_scale = 1.0

    impulse_triggered = False
    stopped_at = -1

    for t in range(int(steps)):
        _set_optimizer_lr(backend, float(ppo_lr_base) * lr_scale)
        _set_clip(backend, max(1e-6, float(ppo_clip_base) * clip_scale))

        rewards_eff = [float(r) * float(reward_scale) for r in rewards_l]

        out = backend.ppo_step(
            prompts=prompts_l,
            completions=completions_l,
            rewards=rewards_eff,
            kl_beta=float(ctrl.beta),
            ref_state=ref_state,
            update_ref=False,
        )

        extra.update({k: float(_as_float(v, 0.0)) for k, v in out.items()})

        if use_fallback:
            _stamp_ppo_audit_from_fallback(extra, out)
        else:
            extra.setdefault("kl_ref_abs", float(_choose_nonneg_kl_proxy(extra)))
            extra.setdefault("kl_ref_sq", float(extra["kl_ref_abs"]) * float(extra["kl_ref_abs"]))

        kl_abs = float(_as_float(extra.get("kl_ref_abs", 0.0), 0.0))
        off = float(compute_offsupport(prompts_l, completions_l))
        rv = float(_reward_var(rewards_l))

        ctrl_out = ctrl.step(kl_abs=kl_abs, offsupport=off, reward_var=rv)

        gate = float(ctrl_out["governor_gate"])
        risk = float(ctrl_out["governor_risk"])

        reward_scale = float(ctrl_out["governor_reward_scale"])
        lr_scale = float(ctrl_out["governor_lr_scale"])
        clip_scale = lr_scale  # stable simplification

        gates.append(gate)
        risks.append(risk)

        if bool(ctrl_out["governor_stop"]):
            impulse_triggered = True
            stopped_at = t
            break

    # restore
    _set_optimizer_lr(backend, float(ppo_lr_base))
    _set_clip(backend, float(ppo_clip_base))

    # post-update snapshot
    completions_post = backend.generate(prompts_l, max_new_tokens=int(max_new_tokens))
    rewards_post = [float(x) for x in rm.score(prompts_l, completions_post)]

    extra["kl"] = float(_choose_nonneg_kl_proxy(extra))

    extra["steps"] = int(len(gates)) if gates else 0
    extra["skipped"] = False
    extra["skip_reason"] = ""

    extra["governor_beta_init"] = float(cfg.beta_init)
    extra["governor_beta_final"] = float(ctrl.beta)

    extra["governor_gate_mean"] = float(sum(gates) / len(gates)) if gates else 0.0
    extra["governor_gate_min"] = float(min(gates)) if gates else 0.0
    extra["governor_gate_last"] = float(gates[-1]) if gates else 0.0

    extra["governor_risk_mean"] = float(sum(risks) / len(risks)) if risks else 0.0
    extra["governor_risk_max"] = float(max(risks)) if risks else 0.0

    extra["governor_impulse"] = bool(impulse_triggered)
    extra["governor_stopped_at"] = int(stopped_at)

    return list(completions_post), list(rewards_post), extra