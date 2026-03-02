# rlhf_eval_lab/train/governor/controller.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def goldbeter_koshland(v1: float, v2: float, j1: float, j2: float) -> float:
    """
    Goldbeter–Koshland function G(v1, v2, J1, J2) in [0,1].

    B = v2 - v1 + J1*v2 + J2*v1
    G = (2*v1*J2) / (B + sqrt(B^2 - 4*(v2 - v1)*v1*J2))

    Stable guards included.
    """
    v1 = max(0.0, float(v1))
    v2 = max(0.0, float(v2))
    j1 = max(1e-12, float(j1))
    j2 = max(1e-12, float(j2))

    b = (v2 - v1) + (j1 * v2) + (j2 * v1)
    disc = (b * b) - (4.0 * (v2 - v1) * v1 * j2)
    if disc < 0.0:
        disc = 0.0

    denom = b + math.sqrt(disc)
    if abs(denom) < 1e-12:
        return 0.5

    g = (2.0 * v1 * j2) / denom
    if not math.isfinite(g):
        return 0.5
    return _clamp(g, 0.0, 1.0)


@dataclass
class GovernorConfig:
    # targets / thresholds
    kl_target: float = 0.08
    kl_max: float = 0.50
    off_max: float = 0.95
    risk_crit: float = 1.20

    # risk weights
    w_kl: float = 1.00
    w_off: float = 0.70
    w_var: float = 0.30

    # GK gating params
    gk_j1: float = 0.02
    gk_j2: float = 0.02

    # reward / lr scaling
    reward_scale_min: float = 0.10
    lr_scale_min: float = 0.05

    # integral feedback on beta
    beta_init: float = 0.10
    beta_min: float = 1e-5
    beta_max: float = 50.0
    kp: float = 0.60
    ki: float = 0.08
    i_min: float = -5.0
    i_max: float = 5.0
    beta_gate_gain: float = 1.00

    impulse: str = "stop"


class GovernorController:
    def __init__(self, cfg: GovernorConfig):
        self.cfg = cfg
        self.i_state = 0.0
        self.beta = float(cfg.beta_init)

    def _risk(self, kl_abs: float, offsupport: float, reward_var: float) -> Tuple[float, float]:
        eps = 1e-12
        kt = max(float(self.cfg.kl_target), eps)
        kl_abs = max(0.0, float(kl_abs))
        offsupport = _clamp(float(offsupport), 0.0, 1.0)
        reward_var = max(0.0, float(reward_var))

        e_kl = (kl_abs - kt) / kt
        e_pos = max(0.0, e_kl)

        r = (
            float(self.cfg.w_kl) * e_pos
            + float(self.cfg.w_off) * offsupport
            + float(self.cfg.w_var) * math.sqrt(reward_var)
        )
        if not math.isfinite(r):
            r = 0.0
        return float(r), float(e_kl)

    def step(self, *, kl_abs: float, offsupport: float, reward_var: float) -> Dict[str, float | int | str | bool]:
        r, e_kl = self._risk(kl_abs=kl_abs, offsupport=offsupport, reward_var=reward_var)

        trust = math.exp(-r)
        trust = _clamp(trust, 1e-9, 1.0)

        gate = goldbeter_koshland(trust, 1.0 - trust, self.cfg.gk_j1, self.cfg.gk_j2)

        rs_min = _clamp(float(self.cfg.reward_scale_min), 0.0, 1.0)
        lr_min = _clamp(float(self.cfg.lr_scale_min), 0.0, 1.0)

        reward_scale = rs_min + (1.0 - rs_min) * gate
        lr_scale = lr_min + (1.0 - lr_min) * gate

        self.i_state = _clamp(self.i_state + float(self.cfg.ki) * e_kl, self.cfg.i_min, self.cfg.i_max)

        beta = float(self.beta)
        beta = beta * math.exp(float(self.cfg.kp) * e_kl + self.i_state)

        gain = _clamp(float(self.cfg.beta_gate_gain), 0.0, 10.0)
        beta = beta * (1.0 + gain * (1.0 - gate))
        beta = _clamp(beta, float(self.cfg.beta_min), float(self.cfg.beta_max))
        self.beta = float(beta)

        stop = False
        if r > float(self.cfg.risk_crit):
            stop = True
        if kl_abs > float(self.cfg.kl_max):
            stop = True
        if offsupport > float(self.cfg.off_max):
            stop = True

        return {
            "governor_risk": float(r),
            "governor_e_kl": float(e_kl),
            "governor_trust": float(trust),
            "governor_gate": float(gate),
            "governor_reward_scale": float(reward_scale),
            "governor_lr_scale": float(lr_scale),
            "governor_beta": float(self.beta),
            "governor_i": float(self.i_state),
            "governor_stop": bool(stop),
        }