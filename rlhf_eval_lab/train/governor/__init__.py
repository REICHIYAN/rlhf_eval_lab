# rlhf_eval_lab/train/governor/__init__.py

"""
Governor: KL-PPO + Impulse Control + GK gating + Integral feedback.

- PPO math stays inside backend.ppo_step()
- Governor wraps PPO with control logic (gating / beta adaptation / early stop)
"""

from .driver import run_governor  # noqa: F401