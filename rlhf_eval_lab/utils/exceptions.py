# rlhf_eval_lab/utils/exceptions.py
# 例外は型で分け、CLI で扱いやすくする

class ConfigError(Exception):
    """設定不正（schema違反・必須欠落など）"""


class ValidationError(Exception):
    """Artifacts や入力データの厳格検証に失敗"""


class DependencyMissingError(Exception):
    """任意依存（transformers等）が無いのに要求された"""
