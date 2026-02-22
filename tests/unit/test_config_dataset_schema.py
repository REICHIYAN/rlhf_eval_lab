from __future__ import annotations

import pytest

from rlhf_eval_lab.config.schema import validate_config_dict
from rlhf_eval_lab.utils.exceptions import ConfigError


def test_dataset_optional_ok() -> None:
    # dataset section absent -> ok
    validate_config_dict({"tiny_lm": {"arch": "gru"}})


def test_dataset_invalid_source_rejected() -> None:
    with pytest.raises(ConfigError):
        validate_config_dict(
            {
                "tiny_lm": {"arch": "gru"},
                "dataset": {"name": "hh_rlhf", "source": "BAD", "path": "x.jsonl"},
            }
        )


def test_dataset_local_requires_path_when_enabled() -> None:
    with pytest.raises(ConfigError):
        validate_config_dict(
            {
                "tiny_lm": {"arch": "gru"},
                "dataset": {"name": "hh_rlhf", "source": "local", "path": ""},
            }
        )
