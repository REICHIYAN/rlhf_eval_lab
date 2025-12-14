# rlhf_eval_lab/registry/__init__.py
# registry は「表の行・列」を定義する SSOT（Single Source of Truth）

from .methods import METHOD_SPECS, METHOD_BY_KEY, METHOD_KEYS, MethodSpec
from .metrics import METRIC_SPECS, MetricSpec
from .datasets import DATASET_SPECS, DatasetSpec
from .specs import TableSpec
