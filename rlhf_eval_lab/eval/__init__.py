# rlhf_eval_lab/eval/__init__.py
# eval は ArtifactsV1 から指標を計算する（学習はしない）

from .runner import evaluate_artifacts, build_table_rows
