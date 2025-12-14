# rlhf_eval_lab/reporting/__init__.py
# reporting は Artifacts を SSOT として、集計・レポート生成を担う

from .artifacts import ArtifactsV1, ProvenanceV1, read_artifacts, write_artifacts
from .aggregate import aggregate_seed_means
from .markdown import render_report_markdown
from .paths import resolve_output_paths
from .provenance import detect_git_commit
