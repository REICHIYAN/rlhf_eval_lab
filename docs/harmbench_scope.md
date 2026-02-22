# HarmBench minimal integration scope (Dd2-0)

Goal: connect a **tiny, OSS-bundled** HarmBench-style prompt set so that Table 2-C is filled with **real, dataset-driven numeric values**, while keeping the full dataset **out of the repo**.

## What is bundled vs not bundled

Bundled (OSS / CI-safe):

- `test_data/harmbench_small.jsonl`
  - tiny subset (order-of-10s lines)
  - intended for **offline evaluation** only
  - safe for CI and cloning without any external downloads

Not bundled (paper / local-only):

- `data/harmbench.jsonl`
  - full dataset is **user-provided**
  - git-ignored (`data/*.jsonl`)
  - referenced by `rlhf_eval_lab/config/presets/paper_harmbench.yaml` (if you use it)

## JSONL schema (minimal, flexible)

Only one field is required:

- `prompt`: string

Optional fields (ignored by the current loader, reserved for future):

- `category`: string
- `tags`: list[str]
- `id`: string

## Presets

- `offline_harmbench_small.yaml` (OSS-bundled wiring preset)
  - `dataset.source: local`
  - `dataset.path: test_data/harmbench_small.jsonl`
  - `subsample_n` small (e.g. 32) for CI speed

CI policy:

- CI must **not** require any external dataset downloads.
- CI may run `offline_harmbench_small` because it is bundled.

## Table 2-C semantics (sanity-tier)

Table 2-C is computed from prompts/completions/rewards produced during `run`.

- `prompt_injection` and `ood_stability` must always be **finite numeric values** (no NaN).
- Definitions are deterministic and sanity-tier; they validate wiring and robustness plumbing.
