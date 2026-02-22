# RLHF-Eval-Lab

An **RLHF-style evaluation harness** that always produces **fully populated, self-auditable Markdown reports** — **no empty cells, no silent failures**.

This repository is intentionally built in **two layers**:

* **Layer 1 (DoD / OSS reliability, frozen):** a torch-only `fallback` backend (**no `transformers`**) that keeps the end-to-end pipeline **unbreakable** on CPU/CI.
* **Layer 2 (Level-C research, active):** an optional Hugging Face `hf` backend for paper-grade experiments, developed on `level-c-research`.

> In the DoD layer, reported values are **sanity-tier**: they validate plumbing, determinism, and auditability — not paper claims.

---

## Definition of Done (DoD)

On **CPU / Colab / CI**, the following must complete **without exceptions**:

```bash
pip install -e .
rlhf-lab run --backend fallback --preset offline_hh_small --seed 0
rlhf-lab report
rlhf-lab validate --report reports
```

### Report invariants (non-negotiable)

The generated report must satisfy:

* Table 1 / Table 2-A / Table 2-B / Table 2-C are always present
* **No empty cells**
* Every cell is either a **number** or column-policy **`N/A`**
* Provenance is recorded (backend, model, tokenizer, config hash, git commit, seed)

Validation entrypoints:

* `rlhf-lab validate` validates artifacts.
* `rlhf-lab validate --report reports` validates artifacts **and** the rendered `reports/report.md` invariants.

---

## Local verification (SSOT)

Run the same checks as CI (guard → DoD E2E → tests → no tracked diffs):

```bash
make check
```

If you prefer the long form:

```bash
rm -rf artifacts reports outputs report.md
rlhf-lab run --backend fallback --preset offline_hh_small --seed 0
rlhf-lab report
rlhf-lab validate --report reports
pytest -q
```

---

## Generated outputs and Git hygiene

This repo generates outputs during normal operation:

* `artifacts/` — per-method artifacts (e.g., `seed_*.json`)
* `reports/` — rendered report (`reports/report.md`)
* `outputs/`, `report.md`, and some root-level caches may also appear depending on presets

**Rule:** do **not** commit generated outputs.

Enforcement:

* CI fails if any generated outputs become **tracked**.
* Use `make check` to catch this locally.

If you need to share an example report, copy it to a non-generated location (e.g., `examples/report_example.md`) rather than committing `reports/report.md`.

---

## Branch policy

* Reliability/DoD guarantees must remain intact.
* Research-oriented development happens on `level-c-research`.

---

## Installation

### Layer 1 (default): DoD / fallback backend

CPU/CI safe and does not depend on `transformers`.

```bash
pip install -e .
```

### Layer 2 (optional): HF backend

Install HF dependencies only if you will run `--backend hf`.

```bash
pip install -r requirements-hf.txt
pip install -e .
```

---

## Methods

This project evaluates RLHF-style methods through a common **ArtifactsV1** schema, ensuring results are comparable and auditable.

* Method keys are defined in: `rlhf_eval_lab/registry/methods.py`
* Metric policy and column-level `N/A` rules are defined in: `rlhf_eval_lab/registry/metrics.py`

The `fallback` backend provides deterministic sanity-tier implementations sufficient for:

* end-to-end execution on CPU/CI
* artifact emission
* metric computation
* fully populated report tables

---

## Datasets

### Bundled offline dataset (recommended for DoD)

A tiny offline prompt set is bundled for deterministic E2E runs:

* `test_data/offline_prompts_small.jsonl`
* Preset: `rlhf_eval_lab/config/presets/offline_hh_small.yaml`

### Research datasets (not bundled)

Real research datasets are **not** bundled.

* Place local JSONL files under `data/` (ignored by git).
* Presets may reference these local files.

Artifacts record dataset identity and a content hash for reproducibility.

---

## HarmBench policy (evaluation-only)

If you use HarmBench, it should be used **strictly for evaluation**, never for training.
Treat local HarmBench JSONL files and generated artifacts as sensitive data depending on your environment.

---

<!-- METRICS_SSOT:START -->

## Metrics SSOT (auto-generated from registry/metrics.py)

This block is updated by `python -m rlhf_eval_lab.utils.update_readme_metrics_ssot`.
It synchronizes metric names, N/A policies, and formatting rules with the code SSOT.

### Table1

| key | label | decimals | N/A | notes |
|---|---|---:|---|---|
| `offsupport` | Off-support ↓ | 4 | - | dir=↓ | dtype=float |
| `tail_var` | Tail Var ↓ | 4 | - | dir=↓ | dtype=float |
| `onsupport` | On-support ↑ | 4 | - | dir=↑ | dtype=float |
| `judge` | Judge ↑ | 4 | - | dir=↑ | dtype=float |
| `win_rate` | Win-rate ↑ | 4 | - | dir=↑ | dtype=float |
| `ppl` | PPL ↓ | 4 | - | dir=↓ | dtype=float |
| `kl` | KL ↓ | 4 | methods: active_pref, dpo, ipo, orpo, rlaif, rrhf | dir=↓ | dtype=float |
| `latency_ms` | Latency (ms) ↓ | 0 | - | dir=↓ | dtype=int |

### Table2A

| key | label | decimals | N/A | notes |
|---|---|---:|---|---|
| `ppo_loss` | PPO Loss ↓ | 4 | methods: 7 | dir=↓ | dtype=float |
| `ratio_mean` | Ratio Mean | 4 | methods: 7 | dtype=float |
| `clipfrac` | Clip Fraction ↓ | 4 | methods: 7 | dir=↓ | dtype=float |
| `kl_ref_abs` | KL Ref Abs ↓ | 4 | methods: 7 | dir=↓ | dtype=float |
| `kl_ref_sq` | KL Ref Sq ↓ | 4 | methods: 7 | dir=↓ | dtype=float |

### Table2B

| key | label | decimals | N/A | notes |
|---|---|---:|---|---|
| `sample_efficiency` | Sample Efficiency ↑ | 4 | methods: adaptive_rm_ppo, kl_ppo_adaptive, kl_ppo_fixed, ppo_standard, safe_ppo, sft | dir=↑ | dtype=float |
| `reward_accuracy` | Reward Accuracy ↑ | 4 | methods: adaptive_rm_ppo, kl_ppo_adaptive, kl_ppo_fixed, ppo_standard, safe_ppo, sft | dir=↑ | dtype=float |
| `label_source` | Label Source | - | - | dtype=str |

### Table2C

| key | label | decimals | N/A | notes |
|---|---|---:|---|---|
| `prompt_injection` | Prompt Injection ↓ | 4 | methods: 7 | dir=↓ | dtype=float |
| `ood_stability` | OOD Stability ↓ | 4 | methods: 7 | dir=↓ | dtype=float |

<!-- METRICS_SSOT:END -->

---

## Project structure

* `rlhf_eval_lab/cli/` — CLI entrypoints (`run`, `validate`, `report`)
* `rlhf_eval_lab/backends/` — backend implementations (`fallback`, `hf`)
* `rlhf_eval_lab/eval/` — evaluation runner and metric computation (SSOT)
* `rlhf_eval_lab/registry/` — method + metric specs (including `N/A` policy)
* `rlhf_eval_lab/reporting/` — aggregation and Markdown report generation
* `tests/` — unit/integration tests
* `test_data/` — bundled offline JSONL for deterministic wiring

---

## License

See `LICENSE`.

## Citation

See `CITATION.cff`.
