# RLHF-Eval-Lab

**RLHF-style methods evaluation harness** that *always* produces fully populated,
self-auditable Markdown reports — **no empty cells, no silent failures**.

This repository has completed the **DoD (Definition of Done) / OSS reliability phase**.
All reported numbers in this phase are **sanity-tier values** intended to validate
the evaluation and reporting pipeline, **not paper-level claims**.

---

## Definition of Done (DoD) — Reliability Phase

On **CPU / Colab / CI**, the following commands must complete **without exceptions**:

```bash
pip install -e .
rlhf-lab run --backend fallback --seed 0
rlhf-lab validate
rlhf-lab report
```

The generated Markdown report must satisfy **all** of the following invariants:

* Table 1 / Table 2-A / Table 2-B / Table 2-C are always present
* **No empty cells**
* All values are either numeric or column-policy `"N/A"`
* Full provenance is recorded (backend, model, tokenizer, config, git commit, seed)

> **Note**
> In the DoD phase, metrics are intentionally **sanity-tier**.
> Their sole purpose is to validate end-to-end plumbing, determinism,
> and reporting correctness — **not** to support scientific claims.

---

## Level-C Research Phase (Active)

The DoD phase is **frozen** and preserved on:

* the `main` branch
* the `v0.1.0-dod` tag

All research-oriented development is conducted **exclusively** on the
`level-c-research` branch.

The DoD guarantees must remain intact throughout the research phase.

---

## Installation

### Layer 1 — DoD / OSS Reliability (default)

* CPU / CI safe
* **No `transformers` dependency**
* Torch-only deterministic `fallback` backend

```bash
pip install -e .
```

### Layer 2 — Level-C Research (HF backend)

* Requires Hugging Face dependencies (`transformers`)
* Used **only** on `level-c-research` or with `--backend hf`

```bash
pip install -r requirements-hf.txt
pip install -e .
```

> The dependency boundary is intentional.
> **DoD reliability must remain independent of `transformers`.**

---

## Fixed Datasets (Level-C Research)

Level-C research uses the following datasets **exclusively**:

* **Preference learning**: HH-RLHF
* **Stress / safety evaluation**: HarmBench

No additional datasets are introduced during this phase.

---

## HarmBench Usage Policy

HarmBench is used **strictly for evaluation** and **never for training**.

Reports contain **only aggregated metrics**.
Raw prompts, completions, or harmful content are **never** stored in artifacts
or rendered in Markdown reports.

---

## Separation from the DoD Phase

All components defined in the DoD phase are reused **as-is**:

* artifacts schema
* evaluation logic
* validation rules
* reporting pipeline

Any research-phase extension **must preserve** the following guarantees:

* no empty cells
* column-level `"N/A"` policy
* strict provenance tracking

Violations are treated as errors, not warnings.

---

## Quickstart (DoD)

### 1) Run (fallback backend, torch-only)

```bash
rlhf-lab run --backend fallback --seed 0
```

Outputs:

* `artifacts/<method_key>/seed_<seed>.json`

---

### 2) Validate (OSS gate)

```bash
rlhf-lab validate
```

Validation guarantees:

* no missing metrics per method
* no empty cells in downstream tables
* `"N/A"` appears **only** where allowed by column policy

---

### 3) Report (Markdown tables)

```bash
rlhf-lab report
```

Outputs:

* `reports/report.md`

---

## How to Read the Report

The generated `report.md` is **self-contained and auditable**.
No external logs or code inspection are required.

### General Rules

* Every cell is filled with either a **numeric value** or **`N/A`**
* `↓` means *lower is better*, `↑` means *higher is better*
* `N/A` does **not** mean missing data; it means the metric is **not applicable**
  to that method by design

### Table 1 — Unified Comparison

Main comparison table across all methods.

* **Off-support ↓**
  Degree to which the policy moves outside the support of the data / reward model
* **Tail Var ↓**
  Variance of the reward tail (lower implies fewer extreme spikes)
* **On-support ↑**
  Average reward within the supported region
* **Judge ↑**
  External judge score
* **Win-rate ↑**
  Pairwise comparison win rate
* **KL ↓**
  KL divergence from the reference policy (policy drift)

### Table 2 Blocks

Method-specific diagnostic tables:

* **Table 2-A (PPO-family Diagnostics)**
  Meaningful only for PPO-based methods; others are `N/A`
* **Table 2-B (Preference-based Diagnostics)**
  Applicable to preference / active learning methods
  `label_source` is one of `pref` or `ai`
* **Table 2-C (Safety / Robustness)**
  Safety and robustness diagnostics, mainly for PPO-based methods

### Provenance

Each report embeds full provenance:

* backend
* model
* tokenizer
* config hash
* git commit
* seed

If any of these differ across methods, report generation fails.
This enforces strict reproducibility and auditability.

---

## Backends

### `fallback` (default, DoD)

* `torch` only
* deterministic tokenizer + GRU tiny language model
* designed to run on CPU and in CI **without `transformers`**

### `hf` (research phase)

* Hugging Face backend
* optional dependency via `requirements-hf.txt`
* used **only** in the Level-C research phase

---

## Project Structure (High Level)

* `rlhf_eval_lab/cli/` — CLI entrypoints (`run`, `validate`, `report`)
* `rlhf_eval_lab/backends/` — backend implementations (`fallback`, `hf`)
* `rlhf_eval_lab/eval/` — metric computation and evaluation runner (SSOT)
* `rlhf_eval_lab/registry/` — method and metric specifications (column N/A policy)
* `rlhf_eval_lab/reporting/` — artifacts, aggregation, Markdown report generation
* `tests/integration/` — end-to-end regression tests (DoD gate)
* `test_data/` — minimal sample JSONL for dataset wiring

---

## Development

### Run end-to-end regression tests

```bash
pytest -q
```

### Clean generated outputs

```bash
rm -rf artifacts reports
```

---

## License

See `LICENSE`.

## Citation

See `CITATION.cff`.
