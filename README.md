# RLHF-Eval-Lab

**RLHF-style methods evaluation harness** that *always* produces fully populated, self-auditable Markdown reports — **no empty cells, no silent failures**.

This repository is intentionally built in **two layers**:

* **Layer 1 (DoD / OSS reliability, frozen):** a torch-only `fallback` backend (**no `transformers`**) that guarantees the end-to-end pipeline is *unbreakable* on CPU/CI.
* **Layer 2 (Level‑C research, active):** an optional Hugging Face `hf` backend used for paper-grade experiments, developed on `level-c-research`.

> In the DoD layer, reported values are **sanity-tier**. They validate plumbing, determinism, and auditability — **not paper claims**.

---

## Definition of Done (DoD) — Reliability Phase

On **CPU / Colab / CI**, the following commands must complete **without exceptions**:

```bash
pip install -e .
rlhf-lab run --backend fallback --seed 0
rlhf-lab validate
rlhf-lab report
```

The generated Markdown report must satisfy **all invariants**:

* Table 1 / Table 2‑A / Table 2‑B / Table 2‑C are always present
* **No empty cells**
* Every cell is either a **number** or column-policy **`N/A`**
* Full provenance is recorded (**backend, model, tokenizer, config hash, git commit, seed**)

---

## Level‑C Research Phase (Active)

* The DoD layer is preserved on:

  * `main`
  * `dod-fallback-v1` (DoD snapshot tag)
* Research-oriented development happens **exclusively** on:

  * `level-c-research`

**Rule:** DoD guarantees must remain intact throughout the research phase.

---

## Installation

### Layer 1 — DoD / OSS Reliability (default)

* CPU / CI safe
* **No `transformers` dependency**
* Torch-only deterministic `fallback` backend

```bash
pip install -e .
```

### Layer 2 — Level‑C Research (HF backend)

* Requires Hugging Face dependencies (`transformers`)
* Used only when you explicitly request it:

  * `--backend hf`

```bash
pip install -r requirements-hf.txt
pip install -e .
```

> The dependency boundary is intentional: **DoD reliability must remain independent of `transformers`.**

---

## Algorithms (Methods)

This project evaluates RLHF-style methods through a common **ArtifactsV1** schema, ensuring the pipeline is comparable and auditable.

### Implemented in DoD (fallback backend)

The fallback backend runs minimal, deterministic “sanity-tier” implementations that are sufficient to:

* run end-to-end on CPU/CI
* produce artifacts
* compute metrics
* generate fully populated tables

Method keys are defined in **SSOT**:

* `rlhf_eval_lab/registry/methods.py`

Typical method families:

* **SFT** (`sft`)

  * Teacher forcing on prompt+completion sequences.
* **PPO family (sanity-tier in DoD)**

  * `ppo_standard`, `kl_ppo_fixed`, `kl_ppo_adaptive`, `safe_ppo`, `adaptive_rm_ppo`
* **Preference-based methods (plumbing in DoD / expanded in Level‑C)**

  * Keys vary by registry; these are required to still produce artifacts + report (no empty cells).

### HF backend policy (research layer)

The `hf` backend is optional and is designed to evolve toward paper-grade training.

* **HF Step1 (generation-only):** methods may run `generate → evaluate → artifacts`, with training explicitly marked as skipped.
* **HF Step2 (minimal training enabled):** currently **SFT** can run minimal training when enabled.

Auditability is enforced per method via `ArtifactsV1.extra`:

* `extra.skipped`: `true|false`
* `extra.skip_reason`: e.g. `"hf_step1_generation_only"` or `""`

This ensures that **every artifact makes the execution mode explicit**.

---

## Evaluation Metrics (Report Semantics)

All metrics are computed by `rlhf_eval_lab/eval/` (single source of truth) and reported through `reports/report.md`.

Metric policy (including column-level `N/A`) is defined in **SSOT**:

* `rlhf_eval_lab/registry/metrics.py`

### Direction

* `↓` means smaller is better
* `↑` means larger is better

### Core metrics (Table 1)

* **Off-support ↓**

  * Degree to which generations move outside the support of the dataset / reward signal.
* **Tail Var ↓**

  * Variance of the reward tail (e.g., top 1%); lower implies fewer extreme spikes.
* **On-support ↑**

  * Average reward within the supported region.
* **Judge ↑**

  * External judge score (optional; may be `N/A` depending on configuration).
* **Win-rate ↑**

  * Pairwise win rate (preference-style comparison; may be `N/A` depending on method).
* **KL ↓**

  * Divergence from the reference policy (policy drift). For PPO-family, this is defined against a reference snapshot per implementation.

### Table 2 blocks

* **Table 2‑A (PPO-family diagnostics)**

  * Meaningful only for PPO-family methods; others are `N/A`.
* **Table 2‑B (Preference-based diagnostics)**

  * Meaningful only for preference / active-learning methods; others are `N/A`.
* **Table 2‑C (Safety / robustness)**

  * Safety-oriented diagnostics (may be `N/A` depending on dataset/method).

### `N/A` policy

`N/A` is **not missing**: it means the metric is **not applicable** to that method by design. Applicability is enforced by registry policy (column-level rules) and validated by `rlhf-lab validate`.

---

## Datasets (Level‑C Research)

Level‑C research targets these datasets:

* **Preference learning:** HH‑RLHF
* **Safety / stress evaluation:** HarmBench

### Data is not bundled

This repository **does not bundle** HH‑RLHF / HarmBench datasets.

Paper presets assume **local JSONL files** placed under `data/` (which is ignored by git).

Dataset placement:

* `data/hh_rlhf.jsonl`
* `data/harmbench.jsonl`

(`data/*.jsonl` is ignored by `.gitignore`.)

### Dataset reproducibility (SSOT)

When a dataset is used, `run` prints and artifacts record:

* `dataset_key` (dataset identity + split + source)
* `dataset_hash` (content hash used for reproducibility)

Preset-controlled fields that determine the dataset slice:

* `split`
* `subsample_n`
* `seed`

These values are intended to make runs **auditable and reproducible** even when data is not bundled.

---

## HarmBench Usage Policy

HarmBench is used **strictly for evaluation** and **never for training**.

* This project’s default reports are designed to be **safe to share**: they contain **aggregated metrics**, not a curated dataset of harmful content.
* **Note (current behavior):** artifacts are an evaluation SSOT and may include prompts/completions depending on your local dataset inputs and configuration.

  * Treat `data/harmbench.jsonl` and generated artifacts as **sensitive local data**.
  * If you plan to distribute artifacts, you must apply appropriate redaction.

---

## Quickstart (DoD)

### 1) Run (fallback backend, torch-only)

```bash
rlhf-lab run --backend fallback --seed 0
```

Outputs:

* `artifacts/<method_key>/seed_<seed>.json`

### 2) Validate (OSS gate)

```bash
rlhf-lab validate
```

Validation guarantees:

* no missing metrics per method
* no empty cells in downstream tables
* `N/A` appears **only** where allowed by column policy

### 3) Report (Markdown tables)

```bash
rlhf-lab report
```

Outputs:

* `reports/report.md`

---

## Quickstart (HF / Level‑C)

### 1) Install HF dependencies

```bash
pip install -r requirements-hf.txt
pip install -e .
```

### 2) Run paper preset (example: HH‑RLHF)

Put the dataset file at `data/hh_rlhf.jsonl`, then:

```bash
rlhf-lab run --backend hf --preset paper_hh --seed 0
rlhf-lab report
rlhf-lab validate --report reports
```

**SFT minimal training (HF Step2):**

* `paper_hh` enables minimal SFT via `train.hf_sft_steps`.
* When enabled, `ArtifactsV1.extra.skipped` is `false` for `sft` and `extra.steps` is set.

---

## Backends

### `fallback` (default, DoD)

* `torch` only
* deterministic tokenizer + GRU tiny language model
* designed to run on CPU and in CI **without `transformers`**

### `hf` (research phase)

* Hugging Face backend
* optional dependency via `requirements-hf.txt`
* used **only** when explicitly requested (`--backend hf`)

---

## Project Structure (High Level)

* `rlhf_eval_lab/cli/` — CLI entrypoints (`run`, `validate`, `report`)
* `rlhf_eval_lab/backends/` — backend implementations (`fallback`, `hf`)
* `rlhf_eval_lab/eval/` — metric computation and evaluation runner (SSOT)
* `rlhf_eval_lab/registry/` — method + metric specifications (column `N/A` policy)
* `rlhf_eval_lab/reporting/` — artifacts, aggregation, Markdown report generation
* `tests/` — unit/integration tests (DoD gate)
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
