# RLHF-Eval-Lab

RLHF-style methods evaluation harness that **always** produces fully-populated Markdown tables (no empty cells).

This repository has **completed the DoD / OSS reliability phase** (sanity-tier numbers, not paper-grade claims).

---

## DoD (Reliability Phase)

On CPU / Colab / CI, the following must complete **without exceptions**:

```bash
pip install -e .
rlhf-lab run --backend fallback --seed 0
rlhf-lab validate
rlhf-lab report
```

And the generated Markdown report must contain:

* Table 1 / 2A / 2B / 2C
* **No empty cells**
* Values are either numeric or column-policy `"N/A"`
* Provenance is recorded in artifacts and report

> Note: In this phase, metrics are **sanity-tier** to validate the plumbing
> (they are **not** intended for paper-level claims).

---

## Level-C Research Phase (Started)

The DoD phase is **frozen** and preserved on the `main` branch and the `v0.1.0-dod` tag.

All research-oriented implementations are conducted **exclusively** on the `level-c-research` branch.

---

## Installation

### Layer 1 — DoD / OSS Reliability (default)

* CPU / CI safe
* **No `transformers` dependency**
* Uses the torch-only `fallback` backend

```bash
pip install -e .
```

### Layer 2 — Level-C Research (HF backend)

* Requires Hugging Face dependencies (`transformers`)
* Used **only** on the `level-c-research` branch or with `--backend hf`

```bash
pip install -r requirements-hf.txt
pip install -e .
```

> The dependency boundary is intentional:
> **DoD reliability must remain independent of `transformers`.**

---

## Fixed Datasets (Level-C)

Level-C research uses the following datasets **exclusively**:

* **Preference learning**: HH-RLHF
* **Stress / safety evaluation**: HarmBench

No additional datasets are introduced in this phase.

---

## HarmBench Usage Policy

HarmBench is used **strictly for evaluation** and **never for training**.

Reports contain **only aggregated metrics**.
Raw prompts or generated harmful content are **never** included in artifacts or Markdown reports.

---

## Separation from DoD Phase

Artifacts schema, evaluation logic, validation rules, and reporting pipelines defined in the DoD phase are **reused as-is**.

Any extension in the research phase must preserve DoD guarantees:

* no empty cells
* column-level `"N/A"` policy
* provenance tracking

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

* No missing metrics per method
* No empty cells downstream
* `"N/A"` only where allowed by column policy

### 3) Report (Markdown tables)

```bash
rlhf-lab report
```

Outputs:

* `reports/report.md`

---

## Backends

### `fallback` (default, DoD)

* `torch` only
* small deterministic tokenizer + GRU tiny LM
* designed to run on CPU and in CI **without `transformers`**

### `hf` (research phase)

* Hugging Face backend
* optional dependency via `requirements-hf.txt`
* used **only** in the Level-C research phase

---

## Project Structure (high level)

* `rlhf_eval_lab/cli/` : CLI entrypoints (`run`, `validate`, `report`)
* `rlhf_eval_lab/backends/` : backend implementations (`fallback`, `hf`)
* `rlhf_eval_lab/eval/` : metric computation + `runner.py` (evaluation SSOT)
* `rlhf_eval_lab/registry/` : method & metric specs (column N/A policy lives here)
* `rlhf_eval_lab/reporting/` : artifacts, aggregation, markdown report generation
* `tests/integration/` : end-to-end regression tests (DoD gate)
* `test_data/` : minimal sample JSONL (seed for future dataset wiring)

---

## Development

### Run E2E regression test

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
