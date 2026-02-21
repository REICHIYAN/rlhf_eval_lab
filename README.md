# RLHF-Eval-Lab

**RLHF-style methods evaluation harness** that *always* produces fully populated, self-auditable Markdown reports — **no empty cells, no silent failures**.

This repository is intentionally built in **two layers**:

* **Layer 1 (DoD / OSS reliability, frozen):** a torch-only `fallback` backend (**no `transformers`**) that guarantees the end-to-end pipeline is *unbreakable* on CPU/CI.
* **Layer 2 (Level-C research, active):** an optional Hugging Face `hf` backend used for paper-grade experiments, developed on `level-c-research`.

> In the DoD layer, reported values are **sanity-tier**. They validate plumbing, determinism, and auditability — **not paper claims**.

---

## Definition of Done (DoD) — Reliability Phase

On **CPU / Colab / CI**, the following commands must complete **without exceptions**:

```bash
pip install -e .
rlhf-lab run --backend fallback --preset offline_hh_small --seed 0
rlhf-lab report
rlhf-lab validate --report reports
```

Notes:

* `rlhf-lab validate` is strict artifact validation.
* `rlhf-lab validate --report reports` additionally validates the **rendered** `reports/report.md` invariants (tables fully populated, `N/A` policy enforced).

The generated report must satisfy **all invariants**:

* Table 1 / Table 2-A / Table 2-B / Table 2-C are always present
* **No empty cells**
* Every cell is either a **number** or column-policy **`N/A`**
* Full provenance is recorded (**backend, model, tokenizer, config hash, git commit, seed**)

---

## Generated outputs & Git hygiene (non-negotiable)

This repo generates outputs during normal operation:

* `artifacts/` — per-method artifacts (`seed_*.json`)
* `reports/` — aggregated Markdown report (`reports/report.md`)

**Rule:** do **not** commit generated outputs (they are git-ignored by default).
If you need to share an example report, copy it to a non-generated location (e.g. `examples/report_example.md`) instead of committing `reports/report.md`.

---

## Two-branch policy

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

### Layer 2 — Level-C Research (HF backend)

* Requires Hugging Face dependencies (`transformers`)
* Used only when explicitly requested (`--backend hf`)

```bash
pip install -r requirements-hf.txt
pip install -e .
```

> The dependency boundary is intentional: **DoD reliability must remain independent of `transformers`.**

---

## Algorithms (Methods)

This project evaluates RLHF-style methods through a common **ArtifactsV1** schema, ensuring results are comparable and auditable.

### Implemented in DoD (fallback backend)

The fallback backend runs minimal, deterministic **sanity-tier** implementations that are sufficient to:

* run end-to-end on CPU/CI
* produce artifacts
* compute metrics
* generate fully populated tables

Method keys are defined in the SSOT:

* `rlhf_eval_lab/registry/methods.py`

Typical families:

* **SFT** (`sft`)
* **PPO family (sanity-tier in DoD)**

  * `ppo_standard`, `kl_ppo_fixed`, `kl_ppo_adaptive`, `safe_ppo`, `adaptive_rm_ppo`
* **Preference-based methods (plumbing in DoD; expanded in Level-C)**

  * Must still produce artifacts + report (no empty cells), even if training is a placeholder.

### HF backend policy (research layer)

The `hf` backend is optional and is designed to evolve toward paper-grade training.

* **HF Step1 (generation-only):** methods may run `generate → evaluate → artifacts`, with training explicitly marked as skipped.
* **HF Step2 (minimal training enabled):** minimal training is enabled only when explicitly implemented/configured.

Auditability is enforced per method via `ArtifactsV1.extra`:

* `extra.skipped`: `true|false`
* `extra.skip_reason`: e.g. `"hf_step1_generation_only"` or `""`
* `extra.steps`: executed training steps (0 if no training)

This ensures that **every artifact makes the execution mode explicit**.

---

## Evaluation Metrics (Report Semantics)

All metrics are computed by `rlhf_eval_lab/eval/` (single source of truth) and reported through `reports/report.md`.

Metric policy (including column-level `N/A`) is defined in the SSOT:

* `rlhf_eval_lab/registry/metrics.py`

### Direction

* `↓` means smaller is better
* `↑` means larger is better

### Core metrics (Table 1)

* **Off-support ↓**: degree to which generations move outside the support of the dataset / reward signal.
* **Tail Var ↓**: variance of the reward tail (e.g. top 1%); lower implies fewer extreme spikes.
* **On-support ↑**: average reward within the supported region.
* **Judge ↑**: external judge score (optional; may be `N/A`).
* **Win-rate ↑**: pairwise win rate (optional; may be `N/A`).
* **KL ↓**: divergence from a reference policy (policy drift).

### Table 2 blocks

* **Table 2-A (PPO-family diagnostics)**: meaningful only for PPO-family methods; others are `N/A`.
* **Table 2-B (Preference-based diagnostics)**: meaningful only for preference/active methods; others are `N/A`.
* **Table 2-C (Safety / robustness)**: safety-oriented diagnostics (may be `N/A` depending on dataset/method).

### `N/A` policy

`N/A` is **not missing**: it means the metric is **not applicable** to that method by design.
Applicability is enforced by registry policy (column-level rules) and validated by:

* `rlhf-lab validate` (artifacts)
* `rlhf-lab validate --report reports` (artifacts + rendered report invariants)

---

## HF: KL & PPO audit diagnostics (proxy semantics)

For `backend=hf`, KL-related values shown in reports are **audit-oriented proxies** derived from sampled trajectory log-probabilities (token-mean), not the full-distribution KL.

* `kl`: preferred **non-negative proxy** used for reporting stability

  * `kl = E[ | logp_post - logp_ref | ]` (token-mean)
* `kl_ref_sq`: squared-difference proxy for drift magnitude

  * `kl_ref_sq = E[ (logp_post - logp_ref)^2 ]` (token-mean)
* `kl_ref` / `kl_ref_pre`: signed mean differences kept for debugging (can be negative)

  * `kl_ref = E[ logp_post - logp_ref ]`, `kl_ref_pre = E[ logp_pre - logp_ref ]`

PPO ratio diagnostics (when emitted) are computed on per-token mean logprobs:

* `ratio_mean_pre`: pre-update `E[ exp(logp_pre - logp_old) ]` (typically ≈ 1)
* `ratio_mean`: post-update `E[ exp(logp_post - logp_old) ]`
* `clipfrac`: fraction of samples where post-update ratio is outside `[1-clip, 1+clip]`

---

## Datasets

### 1) Built-in offline dataset for wiring (bundled)

This repo **bundles a tiny offline prompt set** used for deterministic, offline E2E runs:

* `test_data/offline_prompts_small.jsonl` (63 prompts)
* `offline_hh_small preset subsamples to 32 prompts by default for CI speed.`

A preset is provided:

* `rlhf_eval_lab/config/presets/offline_hh_small.yaml`

This is the recommended starting point for **“reasonable” method-to-method comparison** in the DoD layer (still sanity-tier, but less degenerate than ultra-tiny prompt sets).

### 2) Real research datasets (not bundled)

This repository **does not bundle** HH-RLHF / HarmBench.
Paper presets assume **local JSONL files** placed under `data/` (ignored by git).

Suggested placement:

* `data/hh_rlhf.jsonl`
* `data/harmbench.jsonl`

### Dataset reproducibility (SSOT)

When a dataset is used, `run` prints and artifacts record:

* `dataset_key` (dataset identity + split + source)
* `dataset_hash` (content hash used for reproducibility)

Preset-controlled fields that determine the dataset slice:

* `split`
* `subsample_n`
* `seed`

---

## HarmBench usage policy

HarmBench is used **strictly for evaluation** and **never for training**.

* Default reports are designed to be **safe to share**: they contain aggregated metrics, not a curated dataset.
* Artifacts may contain prompts/completions depending on your local dataset inputs and configuration.

  * Treat `data/harmbench.jsonl` and generated artifacts as **sensitive local data**.
  * If you plan to distribute artifacts, apply appropriate redaction.

---

## Quickstart (DoD)

### 1) Offline E2E (recommended)

Runs fully offline on CPU/CI and is stable enough to compare methods at a basic sanity tier.

```bash
rm -rf artifacts reports
rlhf-lab run --backend fallback --preset offline_hh_small --seed 0
rlhf-lab report
rlhf-lab validate --report reports
```

### 2) Default DoD smoke run (builtin prompts)

This uses a tiny built-in prompt set (3 prompts). It is primarily for quick wiring checks.

```bash
rm -rf artifacts reports
rlhf-lab run --backend fallback --seed 0
rlhf-lab report
rlhf-lab validate --report reports
```

Outputs:

* `artifacts/<method_key>/seed_<seed>.json`
* `reports/report.md`

---

<!-- METRICS_SSOT:START -->

## Metrics SSOT（registry/metrics.py から自動生成）

このブロックは `python -m rlhf_eval_lab.utils.update_readme_metrics_ssot` により更新されます。
列名・意味・N/A規約・表示桁のSSOTを README に同期します。

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

## Quickstart (HF / Level-C)

### 1) Install HF dependencies

```bash
pip install -r requirements-hf.txt
pip install -e .
```

### 2) Run an HF preset (example)

```bash
rm -rf artifacts reports
rlhf-lab run --backend hf --preset paper_hh --seed 0
rlhf-lab report
rlhf-lab validate --report reports
```

---

## Testing

### Run DoD regression tests (CI-equivalent)

```bash
pytest -q
```

---

## Project structure (high level)

* `rlhf_eval_lab/cli/` — CLI entrypoints (`run`, `validate`, `report`)
* `rlhf_eval_lab/backends/` — backend implementations (`fallback`, `hf`)
* `rlhf_eval_lab/eval/` — metric computation and evaluation runner (SSOT)
* `rlhf_eval_lab/registry/` — method + metric specifications (column `N/A` policy)
* `rlhf_eval_lab/reporting/` — artifacts, aggregation, Markdown report generation
* `tests/` — unit/integration tests (DoD gate)
* `test_data/` — bundled offline JSONL for deterministic dataset wiring

---

## Roadmap (short)

### Reliability track (DoD)

* ✅ Bundled offline dataset + preset + E2E integration test (`offline_hh_small`)
* ⏳ README polish (keep SSOT consistent with code + report interpretation)
* ⏳ Release ritual (CHANGELOG, version pin, tag)

### Research track (Level-C)

* ✅ HF audit semantics documented (KL proxy + PPO ratio diagnostics)
* ✅ HF optional E2E test exists (opt-in)
* ⏳ Expand HF training coverage (KL-PPO fixed/adaptive, Safe PPO) while preserving DoD invariants

---

## License

See `LICENSE`.

## Citation

See `CITATION.cff`.

