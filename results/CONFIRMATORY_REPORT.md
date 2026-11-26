# Confirmatory report: Unicode release debt or tokenizer fragmentation?

Generated UTC: 2026-11-26T13:13:16.386666+00:00

This report executes the frozen decision rules in `docs/CONFIRMATORY_PREREGISTRATION.md`. It does not loosen copy, structural-ASCII, tokenizer, or quantization gates after seeing outcomes. Descriptive observations are separated from confirmatory inference.

## Frozen verdicts

| Hypothesis | Verdict | Key result |
|---|---|---|
| H1: fragmentation dose-response | **INCONCLUSIVE** | primary beta and bootstrap CI not estimable: minimum structural-validity gate unmet |
| H2: era equivalence at matched fragmentation | **INCONCLUSIVE** | primary Delta and equivalence interval not estimable: minimum structural-validity gate unmet |
| H3: scalar-listing exact floor | **INCONCLUSIVE** | {"every_legacy_zero_exact": false, "every_newest_zero_exact": false, "minimum_met": false} |
| H4: copy byte-exact ceiling | **REFUTED** | {"every_legacy_byte_exact": false, "every_newest_byte_exact": true, "minimum_met": true} |

**Pivot claim:** H2 era-equivalence is **not confirmed**. The verdict is inconclusive because the frozen minimum structural-validity gate was not met; this run cannot either confirm or refute the paper's pivot claim.

## Freeze-order and completion evidence

VERIFY_REAL: inventory freeze artifact created_utc=2026-11-26T12:00:53.539586+00:00
VERIFY_REAL: inventory freeze commit d06208a555d43ef42419d3387297350fe20838e2 committer_time=2026-11-26T14:02:55+02:00
VERIFY_REAL: first post-freeze memory smoke started=2026-11-26T12:08:32.215798+00:00 (infrastructure smoke only, excluded from confirmatory rows)
VERIFY_REAL: first inventory probe batch completed=2026-11-26T12:08:51.153567+00:00
VERIFY_REAL: last inventory probe batch completed=2026-11-26T13:11:36.593562+00:00
VERIFY_REAL: inventory freeze predates first inventory probe generation: True
VERIFY_REAL: frozen artifact hash verification pass=True
VERIFY_REAL: raw completion audit pass=True
VERIFY_REAL: extractor mutation checks pass=True
VERIFY_REAL: analysis implementation mutation checks pass=True
VERIFY_REAL: raw-response corruption mutation detected=True before_exact=True after_exact=False
VERIFY_REAL: corrupted-artifact hash mutation detected=True target_error_detected=True
VERIFY_REAL: no completed row regenerated=True rows_before_resume=1780 rows_after_completion=2880 duplicate_completion_keys=0
VERIFY_REAL: baseline prefix hash match=True current-startswith-gemma-checkpoint=True

The candidate universe, tokenizer scan, inventory, probe plan, environment freeze, tokenizer freeze, design documents, and inventory-construction scripts were hash-pinned before confirmatory generation. Raw responses were appended to `results/confirmatory/confirmatory_raw.jsonl`; scoring was run only after completion.

## Execution completion and resume audit

| Carrier | Planned rows | Completed rows | Scope |
|---|---:|---:|---|
| `tinyllama-1.1b-chat-v1.0-bf16` | 460 | 460 | primary |
| `qwen2.5-3b-instruct-bf16` | 500 | 500 | primary |
| `llama32-3b-instruct-mlx8` | 500 | 500 | primary |
| `gemma3-4b-it-mlx8-mirror` | 460 | 460 | primary |
| `qwen3-4b-instruct-2507-mlx8` | 500 | 500 | primary |
| `mistral7b-instruct-v03-mlx8` | 460 | 460 | descriptive only |

Resume audit: `True`. The pre-resume baseline contained 1780 rows; the final raw file contains 2880. Actual increments were {"gemma3-4b-it-mlx8-mirror": 140, "llama32-3b-instruct-mlx8": 0, "mistral7b-instruct-v03-mlx8": 460, "qwen2.5-3b-instruct-bf16": 0, "qwen3-4b-instruct-2507-mlx8": 500, "tinyllama-1.1b-chat-v1.0-bf16": 0}. Completion-key duplicates: 0; carrier/probe-key duplicates: 0.

## Frozen scope, resolutions, and execution deviations

- Mistral-7B was demoted to descriptive robustness only in the pre-generation frozen judgment resolutions. It never enters H1-H4 primary carrier minima or verdicts.
- Expanded bf16 comparator downloads were skipped before generation under the same frozen resolution. MLX8 carriers are therefore not automatically excluded, but the intended direct quantization contrast is absent and remains a limitation.
- The pre-inference resolutions otherwise kept the strict gates unchanged: ASCII copy at least 23/24, wrapper failures at most 6/124, ASCII count at least 11/12, ASCII scalar listing at least 11/12.
- During the interrupted prior execution, a Qwen3 load overlapped the tail of Gemma generation, contrary to the intended sequential-loading operations rule. That Qwen3 batch failed before writing any Qwen3 rows. Existing Gemma checkpoint rows were retained under the append-only no-regeneration rule; all work after the interruption was strictly carrier-sequential.
- After that memory error, Qwen3 was rerun at batch size 8 rather than 16, following the predefined downward batch ladder. Batch size is scheduling metadata, not a semantic decode parameter.
- Before final analysis, an implementation-only draft CI-width criterion was removed from the H2 verdict code because it was not part of the frozen ±10 pp decision rule. This correction cannot affect the final classification: the minimum structural-validity gate was already unmet.

## T1. Carrier validity and quantization gates

| Carrier | Primary | ASCII copy | Wrapper failures | ASCII count | ASCII scalar | Copy valid | Structural valid | Quantization status | H1-H3 | H4 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `tinyllama-1.1b-chat-v1.0-bf16` | True | 23/24 | 0/124 | 0/12 | 0/12 | True | False | bf16_no_comparison_required | False | True |
| `qwen2.5-3b-instruct-bf16` | True | 23/24 | 0/124 | 3/12 | 0/12 | True | False | bf16_no_comparison_required | False | True |
| `llama32-3b-instruct-mlx8` | True | 23/24 | 16/124 | 6/12 | 2/12 | False | False | narrow_validation_only_no_bf16_comparator_downloaded_allowed_by_frozen_user_resolution | False | False |
| `gemma3-4b-it-mlx8-mirror` | True | 24/24 | 0/124 | 2/12 | 1/12 | True | False | narrow_validation_only_no_bf16_comparator_downloaded_allowed_by_frozen_user_resolution | False | True |
| `qwen3-4b-instruct-2507-mlx8` | True | 23/24 | 1/124 | 4/12 | 1/12 | True | False | narrow_validation_only_no_bf16_comparator_downloaded_allowed_by_frozen_user_resolution | False | True |
| `mistral7b-instruct-v03-mlx8` | False | 23/24 | 4/124 | 0/12 | 0/12 | True | False | narrow_validation_only_no_bf16_comparator_downloaded_allowed_by_frozen_user_resolution | False | False |

Minimum H1/H2 carrier gate met: **False** (0 structurally valid primary carriers across 0 tokenizer families). Minimum H4 gate met: **True** (4 copy-valid primary carriers across 3 tokenizer families).

Gate outcomes are data. In particular, Llama-3.2 fails the copy-wrapper gate with 16 format failures, and Gemma fails the strict structural-ASCII gate despite passing copy validity. No threshold was adjusted.

## H1. Fragmentation dose-response

Primary pooled estimand: `beta_frag` for ordinal fragmentation level in the covariate-adjusted logistic model. Primary rows available to the confirmatory fit: 0 from 0 structurally valid carriers.

- beta: **NA**
- odds ratio per level: **NA**
- percentile 95% bootstrap CI: **[None, None]**
- valid bootstrap replicates: **0/5000**
- count-based one-sided bootstrap p: NA
- normal-approximate one-sided bootstrap p: NA
- Holm-adjusted p: NA
- nonnegative carrier-specific coefficients: 0 across 0 tokenizer families

Primary estimation was not attempted because the frozen minimum was unmet: fewer than four structurally valid carriers or fewer than three tokenizer families; primary H1 is inconclusive without pooling

Verdict: **INCONCLUSIVE**. This is a gate-driven inconclusive result, not evidence against fragmentation.

## H2. Era equivalence at matched fragmentation

Complete matched pairs in the primary population: 0. Incomplete pair groups excluded without replacement: 0.

- Delta (newest minus legacy failure): **NA**
- percentile 95% bootstrap CI: **None**
- bootstrap SE: NA
- TOST component p-values: upper=NA, lower=NA, max=NA
- Holm-adjusted TOST p: NA
- exact-zero sensitivity p (descriptive): NA
- raw bootstrap tail proportions (diagnostic, not primary): <=-10 pp NA, >=+10 pp NA

Verdict: **INCONCLUSIVE** against the frozen ±10 percentage-point band. The era-equivalence pivot is therefore not confirmed by this run.

## H3. Scalar-listing exact floor

| Eligible carrier | newest exact | legacy exact |
|---|---:|---:|

Verdict: **INCONCLUSIVE**. The cross-carrier exact-floor hypothesis is inconclusive when fewer than four structurally valid quantization-eligible primary carriers exist.

## H4. Copy byte-exact ceiling

| Copy-valid carrier | newest byte-exact | legacy byte-exact |
|---|---:|---:|
| `gemma3-4b-it-mlx8-mirror` | 60/60 | 39/40 |
| `qwen2.5-3b-instruct-bf16` | 60/60 | 40/40 |
| `qwen3-4b-instruct-2507-mlx8` | 60/60 | 40/40 |
| `tinyllama-1.1b-chat-v1.0-bf16` | 60/60 | 40/40 |

Byte-level failures inside copy-valid primary carriers:

| Carrier | Probe | Class | First byte offset | Target hex context | Output hex context |
|---|---|---|---:|---|---|
| `gemma3-4b-it-mlx8-mirror` | `gemma3-4b-it-mlx8-mirror:copy:unicode:0096` | `multi_codepoint_corruption` | 0 | `f0928587` | `746f6f6c5374726970427574746f6e` |

Verdict: **REFUTED** under the exact-ceiling rule. Any byte-level failure in a copy-valid eligible carrier refutes the exact universal form, even when the failure is isolated.

## Required tables and figures

Primary CSV artifacts in `results/confirmatory/tables/`:

- `T1_carrier_validity_quantization_gates.csv`
- `T2_inventory_cells.csv`
- `T3_h1_coefficients.csv`
- `T4_h2_matched_pairs.csv`
- `T5_copy_ceiling.csv`
- `T6_scalar_floor.csv`
- `T7_count_errors.csv`

Primary figures in `results/confirmatory/figures/`:

- `F1_failure_heatmap.png`
- `F2_fragmentation_dose_response.png`
- `F3_era_effect_forest.png`
- `F4_copy_count_scalar_triad.png`
- `F5_copy_near_miss_taxonomy.png`

T2 prints requested and achieved inventory cells, including structurally empty newest-L1 cells and item-type substitutions. Because no carrier entered the primary structural population, H1/H2 adjusted curves and pooled forest estimates are marked not estimable rather than visually invented.

## Pre-registered sensitivity analyses

Sensitivity outputs are stored under `sensitivity_analyses` in `results/confirmatory/confirmatory_analysis.json`. The planned sensitivities include continuous tokens-per-cluster, level fixed effects, legacy-only fits, within-type fits, era interaction, linear probability, pair-mismatch regression, logistic-era average marginal effect, exact-zero testing, and quantization exclusion. With zero structurally valid primary carriers, the structural sensitivities have no confirmatory population to estimate. The quantization-exclusion sensitivity is additionally limited by the pre-generation decision not to download bf16 comparator weights; it cannot estimate the originally intended direct quantization contrast.

## Exploratory observations, not confirmation

**Everything in this section is descriptive.** Every primary carrier failed at least one strict structural-ASCII validity gate, so these numbers must not be reinterpreted as confirmatory structural evidence.

| Carrier | Unicode copy exact | Overlap count exact | Unicode scalar exact | Exploratory matched Delta [bootstrap CI] | Exploratory fragmentation beta [normal CI] |
|---|---:|---:|---:|---:|---:|
| `tinyllama-1.1b-chat-v1.0-bf16` | 100/100 (100.0%) | 0/240 (0.0%) | 0/32 (0.0%) | 0.00 pp [0.00, 0.00] | NA |
| `qwen2.5-3b-instruct-bf16` | 100/100 (100.0%) | 69/240 (28.7%) | 0/32 (0.0%) | -2.50 pp [-8.33, 3.33] | NA |
| `llama32-3b-instruct-mlx8` | 84/100 (84.0%) | 2/240 (0.8%) | 0/32 (0.0%) | 1.67 pp [0.00, 4.17] | NA |
| `gemma3-4b-it-mlx8-mirror` | 99/100 (99.0%) | 131/240 (54.6%) | 0/32 (0.0%) | 15.83 pp [4.98, 25.85] | 0.3105 [-0.570, 1.191] |
| `qwen3-4b-instruct-2507-mlx8` | 100/100 (100.0%) | 44/240 (18.3%) | 0/32 (0.0%) | 0.00 pp [-3.33, 3.33] | NA |
| `mistral7b-instruct-v03-mlx8` | 96/100 (96.0%) | 0/240 (0.0%) | 0/32 (0.0%) | 0.00 pp [0.00, 0.00] | NA |

The descriptive pattern is strong byte-copy performance in several carriers alongside near-total scalar-listing failure and heterogeneous count performance. That pattern motivated the confirmatory design, but the frozen structural-gate failure means these observations remain hypothesis-generating here. Copy failures and near misses are classified in T5/F5; scalar error classes are in T6; count signed errors are in T7.

## Mutation and integrity tests

- Scorer extraction mutations: **True** (`extractor_mutation_checks.json`)
- Frozen artifact hash verification: **True** (`artifact_hash_check.json`)
- Corrupted-artifact detection mutation: **True**; one byte was flipped in a temporary copy and hash verification reported `sha256_mismatch`
- Corrupted raw-response scoring mutation: **True**; one output byte corruption changed task_exact from True to False without touching the retained raw file
- Logistic/TOST/Holm implementation mutation checks: **True**
- Raw completion audit: **True**; raw SHA-256 before scoring `d16c5336a2ff1f97fa63d27b21dd4a19783cfc19d4c2c470ad6e92423c7f8607`
- Power-analysis validation and mutation self-test: **True**
- Power-analysis deterministic rerun match after removing timestamps: **True**
- Inventory audit/mutation reproducibility: **True**
- Chat-template smoke/rendering checks: **True**
- Inference-tokenizer hash verification: **True**

## Limitations

1. **No confirmatory structural population:** no primary carrier passed both strict ASCII structural gates while also passing copy validity. H1, H2, and H3 are therefore inconclusive by design, not negative evidence.
2. **Strict-gate tradeoff:** the ASCII gates intentionally exclude carriers that cannot reliably count short ASCII strings or print exact scalar notation. This protects interpretability but means most measured Unicode structural behavior is descriptive only.
3. **Quantization contrast missing:** direct bf16 comparator inference for MLX8 mirrors was not run under the frozen pre-generation resolution. Quantization cannot be ruled out as a contributor to MLX8 exactness differences.
4. **Sequential-loading deviation in interrupted work:** the prior agent briefly overlapped a Qwen3 load with Gemma generation. No Qwen3 row was written during that failed attempt, and all resumed work was sequential. Existing checkpoints were preserved exactly as required.
5. **Structural empty cells:** newest low-fragmentation complex/ZWJ cells are intrinsically impossible in frozen byte-fallback vocabularies. H2 uses only the L2-L4 overlap core.
6. **Prompt scope:** results apply to the frozen prompts, strict extraction rules, deterministic decoding, local MLX artifacts, and small instruct models tested here. They do not generalize automatically to other prompts, APIs, sampling, or larger models.
7. **Host Unicode stack:** host Python Unicode data is 15.0.0; Unicode 16/17/18 behavior comes from pinned `unicodedata2` 18.0.0 and the explicit segmentation implementation.
8. **Constituent dependence:** probe-level bootstrap does not fully model repeated constituent use across probes, a limitation accepted in the frozen judgment resolutions.

## Reproducibility

Key artifacts:

- `results/confirmatory/environment_freeze.json` sha256 `e9b50aec9d999bbe2866b57cad40df292b51114f81eda4b46cadeb5010beb77f`
- `results/confirmatory/tokenizer_freeze.json` sha256 `7eff2f634c0b035b68c91442d73ff4644532aa146f0997f96b79360a1ad5f1e5`
- `results/confirmatory/inventory_freeze.json` sha256 `532665e42cb4a439522f02a47a4c39d0be2ecf6a4eb514d4f457476cd0827e4c`
- `results/confirmatory/inventory_audit.json` sha256 `92b91d3222859ca41f668c11619aad5afe09cf27bc2d5d7ee85d8ea7a6db7bbf`
- `results/confirmatory/probe_plan.jsonl` sha256 `fc20aea12260e2ec5c61703a81c86c2244b00b0040697e9484fcd715b2d96eda`
- `results/confirmatory/confirmatory_raw.jsonl` sha256 `d16c5336a2ff1f97fa63d27b21dd4a19783cfc19d4c2c470ad6e92423c7f8607`
- `results/confirmatory/raw_completion_audit.json` sha256 `5c6bbfd0242f3474ba24fe134c9281fc4a012a73f6d262323e7e2b780c5573b1`
- `results/confirmatory/confirmatory_scored.jsonl` sha256 `b47016aacefaf294150cf6f643d95d00b790fc2733c09c112ae8a13b0140eb22`
- `results/confirmatory/carrier_gates.json` sha256 `404eb0f7f06968c7f639931f3636a997ca1800033d205cd22ca053c5665d02b7`
- `results/confirmatory/confirmatory_analysis.json` sha256 `635de1b45b071868265d7993d56891c6959bb25703433eb0fc879b950c87a1bd`
- `results/confirmatory/confirmatory_verdict.json` sha256 `80d35a984629cd081a4721a515bb6e1061e7512b21d31a6a212e65d922834d15`
- `results/confirmatory/resume_no_regeneration_audit.json` sha256 `f42475d7450c3d088f7ebf118d58dd4497d80d5983e46d8e9c46db18d80f64fb`

Scored rows: 2880; raw rows: 2880. Runtime, memory, prompt, revision, dtype, template, batch, and finish metadata are in `confirmatory_runtime.jsonl` and `memory_smoke.jsonl`.

