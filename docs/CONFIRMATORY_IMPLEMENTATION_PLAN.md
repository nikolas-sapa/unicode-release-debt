# Confirmatory implementation plan

Status: **implementation plan only**. No confirmatory Unicode model execution has been run on this branch.
Companion document: `docs/CONFIRMATORY_PREREGISTRATION.md`.
Power script already implemented: `src/confirmatory_power_analysis.py`; output `results/confirmatory_power_analysis.json`.

This plan is checkpointed so inventory construction, validation, model execution, scoring, and analysis cannot blur together. Every checkpoint has an artifact, an audit, and a stop condition. Implementation must be additive: do not rewrite round-2 artifacts, README claims, or the existing pilot harness semantics. Shared functions may be reused only when byte-for-byte behavior is preserved.

## 1. Non-goals

1. No execution of the confirmatory probe suite in this design branch.
2. No edits to round-2 raw or scored data.
3. No changes to the frozen copy prompt, structural prompts, extraction rules, gates, hypotheses, alpha, or decision rules after Unicode inference begins.
4. No claim that a statistically significant result proves tokenizer fragmentation causally in the experimental-randomization sense.
5. No attempt to make a complete era x fragmentation factorial where the Unicode/tokenizer system makes cells impossible.
6. No merge to `main` as part of the design branch.

## 2. Planned additive files

Create these files during execution preparation. Names may be implemented exactly as written; if a name changes, update this plan before inference.

```text
src/confirmatory_common.py              # carrier configs, prompts, constants, hashes
src/build_confirmatory_candidates.py    # UCD candidate universe construction
src/scan_confirmatory_candidates.py     # tokenizer fragmentation scan
src/build_confirmatory_inventory.py     # count/copy/scalar-listing inventory and pair matching
src/audit_confirmatory_inventory.py     # structural and mutation audits
src/run_confirmatory_behavioral.py      # checkpointed MLX runner, adapted additively from round 2
src/score_confirmatory_behavioral.py    # strict extraction and taxonomy
src/analyze_confirmatory.py             # gates, bootstrap inference, tables, figures, verdicts
src/check_confirmatory_artifacts.py     # artifact hash/completeness validation
```

Outputs live under `results/confirmatory/`. Large intermediate scans may remain local when regeneration is cheap, but their SHA-256 hashes, row counts, schema versions, and creation commands must be committed in metadata. The final inventory, probe plan, raw outputs, scored outputs, analysis tables, and verdict JSON are committed unless size makes raw-output commit unreasonable; in that case a hash manifest plus a compressed artifact decision must be reviewed explicitly.

## 3. Environment and input freeze

### Checkpoint E0: repository and package freeze

Before building probes, record:

- git commit and branch
- Python version
- macOS/host architecture
- package versions from `requirements.txt` and the active venv, especially:
  - numpy 2.5.5
  - regex 2026.11.24
  - unicodedata2 18.0.0
  - tokenizers 0.23.3
  - sentencepiece 0.2.3
  - transformers 5.24.1
  - mlx 0.32.5
  - mlx-lm 0.32.2
- SHA-256 of every file in `data/unicode/` used by the builder
- Hugging Face cache path and available disk

Artifact: `results/confirmatory/environment_freeze.json`.

Stop if any package differs from the pinned requirement. Disk gating must be explicit before downloads: a cold-cache primary slate requires about 27.4 GiB; the three bf16 quantization comparators add about 21.5 GiB; the worst-case total is about 48.9 GiB. At design time free disk is 64 GiB, leaving only about 15 GiB in that worst case. Require at least 55 GiB free before starting the full plan with quantization comparators, or at least 35 GiB free if the user chooses to defer some comparator downloads. Never silently skip a comparator; record the decision before inference.

## 4. Tokenizer freeze

### Checkpoint T1: acquire and verify tokenizer artifacts

Verify existing round-2 tokenizer inputs for Qwen2.5, Qwen3, Gemma, and TinyLlama exactly as in `results/behavioral_round2/tokenizer_linkage_round2.json` and related scan metadata. Add the two new carrier tokenizers.

Pinned new inference artifacts:

```text
llama32-3b-instruct-mlx8
  inference repo: mlx-community/Llama-3.2-3B-Instruct-8bit
  revision: ff054899609078569493def2823f9acd2780c0c9
  canonical upstream: meta-llama/Llama-3.2-3B-Instruct
  upstream revision: 0cb88a4f764b7a12671c53f0838cd831a0843b95

mistral7b-instruct-v03-mlx8
  inference repo: mlx-community/Mistral-7B-Instruct-v0.3-8bit
  revision: c282bcd11a64be44ac3f130f0b819d48ecc1846d
  canonical upstream: mistralai/Mistral-7B-Instruct-v0.3
  upstream revision: c170c708c41dac9275d15a8fff4eca08d52bab71
```

Expected tokenizer hashes are those listed in the pre-registration. Compute hashes locally after download; do not trust metadata alone. Record tokenizer class, vocabulary size, normalizer, pre-tokenizer, decoder, byte-fallback behavior, chat-template source, and whether a canonical overlay is required. For mirrored artifacts, additionally compare the inference tokenizer and the bf16 comparator tokenizer on a fixed Unicode audit corpus. Tokenizer JSON/config container hashes may differ even when the core vocabulary is equivalent (already observed for the Llama-3.2 and Mistral mirrors); record whole-file hashes and behavioral token-ID equality separately. Inventory construction always uses the exact inference tokenizer, never an assumed canonical equivalent.

Artifacts:

- `results/confirmatory/tokenizer_files/` or hashed cache references
- `results/confirmatory/tokenizer_freeze.json`

Mutation check: mutate one byte in a temporary copy of each tokenizer file and verify that the audit rejects the hash. Never mutate the retained artifact.

Stop condition: any hash mismatch, missing chat template that cannot be overlaid byte-identically, or tokenizer backend that cannot reproduce round-trip/token-count diagnostics.

## 5. Candidate universe construction

### Checkpoint C1: parse pinned Unicode inputs

Use the existing parser logic in `src/ucd_build.py` and `src/testset.py` where possible. Inputs include:

- `data/unicode/UnicodeData.txt`
- `data/unicode/DerivedAge.txt`
- `data/unicode/emoji-data.txt`
- `data/unicode/emoji-test.txt`
- `data/unicode/emoji-frequency.html`
- `data/unicode/publicationdates.html` if present in the execution tree
- grapheme-break files needed for validation

Use only fully-qualified emoji rows. Preserve exact source line, sequence type, version, name, surface family, concept key, frequency row, codepoint list, UTF-8 byte length, plane composition, complexity, and release metadata.

Artifact: `results/confirmatory/candidate_universe.jsonl`.

### Checkpoint C2: non-emoji candidate construction

Eligibility follows the pilot logic: Unicode scalar characters in general categories L, N, P, or S; exclude emoji-property characters, mark categories, controls, surrogates, noncharacters, unassigned codepoints, whitespace separators, and deprecated or otherwise invalid scalar values. Record every exclusion rule in schema metadata.

Candidate types:

1. `atomic_nonemoji`: one eligible scalar
2. `complex_nonemoji`: one eligible scalar plus the pilot combining mark U+20D0, unless the build audit expands this to a documented attested mark inventory before freeze
3. `atomic_emoji`: fully-qualified atomic emoji sequence
4. `zwj_emoji`: fully-qualified ZWJ emoji sequence

Use deterministic seed `20261126` for all sampling. The universe should include all newest eligible non-emoji candidates and a deterministic plane/version-stratified legacy sample large enough to exceed the feasibility margins observed in design. The design-scan reference had 29,599 candidates with the following composition:

```text
newest atomic_nonemoji:  23469
newest complex_nonemoji:  1968
newest atomic_emoji:        23
newest zwj_emoji:           39
legacy atomic_nonemoji:   2500
legacy complex_nonemoji:  1000
legacy atomic_emoji:       503
legacy zwj_emoji:           97
```

The execution build may use a superset, but never a smaller legacy sample unless the pre-inventory feasibility audit still passes with margin. If a superset is used, record exact quotas and hashes before probe selection. Do not adapt the sample after seeing model outputs; models have not run at this point.

Audit requirements:

- no surrogate or noncharacter scalars
- no duplicate `(item_type, text)` candidates within an era
- every candidate has valid UTF-8 and nonzero grapheme count
- era assignment matches the pre-registration
- release dates/versions parse or are explicitly marked unknown
- emoji rows are fully-qualified
- complex items have greater codepoint count than their atomic base

Artifact: `results/confirmatory/candidate_universe_audit.json`.

Mutation checks:

1. Flip one newest candidate era label to legacy in a temporary copy; audit must fail.
2. Introduce one duplicate text in a temporary copy; audit must fail.
3. Replace one fully-qualified status with minimally-qualified in a temporary copy; audit must fail.

## 6. Tokenizer fragmentation scan

### Checkpoint S1: scan candidates on every carrier tokenizer

For each candidate and tokenizer, record:

- token IDs
- token pieces or byte representations where available
- decoded round-trip success
- token count after dummy-prefix stripping
- tokens per cluster for the isolated candidate
- UTF-8 byte length
- grapheme count
- split-codepoint diagnostics when reusable from `src/tokscan.py`

Use backend-specific dummy-prefix handling. Mistral's SentencePiece-style leading `▁` behavior must be explicitly stripped when it is an artifact rather than part of the input. Llama-3 byte-level BPE must be encoded without special tokens. Record exact backend implementation versions.

Artifacts:

- local full scan, path recorded in metadata
- compact committed or regenerable scan metadata in `results/confirmatory/candidate_scan.metadata.json`
- `results/confirmatory/candidate_scan_audit.json`

Stop condition: any tokenizer crash, inability to strip dummy prefixes deterministically, or decode/encode inconsistency that makes token counts unreliable.

Mutation checks:

1. In a temporary scan copy, add one to a token count and verify the inventory audit rejects the derived level if it changes.
2. Remove the dummy-prefix strip in a temporary test input and verify the audit detects the systematic shift.

## 7. Confirmatory inventory construction

### Checkpoint I1: copy and scalar-listing inventories

Construct the copy inventory exactly as pre-registered:

- newest: 10 atomic non-emoji, 20 atomic emoji, 10 complex non-emoji, 20 ZWJ emoji = 60
- legacy: 10 of each type = 40
- the same 24 ASCII copy controls used in round 2

Use deterministic seed `20261126`, stratified by release version and type. Prefer round-2 copy items where available so the replication is close to the pilot. Record whether every selected item existed in round 2.

Construct scalar-listing inventory:

- 16 newest Unicode, 16 legacy Unicode
- 8 atomic and 8 complex/ZWJ within each era
- 12 ASCII controls with one to three ASCII scalars, using the exact cyclic count schedule in the pre-registration

The 12 ASCII count controls likewise use the exact cyclic one-to-five cluster schedule in the pre-registration. Record expected scalar strings and integer answers exactly. Do not normalize targets unless normalization is itself the target; target bytes are the bytes shown to the model.

### Checkpoint I2: count inventory

For each tokenizer independently, construct cells using the exact slot schedule in the pre-registration:

- slot count `(i mod 5)+1`, eight slots per count 1-5
- preferred type cycle `[complex_nonemoji, zwj_emoji, atomic_nonemoji, atomic_emoji]`
- overlap levels L2, L3, L4 in both eras, 40 probes per cell
- legacy L1 always
- newest L1 only when the feasibility audit passes eight unique valid probes at every count 1-5

Probe construction rules:

1. Concatenate homogeneous constituent clusters without separators.
2. Verify the pinned regex `\X` count equals the intended count exactly.
3. Verify no surrogate, newline, prompt delimiter, or wrapper substring appears in the target.
4. Verify target strings are unique within each carrier inventory.
5. Measure whole-probe tokens per cluster using the actual tokenizer backend.
6. Assign fragmentation level from the measured whole-probe value, not isolated constituents.
7. Record constituent IDs, isolated fragmentation, whole-probe fragmentation, byte length, codepoint count, complexity, plane counts, frequency metadata, and construction attempts.

Pair construction is deterministic and uses the hard caps and lexicographic objective in the pre-registration. A pair is invalid if either side falls outside the target whole-probe fragmentation level, even if its isolated constituents looked correct.

Feasibility expectation from the design scan:

- Qwen-family newest-L1 should pass, dominated by atomic items.
- Llama-3.2 newest-L1 should pass narrowly, dominated by atomic items.
- Llama-2/Gemma/Mistral newest-L1 should fail structurally, especially counts 4-5.
- L2-L4 overlap should pass for all tokenizers, although complex_nonemoji L3 newest count-1 can be sparse in byte-fallback tokenizers.

Artifacts:

- `results/confirmatory/inventory.jsonl`
- `results/confirmatory/probe_plan.jsonl`
- `results/confirmatory/inventory_audit.json`
- `results/confirmatory/inventory_cell_table.csv`
- `results/confirmatory/inventory_freeze.json`

### Checkpoint I3: inventory audit and freeze, mandatory stop before model execution

The audit must verify, for every carrier:

1. exact copy count 124, scalar-listing count 44, count ASCII count 12
2. exact requested Unicode count-cell sizes
3. exactly eight probes per expected count in every 40-probe cell
4. homogeneous item type within every count probe
5. correct regex grapheme count
6. no cross-cell duplicate target string unless explicitly justified and flagged
7. exact pair matching constraints and hard residual caps, especially the <=0.5 whole-probe tokens-per-cluster difference cap
8. prompt template hashes match the frozen strings
9. all probe IDs, pair IDs, and completion keys are unique
10. all tokenizer hashes match the inventory scan artifacts
11. empty cells are explicitly enumerated

Mutation checks, all on temporary copies:

1. Change one expected count; audit must fail.
2. Change one target text without changing its hash; audit must fail.
3. Break one pair by changing only the era label; audit must fail.
4. Introduce a duplicate probe ID; audit must fail.
5. Replace a homogeneous two-cluster probe with mixed-type constituents; audit must fail.
6. Move one probe across a fragmentation boundary without changing its level; audit must fail.
7. Pair two probes whose measured tokens per cluster differ by more than 0.5 in a temporary copy; audit must fail.

Only after all audits pass, write `inventory_freeze.json` containing SHA-256 hashes for the inventory, probe plan, candidate metadata, tokenizer metadata, scripts, and pre-registration document. Commit this freeze before any confirmatory model generation. This commit is the point of no confirmatory design change.

## 8. Runner implementation

### Checkpoint R1: additive runner adaptation

Reuse the round-2 runner architecture:

- append-only JSONL output
- deterministic completion keys
- resume by completed keys, never by row position
- batch manifest and runtime rows
- prompt/message hashes
- model revision and dtype recording
- inferred finish reason
- peak memory and wall time per batch
- tokenizer overlay application before model load

Do not edit `src/run_behavioral_round2.py` in place. Import or copy the smallest necessary behavior into the confirmatory runner and keep batching/checkpoint semantics identical.

Fixed probe execution order inside each carrier:

1. ASCII copy
2. Unicode copy
3. ASCII scalar-listing
4. Unicode scalar-listing
5. ASCII count
6. Unicode count

The order is fixed for reproducibility, not because it is expected to affect independent prompts. No adaptive reordering based on observed outputs is allowed.

Carrier execution order, unless memory smoke tests force a documented change before inference:

1. `tinyllama-1.1b-chat-v1.0-bf16`
2. `qwen2.5-3b-instruct-bf16`
3. `llama32-3b-instruct-mlx8`
4. `gemma3-4b-it-mlx8-mirror`
5. `qwen3-4b-instruct-2507-mlx8`
6. `mistral7b-instruct-v03-mlx8`

### Checkpoint R2: template and memory smoke tests

Before generating confirmatory rows:

1. Render one copy message list through each chat template without running the model. Record exact rendered string hash and inspect for assistant-boundary correctness.
2. Load each model artifact at its pinned revision.
3. Run one non-inventory trivial ASCII prompt such as `Copy the input exactly. Input: <IN>test</IN>` if needed to verify generation. This smoke output is infrastructure validation, not a confirmatory probe, and must be stored separately.
4. Record peak memory. If peak memory plus a 1 GiB safety margin exceeds host capacity, lower batch size through the predefined ladder 16 -> 8 -> 4 -> 2 -> 1 before any inventory row is run.
5. If the model still cannot load or generate, declare it memory-infeasible before inference and exclude it descriptively. Do not substitute another model post hoc.

Artifacts: `results/confirmatory/template_rendering_checks.json`, `results/confirmatory/memory_smoke.json`.

## 9. Quantization control execution

### Checkpoint Q1: comparator acquisition and feasibility

Expected bf16 comparators:

- qwen3-4b: `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`
- gemma3-4b: `unsloth/gemma-3-4b-it` at `bf46152c47f5dd20b896357cb51abc4c03b8ee8c` as the accessible bf16 mirror lineage, with canonical `google/gemma-3-4b-it` recorded but gated
- llama32-3b: `unsloth/Llama-3.2-3B-Instruct` at `006f5dcd1393c3add266de40994ba96225e9689d` as an accessible mirror, with canonical Meta repository recorded but manually gated

Run a memory smoke test for each comparator. If a comparator cannot load, label that carrier quantization-untestable. Do not use a different-sized model as a direct comparator.

Mistral-7B bf16 is not memory-feasible on the 16 GB host and is not downloaded for inference. Mistral is therefore expected to be quantization-untestable. This is a limitation accepted in the pre-registration, not a surprise finding.

### Checkpoint Q2: paired quantization probe suite

Run the 72-row quantization suite for each available arm pair:

- 24 ASCII copy
- 24 legacy Unicode copy
- 12 ASCII count
- 12 ASCII scalar-listing

The exact suite is generated before seeing any outputs and hashed in `results/confirmatory/quant_probe_freeze.json`. Both arms use identical user content, decode parameters, scoring, and tokenizer overlays. Store rows separately from confirmatory raw output.

Pass/fail rules are exactly those in the pre-registration. "Agreement" means byte-identical extracted copy output for copy rows and identical parsed answers for count/scalar-listing rows. Count the number of discordant pairs, not merely accuracy differences.

Artifact: `results/confirmatory/quantization_gate.json`.

Mutation checks:

1. Flip one paired copy output in a temporary gate artifact; disagreement count must increase.
2. Mark exactly three copy pairs discordant in a temporary artifact; the copy component must fail the <=2 threshold.

Stop condition: if a carrier is quantization-contaminated, continue running it for descriptive data only if compute allows, but exclude it from primary H1/H2/H3/H4 inference as pre-registered. Do not tune quantization or prompts after seeing discordance.

## 10. Confirmatory generation execution

### Checkpoint G1: ASCII and gate rows first

Run all ASCII copy/count/scalar-listing rows for each carrier before its Unicode rows. Score only these rows and compute gates. Because Unicode outputs must not influence gate thresholds or inventory, this ordering is safe. Save raw rows in the main append-only file with `probe_scope` metadata.

A carrier failing copy validity remains eligible for descriptive Unicode copy near-miss analysis but not H4 confirmation. A carrier failing structural ASCII gates remains eligible for descriptive Unicode structural output but not H1/H2/H3 confirmatory inference.

### Checkpoint G2: Unicode generation

Run all remaining rows in checkpointed batches. Estimated generation-only minutes at the maximum 500 rows/carrier:

| carrier | estimate |
|---|---:|
| tinyllama | 3.3 min |
| qwen2.5-3b bf16 | 6.6 min |
| llama32-3b mlx8 | 11.2 min (qwen3 proxy) |
| gemma3-4b mlx8 | 8.1 min |
| qwen3-4b mlx8 | 11.2 min |
| mistral7b mlx8 | 17.0 min (qwen2.5-7b proxy) |
| total | **57.5 min** |

Realistic wall clock is 2-4 hours including model loading, gates, memory smoke tests, checkpointing, quantization comparisons, scoring, and retries. Downloads are additional and depend on network. A single uninterrupted generation run is not required; checkpoint resume is part of the design.

Batch ladder: start at the smoke-tested size. If an out-of-memory event occurs mid-run:

1. keep completed append-only rows
2. reduce batch size one step
3. resume only missing completion keys
4. record the event and batch-size change in the manifest

Do not rerun completed probes solely because batch size changed. Batch size is a scheduling variable and must be included in runtime metadata.

### Checkpoint G3: completion audit before scoring

Before scoring, verify:

1. every required probe key appears exactly once for each executed carrier
2. no extra probe keys exist
3. all rows have raw text, finish reason, output token count, prompt hash, model revision, dtype, batch id, and runtime metadata
4. prompt hashes match the frozen probe plan
5. completion keys have no duplicates
6. every batch manifest row has matching raw rows
7. raw file hash is computed before scoring

Artifact: `results/confirmatory/raw_completion_audit.json`.

If rows are missing after resume attempts, do not invent or impute them. Declare the affected carrier incomplete. Pair-level exclusions then follow the pre-registration.

## 11. Scoring implementation

### Checkpoint SC1: strict scoring

Implement scoring additively, reusing exact round-2 extraction behavior:

- copy: exactly one `<OUT>`/`</OUT>` regex match, no non-whitespace outside
- count: full-strip signed integer only
- scalar listing: full-strip uppercase `U+[0-9A-F]{4,6}` whitespace-separated tokens only

For every row compute:

- `task_exact`
- `format_failure`
- `content_failure_given_format`
- `total_failure`
- copy taxonomy fields and byte-diff context
- scalar-listing error class
- count signed error where parseable
- near-miss class using the pre-registered precedence

Artifacts:

- `results/confirmatory/scored.jsonl`
- `results/confirmatory/scoring_audit.json`
- `results/confirmatory/extractor_mutation_checks.json`

Mandatory scorer mutation checks:

1. Correct wrapped copy passes; two wrappers fail.
2. Correct wrapped copy passes; non-whitespace outside wrapper fails.
3. Integer `3` passes when expected is 3; `3.` or prose fails.
4. Correct uppercase scalar list passes; lowercase hex fails.
5. Missing ZWJ is classified in the correct precedence class.
6. A truncation at 85% of target bytes is a near miss; a truncation at 50% is not.
7. A one-token scalar-listing error with correct length is a scalar-listing near miss.

If any mutation check fails, scoring implementation is wrong and must be fixed before analysis.

## 12. Analysis implementation

### Checkpoint A1: gates and analysis population

Implement `src/analyze_confirmatory.py` in pure Python plus numpy/matplotlib. Do not add scipy or statsmodels. The script must:

1. load scored rows and the frozen inventory
2. compute copy/structural/quantization/tokenizer validity per carrier
3. assign inclusion flags for H1/H2/H3/H4 exactly as pre-registered
4. write gate mutation checks proving thresholds can fail

Artifact: `results/confirmatory/carrier_gates.json`.

### Checkpoint A2: H1 implementation

Implement logistic regression with numpy IRLS. Requirements:

- deterministic design-matrix column order written to artifact
- no silent dropping of covariate levels
- separation/convergence diagnostics per fit
- stratified bootstrap seed 20261127 and 5000 requested replicates
- valid-replicate counting exactly as pre-registered
- Holm adjustment across H1/H2
- sensitivity outputs in separate fields, never overwriting primary

Before running real bootstrap, include synthetic implementation tests:

1. A monotone synthetic dataset must yield positive `beta_frag`.
2. A flat synthetic dataset must yield an estimate near zero.
3. A perfectly separated synthetic dataset must be flagged rather than silently producing a huge finite coefficient.
4. Resampling one stratum must preserve that stratum's requested size.

### Checkpoint A3: H2 implementation

Implement stratified matched-pair differences exactly as pre-registered. Bootstrap seed 20261128, 5000 replicates. Compute the normal-approximate bootstrap TOST p-value from the bootstrap standard error exactly as pre-registered; do not use raw bootstrap tail proportions as the primary p-value. Verify pair completeness before analysis. Implementation tests:

1. If every newest outcome equals every legacy outcome, `Delta` must be zero.
2. Flipping one newest failure to success must change `Delta` by exactly `-1/(carriers x levels x pairs_per_stratum)` in the equally weighted estimand.
3. Resampling within a stratum must preserve 40 pair draws.
4. A synthetic +20 pp newest effect must produce a positive interval in a smoke simulation.

### Checkpoint A4: H3/H4 and near-miss implementation

Implement exact-count replication tables and classifier precedence. Mutation checks:

1. Flipping one scalar-listing row to exact must refute exact-floor H3 for that carrier.
2. Flipping one copy row to non-exact must refute exact-ceiling H4 for that carrier.
3. A row satisfying multiple copy classes must receive the earliest precedence class, not a later one.

### Checkpoint A5: required outputs

Generate the exact tables and figures listed in the pre-registration. Files:

```text
results/confirmatory/t1_carrier_gates.csv
results/confirmatory/t2_inventory_cells.csv
results/confirmatory/t3_h1_coefficients.csv
results/confirmatory/t4_h2_matched_pairs.csv
results/confirmatory/t5_copy_ceiling.csv
results/confirmatory/t6_scalar_floor.csv
results/confirmatory/t7_count_errors.csv
results/confirmatory/f1_failure_heatmap.png
results/confirmatory/f2_fragmentation_dose_response.png
results/confirmatory/f3_era_effect_forest.png
results/confirmatory/f4_copy_count_scalar_triad.png
results/confirmatory/f5_copy_near_miss_taxonomy.png
results/confirmatory/confirmatory_verdict.json
results/CONFIRMATORY_REPORT.md
```

The verdict JSON must state `confirmed`, `refuted`, or `inconclusive` for H1-H4 separately, plus machine-readable reasons. The report must not soften a refutation into language like "mostly confirmed."

## 13. Reproducibility checks

Before final commit:

1. Rerun `python src/confirmatory_power_analysis.py --self-test` and require exit 0.
2. Rerun the full power script to a temporary output and compare all deterministic values with `results/confirmatory_power_analysis.json`.
3. Rerun inventory audit from the committed inventory and require all checks pass.
4. Rerun scorer mutation checks and require all pass.
5. Rerun artifact hash checker and require no missing/changed frozen artifact.
6. Verify the git diff does not touch round-2 raw/scored artifacts or README claims.
7. Verify at least one mutation or defect injection fails loudly for each critical gate: tokenizer hash, inventory count, wrapper extraction, scalar parser, H1 bootstrap, H2 pair construction, H3 floor, H4 ceiling.

## 14. Failure and ambiguity handling

1. **Inventory deficit:** stop before inference. Do not relax matching caps or cell sizes post hoc.
2. **Tokenizer mismatch:** invalidate affected rows and regenerate only after correcting the artifact; document the wrong rows as superseded.
3. **Model download/gating failure:** record exact HTTP/authentication error. Do not substitute another architecture post hoc.
4. **Memory failure:** use the predefined batch ladder, then declare infeasible if still impossible.
5. **Crash or power loss:** resume by completion key. Never regenerate completed keys unless raw file corruption is proven; if regeneration occurs, mark prior bytes superseded and hash both versions.
6. **Partial carrier data:** no imputation. Pair-missing rows exclude both members of a pair for H2. Cross-carrier inference may become inconclusive.
7. **Quantization gate failure:** label contamination; no ad hoc threshold adjustment.
8. **Structural ASCII failure:** exclude from H1/H2/H3 inference but report descriptive Unicode behavior.
9. **Scoring ambiguity:** strict parser outcome wins. Do not hand-rescue outputs.
10. **Any ambiguity not covered here:** stop and write an amendment before further execution if no model output has been seen; otherwise handle as exploratory.

## 15. Review gates for the user before execution

The following decisions should be explicitly reviewed before the implementation phase starts:

1. Whether the expanded quantization-control suite is worth the additional bf16 downloads, given disk pressure.
2. Whether Mistral-7B should remain in the primary ladder despite quantization-untestable status, or be demoted to a descriptive carrier.
3. Whether strict exact-floor/exact-ceiling replication criteria for H3/H4 are too brittle, given that even one near miss refutes the exact form.
4. Whether the matching hard caps (0.5 tokens/cluster, 4 bytes, 2 codepoints, 2 complexity units per bundle) are strict enough for the intended causal control.
5. Whether probe-level bootstrap is acceptable despite constituent reuse, or whether a more complex multiway/resampled-cluster method should be specified before execution.

## 16. Immediate next implementation actions

Once the design branch is approved:

1. Implement `confirmatory_common.py` with pinned carriers and prompts.
2. Implement candidate and tokenizer scanners; run only tokenizer/candidate audits, no models.
3. Implement inventory builder and iterate until all pre-inference audits pass.
4. Commit the inventory freeze.
5. Only then implement quantization and runner execution.

The design branch itself should contain only the pre-registration, this plan, the reproducible power script, and its deterministic output.
