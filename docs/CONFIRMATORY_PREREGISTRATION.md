# Confirmatory pre-registration: fragmentation, not recency

Status: **design freeze candidate**, written before confirmatory inventory construction and before any confirmatory model execution.
Branch: `confirmatory-design`.
Date: 2026-11-26.
Pilot evidence freeze: round-2 artifacts already committed at `results/behavioral_round2/`.

This document is the pre-analysis plan. It is deliberately exact about hypotheses, probe counts, gates, estimands, inference, decision rules, and empty cells. It is not a result report. If execution requires a deviation, the deviation must be labelled post hoc rather than silently absorbed into the confirmatory analysis.

## 1. Scientific claim

The confirmatory claim is not "models cannot parse new Unicode because it is new." The claim is narrower and mechanistic:

> Small instruction-tuned language models can often reproduce Unicode byte sequences exactly while failing to recover those sequences' grapheme or scalar structure. The dominant measurable proximal mechanism is tokenizer fragmentation: when a user-perceived cluster is split into many tokens, structural operations over that cluster fail. Unicode release recency should have little or no additional effect once fragmentation and stimulus covariates are controlled.

The study therefore uses copy as a byte-plane control, scalar-listing ("decomposition") as a parse probe, and grapheme counting as the primary fragmentation-sensitive structural probe.

The causal language is limited. Tokenizer fragmentation is assigned by the tokenizer, not randomized independently of Unicode item type. The design controls measurable confounders and uses matched homogeneous probes, but complex/ZWJ items cannot exist at low fragmentation in byte-fallback tokenizers when their bytes were never merged. That structural impossibility is part of the phenomenon, not a defect to hide.

## 2. Pilot evidence being carried forward

The power-analysis script audits these exact values from `results/behavioral_round2/round2_scored.jsonl`; see `src/confirmatory_power_analysis.py` and `results/confirmatory_power_analysis.json`.

1. Round 2 contains 1450 scored rows: 5 models x 290 rows. Each model has 182 copy, 48 decomposition, 36 count, and 24 comparison rows.
2. Four models passed the round-2 copy ASCII and wrapper gates:
   - `qwen2.5-3b-instruct-bf16`
   - `qwen3-4b-instruct-2507-mlx8`
   - `gemma3-4b-it-mlx8-mirror`
   - `tinyllama-1.1b-chat-v1.0-bf16`
3. Each of those four copied all 94 newest Unicode copy items exactly and all 64 legacy Unicode copy items exactly.
4. Each of those four produced 0/48 exact scalar-listing answers. `tinyllama` also produced 0/36 exact count answers and is therefore not expected to be structurally valid for H1/H2.
5. Pooled count-probe high-minus-low fragmentation failure gaps were:
   - qwen2.5-3b: 1.000 - 0.500 = **+50.0 pp**
   - qwen3-4b: 0.8889 - 0.6667 = **+22.2 pp**
   - gemma3-4b: 0.6667 - 0.6111 = **+5.6 pp**, much weaker and non-monotonic by era
   - tinyllama: 0 pp because both cells were at a 100% failure floor
6. qwen2.5-7b-instruct-mlx8 was copy-gate-invalid in round 2: 28/182 copy rows failed the wrapper gate, with `<IN>`-mimicry despite byte-perfect inner copies. It is excluded from the confirmatory primary ladder because the frozen measurement instrument is not valid for that carrier.
7. Round-2 copy protocol freeze: `fewshot_copy_v5_multiturn_cueless`, template SHA-256 `1687c1a717cc04c3a77aba9075c8463cf95e9ae084e53435e549bca22ec0d951`, format SHA-256 `e84859819d4873b1b739683534e720c8f17baaf7c6542be35c94eec106692f6d`, recorded in `results/behavioral_round2/protocol_freeze.json`.
8. Round-2 quantization precedent: Qwen2.5-1.5B bf16 and MLX 8-bit each copied 23/24 ASCII controls exactly; the prespecified contamination envelope was +/-2 of 24. This precedent is necessary but not sufficient evidence for every new quantized carrier, so the confirmatory plan includes per-carrier quantization checks where a memory-feasible bf16 comparator exists.

The pilot motivates the hypothesis but is not itself confirmation: its count bins were era-internal median splits, so "low" and "high" overlapped in absolute tokens per cluster across eras and tokenizers. The confirmatory design uses absolute fragmentation levels.

## 3. Hypotheses and numeric predictions

### H1: fragmentation dose-response

**Hypothesis.** Count-probe failure probability shows a material positive dose-response to tokenizer fragmentation after adjustment for carrier, era, expected cluster count, homogeneous item type, byte length, codepoint length, sequence complexity, supplementary-plane composition, and emoji frequency. "Material" is operationally defined as an odds ratio greater than 1.25 per ordinal fragmentation level.

**Primary estimand.** `beta_frag`, the coefficient for ordinal fragmentation level `L` in the prespecified pooled logistic regression. `L` is coded L1=1, L2=2, L3=3, L4=4.

**Numeric point prediction.** The pooled odds ratio per level, `exp(beta_frag)`, is predicted to lie between **1.5 and 4.0**. This range is deliberately conservative relative to the pilot. Manual pilot logistic fits were per tokens-per-cluster unit rather than per two-token-wide level, so they are not unit-identical to the confirmatory coefficient; they nevertheless estimated positive fragmentation effects for qwen2.5-3b (about 4.24 per token), qwen3-4b (about 2.09), and gemma3-4b (about 1.11). The conservative qwen3 high-low failure gap was 22.2 pp.

**Confirmation rule.** H1 is confirmed only if all of the following hold:

1. At least four structurally valid carriers representing at least three tokenizer families are available for confirmatory inference.
2. The Holm-adjusted one-sided bootstrap p-value for `beta_frag > 0` is significant within the H1/H2 familywise alpha of 0.05.
3. The lower bound of the 95% bootstrap confidence interval for `beta_frag` is above `log(1.25)`.
4. At least three structurally valid carriers, spanning at least two tokenizer families, have non-negative carrier-specific fragmentation coefficients. Carrier-specific coefficients come from the primary covariate formula refit separately within each carrier on the same carrier-balanced probe subset, with carrier fixed effects omitted; a carrier whose fit is separation-invalid is missing rather than counted as negative.

**Refutation rule.** H1 is refuted if either:

1. The upper bound of the 95% bootstrap confidence interval for `beta_frag` is below `log(1.25)`, meaning that even the upper plausible effect is weaker than an odds ratio of 1.25 per fragmentation level; or
2. The 95% bootstrap confidence interval lies entirely below zero.

**Inconclusive rule.** Any other outcome, including fewer than four valid carriers or excessive bootstrap model failure, is inconclusive rather than negative evidence.

### H2: no additional era effect at matched fragmentation

**Hypothesis.** Within matched fragmentation levels L2-L4, newest-era probes are not materially more failure-prone than legacy probes after matching item type, expected cluster count, plane composition, and size/complexity covariates.

**Primary estimand.** `Delta`, the stratified matched-pair risk difference:

```text
failure(newest probe) - failure(legacy probe)
```

averaged first within carrier x fragmentation-level strata, then equally across fragmentation levels and structurally valid carriers. Positive values mean newest-era probes fail more often.

**Numeric point prediction.** `Delta` lies in **[-0.05, +0.05]**.

**Equivalence margin.** Practical equivalence is defined as **[-0.10, +0.10]** failure probability. This is not a claim of mathematically exact zero. It is the smallest effect width judged operationally meaningful relative to the pilot fragmentation gaps of +22.2 and +50.0 pp.

**Confirmation rule.** H2 is confirmed only if all of the following hold:

1. At least four structurally valid carriers representing at least three tokenizer families are available.
2. The observed `Delta` is inside [-0.10, +0.10].
3. The percentile 95% bootstrap confidence interval for `Delta` is entirely inside [-0.10, +0.10].
4. The bootstrap two one-sided test for equivalence is significant after Holm adjustment within the H1/H2 family.

**Refutation rule.** H2 is refuted if the 95% bootstrap confidence interval for `Delta` lies entirely outside the equivalence margin in either direction: entirely above +0.10 or entirely below -0.10. A positive interval means newest-era items fail materially more; a negative interval means legacy items fail materially more. If the interval excludes zero but remains inside the equivalence margin, report that as evidence against mathematically exact zero in a sensitivity note; it does not overturn confirmation of practical equivalence.

**Inconclusive rule.** Any other outcome, including a confidence interval that includes zero but extends beyond +/-10 pp, is inconclusive.

### H3: decomposition floor replication

**Hypothesis.** Scalar-listing failure remains at the floor even when copying succeeds.

**Numeric point prediction.** Every structurally valid carrier has **0/32 exact** Unicode scalar-listing answers, and therefore also 0/16 exact on newest items and 0/16 exact on legacy items.

**Confirmation rule.** H3 is confirmed only if every structurally valid and quantization-eligible carrier has exactly 0/32 exact Unicode scalar-listing answers.

**Refutation rule.** H3 is refuted for a carrier if that carrier has one or more exact Unicode scalar-listing answers. The exact universal-floor form of H3 is then false even if performance remains near floor. Near-floor rates and error types are still reported.

**Inconclusive rule.** Fewer than four structurally valid and quantization-eligible carriers makes the cross-carrier hypothesis inconclusive.

### H4: copy ceiling replication

**Hypothesis.** Byte copying remains at ceiling for both newest and legacy Unicode items when the copy gate is valid.

**Numeric point prediction.** Every copy-valid carrier copies all **60/60 newest** and **40/40 legacy** Unicode copy probes byte-exactly.

**Confirmation rule.** H4 is confirmed only if every copy-valid and quantization-eligible carrier is exactly 60/60 newest and 40/40 legacy.

**Refutation rule.** H4 is refuted for a carrier by any byte-level copy failure. Near misses are scientifically important and are classified, but they still refute the exact ceiling.

**Inconclusive rule.** Fewer than four copy-valid carriers representing at least three tokenizer families makes the cross-carrier hypothesis inconclusive.

## 4. Multiplicity and inferential scope

The inferential family contains H1 and H2 only. Familywise alpha is **0.05**, controlled with Holm's procedure:

1. Compute the one-sided bootstrap p-value for H1 and the TOST bootstrap p-value for H2.
2. Rank the two p-values.
3. The smaller is compared with alpha 0.025; if it passes, the larger is compared with alpha 0.05.

H3 and H4 are exact replication checks on finite deterministic inventories. They do not have stochastic null hypotheses in the same sense and are therefore outside the Holm family. They are reported as pass/fail counts with no p-values.

Statistical inference here is conditional on the fixed carrier ladder and is primarily inference over the frozen stimulus ensemble, not over random decoding noise or a hypothetical superpopulation of models. Generation is greedy and deterministic. Bootstrap resampling quantifies sensitivity to stimulus selection and matching, not run-to-run randomness; cross-carrier durability is assessed by the explicit consistency requirements rather than by treating carriers as random samples.

## 5. Carrier ladder

### 5.1 Primary confirmatory carriers

| carrier id | family/tokenizer role | inference artifact | pinned artifact revision | canonical upstream | quantization |
|---|---|---|---|---|---|
| `qwen2.5-3b-instruct-bf16` | Qwen byte-level BPE, newer tokenizer | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | same | bf16 |
| `qwen3-4b-instruct-2507-mlx8` | Qwen byte-level BPE, newer tokenizer | `mlx-community/Qwen3-4B-Instruct-2507-8bit` | `0e42af58449718de7931ee04f28191fbe6c43a56` | `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554` | MLX 8-bit |
| `gemma3-4b-it-mlx8-mirror` | Gemma SentencePiece-style tokenizer | `mlx-community/gemma-3-4b-it-8bit` | `61cfff4de4bc5b975437be840d504f07f0424df2` | `google/gemma-3-4b-it` at `093f9f388b31de276ce2de164bdc2081324b9767` | MLX 8-bit |
| `tinyllama-1.1b-chat-v1.0-bf16` | vintage Llama-2-family tokenizer | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | same | bf16 |
| `llama32-3b-instruct-mlx8` | Llama-3 byte-level BPE, 128k vocabulary | `mlx-community/Llama-3.2-3B-Instruct-8bit` | `ff054899609078569493def2823f9acd2780c0c9` | `meta-llama/Llama-3.2-3B-Instruct` at `0cb88a4f764b7a12671c53f0838cd831a0843b95` | MLX 8-bit |
| `mistral7b-instruct-v03-mlx8` | Mistral SentencePiece-style byte-fallback tokenizer | `mlx-community/Mistral-7B-Instruct-v0.3-8bit` | `c282bcd11a64be44ac3f130f0b819d48ecc1846d` | `mistralai/Mistral-7B-Instruct-v0.3` at `c170c708c41dac9275d15a8fff4eca08d52bab71` | MLX 8-bit |

The canonical Meta repository is manually license-gated for this account. The Llama-3 MLX mirror is accessible and its tokenizer files were separately hashed:

- `tokenizer.json` SHA-256 `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`
- `tokenizer_config.json` SHA-256 `9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e`
- `special_tokens_map.json` SHA-256 `6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec`

Mistral tokenizer file hashes:

- `tokenizer.json` SHA-256 `ad379250e3a923686183c1ee60398c6f278da0a40520dc4b22e69c53cfc9af9e`
- `tokenizer.model` SHA-256 `37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89`
- `tokenizer_config.json` SHA-256 `489746d6758f2e62d01eb30b7b6486e60ebae4728a97cbce4905a2eb7264f5fd`
- `special_tokens_map.json` SHA-256 `6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0`

Approximate weight download sizes for the two added carriers are 3.19 GiB for Llama-3.2-3B-8bit and 7.18 GiB for Mistral-7B-8bit, total 10.36 GiB. If the local cache is cold, the full six-carrier primary slate is about 27.4 GiB. The three bf16 quantization comparators would add about 21.5 GiB, for a worst-case new-download total near 48.9 GiB. At design time the host has 64 GiB free and an existing 41 GB Hugging Face cache. The full plan is therefore feasible but tight, leaving only roughly 15 GiB in the worst case; the expanded quantization gate is the main discretionary disk cost. Mistral-7B is memory-feasible only if the 8-bit artifact stays within the 16 GB host envelope after runtime overhead; a load smoke test is mandatory.

### 5.2 Why qwen2.5-7b is absent

`qwen2.5-7b-instruct-mlx8` is not in the primary ladder because round 2 already showed that the frozen copy instrument is invalid for that carrier: 28/182 copy rows had wrapper-format failures. Re-running it as a primary carrier would spend compute on a known measurement-instrument failure. It may be run only after all primary confirmatory rows are complete, as a clearly labelled nonconfirmatory robustness diagnostic.

### 5.3 Family interpretation

Qwen2.5 and Qwen3 are separate model generations but not independent tokenizer families. Gemma, Llama-3, Mistral, and TinyLlama/Llama-2 supply distinct tokenizer mechanisms or vintages. Cross-carrier confirmation therefore requires both a minimum carrier count and a minimum tokenizer-family count.

## 6. Frozen measurement protocol

### 6.1 Decoding

All confirmatory generation uses greedy/argmax decoding, temperature 0.0, no top-p or top-k sampling, maximum 64 output tokens, and the same batch runner logic as `src/run_behavioral_round2.py` unless a memory-forced batch-size reduction is needed. Batch size affects scheduling only, not the semantic decode parameters. Every batch row must record model id, revision, dtype/quantization class, prompt hash, completion key, output token count, runtime, peak memory, and inferred finish reason.

### 6.2 Copy prompt

Copy uses exactly `fewshot_copy_v5_multiturn_cueless`:

```text
Copy the input exactly. Input: <IN>{target}</IN>
```

The probe user turn is preceded by the four fixed ASCII exemplar user/assistant pairs frozen in round 2:

1. `bO5*Fe%Yx^9pM=5g`
2. `;@N@fd=K\+w=kEJ3`
3. `mX3 *vQ9!zB 2#7R`
4. `p~4tD14z/4g;[zM!`

Assistant exemplar outputs are exactly `<OUT>{exemplar}</OUT>`. The user-content message list is byte-identical across models; only each model's own chat-template wrapper may differ.

### 6.3 Count prompt

```text
Count the extended grapheme clusters (user-perceived characters), not bytes or code points. Reply with one integer only. ZWJ joins emoji components into one cluster; combining marks and variation selectors attach to their base character. Input:
{target}
```

A correct answer is a full-strip integer equal to the number of extended grapheme clusters computed with the pinned `regex` `\X` implementation used at inventory construction.

### 6.4 Scalar-listing prompt

```text
List the Unicode scalar values in order as hexadecimal codepoints. Use exactly this format: `U+1F600 U+200D`. Input:
{target}
```

A correct answer is a full-strip uppercase whitespace-separated `U+XXXX` token list equal to the exact scalar sequence. No prose, lowercase hex, comma, bracket, or explanation is accepted. This is scalar enumeration, not Unicode canonical decomposition: a precomposed character is represented by its own scalar unless that scalar is literally present in the input sequence.

### 6.5 Extraction strictness

Copy extraction is the round-2 strict rule: exactly one `<OUT>` and one `</OUT>`, exactly one regex match, and no non-whitespace outside the wrapper. Count parsing is full-strip integer only. Scalar-listing parsing is full-strip uppercase `U+[0-9A-F]{4,6}` tokens separated by whitespace only. No lenient rescue rules are used in confirmatory scoring.

## 7. Probe inventory

### 7.1 Unicode era definitions

- **Newest:** emoji first fully qualified in Emoji 15.1, 16.0, 17.0, or 18.0; non-emoji scalar items whose maximum codepoint Unicode age is 15.1, 16.0, 17.0, or 18.0.
- **Legacy:** emoji first fully qualified in Emoji 15.0 or earlier; non-emoji scalar items whose maximum codepoint Unicode age is 15.0 or earlier.

The inventory builder records both codepoint maximum Unicode age and emoji sequence version because they can diverge for atomic emoji.

### 7.2 Homogeneous item types

Every Unicode count probe is homogeneous in exactly one of four item types:

1. `atomic_nonemoji`
2. `atomic_emoji`
3. `complex_nonemoji`
4. `zwj_emoji`

No count probe mixes types. Complex non-emoji items use the same valid base-plus-combining-mark construction as the pilot unless the implementation plan's audit explicitly documents a broader attested combining-mark inventory. ZWJ emoji are fully-qualified RGI sequences.

### 7.3 Absolute fragmentation levels

For a probe with `n` expected clusters and `T` tokenizer tokens after dummy-prefix removal, `tokens_per_cluster = T / n`. Levels are:

| level | tokens per cluster |
|---|---|
| L1 | [1.0, 2.5) |
| L2 | [2.5, 4.5) |
| L3 | [4.5, 6.5) |
| L4 | [6.5, infinity) |

The measured whole-probe token count is decisive, not the sum predicted by isolated constituents. The inventory builder must record both isolated constituent fragmentation and whole-probe fragmentation.

### 7.4 Count-probe cell counts

For every carrier tokenizer:

| inventory component | probes |
|---|---:|
| L2 newest | 40 |
| L2 legacy | 40 |
| L3 newest | 40 |
| L3 legacy | 40 |
| L4 newest | 40 |
| L4 legacy | 40 |
| L1 legacy | 40 |
| L1 newest, only if structurally feasible | 0 or 40 |
| ASCII count controls | 12 |

Thus the Unicode count inventory is 280 probes per carrier when newest-L1 is impossible and 320 when it is feasible. The overlap core used by H2 is exactly 240 probes per carrier: three levels x two eras x 40. Each 40-probe cell has exactly eight probes at each expected cluster count 1, 2, 3, 4, and 5.

### 7.5 Type-count slot schedule

Within each 40-probe count cell, slot `i` (0 through 39) has:

```text
expected_count = (i mod 5) + 1
preferred_item_type = item_type_cycle[i mod 4]
item_type_cycle = [complex_nonemoji, zwj_emoji, atomic_nonemoji, atomic_emoji]
```

This yields exactly two slots for every item-type x expected-count combination when all four types are feasible. If the preferred type cannot supply a valid probe in both eras for an overlap pair, the builder tries the next types in cyclic order and records the substitution. Substitutions are design data, not post hoc exclusions.

### 7.6 Matched-pair construction for H2

Each overlap slot produces one newest and one legacy probe joined by a `match_pair_id`. Pairs must match exactly on:

- carrier tokenizer
- fragmentation level
- expected cluster count
- homogeneous item type
- number of supplementary-plane clusters

They must additionally satisfy these hard residual caps:

- absolute whole-probe tokens-per-cluster difference <= 0.5, using the measured tokenization of each complete bundle
- absolute total UTF-8 byte difference <= 4 bytes per bundle
- absolute total codepoint-count difference <= 2 per bundle
- absolute mean sequence-complexity difference <= 2

The tokens-per-cluster cap is essential: sharing a wide fragmentation bin is not sufficient matching when within-bin fragmentation can still differ systematically by era. The deterministic matching objective minimizes, in lexicographic order:

1. whole-probe tokens-per-cluster difference
2. total UTF-8 byte difference
3. total codepoint-count difference
4. mean complexity difference
5. emoji frequency-row difference, treating missing frequency as a large but finite penalty
6. constituent reuse penalty
7. SHA-256 of the canonical candidate-pair serialization as a deterministic tie-breaker

If no pair inside the hard caps exists, the slot is not filled with a worse pair. The cell is declared structurally insufficient, the audit records the deficit, and the execution checkpoint stops before model inference. Feasibility scans indicate ample supply in the overlap core overall, but not uniformly by type: in a Llama-2 spot check, complex non-emoji bundles had essentially exact era overlap in tokens per cluster, while newest ZWJ emoji were systematically more fragmented than legacy ZWJ emoji even within L4. Some ZWJ substitutions or deficits are therefore plausible and must be reported rather than hidden.

### 7.7 Structurally empty low-fragmentation newest cells

The newest-L1 cell is conditional because it is not universally constructible. Feasibility scans over a 29,599-candidate stratified universe, with tokenizer-specific whole-bundle rescans, showed:

- Qwen-family tokenizers: newest-L1 bundles can be constructed at all expected counts 1-5, but almost entirely from atomic items.
- Llama-3.2 tokenizer: newest-L1 bundles can be constructed at counts 1-5, again dominated by atomic items; only 20 direct count-1 newest candidates were found in the sample, still above the required eight.
- Llama-2, Gemma, and Mistral byte-fallback tokenizers: only three direct newest-L1 candidates were found in the sample; concatenation yielded at most six unique low-fragmentation hits at counts 2-3 and zero at counts 4-5. The required eight per count cannot be met.

Therefore:

1. H2 uses only overlap levels L2-L4.
2. L1 newest is included only for tokenizers satisfying the feasibility audit.
3. Complex/ZWJ newest low-fragmentation cells are expected to be empty in every tokenizer. This is structural: a genuinely new complex sequence cannot already be merged as a low-token unit in a frozen vocabulary. These empty cells must be printed in the inventory table rather than hidden.
4. H1 uses legacy L1 as the low-end reference plus the overlap range in both eras. Newest L1 is a family-specific sensitivity, not a primary overlap cell.

### 7.8 Copy inventory

Each carrier receives 124 copy probes:

- 60 newest Unicode copy probes
- 40 legacy Unicode copy probes
- 24 ASCII copy controls

Unicode copy probes are a deterministic stratified subset of the round-2 copy universe:

| era | atomic_nonemoji | atomic_emoji | complex_nonemoji | zwj_emoji | total |
|---|---:|---:|---:|---:|---:|
| newest | 10 | 20 | 10 | 20 | 60 |
| legacy | 10 | 10 | 10 | 10 | 40 |

The ASCII copy controls are the same 24 strings used in round 2 so the gate is exactly comparable.

### 7.9 ASCII structural control schedule

The 12 ASCII count controls are deterministic random printable-ASCII strings without spaces or wrapper substrings. For control slot `i` (0 through 11), the cluster count is `(i mod 5) + 1`; thus counts 1 and 2 occur three times each and counts 3, 4, and 5 occur twice each. ASCII characters are drawn without replacement within a string so the answer is not trivially repetition counting.

The 12 ASCII scalar-listing controls are deterministic random printable-ASCII strings. For control slot `i`, the number of scalars is `(i mod 3) + 1`; each length occurs four times. Expected outputs use uppercase zero-padded forms such as `U+0041`.

### 7.10 Scalar-listing inventory

Each carrier receives 44 scalar-listing probes:

- 16 newest Unicode items
- 16 legacy Unicode items
- 12 ASCII controls

Within each era, eight probes are atomic and eight are complex/ZWJ. Unicode scalar-listing targets are drawn from the same item universe as copy, with disjointness from count targets preferred but not required. ASCII controls have one to three ASCII scalars and test whether the carrier can emit exact uppercase `U+00XX` notation.

### 7.11 Total rows

Per carrier:

| component | minimum | maximum |
|---|---:|---:|
| copy | 124 | 124 |
| scalar-listing | 44 | 44 |
| count | 292 | 332 |
| total | **460** | **500** |

Across six primary carriers: **2760-3000 confirmatory rows**, excluding quantization-gate probe rows. This is approximately twice the round-2 row count because H2 requires adequate matched-pair precision.

## 8. Validity gates frozen before execution

### 8.1 Copy-valid carrier

A carrier is copy-valid only if both hold:

1. ASCII copy exact count is at least 23/24.
2. Copy wrapper format failure rate is at most 5% across all 124 copy rows. With 124 rows, at most 6 format failures pass; 7 fail.

Copy validity is required for interpreting H4 and is part of structural validity. It is not sufficient for structural validity.

### 8.2 Structurally valid carrier

A carrier is structurally valid only if all hold:

1. It is copy-valid.
2. ASCII count exact count is at least 11/12.
3. ASCII scalar-listing exact count is at least 11/12.

The structural ASCII gates are intentionally strict. A carrier unable to count short ASCII strings or print ASCII scalar notation cannot produce interpretable Unicode structural failures. TinyLlama is therefore expected to be copy-valid but structurally invalid based on the pilot; this is a prediction, not an exemption.

### 8.3 Tokenizer validity

For every carrier, the exact inference tokenizer files or overlays must hash to the pinned tokenizer artifact used for inventory construction. If an MLX artifact omits or alters tokenizer/chat-template files, the same canonical-overlay mechanism used in round 2 must be applied and hashed before inference. A tokenizer mismatch invalidates all rows for that carrier until corrected and regenerated.

### 8.4 Quantization validity

For qwen3-4b, gemma3-4b, and llama32-3b, a memory-feasible bf16 comparator is expected. Before Unicode inference, run a paired quantization-control suite:

- 24 ASCII copy probes
- 24 legacy Unicode copy probes, stratified across atomic and complex/ZWJ items
- 12 ASCII count probes
- 12 ASCII scalar-listing probes

A quantization arm passes only if all hold relative to its bf16 comparator:

1. The number of discordant ASCII copy pairs is at most 2/24, where agreement means byte-identical extracted output.
2. The number of discordant legacy Unicode copy pairs is at most 2/24, where agreement means byte-identical extracted output.
3. The number of discordant ASCII count pairs is at most 1/12, where agreement means identical parsed integer answers.
4. The number of discordant ASCII scalar-listing pairs is at most 1/12, where agreement means identical parsed scalar-token lists.

A carrier failing this gate is excluded from primary H1/H2/H3/H4 inference and labelled quantization-contaminated, though its measurements remain descriptive artifact-level behavior. A carrier is **quantization-eligible** if it either passes this gate or is a bf16 carrier needing no comparison. Quantization-untestable carriers are conditionally eligible only under the sensitivity rule in the next sentence; contaminated carriers are never eligible. If no comparator can be run because bf16 is not memory-feasible, the carrier is labelled quantization-untestable. Mistral-7B is expected to be in this category. Primary inference includes quantization-untestable carriers only if the prespecified sensitivity excluding them does not change any confirmation/refutation classification; otherwise affected hypotheses are inconclusive. This is a conservative but imperfect solution and is flagged for user review.

### 8.5 Minimum carrier count

H1/H2 confirmatory inference requires at least four structurally valid and quantization-eligible carriers spanning at least three tokenizer families. H4 cross-carrier replication requires at least four copy-valid and quantization-eligible carriers spanning at least three tokenizer families. If either minimum is not met, the corresponding cross-carrier hypothesis is inconclusive. Individual carrier results are still reported descriptively.

## 9. Primary analysis plan

### 9.1 Outcome definition

For all tasks, `failure` is `not task_exact`. Format failure is failure. This avoids rescuing structurally incorrect behavior through lenient parsing. Content-failure-given-format is reported secondarily.

### 9.2 H1 model

The primary pooled model is fit on a carrier-balanced Unicode count subset from structurally valid carriers: legacy L1 plus both eras at L2-L4. Conditional newest-L1 probes are excluded from the primary fit even when they exist, because they are not universally constructible; they enter only the prespecified sensitivity analysis. The model is:

```text
logit P(failure = 1) =
    alpha0
  + carrier fixed effects
  + beta_frag * L
  + beta_era * I[newest]
  + expected-count fixed effects for counts 2,3,4,5 relative to 1
  + item-type fixed effects relative to atomic_nonemoji
  + lambda1 * utf8_bytes_per_cluster
  + lambda2 * codepoints_per_cluster
  + lambda3 * mean_sequence_complexity
  + lambda4 * supplementary_cluster_fraction
  + lambda5 * emoji_frequency_row_mean
  + lambda6 * I[emoji probe with any frequency row missing]
```

`L` is ordinal level 1-4. For frequency adjustment, non-emoji probes are coded zero on both frequency covariates because no emoji frequency row is applicable; the missingness indicator is reserved for emoji probes with missing frequency data. This avoids exact collinearity with item-type fixed effects. No interaction terms are in the primary model. The interaction is a sensitivity because the primary H1 claim is a common fragmentation dose-response after additive era adjustment.

### 9.3 H1 bootstrap inference

Use 5000 stratified bootstrap replicates with seed 20261127. Within each carrier x era x fragmentation-level stratum, resample probe IDs with replacement to the stratum's original size. Refit the full model. The one-sided p-value for a positive fragmentation effect is:

```text
(1 + number of valid replicates with beta_frag <= 0) / (1 + number of valid replicates)
```

A replicate is invalid if the logistic fit fails to converge or exhibits complete separation that prevents finite coefficient estimation. If fewer than 4000 valid replicates remain, H1 inference is automatically inconclusive. Let `se_boot` be the standard deviation of `beta_frag` over valid replicates. The one-sided normal-approximate bootstrap p-value is `1 - Phi(beta_hat / se_boot)`, where `Phi` is the standard normal CDF. Report both this p-value and the percentile 95% confidence interval over valid replicates.

### 9.4 H1 sensitivity analyses

Prespecified sensitivities, none of which replace the primary result:

1. Continuous measured `tokens_per_cluster` in place of ordinal level.
2. Fragmentation-level fixed effects to inspect nonlinearity.
3. Legacy-only fit, because legacy spans L1-L4 in every tokenizer.
4. Within-type fits for `complex_nonemoji` and `zwj_emoji` separately where support exists.
5. Era x fragmentation interaction fit, reporting era-specific slopes.
6. Linear probability model with the same covariates, to avoid logistic-separation dependence.
7. Exclusion of quantization-untestable and quantization-contaminated carriers.
8. Unadjusted model with only carrier, era, fragmentation, and expected count.

### 9.5 H2 matched-pair analysis

For each carrier x level stratum compute the mean paired difference:

```text
d[c,L] = mean_over_pairs(failure_newest - failure_legacy)
```

Then compute:

```text
Delta = mean_over_structurally_valid_carriers( mean_over_L2,L3,L4( d[c,L] ) )
```

Each carrier and each of the three levels receives equal conceptual weight. Because generation must complete every probe, missing pairs are not expected. If a runner failure leaves a missing pair after checkpoint recovery, both sides of that pair are excluded and the deficit is reported; no single-sided replacement is allowed.

Use 5000 stratified bootstrap replicates with seed 20261128. Within each carrier x level stratum, resample matched pair IDs with replacement to 40 pairs. Compute `Delta` for each replicate. As diagnostics, report the raw bootstrap tail proportions:

```text
tail_lower = proportion(Delta_boot <= -0.10)
tail_upper = proportion(Delta_boot >= +0.10)
```

These raw proportions are not the primary p-value. Let `se_boot` be the standard deviation of the bootstrap `Delta` replicates. Using the standard normal CDF `Phi`, the TOST p-value is:

```text
p_upper_margin = Phi((Delta_hat - 0.10) / se_boot)
p_lower_margin = 1 - Phi((Delta_hat + 0.10) / se_boot)
p_TOST = max(p_upper_margin, p_lower_margin)
```

This normal-approximate bootstrap TOST is asymptotically aligned with requiring a two-sided 95% interval to lie inside the equivalence margin. Report the percentile 95% confidence interval, the bootstrap standard error, and all three p-value components.

### 9.6 H2 sensitivity analyses

1. Carrier-specific paired differences and forest plot.
2. Level-specific differences at L2, L3, and L4.
3. Type-restricted differences for complex_nonemoji and zwj_emoji.
4. Linear regression of pair differences on residual byte, codepoint, complexity, and frequency mismatches.
5. Logistic regression era coefficient with carrier and covariates, converted to an average marginal effect.
6. Sensitivity including newest-L1 pairs only in tokenizers where that cell exists. Boundary L1 probes are pair-constructed with the same matching rules when newest-L1 is feasible; legacy L1 probes in tokenizers without a newest counterpart remain unpaired H1 reference probes.
7. Exclusion of quantization-untestable and quantization-contaminated carriers.
8. Exact-zero sensitivity: a normal-approximate bootstrap test of `Delta = 0`, reported descriptively because statistically exact zero is a sharper claim than the practical-equivalence primary hypothesis.

Quantization sensitivity applies to every hypothesis: after the primary H1-H4 classifications are computed, recompute them excluding quantization-untestable and quantization-contaminated carriers. If exclusion changes a classification, the primary classification for that hypothesis is inconclusive as specified in the gate rule.

### 9.7 Copy analysis

Report exact copy rates by carrier, era, item type, Unicode/emoji version, plane, byte length, codepoint count, and tokenizer fragmentation. H4 uses only exact byte equality. Additional copy diagnostics:

- canonical-equivalence-only mismatch
- missing ZWJ
- missing VS16
- decomposed RGI sequence
- truncation
- extra text after exact target
- insertion/deletion/substitution counts
- replacement character emission
- reordering
- codepoint Levenshtein distance
- first UTF-8 byte mismatch offset and hex context

### 9.8 Scalar-listing analysis

Report exact rates by carrier, era, item type, target length, and whether the expected output contains only BMP or supplementary scalars. Classify errors as:

1. format failure
2. wrong number of scalar tokens
3. correct count but one wrong scalar
4. correct count but multiple wrong scalars
5. byte/content transcription mismatch
6. normalization substitution
7. placeholder/replacement-character output
8. other

### 9.9 Count secondary diagnostics

Report exact count rates and format rates by carrier, era, fragmentation level, item type, expected count, byte length, and measured tokens per cluster. Report signed integer error `parsed - expected` among parseable responses. A wrong-by-one answer is a near miss descriptively, not a success.

## 10. Near-miss classifier

Copy rows receive exactly one primary near-miss class using this precedence order:

1. `format_failure_wrapper_mimicry`: strict format failure and `<IN>`/instruction scaffolding appears in the completion
2. `format_failure_other`
3. `exact_copy`
4. `normalization_only_mismatch`
5. `decomposed_rgi_sequence_or_missing_zwj`
6. `missing_vs16`
7. `truncation_near_miss`: truncation with common prefix at least 80% of target bytes
8. `truncation_other`
9. `extra_text_after_exact_target`
10. `replacement_character_emitted`
11. `single_codepoint_edit`: Levenshtein distance exactly 1
12. `two_codepoint_edits`: Levenshtein distance exactly 2
13. `reordering`
14. `multi_codepoint_corruption`
15. `other_unclassified_mismatch`

A copy failure is called a **near miss** if it is well formatted and at least one of the following holds: canonical-equivalent only, Levenshtein distance <= 2, only ZWJ/VS16 structure is missing, exact target followed by extra text, or truncation preserves at least 80% of target bytes. Near-miss status never changes exact-success outcomes.

Count near miss: well-formatted integer answer differing from expected by exactly +/-1.
Scalar-listing near miss: well-formatted answer with the correct number of scalar tokens and exactly one wrong scalar.

## 11. Power analysis

The reproducible pure-NumPy script is `src/confirmatory_power_analysis.py`; output is `results/confirmatory_power_analysis.json`. It uses the pilot high-low failure gaps, includes a heterogeneous-carrier scenario, validates round-2 pilot facts against the raw scored rows, and includes a mutation check proving that flipping one qwen2.5 high-fragmentation outcome changes the audited gap.

At the frozen design size of 40 probes per era x fragmentation-level cell:

- H1 material-effect power is 1.000 in 2000 simulations under the conservative 22.2 pp pilot-calibrated scenario with either five or four valid core carriers.
- H1 material-effect power is 1.000 under the strong qwen2.5-like scenario and 0.999 in a heterogeneous scenario with two strong carriers, one moderate carrier, and two null carriers.
- The flat-null H1 material-effect rejection rate is 0.000 in these simulations because confirmation requires the lower confidence bound to exceed `log(1.25)`, not merely zero. This is a check on the simulation criterion, not a guarantee of exact finite-sample size.
- H2 equivalence-confirmation power under true `Delta = 0` is:
  - 0.996 with five valid carriers
  - 0.977 with four valid carriers
  - 0.9175 with three valid carriers
- H2 power to refute a material era effect is asymmetric and effect-size dependent. With a nominal +15 pp input shift, ceiling compression produces a realized mean difference of only about +11 pp; strict outside-margin refutation power is just 0.0845, 0.069, and 0.047 with five, four, and three carriers. With a nominal +25 pp input shift, the realized mean difference is about +14.6 pp and refutation power is 0.800, 0.686, and 0.5745. Thus the design is well powered to confirm equivalence when the effect is truly near zero and adequately powered only for large era effects, not for effects barely above the 10 pp margin.

The H1 simulation is optimistic because it omits item-type fixed effects and within-inventory dependence. The H2 simulation is conservative in treating matched outcomes as conditionally independent given pair difficulty, but optimistic in treating distinct probes as independent. Both limitations are explicit. The design minimum of four valid carriers is chosen so H2 equivalence confirmation remains adequately powered even if one core carrier fails structural validity; it does not solve the intrinsic difficulty of refuting effects close to the equivalence margin under binary ceiling compression.

Estimated generation time for the maximum 500 rows per carrier, using measured round-2 seconds/probe and conservative proxies for the two new carriers, is **57.5 minutes of generation** across all six carriers. Load time, tokenizer scans, gates, scoring, and failure recovery make a realistic wall-clock budget **2-4 hours** on the 16 GB M3 host, excluding model downloads.

## 12. Required tables and figures

The final report must contain exactly these primary tables and figures, though additional supplementary diagnostics are allowed:

### Tables

1. **T1 Carrier validity and quantization gate table**: every carrier, revision, tokenizer hash status, ASCII copy count, wrapper failures, ASCII count/scalar-listing counts, quantization status, inclusion in H1/H2/H3/H4.
2. **T2 Inventory cell table**: requested and achieved counts by carrier, era, fragmentation level, expected count, and item type; all structurally empty cells explicitly printed.
3. **T3 H1 pooled and carrier-specific coefficient table**: beta, odds ratio, bootstrap CI, p-value, valid replicate count.
4. **T4 H2 matched-pair table**: pooled Delta, carrier-specific Delta, level-specific Delta, bootstrap CI, TOST and Holm results.
5. **T5 Copy ceiling table**: exact counts by carrier, era, type, plus near-miss counts.
6. **T6 Scalar-listing floor table**: exact counts by carrier, era, type, plus error classes.
7. **T7 Count error table**: exact counts, wrong-by-one counts, signed error distribution, and format failure counts by level and era.

### Figures

1. **F1 Failure heatmap**: carrier rows, era x fragmentation-level columns, count-probe failure rate.
2. **F2 Fragmentation dose-response**: observed failure by measured tokens per cluster with adjusted model curve, faceted by carrier.
3. **F3 Matched era-effect forest plot**: carrier-specific and pooled `Delta` with bootstrap intervals and +/-10 pp equivalence band.
4. **F4 Copy/count/scalar triad**: per-carrier exact rates for copy, count overlap cells, and scalar listing.
5. **F5 Copy near-miss taxonomy**: counts by carrier and item type, with byte-offset examples in a supplement.

## 13. Threats to validity

1. **No random assignment of fragmentation.** Fragmentation co-varies with item type, byte length, age, plane, and sequence complexity. Matching and regression reduce but do not eliminate confounding.
2. **Structural empty cells.** New complex/ZWJ items cannot occupy the low-fragmentation region in frozen byte-fallback vocabularies. The design therefore cannot estimate a complete era x fragmentation factorial. This is intrinsic to the mechanism.
3. **Item-type collinearity.** Atomic items dominate L1/L2 and complex/ZWJ items dominate L3/L4. Within-type sensitivities are essential.
4. **Artificial complex non-emoji stimuli.** Base plus U+20D0 probes are valid graphemes but not necessarily attested words. They test structure manipulation rather than natural text frequency.
5. **Tokenizer and model family entanglement.** Tokenizer vintage is not randomized independently of model architecture or training data. Cross-family consistency mitigates but does not eliminate this.
6. **Quantization.** Mistral-7B cannot be compared with bf16 on a 16 GB host. Quantization may affect exactness. Quantization-untestable sensitivity is mandatory.
7. **Mirror provenance.** Llama-3 uses an MLX community artifact because canonical Meta weights are gated. Hash pinning documents exact artifacts but does not prove bit-equivalence to canonical bf16 weights. Tokenizer container files can also differ between mirrors even when core tokenization is behaviorally identical; the inventory therefore uses the exact inference tokenizer rather than assuming canonical equivalence.
8. **Stimulus dependence.** Constituents recur across probes. Probe-level bootstrap does not fully model multi-way constituent dependence.
9. **Frequency data incompleteness.** Emoji frequency rows are sparse and not available for non-emoji. Frequency adjustment is necessarily partial.
10. **Scalar-listing capability.** Failure may reflect hexadecimal notation or instruction following as well as Unicode structure. ASCII scalar-listing gates and error classification mitigate but do not erase this.
11. **Prompt generality.** Results apply to these exact prompts and strict output formats, not all possible elicitation strategies.
12. **Host environment.** MLX, tokenizer, regex, Python, and Unicode library versions affect results. All must be pinned and hashed.

## 14. Freeze order and amendment rule

The execution freeze order is:

1. Commit and push this pre-registration and implementation plan before any confirmatory model output is generated.
2. Build tokenizer inventories and count/copy/scalar-listing probe artifacts without running models.
3. Run inventory audits and mutation checks. Fix build bugs before proceeding; any substantive design change requires an explicit amendment section and a new commit before inference.
4. Freeze artifact hashes in `results/confirmatory/inventory_freeze.json`.
5. Run ASCII gates, quantization controls, and memory smoke tests.
6. Run Unicode confirmatory probes in checkpointed batches.
7. Score and analyze only after all primary rows are complete or a carrier is formally declared infeasible/invalid.

No post-execution change to hypotheses, gates, extraction rules, cell definitions, alpha, decision rules, or primary estimands is confirmatory. Such changes may be reported only as exploratory sensitivity analyses.

## 15. Judgment calls explicitly flagged for review

These choices are frozen for design purposes but are the most important judgment calls for user review before execution:

1. **Newest-L1 is conditional rather than forced.** Byte-fallback tokenizers structurally cannot supply the required count-4/5 low-fragmentation newest probes. H2 therefore uses L2-L4 overlap only.
2. **H2 uses practical equivalence, not exact zero.** The margin is +/-10 pp; refutation requires the bootstrap interval to lie entirely outside that margin. Exact-zero sensitivity is reported separately. This is coherent but intrinsically underpowered for effects only slightly larger than the margin.
3. **Mistral-7B is included despite an untestable bf16 quantization contrast.** It adds a distinct tokenizer family and 7B scale but is the largest inferential risk.
4. **Structural ASCII gates are strict at 11/12.** This protects interpretability but may exclude carriers that are imperfectly competent rather than incapable.
5. **The <=0.5 tokens-per-cluster pair cap may cause type substitutions or build deficits.** It is necessary for genuine fragmentation matching, especially because newest ZWJ items are systematically more fragmented within broad L4 bins.
6. **H1 uses a pooled ordinal-level logistic coefficient as primary.** Continuous-tpc, within-type, era-interaction, and linear-probability analyses are sensitivities rather than competing primary tests.
7. **Copy/decomposition inventories are reduced relative to round 2, and comparison probes are omitted** so the matched count design can be adequately powered within hours rather than days on the local host. Comparison is a byte-order control that was secondary to the fragmentation mechanism in the pilot.
