# Unicode Release Debt — exact-copy failure across post-tokenizer emoji and graphemes

**Status:** research spec, pre-kill-test  
**Date:** 2026-11-25  
**Owner:** solo independent researcher  
**Core artifact budget:** 575 LOC maximum

## 1. One-paragraph pitch

Unicode adds characters and recommended emoji sequences every year, but deployed tokenizer vocabularies are frozen artifacts. This project tests whether modern LLMs and hosted services accrue “Unicode release debt”: a rising probability of failing byte-exact copying and grapheme-structure operations as test items become newer than the tokenizer artifact, even when the task requires no semantic knowledge of the new character. The study separates two claims that are often conflated: an **operational round-trip claim** (can the system reproduce a valid Unicode sequence byte-for-byte?) and a **structural representation claim** (can it count or decompose extended grapheme clusters, especially ZWJ emoji sequences?). Unicode 15.1 is a useful natural experiment: it added 627 non-emoji characters, mostly CJK, while its emoji update introduced 118 RGI ZWJ sequences built from existing emoji components rather than new emoji codepoints. Ground truth is public and exact: the Unicode Character Database, `emoji-test.txt`, official grapheme-break tests, CLDR annotations, and Unicode release dates. The artifact is small: stratified test data, a verifiable tokenizer-vintage dataset, deterministic behavioral probes, byte-exact scoring, and regression analysis in under 600 lines of glue code. The project explicitly does **not** claim semantic understanding of new emoji or characters; naming/meaning probes are exploratory only and are confounded by training-data cutoffs.

## 2. Exact falsifiable claims

### 2.1 Operational definitions

For item `x`, tokenizer/model/system `m`:

- `B(x)`: UTF-8 bytes of the exact target string. All targets are valid Unicode scalar sequences; no unassigned code points, private-use code points, noncharacters, malformed ZWJ sequences, or byte garbage.
- `D(x)`: primary release date. For non-emoji characters it comes from UCD `DerivedAge`; for multi-codepoint non-emoji sequences it is the date of the maximum component age. For atomic emoji it is the Unicode release date of the underlying codepoint, not merely the later date at which an existing symbol became emoji-qualified. For ZWJ and other multi-codepoint RGI emoji it is the emoji release date from the `E` field in `emoji-test.txt`, because the sequence arrangement is the object of study. Store both `D_codepoint_max` and `D_emoji` for every emoji item and use `D_emoji` for atomic items only in sensitivity analysis.
- `V(m)`: conservative tokenizer-vintage upper bound, normally the earliest verifiable public artifact or model release date. This is not assumed to be the true training-freeze date.
- `lag(x,m) = max(0, D(x) - V(m)) / 365.25` in years. Signed sensitivity analyses also use the unclamped difference.
- `G(x)`: number of extended grapheme clusters under the pinned Unicode 18.0 UAX #29 reference implementation, validated against `GraphemeBreakTest.txt`.
- `toks(x,m)`: token count under tokenizer `m`, with special tokens disabled.
- `tokens_per_cluster(x,m) = toks(x,m) / G(x)`.
- `whole_sequence_token(x,m)`: 1 if the entire target item is exactly one non-special token.
- `tokenizer_roundtrip_exact(x,m)`: 1 if `decode(encode(x))` with special tokens disabled and documented decode settings reproduces `B(x)` exactly. This is a tokenizer-mechanism covariate, not the behavioral copy endpoint.
- `explicit_byte_fallback_count(x,m)`: count of explicit `<0xNN>`-style byte-fallback pieces for SentencePiece-like tokenizers.
- `partial_utf8_token_count(x,m)`: byte-level-BPE analogue counting tokens whose byte strings are not independently valid UTF-8, where token bytes are recoverable. This is not called “byte fallback” for byte-level BPE because those bytes are ordinary BPE atoms, not a separate fallback mechanism.
- `split_codepoint_count(x,m)`: number of Unicode scalar values whose UTF-8 bytes span more than one output token, computed by aligning concatenated token bytes to UTF-8 scalar boundaries. Missing when token bytes cannot be recovered; never silently zero-filled.
- `nfc_nfd_divergence(x,m)`: token-count and token-ID difference between NFC and NFD forms of the same canonical-equivalence pair.

### 2.2 Behavioral metrics

No LLM judges. All scoring is deterministic.

**Copy metric.** The model is instructed to wrap the copied text in ASCII tags. `extract(out)` extracts the contents of exactly one `<OUT>...</OUT>` pair using a Unicode-aware regex over Python strings; it never normalizes. Formatting failure is recorded separately.

`copy_exact(x,m) = 1[extract(out).encode("utf-8") == B(x)]`  
`F_copy(x,m) = 1 - copy_exact(x,m)`

Secondary deterministic diagnostics: normalization-only mismatch, missing U+200D, missing U+FE0F, fallback decomposition into separate emoji, truncation, insertion, substitution, U+FFFD emission, format failure, and codepoint-level Levenshtein distance. These are error taxonomy variables, not the primary score.

**Structural metrics.**

- `count_exact`: parsed integer answer equals the exact number of extended grapheme clusters requested; parse/format failure counts as task failure in the primary endpoint and is also reported separately.
- `decomp_exact`: parsed ordered list of `U+[0-9A-F]{4,6}` tokens exactly equals the input codepoint sequence.
- `compare_exact`: exact label (`SAME`, `REORDERED`, or `DIFFERENT`) equals ground truth.
- `F_count`, `F_decomp`, and `F_compare` are one minus the respective exactness indicators.

### 2.3 Confirmatory claims

**C1a, operational release-debt claim.** Within a fixed model or service, and after adjustment for item type, UTF-8 byte length, plane composition, sequence complexity, and (in emoji-only sensitivity analyses) available frequency proxies, `P(F_copy=1)` increases with positive `lag(x,m)`. In the prespecified logistic model this is `β_lag > 0`.

**C1b, operational fragmentation claim.** Within the same item universe, `P(F_copy=1)` increases with tokenizer fragmentation, operationalized primarily as `tokens_per_cluster` and secondarily as `split_codepoint_count`, explicit byte-fallback count where applicable, and lack of `whole_sequence_token`. In the prespecified logistic model this is `γ_frag > 0` for the primary fragmentation measure.

**C2a, structural count claim.** `P(F_count=1)` increases with `tokens_per_cluster` and with positive release lag after the same controls.

**C2b, structural decomposition claim.** `P(F_decomp=1)` increases with `tokens_per_cluster` and with positive release lag after the same controls. “In proportion to fragmentation” is operationalized as a monotone dose-response in logistic odds, not literal linear proportionality.

**Mechanistic prediction, secondary.** If input-side representation is the bottleneck, generating the same target bytes from an ASCII codepoint spelling (for example `U+1F642 U+200D U+2194 U+FE0F`) should succeed more often than copying the glyph sequence directly. If both fail equally, output-side byte assembly or decoding is implicated instead. This is analyzed as a paired contrast, not as a semantic test.

**Support criterion.** A claim is supported only if the adjusted coefficient is positive, its cluster-bootstrapped 95% CI excludes zero, and the sign is consistent in at least two independent tokenizer families. A result confined to one small model is a case study, not validation of the general claim. Nulls are informative if the week-1 kill criteria in section 8 are met.

**Explicit non-claim.** The study does not infer that a model “understands” an emoji, knows its name, or understands Unicode as a standard. Naming/meaning performance is cutoff-confounded and exploratory only.

## 3. Prior art table with verification status

| Prior-art claim or adjacent work | Status | Verification result | URL |
|---|---:|---|---|
| Di Cristofaro, “Data interference: emojis, homoglyphs, and issues of data fidelity in corpora and their results” | **Confirmed, narrowed** | Confirmed that the tested corpus tools AntConc, LancsBoxX, and SketchEngine did not correctly tokenize all current Unicode v15.1 emoji; ZWJ sequences were split or invalid types produced. This is evidence about corpus tools, not modern LLM tokenizers or behavioral copying. | https://arxiv.org/abs/2507.01764 |
| Peng et al., “EmojiLM: Modeling the New Emoji Language” | **Confirmed, narrowed** | Confirmed: EmojiLM augments BART/T5 vocabularies for emoji and deliberately tokenizes composed emoji into constituent pieces separated by a U+200D token for text-emoji translation. It is task-specific modeling evidence, not evidence about general-purpose frontier LLMs. The joiner is ZWJ U+200D, not ZWNJ. | https://arxiv.org/abs/2311.01751 |
| Hanay, “An Empirical Study on Sub-Word Tokenizer Fragmentation via Zero-Width Unicode Injection” | **Corrected** | The correct versioned record is `10.5281/zenodo.22064524`, not the initially supplied `10.5281/zenodo.22064523`. The preprint concerns invisible ZWNJ U+200C injection into ordinary text, not emoji ZWJ U+200D sequences. It reports cl100k token inflation and failures in tiny QA/summarization models; it explicitly did not evaluate large chat models and is not peer-reviewed. | https://doi.org/10.5281/zenodo.22064524 |
| Akritai practitioner post, “Does text survive a tokenizer round trip?” | **Confirmed** | Reports direct `decode(encode(text))` measurements on pinned public tokenizers; Qwen2.5 differences were NFC-normalization differences, while phi-4 and Gemma 4 had no differences in its sampled fields. Practitioner evidence, not peer-reviewed, but directly relevant to round-trip methodology. | https://akritai.dev/en/blog/tokenizer-roundtrip-in-training-data |
| Big Iron practitioner guide, “Reimplementing a BPE Tokenizer Without Dependencies: The Traps” | **Confirmed** | Documents Qwen2.5’s NFC normalizer, Unicode-version drift in normalization tables, incomplete UTF-8 per-token decoding, and special-token injection traps. Practitioner evidence, not peer-reviewed. | https://www.bigiron.cc/guides/reimplementing-a-bpe-tokenizer-without-dependencies-the-traps |
| Shoeb and de Melo, “Assessing Emoji Use in Modern Text Processing Tools” | **Confirmed** | ACL 2021 paper found notable shortcomings in NLP/text-processing tools on emoji tokenization, POS, parsing, and sentiment; ZWJ emoji handling was particularly difficult. Predates modern LLM services. | https://aclanthology.org/2021.acl-long.110/ |
| Petrov et al., “Language Model Tokenizers Introduce Unfairness Between Languages” | **Confirmed** | NeurIPS 2023 paper showed large cross-language token-length disparities, up to 15x in some cases. It motivates fragmentation-cost controls but does not measure Unicode-version-stratified copying or grapheme behavior. | https://arxiv.org/abs/2305.15425 |
| Edman et al., “CUTE: Measuring LLMs’ Understanding of Their Tokens” | **Confirmed, adjacent** | EMNLP 2024 benchmark of orthographic/token knowledge found models often know token spellings but fail to manipulate them. It does not stratify by Unicode release or focus on emoji/RGI sequences. | https://arxiv.org/abs/2409.15452 |
| Uzan and Pinter, “CharBench” | **Confirmed, adjacent** | Large character-level benchmark; counting correctness was only weakly correlated with tokenization properties, while positional tasks correlated with containing-token length. It is close methodologically but not release-stratified or grapheme/RGI-centered. | https://arxiv.org/abs/2508.02591 |
| Firestone et al., “UTF-8 Plumbing” | **Confirmed, adjacent** | Formalizes why byte-level tokenizer vocabularies can emit ill-formed UTF-8 and studies serving/constrained-generation consequences. It does not measure release recency or exact-copy behavior. | https://arxiv.org/abs/2511.05578 |
| Jang et al., “Improbable Bigrams Expose Vulnerabilities of Incomplete Tokens in Byte-Level Tokenizers” | **Confirmed, adjacent** | Shows hallucination vulnerabilities tied to incomplete byte-level tokens and unusual token adjacency. Relevant mechanism, not a release-stratified Unicode study. | https://arxiv.org/abs/2410.23684 |
| TokSuite | **Confirmed, adjacent** | Fourteen otherwise-matched pretrained models isolate tokenizer choice and include multilingual perturbation benchmarks. It is controlled tokenizer science, but not a Unicode-release-debt measurement. | https://arxiv.org/abs/2512.20757 |
| Zhang et al., “When Copying Is Hard” | **Confirmed, adjacent** | NeurIPS 2026 paper reports exact-span reproduction failures and a copy-constrained decoding method. Its domains are contracts, scientific literature, code, and random sequences, not Unicode-version-stratified emoji/graphemes. | https://arxiv.org/abs/2610.28393 |
| Qwen tokenization note | **Confirmed implementation evidence** | Qwen documents byte-level BPE over UTF-8, fallback to single bytes for uncommon text, and replacement characters when incomplete generated byte sequences are decoded with replacement behavior. | https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md |
| Claimed literature gap: no release-stratified exact-copy plus grapheme behavioral measurement | **Unverified/provisional** | Targeted searches found close work on copying, character reasoning, tokenizer fairness, emoji tools, malformed UTF-8, and tokenizer robustness, but no direct study matching all four criteria: release stratification, tokenizer-vintage alignment, byte-exact round-trip, and grapheme-structure tasks. Absence of evidence is not proof; this is the first Gemini Deep Research question. | See `GEMINI_DEEP_RESEARCH.md` |

## 4. Artifact architecture

Python, pinned dependencies, JSONL intermediate artifacts, append-only raw responses. The artifact is glue code, not a reimplementation: delegate grapheme segmentation, BPE/SentencePiece/tiktoken loading, and model inference to pinned libraries such as `regex`, `tokenizers`, `sentencepiece`, `tiktoken`, and `transformers`. The budget includes handwritten validation code but excludes downloaded Unicode data, generated JSONL data, dependency lockfiles, raw model outputs, and paper text.

| Module | LOC cap | Responsibility |
|---|---:|---|
| `ucd_build.py` | 100 | Download versioned Unicode/CLDR inputs; record URL, retrieval date, SHA256; parse `DerivedAge`, `UnicodeData` including range entries, `emoji-data`, `emoji-test.txt`, `GraphemeBreakTest`, and CLDR annotations. |
| `testset.py` | 80 | Surface-family/concept-cluster keys, eligibility filters, stratified sampling, matching variables, normalization pairs, deterministic item IDs. |
| `tokscan.py` | 90 | Load `tokenizer.json`, SentencePiece, and tiktoken artifacts; compute fragmentation, exact-sequence-token coverage, direct decode(encode) round-trip, byte-boundary metrics, NFC/NFD divergence, vocabulary age coverage. |
| `probes.py` | 60 | Prompt templates and deterministic probe-instance generation for copy, count, decomposition, comparison, codepoint conversion, and exploratory naming. |
| `runners.py` | 85 | Local Hugging Face generation plus minimal HTTP API clients; cache by prompt/model/decoding hash; record raw response, endpoint, date, latency, errors. |
| `score.py` | 55 | Unicode-aware extraction, byte equality, integer/codepoint/label parsers, deterministic error taxonomy. No fuzzy matching and no model judges. |
| `analyze.py` | 70 | Join behavior to tokenizer metrics; logistic regressions, matched sensitivity, cluster bootstrap, tables and plots. |
| `validate.py` | 35 | Checksums, schema validation, delimiter collision checks, ASCII sanity tests, official grapheme-break conformance, token-scan invariant tests. |
| **Total** | **575** | Hard cap. If exceeded, remove optional hosted-API or naming functionality rather than weakening validation. |

Implementation constraints:

1. All inputs and outputs immutable once written; corrections create a new content-addressed batch.
2. Every item stores source-file version, source line, codepoints, byte length, plane mix, sequence type, surface-family key, concept-cluster key, primary release version/date, codepoint-maximum date, emoji-sequence date where applicable, frequency proxy, and match covariates.
3. Every generation stores model revision, endpoint, decoding parameters, prompt hash, raw text, extraction status, and wall-clock date.
4. Dependency versions and tokenizer implementation Unicode-table versions are recorded. A tokenizer scan that cannot extract token bytes must emit `null`, not zero, for byte-boundary metrics.
5. The grapheme segmenter must pass 100% of the pinned official `GraphemeBreakTest.txt` cases before any derived count is trusted. If no lightweight library passes, the constructor may use known test-case boundaries directly rather than implementing a partial segmenter.

## 5. Test-set construction protocol

### 5.1 Frozen public inputs

Use Unicode 18.0 as the initial ground-truth snapshot:

- UCD `DerivedAge`: https://www.unicode.org/Public/18.0.0/ucd/DerivedAge.txt
- UCD `UnicodeData`: https://www.unicode.org/Public/18.0.0/ucd/UnicodeData.txt
- UCD `GraphemeBreakTest`: https://www.unicode.org/Public/18.0.0/ucd/auxiliary/GraphemeBreakTest.txt
- UCD `emoji-data` properties: https://www.unicode.org/Public/18.0.0/ucd/emoji/emoji-data.txt
- Emoji 18 `emoji-test.txt`: https://www.unicode.org/Public/18.0.0/emoji/emoji-test.txt
- Unicode release dates: https://www.unicode.org/history/publicationdates.html
- UAX #29 segmentation definition: https://unicode.org/reports/tr29/
- UTS #51 emoji definitions and versioning: https://unicode.org/reports/tr51/
- CLDR English annotations, pinned release-49 because CLDR 49 is explicitly updated for Unicode 18 emoji annotations (release note: https://cldr.unicode.org/downloads/cldr-49): https://raw.githubusercontent.com/unicode-org/cldr/release-49/common/annotations/en.xml
- Unicode emoji frequency ranking: https://www.unicode.org/emoji/frequency.html

**Path-version warning.** The emoji data location changes in newer releases. Emoji 15.1/16.0 have legacy `/Public/emoji/<version>/` directories; Emoji 17/18 are under `/Public/<unicode-version>/emoji/`. The builder must resolve paths from the versioned `ReadMe` rather than hard-coding only the legacy pattern. Every download records SHA256.

### 5.2 Item universe

**Panel A: RGI emoji.** Use only `fully-qualified` entries from `emoji-test.txt`.

- **Atomic emoji:** one base codepoint, optionally followed by U+FE0F; no ZWJ, skin-tone modifier, tag sequence, keycap sequence, or regional-indicator pair.
- **ZWJ emoji:** fully-qualified sequences containing U+200D.
- Exclude minimally-qualified and unqualified forms from the primary panel; use them only in sensitivity probes.
- Build `surface_family_key` by removing only skin-tone modifiers U+1F3FB through U+1F3FF. Use it for deduplication and allocation: sample at most one surface form per surface family. This prevents 20 skin-tone variants from counting as 20 independent items while retaining genuinely different gender/person codepoint compositions as distinct structural surface forms.
- Also record `concept_cluster_key`, a manually audited coarse key that merges only obvious gender/person variants of the same activity or object. Use it for cluster-bootstrap inference. Never merge distinct objects or activities. Publish the mapping with the test set; a sensitivity analysis uses the more granular surface key instead.

**Panel B: non-emoji atomic characters.** Assigned characters from `UnicodeData`/`DerivedAge`, categories L/M/N/P/S, excluding emoji-property characters, controls, combining marks alone in primary atomic items, noncharacters, private use, and unassigned code points. Correctly expand `First/Last` range records, especially CJK ranges. Prefer rare supplementary-plane letters/symbols/CJK that match recent emoji in UTF-8 length and plane.

**Panel C: non-emoji complex graphemes.** Use two sources. First, filter official `GraphemeBreakTest` sequences to assigned, non-control, non-private-use strings with valid scalar values. Second, because official tests contain few or no sequences whose maximum component age is recent, construct simple valid sequences by combining assigned non-emoji base characters with combining marks or other well-defined Extend/SpacingMark components. Do not use unassigned code points or claim these are orthographically valid words. Compute exact segmentation with the pinned implementation validated against the official test file. Classify sequence age by maximum component `DerivedAge`. This captures combining marks, conjunct/linker behavior where official examples exist, ZWJ/ZWNJ, regional indicators, and keycap-like structure without relying solely on the small official test corpus.

**Panel D: controls.**

- ASCII random strings of exactly 16 printable characters, excluding `<`, `>`, leading/trailing whitespace, and prompt delimiters.
- Common older emoji, oversampling Unicode frequency rows 0-4 where available.
- Rare older CJK/symbol characters matched to newer atomic items by byte length and plane.
- Canonical normalization pairs: NFC and NFD forms of the same string where canonically distinct.
- Explicit codepoint-spelling representations for paired conversion probes.

### 5.3 Version strata

Primary version bands:

1. Legacy: Unicode age up to 13.0 for characters or emoji age up to E13.1 for emoji sequences (there is no Unicode 13.1 release)
2. 14.0, released 2021-09-14
3. 15.0, released 2022-09-13
4. 15.1, released 2023-09-12
5. 16.0, released 2024-09-10
6. 17.0, released 2025-09-09
7. 18.0, released 2026-09-16

Unicode 15.1 must be split conceptually into two substrates: its new non-emoji character repertoire and its new RGI emoji ZWJ sequences. It added 627 characters overall, but its emoji update used existing emoji components in new sequences. This distinction separates “new codepoint” effects from “new arrangement of old codepoints” effects.

### 5.4 Sampling and matching

All random sampling uses `random.Random(20261125)` after sorting candidates deterministically by sequence type, version, and hexadecimal codepoint tuple. The sampled manifest records input sort order, seed, and exclusion reasons. Target main-study allocation before availability audit:

| Panel | Allocation | Approximate N |
|---|---|---:|
| Emoji atomic | 12 each from legacy/E14/E15; all strict atomic surface families from E16/E17/E18 (7/7/9 in the Unicode 18 snapshot) | 59 |
| Emoji ZWJ | 12 legacy; the single skin-tone-stripped E14 and E15 families; 12 of the 28 E15.1 surface families; all 11 E17 surface families | 37 |
| Non-emoji atomic | 16 per version band, 7 bands | 112 |
| Non-emoji complex graphemes | up to 8 per version band, 7 bands | <=56 |
| ASCII controls | random printable strings, exactly 16 characters | 24 |
| High-frequency older emoji controls | frequency rows 0-4 where available | 12 |
| Normalization pairs | 12 pairs, both forms included | 24 |
| **Total copy inputs** | availability-dependent | **about 324** |

The caps are deliberate. The recent emoji supply is small once skin-tone variants are deduplicated; do not inflate it. In the Unicode 18 snapshot, Emoji 15.1 has 118 fully qualified new ZWJ surface forms but only 28 skin-tone-stripped surface families; Emoji 17 has 146 new fully qualified ZWJ forms but only 11 such families. Concept clusters are fewer still. Cells with fewer than seven independent items are retained for description and pooled regression but not interpreted as stand-alone stratum estimates.

Matching variables, in priority order:

1. Exact UTF-8 byte length
2. Plane composition: BMP-only versus containing supplementary-plane codepoints
3. Item type: atomic emoji, ZWJ emoji, atomic non-emoji, complex non-emoji
4. Codepoint count and grapheme-cluster count
5. Sequence complexity proxy: `ZWJ instances + skin-tone modifiers + variation selectors + keycap/tag components + max(0, distinct pictograph bases - 1)`. This is a transparent sequence-complexity proxy, not a validated perceptual complexity measure.
6. Unicode emoji frequency row when available. Missing frequency is explicit, never imputed as zero.

Use greedy nearest-neighbor matching without replacement for old-versus-new matched sensitivity sets. Process new items in deterministic sorted order. A match is acceptable only when byte length and plane composition are exact and item type is identical or in a prespecified cross-type comparison. Within that admissible set choose the unused control minimizing the ordered tuple `(absolute codepoint-count difference, absolute grapheme-count difference, absolute complexity difference, frequency-row distance, hexadecimal codepoint tuple)`. For frequency matching, both-missing distance is 0 and exactly-one-missing distance is 99. If the first two differences exceed 1 or the complexity difference exceeds 2, leave the item unmatched rather than forcing a poor match. The primary regression uses all eligible items with covariates; the matched set is a robustness analysis, not the sole evidence.

### 5.5 Validity checks

- 100% of source rows either parse or produce a logged parse error; silent drops are failures.
- Copy targets are byte-distinct across all primary panels and controls; overlapping control candidates are replaced before freezing the test set.
- Every target passes UTF-8 round-trip encoding and contains no surrogate code points.
- Emoji targets are fully-qualified RGI entries in the frozen reference version.
- For strict atomic emoji, record any mismatch between emoji-qualification version and underlying codepoint age. In the Unicode 18 snapshot the E14-E18 strict-atomic target cells have zero such mismatches; any future mismatch is sensitivity-tagged rather than silently pooled.
- No target contains the exact ASCII input or output delimiter strings `<IN>`, `</IN>`, `<OUT>`, or `</OUT>`.
- No target has leading/trailing whitespace unless whitespace is the explicit object of a control.
- At least 30 items or 5% of items, whichever is larger, receive manual inspection of codepoints, version, surface-family key, concept-cluster key, and expected answer.
- Font rendering is not required. “Visually valid” means valid Unicode/RGI text, not guaranteed support in the researcher’s local font stack.

## 6. Tokenizer-vintage dataset: key sub-artifact

### 6.1 Initial tokenizer inventory

Scan the following, subject to license access:

| Family | Initial artifacts |
|---|---|
| OpenAI | `r50k_base`, `p50k_base`, `cl100k_base`, `o200k_base` from tiktoken. Official model-encoding mapping: https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken |
| Meta Llama | Llama 1/2 SentencePiece artifacts; Llama 3/3.1/3.2 tiktoken-based artifacts. Meta’s table gives launch dates and tokenizer families: https://github.com/meta-llama/llama-models/blob/main/README.md |
| Alibaba Qwen | Qwen `qwen.tiktoken`, Qwen2.5 `tokenizer.json`, Qwen3 tokenizer artifacts. Qwen 2023 started from cl100k and augmented multilingual vocabulary: https://arxiv.org/html/2309.16609; Qwen2.5 tokenizer details: https://arxiv.org/html/2412.15115; Qwen3 report: https://arxiv.org/abs/2505.09388 |
| Google Gemma/Gemini-derived | Gemma 1/2 artifacts and Gemma 3 artifact. Gemma 3 uses the Gemini 2.0 SentencePiece tokenizer with split digits, preserved whitespace, byte-level encodings, and about 262k entries: https://arxiv.org/abs/2503.19786 |
| Optional expansion | Mistral/Tekken or another family only after the four-family core is complete. |

Initial behavioral open models, chosen for CPU feasibility rather than maximal capability:

- Qwen3-0.6B or Qwen3-1.7B, released 2025-04-29: https://qwenlm.github.io/blog/qwen3/
- Gemma-3-1B-it, released 2025-03-12: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-3/
- Llama-3.2-1B-Instruct, released 2024-09-25: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
- TinyLlama-1.1B-Chat-v1.0 as a vintage-carrier control: its model card states that it uses exactly the Llama 2 tokenizer, making Unicode 15.1 and later post-vintage while remaining CPU-feasible: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

If a license or download blocks one checkpoint, substitute the smallest legally accessible instruct checkpoint in the same family and record the substitution as a protocol deviation. For TinyLlama specifically, the substitute must be a comparably small chat model demonstrably using the unmodified Llama 2 tokenizer; a newer Llama tokenizer is not an acceptable substitute because it removes the vintage contrast. Scan both the canonical Llama 2 artifact and the TinyLlama-packaged tokenizer files; if their regular vocabularies or merges differ, treat TinyLlama as its own tokenizer artifact rather than assuming continuity from the model card. Tokenizer scans should still proceed for all obtainable artifacts, including artifacts whose full models are too large for local behavior tests.

### 6.2 Vintage evidence protocol

Do **not** equate model release date, knowledge cutoff, tokenizer training date, and Unicode coverage. Record them as separate fields.

For each tokenizer artifact:

1. **Artifact identity.** Canonical URL, repository, revision/commit, file name, SHA256, file size, license status, and retrieval date.
2. **Existence upper bound.** Earliest verifiable public date at which the bitwise artifact or model using it existed: official release post, paper date, git commit, HF revision, package release, or archived download. Prefer primary sources. This date is conservative: anything released after it is definitely post-public-vintage.
3. **Construction window.** Directly documented tokenizer construction/training window only if a primary source states it. Otherwise `unknown..existence_upper_bound`. Never infer a precise freeze date from a knowledge cutoff.
4. **Normalizer and preprocessing.** Extract normalizer type, pre-tokenizer regex, byte-fallback mode, split-digit behavior, whitespace handling, special-token behavior, and implementation/library version.
5. **Empirical Unicode coverage.** For every single-codepoint vocabulary entry, join to `DerivedAge` and compute coverage fraction and maximum represented age per Unicode version. For every test emoji sequence, record both exact byte-sequence membership in the vocabulary and whether the full tokenizer pipeline actually encodes the sequence as one token; these can differ because normalization or pretokenizer boundaries may prevent a vocabulary entry from being produced. Presence is direct evidence of representation in the vocabulary; absence is not proof of ignorance because selective vocabularies omit many assigned characters and byte fallback can still encode them.
6. **ZWJ-sequence coverage curve.** For each emoji version, compute fraction of skin-tone-stripped RGI surface families that are single tokens, mean tokens per cluster, and mean split-codepoint count.
7. **Cross-generation continuity.** Compare artifact hashes and regular-vocabulary hashes across model generations. Do not assume continuity from marketing language. A local pre-audit on 2026-11-25 found that the HF revisions `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775` and `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` have exactly equal regular vocabularies (151,643 entries) and ordered merge lists (151,387 entries) after normalizing the JSON merge serialization; Qwen3 adds four control tokens (151,665-151,668). This verifies continuity for those two snapshots only, not the whole Qwen line.
8. **Hosted services.** Treat each dated hosted endpoint as a black-box system. Record vendor, endpoint snapshot, request date, official tokenizer mapping if one exists, and `unknown` otherwise. Never assign a public tokenizer measurement to a proprietary endpoint without primary evidence.

Required output schema fields:

`tokenizer_id, family, artifact_url, revision, artifact_sha256, artifact_size, license_status, first_public_date, first_public_evidence_url, release_date, construction_window_start, construction_window_end, vintage_confidence, algorithm, vocab_size, regular_vocab_sha256, normalizer, pre_tokenizer_hash, byte_fallback_mode, unicode_table_version, max_atomic_age, coverage_by_unicode_version, rgi_single_token_rate_by_emoji_version, notes`

### 6.3 Verifiability requirements

- Two independent downloads of the same pinned artifact must produce identical SHA256 values.
- Every date requires a URL and a quoted evidence snippet in the dataset documentation. If only one weak source exists, `vintage_confidence` is low.
- The scan is deterministic: same artifact bytes plus same test items must produce identical metrics.
- License-restricted artifacts are not redistributed. Publish URLs, hashes, scripts, and derived non-sensitive measurements.
- Primary analysis uses the conservative public-vintage upper bound. Sensitivity analyses use documented construction windows if deep research uncovers them.

## 7. Experimental design

### 7.1 Probes

All prompts include a short task-specific instruction and forbid explanation. Use the model’s native chat template for instruction-tuned models. Keep system prompts empty unless the vendor requires one. Examples below use `{TARGET}` placeholders.

| Probe | Prompt pattern | Required output | Primary score |
|---|---|---|---|
| Copy exactly | “Copy the text between `<IN>` and `</IN>` exactly. Preserve every Unicode code point, including invisible joiners and variation selectors. Put only the copied text inside `<OUT>` and `</OUT>`. Input: `<IN>\n{TARGET}\n</IN>`” | `<OUT>{exact target}</OUT>` | `copy_exact` |
| Count graphemes | “Count the extended grapheme clusters (user-perceived characters), not bytes or code points. Reply with one integer only.” Definitions of ZWJ and combining behavior are supplied generically, without the answer. | integer | `count_exact` |
| Decompose sequence | “List the Unicode scalar values in order as hexadecimal codepoints. Use exactly this format: `U+1F600 U+200D`.” | ordered codepoints | `decomp_exact` |
| Compare/reorder | “Compare A and B byte-exactly. If identical output `SAME`; if same codepoints in a different order output `REORDERED`; otherwise output `DIFFERENT`.” | one label | `compare_exact` |
| Codepoint conversion | “Convert this ASCII codepoint spelling into the exact Unicode string and wrap it in `<OUT>` tags.” | exact target bytes | `copy_exact` |
| Exploratory naming | “Give the official Unicode/CLDR name.” | name | exact/normalized name match, exploratory only |

In probe templates, `\n` denotes an actual newline. Targets are visually and tokenization-wise isolated from delimiter letters as much as possible; tokenizer-scan covariates are nevertheless measured on the target string alone, with prompt-context tokenization retained only as a sensitivity diagnostic. Copy inputs contain a single target. Count inputs contain one to five validated clusters, with expected count known by construction. Where adjacency could change segmentation, construction uses the official segmentation implementation to verify expected boundaries rather than assuming them.

### 7.2 Systems and decoding

Primary local/open arm: the four small instruct models in section 6.1. TinyLlama is mandatory because it is the vintage-carrier case that makes Unicode 15.1 post-vintage; the other three cover newer tokenizer families. Deterministic decoding (`temperature=0` or do-not-sample), maximum output length 64 tokens, repetition penalty disabled unless needed to terminate. One deterministic run is the primary observation. A second sampled run at temperature 0.7 is sensitivity-only.

Optional hosted arm: cheapest stable dated text endpoints from OpenAI, Google, and/or Anthropic, plus Alibaba if convenient. Run a 72-item stratified audit only if existing credits are available. Hard cap USD 10; otherwise omit. Analyze hosted endpoints separately because hidden preprocessing, safety layers, detokenization, and endpoint drift prevent clean tokenizer attribution.

### 7.3 Scoring and error taxonomy

Primary scoring is byte equality or exact deterministic answer equality. Extraction failure is a task failure, but report three rates separately:

1. `format_failure`: no unique valid wrapper/parseable answer
2. `content_failure_given_format`: extracted/parseable answer is wrong
3. `total_failure`: union, the primary endpoint

For copy mismatches, deterministic classifiers record: NFC/NFD-only difference, missing ZWJ, missing variation selector, emoji sequence split into components, truncation, extra text, substitution, replacement character, reordering, and other. Classes may overlap.

Copy results are always stratified by `tokenizer_roundtrip_exact`. The system-level endpoint includes tokenizer-lossy items because users experience the whole pipeline. A separate model-representation analysis excludes or controls for items whose tokenizer round trip already alters the target; failures on those items must not be attributed to model weights.

### 7.4 Statistical analysis

Primary unit: one deterministic response per item x model x probe. Primary regression for copy:

`logit P(F_copy=1) = α_model + α_itemtype + β1 lag_positive + β2 utf8_bytes + β3 plane_composition + β4 log(codepoints) + β5 complexity`

Here `α_itemtype` distinguishes atomic emoji, ZWJ emoji, atomic non-emoji, complex non-emoji, and controls, so a separate ZWJ indicator is collinear and omitted. Unicode frequency rows are not estimable across the full item universe because non-emoji rows are structurally missing; enter frequency as a categorical control only in emoji-only sensitivity analyses, with missing frequency as its own level and never as zero.

Structural regressions replace the outcome with `F_count` or `F_decomp` and add constructed cluster count where relevant. Fragmentation regressions use:

`logit P(F=1) = α_model + α_itemtype + γ1 tokens_per_cluster + same non-lag controls`

A combined model enters `lag_positive` and `tokens_per_cluster` together. Attenuation of `β1` after fragmentation is descriptive evidence for mediation, not proof of causation.

Inference:

- 5,000 cluster bootstrap replicates clustered by `concept_cluster_key` (or surface-family key in sensitivity), preserving model structure.
- Model and item type are fixed effects. Item families are not treated as random samples from a large population because recent cells nearly enumerate the population.
- Holm correction across six confirmatory tests: copy-lag, copy-fragmentation, count-lag, count-fragmentation, decomposition-lag, and decomposition-fragmentation.
- If complete separation occurs, report exact counts and use penalized/Firth logistic regression or a prespecified permutation trend test; do not silently accept unstable huge coefficients.

Sensitivity analyses:

1. Matched-only subset.
2. Signed lag rather than positive-part lag.
3. Atomic-emoji date definition: underlying codepoint age in the primary analysis versus emoji-qualification date in sensitivity.
4. Alternative vintage assumptions: public release date versus documented construction window.
5. Excluding frequency controls; then restricting to items with observed frequency within emoji-only analyses.
6. Unicode 17 versus Unicode 18 grapheme reference where rules changed.
7. Sampled decoding rather than deterministic decoding.
8. Excluding format failures and scoring content conditional on format.
9. Atomic-only and ZWJ-only subsets.
10. Excluding Unicode 18 items because they are newest and sparsely represented.

## 8. Week-1 kill test with numeric thresholds

### 8.1 Pilot sample and workload

Copy pilot:

- All Emoji 15.1/16/17/18 surface families available in the Unicode 18 snapshot: exactly 62 items after skin-tone deduplication if availability matches the audit (28 E15.1 ZWJ + 11 E17 ZWJ + 7/7/9 strict atomic E16/E17/E18).
- 32 older emoji controls: 16 atomic and 16 ZWJ, matched to the new emoji set as well as feasible.
- 32 new non-emoji items from Unicode 15.1-18: 16 atomic and 16 complex where availability permits.
- 32 older non-emoji controls: 16 atomic and 16 complex, matched by byte length/plane as well as feasible.
- 24 ASCII controls.

Expected copy total: 182 items x 4 local models = 728 deterministic generations. The newest-bin operational subset contains 94 items per model (62 new emoji plus 32 new non-emoji).

Structural pilot:

- 48 decomposition probes per model: 24 newest surface-family ZWJ items (12 E15.1 plus all 11 E17 plus one additional E15.1 item) and 24 legacy ZWJ controls matched as closely as feasible on codepoint count and UTF-8 byte length.
- 36 count bundles per model: 18 constructed only from Unicode/emoji 15.1-or-newer clusters and 18 constructed only from legacy clusters, paired by intended cluster count and approximate byte length. Within each era half are high-fragmentation and half low-fragmentation under that model's own tokenizer scan, using the median `tokens_per_cluster` within that era's candidate pool for that model.
- 24 comparison probes per model: 12 newest and 12 legacy.

Expected total: 728 copy plus 432 structural generations = 1,160 short local generations in week 1. This is larger than a three-model pilot but is justified because TinyLlama supplies the only clean pre-Unicode-15.1 tokenizer-vintage carrier in the selected CPU-feasible set. If compute becomes binding, reduce optional comparison probes first, never the copy or decomposition cells. Hosted APIs are not required for the kill test.

### 8.2 Harness-validity gates

Before interpreting any result:

- ASCII byte-copy accuracy must be at least 23/24 (95.8%) for each model. If not, the prompt/extraction harness or instruction-following setup is broken; fix it.
- Output-wrapper format failure must be at most 5% per model in the copy task. If not, treat total failure as a service-behavior result but do not interpret it as representation failure until content-conditional scoring is checked.
- Tokenizer scans must produce metrics for at least 95% of pilot items per open tokenizer. Otherwise the measurement artifact, not the hypothesis, is broken.
- The grapheme implementation must pass 100% of the pinned official test file.

### 8.3 Explicit kill criteria

**Hard kill, operational arm:** every one of the four models copies at least 92/94 newest-bin items byte-exactly (97.9%) at temperature 0, and the newest-bin failure rate minus the pooled old-control failure rate is at most 2 percentage points for every model. In that case the operational release-debt claim is dead in its strongest form; do not continue spending on copy probes.

**Hard kill, structural arm:** every model achieves at least 22/24 exact responses (91.7%) on the newest decomposition probes and at least 17/18 exact responses (94.4%) on the newest-only count bundles, and the high-fragmentation minus low-fragmentation error gap is no more than 2 percentage points in every model. If so, the structural claim is dead in this design.

**Whole-project hard kill:** both arm-level hard kills occur. Publish nothing or reduce to a one-page negative-note artifact; do not manufacture a semantic-cutoff story.

**Regression-based stop rule:** if the pilot is not perfect but the adjusted release-lag slope is non-positive and the upper 95% confidence bound is below an odds ratio of 1.10 per year of positive lag in both copy and structural endpoints, stop. The design has no evidence of a practically meaningful monotone release effect.

**Proceed rule:** proceed to the full study if any of the following holds:

1. Newest-bin copy failure exceeds matched old-control failure by at least 5 percentage points in at least two of four models.
2. High-fragmentation structural failure exceeds low-fragmentation failure by at least 15 percentage points in at least two of four models.
3. A systematic edit class appears in at least five distinct newest items in any model: missing ZWJ, missing FE0F, normalization-only alteration, replacement character, or decomposition of a single RGI sequence.
4. Tokenizer scan shows a monotone drop in exact single-token coverage or rise in split-codepoint count across emoji versions in at least two families, even if behavior is mixed. In this case proceed but narrow the paper toward mechanism plus structural behavior.

**Pivot rules:** if copy is hard-killed but structural survives, drop the operational arm and make grapheme structure the paper. If structural is hard-killed but copy survives, drop structural claims and make byte round-trip the paper. If failures are similar in ASCII and Unicode controls, debug instruction following and do not interpret as Unicode release debt.

**Incomplete-decision rule:** if neither a hard kill nor a proceed rule fires and the regression-based stop rule is not met, the pilot is inconclusive. Do not write up the inconclusive pilot as evidence. The default is to stop; at most one bounded three-day harness refinement is allowed when a specific mechanical defect has been identified, followed by exactly one rerun.

## 9. Execution plan

| Phase | Time | Work | Decision gate |
|---|---|---|---|
| 0. Snapshot and inventory | Days 1-2 | Download/checksum Unicode inputs; implement `ucd_build`, item parser, tokenizer artifact inventory skeleton. | G0: all URLs, hashes, and parses valid; otherwise stop and fix data pipeline. |
| 1. Kill test | Days 3-7 | Implement tokenizer scan, copy/decomposition/count probes, deterministic runner, scoring; run pilot on four local models. | G1: apply section 8. Kill, pivot, or proceed. |
| 2. Full test set | Week 2 | Construct all strata, matching variables, normalization pairs, surface-family keys, concept-cluster keys; manually audit sample. Freeze test set hash. | G2: no target/delimiter/ground-truth errors; recent-cell scarcity documented. |
| 3. Main runs | Weeks 2-3 | Run deterministic local main study, sampled sensitivity, optional hosted audit; scan all tokenizer artifacts. | G3: harness validity repeated; no missingness above 5% unexplained. |
| 4. Analysis | Week 4 | Regressions, bootstrap, matched sensitivity, error taxonomy, tokenizer-vintage dataset documentation. | G4: decide whether evidence is multi-family general, single-family case study, or negative. |
| 5. Deep-research integration | Week 4, parallel | Run the four Gemini Deep Research prompts; reconcile citations; correct prior-art table. | G5: if direct prior art exists, reframe as replication/extension or kill novelty claim. |
| 6. Writing and release | Weeks 5-6 | Short paper, artifact README, anonymized repository, reproducibility checklist. | G6: submit only if analysis and artifact are complete; otherwise release preprint/artifact later. |

If the ACL 2027 January ARR deadline is the target, weeks 1-5 are immovable. Do not expand scope to more model families, more languages, semantic benchmarks, human subjects, or multimodal rendering before that gate.

## 10. Venue plan

| Venue | Fit | Verified deadline/status | Plan |
|---|---|---|---|
| ACL 2027 main/Findings | Medium to high if framed as evaluation/reproducibility/resource plus tokenizer analysis | ARR submission deadline January 4, 2027; workshops August 18-19; main conference August 20-22. https://2027.aclweb.org/calls/main/ | Ambitious but feasible only if week-1 kill test passes and the artifact remains small. Submit as short paper. |
| Eval4NLP | Highest methodological fit: exact metrics, evaluation design, reproducibility | 2025 deadline was September 29, 2025; workshop December 23, 2025. No verified 2026/2027 call found as of this spec. https://eval4nlp.github.io/2025/cfp.html | Preferred workshop if a future iteration appears. ARR reviews can later be committed if rules permit. |
| BlackboxNLP | Medium if framed as interpretability/representation failure and tokenizer mechanism | 2026 direct submission deadline July 17, 2026, passed. https://www.aclweb.org/portal/content/blackboxnlp-2026-9th-workshop-analyzing-and-interpreting-neural-networks-nlp | Target next cycle only if internal/tokenizer mechanism is central; otherwise too interpretability-specific. |
| Workshop on Multilingual Multimodal Learning | Low as currently scoped | Verified instance was ACL 2022, submission deadline February 28, 2022. https://mml-workshop.github.io/ | Correction to initial venue intel: not an obviously active recurring venue and the project is not inherently multimodal. Do not plan around it unless revived and reframed. |
| arXiv plus artifact | High fallback | No deadline | If kill/pivot evidence is interesting but too small for a venue, release reproducible artifact and negative/positive note. |

## 11. Risks and honest limitations

1. **Tokenizer vintage is not directly observable.** Public release date is an upper bound, not a construction date. Conservative analyses may underestimate true lag; aggressive assumptions can invent false lag.
2. **Tokenizer coverage is not Unicode support.** Byte-level and byte-fallback tokenizers can encode unseen characters. Exact vocabulary membership is evidence about compression/lexicalization, not proof that a model knows or does not know a Unicode version.
3. **Recency and frequency are confounded.** Unicode itself notes that newer emoji tend to have lower measured frequency because platform rollout takes time. Matching can reduce but not remove this confound.
4. **Small-cell problem.** Recent emoji releases contain few independent atomic families after skin-tone deduplication. Non-emoji characters provide power, but then the claim broadens beyond emoji.
5. **Model quality confound.** Small CPU-friendly models may fail copy/instruction tasks for reasons frontier models do not. Hosted endpoints are black boxes and cannot cleanly attribute failure to tokenization.
6. **Instruction following versus representation.** Wrapper failure is not Unicode failure. Content-conditional and total-failure analyses must be shown side by side.
7. **Copying can succeed without structural representation.** Attention-based copy mechanisms may reproduce bytes even when counting/decomposition fails. Hence the two arms are separate, not redundant.
8. **Normalization is both a mechanism and a target alteration.** NFC-normalizing tokenizers make byte-exact NFD round trips impossible at input encoding. This must be reported as tokenizer behavior, not blamed on model weights alone.
9. **Unicode implementations themselves drift.** Regex property tables, normalization tables, and grapheme rules have versions. The tokenizer’s implementation Unicode version may differ from both Python and the UCD snapshot.
10. **Observational design.** Version distance is not randomized. Regression and matching control measured confounders only.
11. **No invalid UTF-8 through normal text APIs.** Hosted APIs generally expose valid Unicode strings, so raw malformed token sequences may be sanitized before observation. Open-model logits/serving internals would be needed for that separate question.
12. **Multiple comparisons.** Version bands, tasks, models, and fragmentation metrics create many possible tests. Only section 2.3 is confirmatory; everything else is exploratory.
13. **Visual validity is not font validity.** New Unicode 18 emoji may render as fallback boxes on old systems while still being valid text inputs.
14. **ZWJ/ZWNJ confusion is easy.** Emoji composition uses ZWJ U+200D. The corrected Zenodo prior art concerns ZWNJ U+200C injection. The paper must not conflate them.
15. **Structural tasks are not pure representation probes.** Counting, decomposition, and comparison also test instruction following and familiarity with `U+` notation. ASCII controls, format-conditional analyses, fragmentation controls, and the conversion arm reduce but do not eliminate this confound.

## 12. Open questions requiring deep external research

1. Does any 2022-2026 work already measure exact-copy or grapheme-structure behavior stratified by Unicode/emoji release version relative to tokenizer vintage? The novelty claim depends on this.
2. What is the verifiable timeline of major tokenizer artifacts, including first public dates, vocabulary continuity across generations, normalizers, byte-fallback modes, and any documented tokenizer training windows?
3. Which Unicode versions can each tokenizer plausibly represent directly in its vocabulary, based on auditable artifact evidence rather than model knowledge cutoffs?
4. What exactly happens in byte-level BPE, SentencePiece byte fallback, normalization layers, streaming detokenizers, and hosted API preprocessing when unseen emoji ZWJ sequences or NFC/NFD variants appear?
5. Do hosted services normalize, repair, strip, or re-encode Unicode input/output outside the model? If yes, byte-level failures may be service-layer behavior rather than model behavior.

`GEMINI_DEEP_RESEARCH.md` operationalizes these questions as copy-paste-ready external research prompts.
