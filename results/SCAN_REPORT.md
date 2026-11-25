# Unicode release debt: tokenizer scan report

Scope: Phase 0 plus tokenizer-scan portion of Phase 1. No model weights or behavioral inference were used. Tokenization uses no special tokens. SentencePiece artifacts are encoded with the native backend and cross-checked against raw `tokenizer.json`; Qwen uses raw `tokenizer.json` directly, never `transformers.AutoTokenizer`.

## Data and harness validity

- Unicode/emoji inputs: Unicode 18.0.0, Emoji 18, CLDR release-49 annotations; 32 recorded input files are SHA-256 pinned and passed independent second-download hash checks.
- Audit counts reproduced: E15.1 ZWJ 28 surface families (118 raw), E17 ZWJ 11 (146 raw), strict atomic E16/E17/E18 = 7/7/9.
- Grapheme conformance: 853/853 official tests (100.0%), UAX #29 revision 49 / Unicode 18.0.0 extended grapheme clusters, implementation `regex 2026.11.24 \X`. A split-every-codepoint negative control fails 516 cases, so the conformance check can fail.
- Copy pilot: 182 byte-distinct items (62 new emoji, 32 old emoji controls, 32 new non-emoji, 32 old non-emoji controls, 24 ASCII). Structural base probes: 48 decomposition and 24 comparison; 36 count bundles per scanned tokenizer. Five canonical-form scan-only diagnostics were added for NFC/NFD behavior.
- Tokenizer-scan completeness gate: 100.0% minimum over copy items and tokenizers (threshold 95%); native SentencePiece/fast-tokenizer ID agreement is 100.0%. The no-model-weights artifact gate passes.
- Host normalization versions are explicit: host Python 3.12.13 `unicodedata` 15.0.0; canonical diagnostics use pinned `unicodedata2` 18.0.0.

## Tokenizer artifacts

- qwen2.5-0.5b-instruct vs qwen3-0.6b: regular vocab equal=True, merges equal=True (151643/151643 regular entries).
- tinyllama-1.1b-chat vs llama2-7b-hf-mirror: regular vocab equal=True, merges equal=True (31997/31997 regular entries).
- Gemma 3 and canonical Llama 2 repositories are license-gated; scans use pinned unsloth Gemma and NousResearch Llama mirrors. TinyLlama's `tokenizer.model` SHA-256 equals the Llama mirror's, and their regular vocabularies/merges are identical.
- Maximum Unicode age with at least one exact single-codepoint vocabulary entry: Gemma 15.0, Qwen2.5/Qwen3 12.0, Llama2/TinyLlama 6.0. This is sparse lexical coverage, not a completeness threshold.

## Newest-cohort exact vocabulary membership

- gemma3-1b-it-mirror: pipeline 0/62, exact-vocab 0/62; llama2-7b-hf-mirror: pipeline 0/62, exact-vocab 0/62; qwen2.5-0.5b-instruct: pipeline 0/62, exact-vocab 0/62; qwen3-0.6b: pipeline 0/62, exact-vocab 0/62; tinyllama-1.1b-chat: pipeline 0/62, exact-vocab 0/62.
- Any newest-cohort emoji as one exact vocabulary entry in any scanned tokenizer: **no**. Any newest-cohort emoji as one token through the full pipeline: **no**. This does not weaken the lexical-staleness mechanism, but absence is not proof of ignorance because byte fallback can still encode the sequences.

## Proceed-rule 4 evidence

Interpretation used: primary emoji bands from the spec, all RGI surface families pooled, monotone means non-increasing/non-decreasing with at least one strict adjacent change. Flat all-zero series are not counted as a coverage drop. Because atomic and ZWJ composition varies by release, atomic-only series are the cleaner sensitivity and are shown here.

- gemma3-1b-it-mirror (Gemma): pooled pipeline single-token coverage legacy=0.603 (n=1561), E14.0=0.263 (n=38), E15.0=0.095 (n=21), E15.1=0.000 (n=28), E16.0=0.000 (n=7), E17.0=0.000 (n=18), E18.0=0.000 (n=9); atomic pipeline coverage legacy=0.714 (n=1320), E14.0=0.270 (n=37), E15.0=0.100 (n=20), E16.0=0.000 (n=7), E17.0=0.000 (n=7), E18.0=0.000 (n=9); atomic mean split-codepoint count legacy=0.149, E14.0=0.730, E15.0=0.900, E16.0=1.000, E17.0=1.000, E18.0=1.000; pooled signal=True, atomic sensitivity signal=True.
- llama2-7b-hf-mirror (Llama): pooled pipeline single-token coverage legacy=0.002 (n=1561), E14.0=0.000 (n=38), E15.0=0.000 (n=21), E15.1=0.000 (n=28), E16.0=0.000 (n=7), E17.0=0.000 (n=18), E18.0=0.000 (n=9); atomic pipeline coverage legacy=0.002 (n=1320), E14.0=0.000 (n=37), E15.0=0.000 (n=20), E16.0=0.000 (n=7), E17.0=0.000 (n=7), E18.0=0.000 (n=9); atomic mean split-codepoint count legacy=0.988, E14.0=1.000, E15.0=1.000, E16.0=1.000, E17.0=1.000, E18.0=1.000; pooled signal=True, atomic sensitivity signal=True.
- qwen2.5-0.5b-instruct (Qwen): pooled pipeline single-token coverage legacy=0.561 (n=1561), E14.0=0.000 (n=38), E15.0=0.000 (n=21), E15.1=0.000 (n=28), E16.0=0.000 (n=7), E17.0=0.000 (n=18), E18.0=0.000 (n=9); atomic pipeline coverage legacy=0.664 (n=1320), E14.0=0.000 (n=37), E15.0=0.000 (n=20), E16.0=0.000 (n=7), E17.0=0.000 (n=7), E18.0=0.000 (n=9); atomic mean split-codepoint count legacy=0.189, E14.0=1.000, E15.0=1.000, E16.0=1.000, E17.0=1.000, E18.0=1.000; pooled signal=True, atomic sensitivity signal=True.
- qwen3-0.6b (Qwen): pooled pipeline single-token coverage legacy=0.561 (n=1561), E14.0=0.000 (n=38), E15.0=0.000 (n=21), E15.1=0.000 (n=28), E16.0=0.000 (n=7), E17.0=0.000 (n=18), E18.0=0.000 (n=9); atomic pipeline coverage legacy=0.664 (n=1320), E14.0=0.000 (n=37), E15.0=0.000 (n=20), E16.0=0.000 (n=7), E17.0=0.000 (n=7), E18.0=0.000 (n=9); atomic mean split-codepoint count legacy=0.189, E14.0=1.000, E15.0=1.000, E16.0=1.000, E17.0=1.000, E18.0=1.000; pooled signal=True, atomic sensitivity signal=True.
- tinyllama-1.1b-chat (Llama): pooled pipeline single-token coverage legacy=0.002 (n=1561), E14.0=0.000 (n=38), E15.0=0.000 (n=21), E15.1=0.000 (n=28), E16.0=0.000 (n=7), E17.0=0.000 (n=18), E18.0=0.000 (n=9); atomic pipeline coverage legacy=0.002 (n=1320), E14.0=0.000 (n=37), E15.0=0.000 (n=20), E16.0=0.000 (n=7), E17.0=0.000 (n=7), E18.0=0.000 (n=9); atomic mean split-codepoint count legacy=0.988, E14.0=1.000, E15.0=1.000, E16.0=1.000, E17.0=1.000, E18.0=1.000; pooled signal=True, atomic sensitivity signal=True.

**Verdict:** proceed-rule 4 FIRES under the non-increasing/non-decreasing interpretation; firing families: Gemma, Llama, Qwen. Even if near-ceiling Llama is discounted, Gemma and Qwen independently satisfy the two-family threshold. Llama's coverage drop is small in absolute terms (0.002 to zero) and its atomic split rise starts at 0.988, so it is ceiling-sensitive rather than primary evidence.

## NFC/NFD tokenizer behavior

- gemma3-1b-it-mirror: NFC/NFD token-count pairing divergent-counts (40.0% equal), token-ID divergence 100.0%, NFC round-trip 100.0%, NFD round-trip 100.0% (n=5).
- llama2-7b-hf-mirror: NFC/NFD token-count pairing divergent-counts (0.0% equal), token-ID divergence 100.0%, NFC round-trip 100.0%, NFD round-trip 100.0% (n=5).
- qwen2.5-0.5b-instruct: NFC/NFD token-count pairing equal-counts (100.0% equal), token-ID divergence 0.0%, NFC round-trip 100.0%, NFD round-trip 0.0% (n=5).
- qwen3-0.6b: NFC/NFD token-count pairing equal-counts (100.0% equal), token-ID divergence 0.0%, NFC round-trip 100.0%, NFD round-trip 0.0% (n=5).
- tinyllama-1.1b-chat: NFC/NFD token-count pairing divergent-counts (0.0% equal), token-ID divergence 100.0%, NFC round-trip 100.0%, NFD round-trip 100.0% (n=5).

## Caveats and protocol deviations

- The spec's global surface-family deduplication instruction conflicts with its 62-item audit: 13 ZWJ surface keys occur in multiple emoji versions, including 4 E17 skin-tone-bearing families that collapse onto older no-tone E4.0 surface keys if deduplication is global. Counts and the pilot therefore dedupe within emoji version, as required to reproduce 28/11/7/7/9 and 62 items.
- Proceed-rule 4 does not define the version series or whether monotone means strict at every adjacent step. This report uses the spec's primary bands and non-strict monotonicity with at least one strict adjacent change. Under a stricter every-adjacent-step decrease requirement, coverage cannot fire after it reaches zero; the atomic split-codepoint prong remains the cleaner positive signal.
- The user prompt omitted Llama 3.2, although the spec's behavioral four-model arm includes it. Llama 3.2 tokenizer artifacts and model selection remain unresolved before behavioral inference.
- Unicode 18.0 itself is dated 2026-09-16 in publicationdates.html, while Emoji 18.0 is dated 2026-09-15 in UTS #51. This is not a conflict: atomic characters use Unicode dates and emoji sequences use emoji dates.
- The current Phase-0/scan codebase totals 1507 LOC across src/*.py (analyze_scan 302, testset 417, tokscan 405, ucd_build 240, validate 143), already exceeding the spec's eventual 575-line total artifact budget before behavioral runners exist. The cap appears infeasible at the specified rigor (hand-rolled SentencePiece protobuf parsing, inventory vintage analysis, audit reproduction are not delegable); explicit budget renegotiation is needed before Phase 1 behavioral work.

Machine-readable outputs: `manifest.json`, `ucd_audit.json`, `testset_audit.json`, `validation.json`, `tokenizer_inventory.json`, `tokenizer_scan.csv/jsonl`, `copy_scan_summary.csv`, `normalization_summary.csv`, `rgi_version_curve.csv`, `rgi_primary_band_curve.csv`, and `scan_verdict.json`.
