# Unicode release debt: behavioral round 2 report

Scope: deterministic local behavioral round 2 after the round-1 validity-gate failure. Probe item identities are byte-for-byte unchanged from round 1; only the copy prompt/extraction instrument and model slate changed. Inference used MLX `batch_generate`, greedy/argmax decoding, max 64 output tokens, no repetition penalty, no vLLM, and one model process at a time. Raw responses are append-only in `results/behavioral_round2/round2_raw.jsonl`.

## Instrument fixes and freeze order

- Quantization sensitivity was run first, before calibration or cohort inference: Qwen2.5-1.5B ASCII exact was 23/24 bf16 versus 23/24 MLX 8-bit; drop=0/24. Verdict: **q8 within the prespecified +/-2-of-24 ASCII noise envelope; acceptable for 7B-class models**.
- Copy protocol frozen before any emoji-cohort or structural generation in `protocol_freeze.json`: `fewshot_copy_v5_multiturn_cueless`, template SHA-256 `1687c1a717cc04c3a77aba9075c8463cf95e9ae084e53435e549bca22ec0d951`, full-format SHA-256 `e84859819d4873b1b739683534e720c8f17baaf7c6542be35c94eec106692f6d`. The four exemplars are held-out ASCII filler strings, with zero exact overlap against copy probes.
- Protocol selection: eight candidates were calibrated on 48 ASCII control items across two harness-test models before any emoji-cohort inference. Single-turn few-shot and instruction variants failed the ASCII gate badly (9-18/48 exact). `fewshot_copy_v4_multiturn` and `fewshot_copy_v5_multiturn_cueless` tied numerically (46/48 exact, 0 format failures, ASCII gate pass on both test models). v5 was selected over v4 because its probe user turn carries no output cue: the `<OUT>` answer format must transfer from the exemplar pairs alone, so format compliance measures exemplar-driven task inference rather than cue echoing, the more conservative instrument. v6 (minimal, no prose) regressed to 43/48 and was rejected. The v1 freeze attempt was superseded before any emoji-cohort inference; supersession is recorded in `protocol_freeze_attempts.jsonl`.
- Strict extraction: exactly one `<OUT>`/`</OUT>` pair, no non-whitespace outside the wrapper, and no stripping/normalization/repair of extracted bytes. Newline padding inside the wrapper is a byte-copy failure, not a valid copy.
- Runtime estimate before cohort inference: 43.58227973023814 minutes projected for 5 models; below the 6-hour trimming threshold.

## Models, provenance, and tokenizer linkage

| model | inference artifact | upstream artifact | quant | strict tokenizer files identical | scan linkage acceptable | rows | generation time | peak MLX memory |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | Qwen/Qwen2.5-3B-Instruct@aa8e72537993 | Qwen/Qwen2.5-3B-Instruct@aa8e72537993 | bf16 | True | True | 290 | 3.9 min | 6.9 GB
| qwen3-4b-instruct-2507-mlx8 | mlx-community/Qwen3-4B-Instruct-2507-8bit@0e42af584497 | Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c0 | mlx8 | False | True | 290 | 6.5 min | 6.6 GB
| qwen2.5-7b-instruct-mlx8 | mlx-community/Qwen2.5-7B-Instruct-8bit@df11cb390bd6 | Qwen/Qwen2.5-7B-Instruct@a09a35458c70 | mlx8 | True | True | 290 | 9.9 min | 9.0 GB
| gemma3-4b-it-mlx8-mirror | mlx-community/gemma-3-4b-it-8bit@61cfff4de4bc | google/gemma-3-4b-it@093f9f388b31 | mlx8 | False | True | 290 | 4.7 min | 6.0 GB
| tinyllama-1.1b-chat-v1.0-bf16 | TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffed | TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffed | bf16 | True | True | 290 | 1.9 min | 2.9 GB

- Canonical `meta-llama/Llama-2-7b-chat-hf` was attempted first and returned HTTP 403 gated-repository access. The accessible NousResearch mirror has tokenizer files matching the scanned llama2 mirror but no weight-hash-equivalence documentation; per the round-2 rule it was not used. TinyLlama is therefore the vintage Llama-2-tokenizer carrier.
- Strict per-file tokenizer byte identity is reported separately from scan-covariate usability. MLX-community Qwen2.5 8-bit repositories were found to package a non-byte-identical Qwen3-like tokenizer, so canonical Qwen2.5 tokenizer files were overlaid before accepted q8 inference; wrong-tokenizer quant-control rows are preserved only as superseded artifacts. Qwen3-4B-Instruct-2507 and Gemma-3-4B differ in auxiliary tokenizer/config files but share core `tokenizer.json`; probe-text token-id equality is explicit in `tokenizer_linkage_round2.json`.
- Chat-template verification (`qwen3_template_rendering_check.json`, `gemma_template_rendering_check.json`): the Qwen3 q8 artifact's `chat_template.jinja` file differs textually from the canonical embedded template, but both render the round-2 multi-turn message structure byte-identically, matching the actual inference prompt hash. Gemma's artifact template is byte-identical to the canonical embedded template. Rendered prompts, not just token IDs, are therefore verified.
- Gemma-3-4B inference used the mlx-community multimodal artifact because no clean official text-only Gemma-3-4B MLX 8-bit artifact exists; only the text backbone was exercised (no vision inputs). Its card states conversion from `google/gemma-3-4b-it` with mlx-vlm 0.1.18; canonical upstream is license-gated, so the unsloth mirror (bf46152c) served as the accessible canonical reference for tokenizer/template comparison. Weight-hash equivalence to canonical cannot be independently verified for any MLX conversion; pinned revisions are the artifact identity.

## Harness-validity gates

| model | ASCII exact | ASCII rate | ASCII gate | wrapper failures | wrapper rate | wrapper gate |
|---|---|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | 23/24 | 95.8% | True | 0/182 | 0.0% | True
| qwen3-4b-instruct-2507-mlx8 | 23/24 | 95.8% | True | 1/182 | 0.5% | True
| qwen2.5-7b-instruct-mlx8 | 24/24 | 100.0% | True | 28/182 | 15.4% | False
| gemma3-4b-it-mlx8-mirror | 24/24 | 100.0% | True | 0/182 | 0.0% | True
| tinyllama-1.1b-chat-v1.0-bf16 | 23/24 | 95.8% | True | 0/182 | 0.0% | True

Mutation controls: ASCII-gate mutation detected=True; wrapper-gate mutation detected=True; extractor mutation battery passed=True. Completeness: qwen2.5-3b-instruct-bf16 290/290; qwen3-4b-instruct-2507-mlx8 290/290; qwen2.5-7b-instruct-mlx8 290/290; gemma3-4b-it-mlx8-mirror 290/290; tinyllama-1.1b-chat-v1.0-bf16 290/290.

## Copy exact-match rates by cohort

| model | ASCII | legacy emoji | legacy non-emoji | E15.1 emoji | E16.0 emoji | E17.0 emoji | E18.0 emoji | new non-emoji |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | 23/24 (95.8%) | 32/32 (100.0%) | 32/32 (100.0%) | 28/28 (100.0%) | 7/7 (100.0%) | 18/18 (100.0%) | 9/9 (100.0%) | 32/32 (100.0%)
| qwen3-4b-instruct-2507-mlx8 | 23/24 (95.8%) | 32/32 (100.0%) | 32/32 (100.0%) | 28/28 (100.0%) | 7/7 (100.0%) | 18/18 (100.0%) | 9/9 (100.0%) | 32/32 (100.0%)
| qwen2.5-7b-instruct-mlx8 | 24/24 (100.0%) | 28/32 (87.5%) | 32/32 (100.0%) | 21/28 (75.0%) | 4/7 (57.1%) | 9/18 (50.0%) | 3/9 (33.3%) | 32/32 (100.0%)
| gemma3-4b-it-mlx8-mirror | 24/24 (100.0%) | 32/32 (100.0%) | 32/32 (100.0%) | 28/28 (100.0%) | 7/7 (100.0%) | 18/18 (100.0%) | 9/9 (100.0%) | 32/32 (100.0%)
| tinyllama-1.1b-chat-v1.0-bf16 | 23/24 (95.8%) | 32/32 (100.0%) | 32/32 (100.0%) | 28/28 (100.0%) | 7/7 (100.0%) | 18/18 (100.0%) | 9/9 (100.0%) | 32/32 (100.0%)

## Newest-bin versus old-copy controls

| model | newest exact | newest rate | old exact | old rate | failure gap | meets operational hard-kill conditions | rule-1 signal (>=5 pp) |
|---|---|---|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | 94/94 | 100.0% | 64/64 | 100.0% | 0.0 pp | True | False
| qwen3-4b-instruct-2507-mlx8 | 94/94 | 100.0% | 64/64 | 100.0% | 0.0 pp | True | False
| qwen2.5-7b-instruct-mlx8 | 69/94 | 73.4% | 60/64 | 93.8% | 20.3 pp | False | True
| gemma3-4b-it-mlx8-mirror | 94/94 | 100.0% | 64/64 | 100.0% | 0.0 pp | True | False
| tinyllama-1.1b-chat-v1.0-bf16 | 94/94 | 100.0% | 64/64 | 100.0% | 0.0 pp | True | False

## Cohort-trend check

| model | E15.1/E16/E17/E18 exact rates | new-only monotone non-increasing | legacy+new monotone | failure slope pp/step | ASCII exact |
|---|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | 100.0%, 100.0%, 100.0%, 100.0% | True | True | 0.0 | 95.8%
| qwen3-4b-instruct-2507-mlx8 | 100.0%, 100.0%, 100.0%, 100.0% | True | True | 0.0 | 95.8%
| qwen2.5-7b-instruct-mlx8 | 75.0%, 57.1%, 50.0%, 33.3% | True | True | 13.2 | 100.0%
| gemma3-4b-it-mlx8-mirror | 100.0%, 100.0%, 100.0%, 100.0% | True | True | 0.0 | 100.0%
| tinyllama-1.1b-chat-v1.0-bf16 | 100.0%, 100.0%, 100.0%, 100.0% | True | True | 0.0 | 95.8%

Interpretation: positive slope means failure rises from E15.1 to E18. These raw cohort contrasts are descriptive and item-type-confounded because E15.1/E17 are mostly ZWJ families while E16/E18 are atomic; they are not the adjusted confirmatory regression.

## Cross-model vintage contrast

| model | vintage group | gate-valid | newest exact | old exact | newest-old failure gap |
|---|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | newer_tokenizer | True | 100.0% | 100.0% | 0.0 pp
| qwen3-4b-instruct-2507-mlx8 | newer_tokenizer | True | 100.0% | 100.0% | 0.0 pp
| qwen2.5-7b-instruct-mlx8 | newer_tokenizer | False | 73.4% | 93.8% | 20.3 pp
| gemma3-4b-it-mlx8-mirror | newer_tokenizer | True | 100.0% | 100.0% | 0.0 pp
| tinyllama-1.1b-chat-v1.0-bf16 | vintage_llama2_tokenizer | True | 100.0% | 100.0% | 0.0 pp

Group summary: vintage valid mean gap=0.0 pp; newer valid mean gap=0.0 pp; newer-minus-vintage difference=0.0 pp. Interpretable=True.

## Structural probes

| model | newest decomp | decomp rate | newest count | count rate | high-low count failure gap | meets structural hard-kill conditions | rule-2 signal |
|---|---|---|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | 0/24 | 0.0% | 5/18 | 27.8% | 50.0 pp | False | True
| qwen3-4b-instruct-2507-mlx8 | 0/24 | 0.0% | 2/18 | 11.1% | 22.2 pp | False | True
| qwen2.5-7b-instruct-mlx8 | 0/24 | 0.0% | 9/18 | 50.0% | 72.2 pp | False | True
| gemma3-4b-it-mlx8-mirror | 0/24 | 0.0% | 6/18 | 33.3% | 5.6 pp | False | False
| tinyllama-1.1b-chat-v1.0-bf16 | 0/24 | 0.0% | 0/18 | 0.0% | 0.0 pp | False | False

| model | probe | era | fragmentation | exact | rate | format-failure rate |
|---|---|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | decomposition | legacy | all | 0/24 | 0.0% | 4.2%
| qwen2.5-3b-instruct-bf16 | decomposition | newest | all | 0/24 | 0.0% | 20.8%
| qwen2.5-3b-instruct-bf16 | count | legacy | high | 0/9 | 0.0% | 0.0%
| qwen2.5-3b-instruct-bf16 | count | legacy | low | 4/9 | 44.4% | 0.0%
| qwen2.5-3b-instruct-bf16 | count | newest | high | 0/9 | 0.0% | 0.0%
| qwen2.5-3b-instruct-bf16 | count | newest | low | 5/9 | 55.6% | 0.0%
| qwen2.5-3b-instruct-bf16 | comparison | legacy | all | 11/12 | 91.7% | 0.0%
| qwen2.5-3b-instruct-bf16 | comparison | newest | all | 10/12 | 83.3% | 0.0%
| qwen3-4b-instruct-2507-mlx8 | decomposition | legacy | all | 0/24 | 0.0% | 29.2%
| qwen3-4b-instruct-2507-mlx8 | decomposition | newest | all | 0/24 | 0.0% | 83.3%
| qwen3-4b-instruct-2507-mlx8 | count | legacy | high | 1/9 | 11.1% | 0.0%
| qwen3-4b-instruct-2507-mlx8 | count | legacy | low | 5/9 | 55.6% | 0.0%
| qwen3-4b-instruct-2507-mlx8 | count | newest | high | 1/9 | 11.1% | 0.0%
| qwen3-4b-instruct-2507-mlx8 | count | newest | low | 1/9 | 11.1% | 0.0%
| qwen3-4b-instruct-2507-mlx8 | comparison | legacy | all | 12/12 | 100.0% | 0.0%
| qwen3-4b-instruct-2507-mlx8 | comparison | newest | all | 12/12 | 100.0% | 0.0%
| qwen2.5-7b-instruct-mlx8 | decomposition | legacy | all | 0/24 | 0.0% | 0.0%
| qwen2.5-7b-instruct-mlx8 | decomposition | newest | all | 0/24 | 0.0% | 4.2%
| qwen2.5-7b-instruct-mlx8 | count | legacy | high | 0/9 | 0.0% | 0.0%
| qwen2.5-7b-instruct-mlx8 | count | legacy | low | 8/9 | 88.9% | 0.0%
| qwen2.5-7b-instruct-mlx8 | count | newest | high | 2/9 | 22.2% | 0.0%
| qwen2.5-7b-instruct-mlx8 | count | newest | low | 7/9 | 77.8% | 0.0%
| qwen2.5-7b-instruct-mlx8 | comparison | legacy | all | 12/12 | 100.0% | 0.0%
| qwen2.5-7b-instruct-mlx8 | comparison | newest | all | 12/12 | 100.0% | 0.0%
| gemma3-4b-it-mlx8-mirror | decomposition | legacy | all | 0/24 | 0.0% | 0.0%
| gemma3-4b-it-mlx8-mirror | decomposition | newest | all | 0/24 | 0.0% | 0.0%
| gemma3-4b-it-mlx8-mirror | count | legacy | high | 2/9 | 22.2% | 0.0%
| gemma3-4b-it-mlx8-mirror | count | legacy | low | 5/9 | 55.6% | 0.0%
| gemma3-4b-it-mlx8-mirror | count | newest | high | 4/9 | 44.4% | 0.0%
| gemma3-4b-it-mlx8-mirror | count | newest | low | 2/9 | 22.2% | 0.0%
| gemma3-4b-it-mlx8-mirror | comparison | legacy | all | 8/12 | 66.7% | 0.0%
| gemma3-4b-it-mlx8-mirror | comparison | newest | all | 11/12 | 91.7% | 0.0%
| tinyllama-1.1b-chat-v1.0-bf16 | decomposition | legacy | all | 0/24 | 0.0% | 100.0%
| tinyllama-1.1b-chat-v1.0-bf16 | decomposition | newest | all | 0/24 | 0.0% | 100.0%
| tinyllama-1.1b-chat-v1.0-bf16 | count | legacy | high | 0/9 | 0.0% | 100.0%
| tinyllama-1.1b-chat-v1.0-bf16 | count | legacy | low | 0/9 | 0.0% | 100.0%
| tinyllama-1.1b-chat-v1.0-bf16 | count | newest | high | 0/9 | 0.0% | 100.0%
| tinyllama-1.1b-chat-v1.0-bf16 | count | newest | low | 0/9 | 0.0% | 100.0%
| tinyllama-1.1b-chat-v1.0-bf16 | comparison | legacy | all | 0/12 | 0.0% | 100.0%
| tinyllama-1.1b-chat-v1.0-bf16 | comparison | newest | all | 0/12 | 0.0% | 100.0%

## Copy byte-error taxonomy

| model | cohort | failures | ZWJ dropped | VS16 dropped | NFC/NFD variant | truncation | newline padding | other |
|---|---|---|---|---|---|---|---|---|
| gemma3-4b-it-mlx8-mirror | E15.1 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| gemma3-4b-it-mlx8-mirror | E16.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| gemma3-4b-it-mlx8-mirror | E17.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| gemma3-4b-it-mlx8-mirror | E18.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| gemma3-4b-it-mlx8-mirror | new non-emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-3b-instruct-bf16 | E15.1 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-3b-instruct-bf16 | E16.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-3b-instruct-bf16 | E17.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-3b-instruct-bf16 | E18.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-3b-instruct-bf16 | new non-emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-7b-instruct-mlx8 | E15.1 emoji | 7 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-7b-instruct-mlx8 | E16.0 emoji | 3 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-7b-instruct-mlx8 | E17.0 emoji | 9 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-7b-instruct-mlx8 | E18.0 emoji | 6 | 0 | 0 | 0 | 0 | 0 | 0
| qwen2.5-7b-instruct-mlx8 | new non-emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen3-4b-instruct-2507-mlx8 | E15.1 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen3-4b-instruct-2507-mlx8 | E16.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen3-4b-instruct-2507-mlx8 | E17.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen3-4b-instruct-2507-mlx8 | E18.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| qwen3-4b-instruct-2507-mlx8 | new non-emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| tinyllama-1.1b-chat-v1.0-bf16 | E15.1 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| tinyllama-1.1b-chat-v1.0-bf16 | E16.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| tinyllama-1.1b-chat-v1.0-bf16 | E17.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| tinyllama-1.1b-chat-v1.0-bf16 | E18.0 emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0
| tinyllama-1.1b-chat-v1.0-bf16 | new non-emoji | 0 | 0 | 0 | 0 | 0 | 0 | 0

## Systematic copy-error classes in newest items

| model | missing_zwj | missing_vs16 | normalization_only_mismatch | replacement_character_emitted | decomposed_rgi_sequence | rule-3 signal |
|---|---|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | 0 | 0 | 0 | 0 | 0 | False
| qwen3-4b-instruct-2507-mlx8 | 0 | 0 | 0 | 0 | 0 | False
| qwen2.5-7b-instruct-mlx8 | 0 | 0 | 0 | 0 | 0 | False
| gemma3-4b-it-mlx8-mirror | 0 | 0 | 0 | 0 | 0 | False
| tinyllama-1.1b-chat-v1.0-bf16 | 0 | 0 | 0 | 0 | 0 | False

## Manual failure inspection

- qwen2.5-3b-instruct-bf16 ASCII near-miss: target=")(]N*]9NIbUYN?!{"; extracted=")][N*]9NIbUYN?!{"; newline_padding=False; first_byte_mismatch=1.
- qwen3-4b-instruct-2507-mlx8 ASCII near-miss: target="oM07U$ylJ-R&U3t["; extracted=null; newline_padding=None; first_byte_mismatch=None.
- qwen2.5-7b-instruct-mlx8 newest copy failure example: item=copy-002; target="🍄‍🟫"; extracted=null; classes=format_failure; raw_prefix="<IN>🍄‍🟫</IN>".
- tinyllama-1.1b-chat-v1.0-bf16 ASCII near-miss: target="8VR[h/;lU;R}[yH#"; extracted="8VR[h/;lU;R][yH#"; newline_padding=False; first_byte_mismatch=11.

## Wrapper-direction mimicry analysis (post-hoc)

| model | IN-tag mimicry failures | inner byte-exact | inner byte-diff | by cohort |
|---|---|---|---|---|
| qwen2.5-3b-instruct-bf16 | 0 | 0 | 0 | {}
| qwen3-4b-instruct-2507-mlx8 | 0 | 0 | 0 | {}
| qwen2.5-7b-instruct-mlx8 | 28 | 28 | 0 | {"E15.1 emoji": 7, "E16.0 emoji": 3, "E17.0 emoji": 9, "E18.0 emoji": 6, "legacy emoji": 3}
| gemma3-4b-it-mlx8-mirror | 0 | 0 | 0 | {}
| tinyllama-1.1b-chat-v1.0-bf16 | 0 | 0 | 0 | {}

Strict-protocol extraction treats `<IN>...</IN>` outputs as format failures, and the validity gates account them as such; that accounting is unchanged. This post-hoc check asks a separate scientific question: is the byte-copy operation itself intact when the wrapper direction is wrong? For qwen2.5-7b-instruct-mlx8, **all 28 mimicry failures contain byte-perfect copies of the target inside the wrong-direction wrapper**. The model mirrors the user-turn `<IN>` tag instead of the assistant exemplar `<OUT>` tag, concentrated in emoji cohorts (E15.1=7, E16=3, E17=9, E18=6, legacy emoji=3; zero in ASCII, legacy non-emoji, and new non-emoji). No other model produced any mimicry output.

Interpretation consequence: the qwen2.5-7b newest-minus-old copy gap of 20.3 pp visible in the tables above is **not evidence of byte-copy release debt**; the underlying copied bytes are exact in every mimicry case. The model is gate-invalid (15.4% wrapper failure) and therefore excluded from every proceed/hard-kill rule, and the gap is reported descriptively only. The single non-mimicry 7B copy failure (`copy-072`) emitted a replacement character plus non-target text (`<OUT>\ufffdfindViewById</OUT>`), the only true byte-level copy corruption observed in any model outside single-character ASCII substitutions.

## Proceed/kill verdict under spec 8.3

- Behavioral round 2 conclusive: **True** (4 gate-valid models; at least two required). Final behavioral interpretation: **proceed**.
- Operational hard kill among gate-valid models: **True**. Structural hard kill among gate-valid models: **False**. Whole-project hard kill: **False**.
- Behavioral proceed rule 1 (newest-minus-old copy failure >=5 pp in >=2 gate-valid models): **False**; models: none.
- Behavioral proceed rule 2 (high-minus-low fragmentation structural failure >=15 pp in >=2 models): **True**; models: qwen2.5-3b-instruct-bf16, qwen3-4b-instruct-2507-mlx8.
- Behavioral proceed rule 3 (systematic edit class in >=5 distinct newest items in any model): **False**; models: none.
- Tokenizer-scan proceed rule 4: **True** (already established). Overall proceed-rule status: **True**.
- Regression-based stop rule: not needed for round-2 decision because tokenizer-scan rule 4 already fires.

## Synthesis

The two arms of the study separate cleanly in round 2. The **operational copy arm is hard-killed**: every gate-valid model, including the vintage Llama-2-tokenizer TinyLlama-1.1B, copies every newest-cohort item byte-exactly (94/94 newest vs 64/64 legacy controls in each valid model, zero missing-ZWJ/VS16/normalization errors anywhere). Unicode release recency has no measurable effect on byte-copy fidelity, and the only copy failures observed anywhere are single-character substitutions inside adversarial ASCII control strings plus the qwen2.5-7b wrapper-direction mimicry artifact, whose inner bytes are nevertheless exact. Byte-level copying of unfamiliar Unicode is attention-mediated and tokenizer-release-independent at these model scales.

The **structural arm survives and carries the project forward**: decomposition is an era-independent floor failure (0/48 everywhere, legacy included), while the count probe shows the prespecified fragmentation contrast, high-fragmentation failure exceeding low-fragmentation failure by 50.0 pp (qwen2.5-3b) and 22.2 pp (qwen3-4b) pooled over eras, firing behavioral proceed rule 2 in two gate-valid models. Count errors overcount ZWJ sequences and combining-mark clusters as multiple characters, i.e. models count code points rather than grapheme clusters in both eras. Because the same fragmentation pattern appears in legacy items, round-2 behavioral evidence supports a fragmentation-sensitivity mechanism rather than a release-recency mechanism; the release-cohort contrast survives only in the tokenizer-scan covariates (rule 4) and must be tested in the regression-adjusted full-study analysis, not in copy behavior.

## Interpretation caveats

- The round-1 analysis implementation inverted the spec's high-success hard-kill logic and used 20 pp instead of the spec's 5 pp for proceed rule 1. Round-2 machine verdicts use the spec thresholds directly; this correction is analysis-only and does not change probe items or raw responses.
- Validity gates remain the primary guard against confusing instruction-following failure with Unicode release debt. Invalid models are tabulated descriptively but excluded from proceed/hard-kill rule evaluation.
- Copy attention can succeed even when structural operations fail; the copy and structural arms are therefore reported separately.
- MLX 8-bit inference is accepted only because the prespecified ASCII sensitivity check was within noise; this does not prove quantization neutrality for every Unicode item. In particular, qwen2.5-7b ran only in MLX 8-bit (bf16 7B exceeds the 16 GB host budget), so its wrapper-direction mimicry cannot be isolated from a possible scale-by-quantization interaction; the q8-acceptance control was measured on Qwen2.5-1.5B, not at 7B scale.
- Decomposition exact accuracy is 0/48 in every model for both eras, including legacy emoji sequences that predate every tokenizer here. The decomposition arm therefore cannot discriminate release cohorts at all: it is an era-independent floor failure (models hallucinate plausible-looking but wrong codepoint sequences even for legacy emoji). Fragmentation sensitivity is instead visible in the count arm, where errors systematically count code points rather than grapheme clusters (ZWJ families and VS16/combining-mark sequences overcounted), in both eras.
- TinyLlama structural results are uninterpretable beyond format compliance: 100% format failure on decomposition, count, and comparison (the model does not emit the required answer shapes at all), while its copy arm passes both gates with 181/182 byte-exact copies. The vintage contrast is therefore a copy-arm contrast only.
- Probe-text token-id equality between scan and behavior tokenizers held on every checked probe text for every model (243 unique texts for the Qwen/Gemma scan tokenizers, 242 for TinyLlama; the count difference is a dedup artifact of per-scan-tokenizer probe construction, not an encoding failure).
- The regression-adjusted full-study analysis is not replaced by these round-2 tables.

Machine-readable outputs: `round2_manifest.json`, `probe_plan_round2.jsonl`, `round2_raw.jsonl`, `round2_scored.jsonl`, `round2_verdict.json`, `quantization_sensitivity.json`, `protocol_freeze.json`, `tokenizer_linkage_round2.json`, and round2 CSV summaries.
