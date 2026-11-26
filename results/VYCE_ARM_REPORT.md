# Vyce hosted scale/diversity arm report

**Arm status: EXPLORATORY scale/diversity extension. This is not confirmatory evidence and does not replace the local confirmatory scale-up.**

This arm ports the frozen round-2 probe items, prompt wrappers, few-shot copy protocol, and byte-exact scoring to the OpenAI-compatible vyce relay. It uses a stratified 104-probe subsample per model: all 24 ASCII gate controls, 40 additional copy probes, and 40 structural probes (24 count + 16 decomposition). The count high/low labels are frozen round-2 strata measured under `qwen2.5-0.5b-instruct`; no hosted-endpoint tokenizer relabeling is assumed. Scoring remains byte-exact and deterministic. No normalization, stripping, repair, fuzzy matching, or model judging is used.

## Availability and relay access

Working target models at analysis time: none. Nonworking or unverified target models: ['gpt-oss-120b', 'deepseek-v4-flash', 'glm-5.3-flash', 'deepseek-v4-flash-lr'].

| model | HTTP | responds | error | artifacts |
|---|---|---|---|---|
| gpt-oss-120b | 503 | False | This model is currently unavailable. Please try again later. | api_error_object, content_missing, http_error, no_choices, usage_missing
| deepseek-v4-flash | 503 | False | This model is temporarily disabled | api_error_object, content_missing, http_error, no_choices, usage_missing
| glm-5.3-flash | 503 | False | This model is currently unavailable. Please try again later. | api_error_object, content_missing, http_error, no_choices, usage_missing
| deepseek-v4-flash-lr | 403 | False | You do not have access to this resource. | api_error_object, content_missing, http_error, no_choices, usage_missing

The relay's sanitized `/v1/me` metadata reports tier=`free`, verified=`False`, enabled=`True`, rateLimit=`60`. No account verification or tier change was attempted. The measured target-model usage delta after the first metadata snapshot was `{"deepseek-v4-flash": {"cost": 0.0, "inputTokens": 0, "outputTokens": 0}, "deepseek-v4-flash-lr": {"cost": 0, "inputTokens": 0, "outputTokens": 0}, "glm-5.3-flash": {"cost": 0, "inputTokens": 0, "outputTokens": 0}, "gpt-oss-120b": {"cost": 0, "inputTokens": 0, "outputTokens": 0}}`.

## Harness-validity gates

| model | ASCII exact | ASCII gate | copy rows | wrapper failures | wrapper gate | exploratory valid |
|---|---|---|---|---|---|---|

## Copy arm: newest versus legacy controls

| model | newest exact | legacy-control exact | failure gap |
|---|---|---|---|

## Structural arm: decomposition

| model | newest decomposition exact | legacy decomposition exact |
|---|---|---|

## Structural arm: count fragmentation

| model | low-frag failure | high-frag failure | high-minus-low gap |
|---|---|---|---|

## Relay artifacts and untrusted observability

The vyce relay is untrusted observability: hidden preprocessing, injected instructions, safety layers, upstream substitution, detokenization, caching, and endpoint drift cannot be excluded. Raw response bodies are preserved verbatim as base64 plus SHA-256 hashes before scoring. Format failures are therefore reported as measurement outcomes, not silently repaired. Usage tokens are relay-reported when present and conservatively upper-bounded when absent.

| model | raw rows | ok | error | finish=length | usage missing | model mismatch | reasoning field | artifact counts |
|---|---|---|---|---|---|---|---|---|

Diagnostic relay probes, including non-target infrastructure controls and alias checks, are first-class relay artifacts rather than scientific battery rows:

| purpose | model | HTTP | content | note |
|---|---|---|---|---|
| www_host_diagnostic | gpt-oss-120b | 404 | None | body reconstructed and hash-verified against the cost-ledger row because this initial infrastructure diagnostic was logged before the diagnostic jsonl existed
| non_target_infrastructure_control | wave | 200 | 'OK' | non-target control only: establishes whether the key/relay can complete any chat request; not part of the scientific battery
| alias_model_diagnostic | openai/gpt-oss-120b | 503 | None | alias diagnostic only: checks whether an upstream-style model identifier is routed; not part of the scientific battery unless separately justified
| alias_model_diagnostic | deepseek-ai/DeepSeek-V4-Flash | 503 | None | alias diagnostic only: checks whether an upstream-style model identifier is routed; not part of the scientific battery unless separately justified
| alias_model_diagnostic | deepseek-ai/deepseek-v4-flash | 503 | None | alias diagnostic only: checks whether an upstream-style model identifier is routed; not part of the scientific battery unless separately justified
| alias_model_diagnostic | zai-org/GLM-5.3-Flash | 503 | None | alias diagnostic only: checks whether an upstream-style model identifier is routed; not part of the scientific battery unless separately justified
| alias_model_diagnostic | zai-org/glm-5.3-flash | 503 | None | alias diagnostic only: checks whether an upstream-style model identifier is routed; not part of the scientific battery unless separately justified

## Cost accounting

| model | calls | input tokens | output tokens | proxy-estimated cost | conservative cap cost |
|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V4-Flash | 1 | 102 | 4 | $0.000114 | $0.000114
| deepseek-ai/deepseek-v4-flash | 1 | 102 | 4 | $0.000114 | $0.000114
| deepseek-v4-flash | 4 | 408 | 40 | $0.000119 | $0.000528
| deepseek-v4-flash-lr | 4 | 408 | 40 | $0.000528 | $0.000528
| glm-5.3-flash | 4 | 408 | 40 | $0.000091 | $0.000528
| gpt-oss-120b | 5 | 510 | 44 | $0.000089 | $0.000642
| non_completion_endpoint | 7 | 0 | 0 | $0.000000 | $0.000000
| openai/gpt-oss-120b | 1 | 102 | 4 | $0.000114 | $0.000114
| wave | 1 | 15 | 15 | $0.000060 | $0.000060
| zai-org/GLM-5.3-Flash | 1 | 102 | 4 | $0.000114 | $0.000114
| zai-org/glm-5.3-flash | 1 | 102 | 4 | $0.000114 | $0.000114

Total proxy-estimated cost: **$0.001458**. Total conservative cap-accounted cost: **$0.002856** against the hard cap of **$8.000000**; remaining conservative headroom: **$7.997144**. These are conservative accounting bounds, not observed debit. The relay's `/v1/me` usage meter showed zero target-model token/cost delta after the first metadata snapshot; because all target calls returned HTTP errors before generation and the non-target wave control showed no metered token/cost delta, observed incremental spend is consistent with **$0**. Exact debit for the earliest diagnostics cannot be isolated from the key's large pre-existing lifetime usage, so the conservative cap ledger remains the binding audit trail.

## Harness validation evidence

| check | pass | detail |
|---|---|---|
| tiny-cap preflight abort | True | {"budget_abort": "projected conservative cost for logical batch would exceed hard cap", "cap_usd": 1e-09, "cumulative_cap_cost_usd": 0.0, "projection": {"cost_bound_method": "sum of per-call UTF-8 byte/overhead input upper bounds plus max_tokens output upper bounds", "input_tokens_upper_bound": 1192, "output_tokens_upper_bound": 128, "projected_cap_cost_usd": 0.001576}}
| cost-ledger consistency | True | rows=30, problems=0
| synthetic scorer mutations | True | 
| corrupted-response negative control | True | 
| live scored-response mutation | False | no scored byte-exact copy response yet
| overall harness validation | False | overall is false until a live scored byte-exact copy response exists

## Tokenizer scan extension

Open tokenizer artifacts were pinned by exact Hugging Face revision and SHA-256, then scanned locally with the existing scanner. This scan is free of relay API cost.

| tokenizer | family | vocab | newest exact/pipeline single | E15.1 ZWJ TPC | E17 ZWJ TPC |
|---|---|---|---|---|---|
| deepseek-v4-flash | DeepSeek | 128000 | 0/0 of 62 | 8.179 | 9.000
| deepseek-v4.1-flash | DeepSeek | 128000 | 0/0 of 62 | 8.179 | 9.000
| glm-5.3-flash | GLM | 154820 | 0/0 of 62 | 8.321 | 7.909
| openai-gpt-oss-120b | OpenAI-gpt-oss | 199998 | 0/0 of 62 | 8.107 | 8.909

DeepSeek V4 Flash and V4.1 Flash tokenizer continuity: `[{"left": "deepseek-v4-flash", "left_merges": 127741, "left_vocab": 127997, "merges_equal": true, "regular_vocab_equal": true, "right": "deepseek-v4.1-flash", "right_merges": 127741, "right_vocab": 127997}]`.

Frozen count-stratum covariates under the extension tokenizers (TPC = tokens per grapheme cluster; split CP = split codepoints):

| tokenizer | frozen stratum | n | mean TPC | TPC range | mean split CP |
|---|---|---|---|---|---|
| deepseek-v4-flash | low | 12 | 2.896 | 2.000-5.000 | 3.250
| deepseek-v4-flash | high | 12 | 6.207 | 4.333-7.600 | 6.167
| deepseek-v4.1-flash | low | 12 | 2.896 | 2.000-5.000 | 3.250
| deepseek-v4.1-flash | high | 12 | 6.207 | 4.333-7.600 | 6.167
| glm-5.3-flash | low | 12 | 2.618 | 1.000-7.000 | 1.750
| glm-5.3-flash | high | 12 | 6.667 | 4.000-9.200 | 4.750
| openai-gpt-oss-120b | low | 12 | 2.907 | 1.800-5.000 | 3.167
| openai-gpt-oss-120b | high | 12 | 6.183 | 4.333-8.400 | 6.000

## Interpretation

No hosted battery rows were run because no target model passed the live availability probe. The scale hypothesis is therefore untested in this arm, not supported and not refuted.

## Protocol and verification notes

- Copy prompts use the frozen `fewshot_copy_v5_multiturn_cueless` round-2 protocol, including the same four held-out ASCII exemplars as alternating user/assistant chat pairs. Structural probes use the same single-user prompt text. No system prompt is added.
- Temperature is 0.0 and maximum output is 64 tokens where the relay accepts them.
- The adaptation is frozen on ASCII controls first: a model requires at least 23/24 ASCII byte-exact copies before Unicode/structural spend. The wrapper gate uses the same <=5% failure-rate logic as round 2, applied to the 64-copy exploratory subsample rather than the full 182-copy battery.
- Raw rows are append-only JSONL. Each row stores the response body verbatim as base64, a SHA-256 body hash, usage, finish reason, latency, relay artifacts, cost ledger linkage fields, and the exact chat-message hash.
- Mutation controls verify that exact copies are accepted, byte/codepoint corruption is rejected, missing ZWJ/VS16 are classified, NFC/NFD alterations are rejected, newline padding is rejected, echoed prompts and extra text are rejected, and truncations are rejected.
- This subsample cannot reproduce the round-2 hard-kill cell sizes (94 newest copy items, 24 newest decomposition probes, 18 newest count probes). It is a scale/diversity screen only; inferential use is descriptive and exploratory.

## Honest limitations

- Hosted endpoints are black-box systems. Tokenizer scans of open artifacts do not prove that the relay serves those exact weights, tokenizers, preprocessing, or detokenization paths.
- `deepseek-v4-flash-lr` has no confidently matching public tokenizer artifact; DeepSeek V4/V4.1 scans are tokenizer-family evidence only.
- Relay-reported usage may include hidden prompt or reasoning tokens and cannot be independently audited.
- Small stratified cells are fragile: 28 newest copy items, 12 legacy copy controls, 12 low/high count rows pooled over eras, and 8 decomposition rows per era per model.
- Absence of a working endpoint is a relay availability result, not a model capability result.

Machine-readable outputs: `results/vyce_hosted/vyce_probe_plan.jsonl`, `vyce_selection_manifest.json`, `availability_probe.jsonl`, `cost_ledger.jsonl`, `vyce_raw.jsonl`, `vyce_scored.jsonl`, `vyce_analysis.json`, `vyce_cost_summary.json`, `vyce_scoring_validation.json`, `tokenizer_scan_extension.jsonl`, `tokenizer_inventory_extension.json`, and `vyce_artifact_hashes.json`.
