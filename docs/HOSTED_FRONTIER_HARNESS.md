# Hosted-frontier execution harness

Status: **prepared, not yet live-executed**. No first-party provider calls have been made from this branch. All end-to-end evidence currently comes from deterministic fake transports that capture the exact outbound request bytes and inject provider-shaped canned responses.

## Purpose

The harness ports the frozen confirmatory ASCII/unicode structural battery to three first-party hosted endpoints:

| provider | default pinned model | API shape | decode request |
|---|---|---|---|
| OpenAI | `gpt-5.5-2026-04-23` | `POST https://api.openai.com/v1/chat/completions` | `temperature: 0.0`, `max_completion_tokens: 64`, `reasoning_effort: "none"` |
| Anthropic | `claude-opus-4-5-20251101` | `POST https://api.anthropic.com/v1/messages`, `anthropic-version: 2023-06-01` | `temperature: 0.0`, `max_tokens: 64` |
| Google Gemini | `gemini-3.1-pro-preview` | `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` | `temperature: 0.0`, `maxOutputTokens: 64`, `candidateCount: 1`, `thinkingLevel: LOW`, `includeThoughts: false` |

The OpenAI default is the newest dated GPT flagship-class snapshot documented with a frozen snapshot identifier at preparation time; the current undated GPT-6.x identifiers are not used because the harness requires a non-floating pin. Anthropic's current 5.x identifiers are provider-pinned but dateless; the dated Opus 4.5 snapshot is used because it still accepts explicit `temperature: 0.0` under the pinned Messages API version. Gemini has no dated first-party Pro string with the required API shape, so the exact public preview version string is pinned and the served `modelVersion` is archived. Any served-model mismatch aborts the run.

## Frozen probe reuse and scientific caveat

The default execution inventory is derived semantically byte-for-byte from the 500 frozen confirmatory rows for `qwen2.5-3b-instruct-bf16`. That carrier is the reference because it is a primary bf16 carrier and has the full 500-row inventory, including the conditional newest-L1 cell. The transformation changes only carrier-scoped identifiers (`carrier_id`, `probe_key`, `probe_id`). Targets, expected answers, prompt hashes, inventory covariates, match identifiers, and row metadata are verified unchanged before execution. Scoring imports and calls the same byte-exact confirmatory `score_row` implementation; no hosted-specific rescue or normalization is added.

Important interpretation constraint: count-probe fragmentation levels and tokenizer covariates remain measurements of the frozen reference-carrier tokenizer. OpenAI, Anthropic, and Gemini do not expose their inference tokenization through these endpoints. Consequently, in a hosted run those labels are fixed-stimulus covariates, not verified hosted-carrier fragmentation measurements. ASCII-gated structural outputs remain interpretable as black-box structural behavior under the frozen stimulus ensemble, but tokenizer-fragmentation causal inference requires a separately justified tokenizer-linkage decision.

`--reference-carrier` exists for explicit sensitivity work. Changing it writes a separate artifact namespace and must be interpreted as a different fixed-stimulus inventory, not as a correction to hosted tokenizer fragmentation.

## Commands

Dry-run the complete harness without keys or network. Each default dry run uses a fresh ignored directory under `out/hosted_frontier_dry_run/`. Fake dry-run artifacts are refused under `results/hosted_frontier/` so they cannot contaminate the live ledger or scientific raw files:

```sh
.venv/bin/python src/run_hosted_frontier.py --provider openai --dry-run
.venv/bin/python src/run_hosted_frontier.py --provider anthropic --dry-run
.venv/bin/python src/run_hosted_frontier.py --provider gemini --dry-run
```

Run the validation suite, including mutation tests, entirely against fake transports:

```sh
.venv/bin/python src/validate_hosted_frontier.py
```

Once keys exist, the live commands are:

```sh
.venv/bin/python src/run_hosted_frontier.py --provider openai
.venv/bin/python src/run_hosted_frontier.py --provider anthropic
.venv/bin/python src/run_hosted_frontier.py --provider gemini
```

Optional cap override example:

```sh
.venv/bin/python src/run_hosted_frontier.py --provider openai --cap-usd 2.50
```

The default hard cap is **$4 per provider**. The ledger is provider-level (`results/hosted_frontier/<provider>/cost_ledger.jsonl`), so multiple models or resumed attempts under the same provider accumulate against that provider cap. Failed transport attempts are conservatively debited because provider-side metering cannot be audited.

## Key handling

Keys are read immediately before each call and are never written to artifacts, logs, request bodies, or this repository.

| provider | environment variables | fallback files |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `~/.secrets/openai-api-key`, `~/.secrets/openai` |
| Anthropic | `ANTHROPIC_API_KEY` | `~/.secrets/anthropic-api-key`, `~/.secrets/anthropic`, `~/.secrets/claude-api-key` |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `~/.secrets/gemini-api-key`, `~/.secrets/google-api-key`, `~/.secrets/gemini` |

Gemini uses the `x-goog-api-key` header, not a query-string key. Authorization headers are never archived. If a key is reflected in a response body, the harness redacts the archived copy, refuses the response, and aborts fail-closed.

## Spend-gated execution order

For each provider/model artifact namespace:

1. Validate key presence (live mode only).
2. Build and verify the hosted reference inventory.
3. Compute a worst-case cost projection for the availability call plus all remaining rows before any transport.
4. Abort before transport if the projection plus ledgered provider spend exceeds the cap.
5. Make one minimal availability call and require HTTP 200, a parseable provider-specific response, usage metadata when supplied by the API contract, and an exact served-model match.
6. Run the 48-row ASCII gate battery first:
   - 24 ASCII copy controls;
   - 12 ASCII scalar-listing controls;
   - 12 ASCII count controls.
7. Score with the frozen byte-exact scorer. Continue only if ASCII copy is at least 23/24, ASCII scalar listing at least 11/12, and ASCII count at least 11/12.
8. Run the 100 Unicode copy probes. Continue to Unicode structural probes only if the full 124-copy instrument is complete, the ASCII copy threshold still holds, and wrapper format failures are at most 6/124.
9. Run 32 Unicode scalar-listing probes and 320 Unicode count probes.
10. Write final scored rows, exact gate outcomes, runtime events, and the run manifest. Gate failure is recorded as data and exits successfully with `status: stopped_by_gate`; malformed transport, integrity failure, API failure, or budget failure exits nonzero.

Raw rows are append-only and resumable by deterministic completion key. Re-running a completed live command performs the availability/preflight call again but skips completed probe rows.

## Cost model

The committed pricing table is `src/hosted_frontier_pricing.json`, checked against provider documentation on **2026-11-26**. It is a list-price table, not an invoice feed.

Each ledger row distinguishes:

- `priced_usage_cost_usd`: list-price cost computed from provider-reported usage when valid usage is present; otherwise `null`.
- `conservative_cap_debit_usd`: the amount debited against the hard cap. This is the conservative prompt-byte/overhead plus `max_tokens` projection, raised to provider-reported priced usage if reported usage is larger.
- `debit_basis`: which of those accounting rules was used.

The byte-count input bound is conservative for tokenization but cannot observe hidden provider instructions or middleware. That limitation is explicit in the ledger projection metadata. When providers report usage, any overage relative to the preflight bound is debited immediately and blocks further transport once the cap is reached.

Observed fake-transport validation totals for the 500-probe battery plus one availability call were:

| provider | conservative cap debit | priced-usage total in fake validation |
|---|---:|---:|
| OpenAI | $1.945875 | $0.411820 |
| Anthropic | $1.785555 | $0.420725 |
| Gemini | $0.778350 | $0.181386 |

The fake priced-usage totals are interface validations, not forecasts. The conservative cap debit is the relevant hard-cap accounting bound under the committed token-estimation rule.

## Integrity and archival guarantees

For every transport attempt, including retries and the availability call:

- exact outbound request body bytes are archived as base64, without authorization headers;
- exact response body bytes are archived as base64 unless secret redaction is required;
- request and response SHA-256 hashes are stored;
- latency, HTTP status, selected non-secret response headers, adapter identifier, purpose, attempt, and integrity status are stored;
- the cost ledger is append-only and hash-chained (`sequence`, `previous_entry_sha256`, `entry_sha256`);
- malformed UTF-8/JSON or structurally invalid provider responses are archived but never admitted as scientific rows;
- HTTP 429 and transient 5xx/network failures use deterministic exponential backoff, capped at 30 seconds, up to the retry limit;
- non-200 responses after retry policy abort the run without scoring them as model behavior;
- response-served model strings must exactly equal the requested pin.

## Validation evidence model

`src/validate_hosted_frontier.py` verifies, per provider:

- fake-transport end-to-end execution of all 500 reference probes;
- exact ASCII-first ordering and all ASCII gates at 24/24 copy, 12/12 scalar, and 12/12 count in the correct-response fake;
- byte-exact scoring of every correct fake response;
- exact outbound payload capture and byte equality with the archived request payload;
- provider-specific payload shape and decode fields;
- hash-chain validity and priced-versus-conservative ledger separation;
- absence of validation secrets from artifacts.

Mutation/negative controls verify:

- corrupting one canned copy response makes exactly that row byte-inexact and degrades ASCII copy to 23/24;
- truncating response payload bytes triggers provider adapter integrity failure before any scientific row is accepted;
- a `$1e-9` cap aborts before any transport attempt;
- mutating a ledger cost breaks the hash chain;
- mutating an archived response payload breaks its pinned hash;
- one injected HTTP 429 is retried with backoff and the battery completes;
- a reflected secret is redacted and refused;
- a served-model mismatch aborts after the availability call.

The generated machine-readable evidence file is `results/hosted_frontier/harness_validation.json` when the validation command is run with its default output path.

## Honest gaps before live execution

1. No live provider response has validated current parameter acceptance. In particular, reasoning-model sampling rules can differ by provider and model generation; the first live availability call is deliberately fail-closed.
2. Anthropic's dated Opus 4.5 snapshot was deprecated on 2026-11-23 and is documented as retiring on 2027-01-22. After retirement, choose a successor deliberately, update the pricing table, and rerun fake validation; do not silently switch aliases.
3. Gemini 3.1 Pro is a public-preview string. If Google changes availability or served `modelVersion`, the harness aborts rather than silently drifting.
4. Hosted endpoints are black-box systems. Hidden preprocessing, safety mediation, detokenization, caching, retries, or model substitution cannot be excluded. Exact archives preserve evidence but do not eliminate those artifacts.
5. Temperature 0 is a sampling control, not a proof of bitwise determinism in hosted systems. Reasoning effort/thinking settings minimize hidden reasoning but do not make black-box decoding fully auditable.
6. Fragmentation covariates are reference-tokenizer labels, as described above. Hosted-carrier tokenizer fragmentation remains an open measurement problem.
