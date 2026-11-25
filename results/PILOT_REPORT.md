# Unicode release debt: behavioral pilot report

Scope: deterministic local behavioral pilot only. Inputs are the frozen 182-item copy pilot and already-constructed structural probes. Inference used MLX `batch_generate` with greedy/argmax decoding, max 64 output tokens, no repetition penalty, and no vLLM. Raw responses are append-only in `results/behavioral/behavioral_raw.jsonl`.

## Models and protocol deviations

| model | inference artifact | upstream artifact | dtype/quantization | rows | generation time |
|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0-bf16 | TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffed | TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffed | unquantized bfloat16 safetensors (config torch_dtype=bfloat16) | 290 | 0.9 min |
| qwen2.5-1.5b-instruct-bf16 | Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4c | Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4c | unquantized bfloat16 safetensors (config torch_dtype=bfloat16) | 290 | 0.7 min |
| qwen3-1.7b-bf16 | Qwen/Qwen3-1.7B@70d244cc86cc | Qwen/Qwen3-1.7B@70d244cc86cc | unquantized bfloat16 safetensors (config torch_dtype=bfloat16) | 290 | 1.0 min |
| gemma3-1b-it-bf16-mirror | unsloth/gemma-3-1b-it@5b11413a10db | google/gemma-3-1b-it@dcc83ea841ab | unquantized bfloat16 safetensors (config torch_dtype=bfloat16) | 290 | 0.4 min |

- Llama 3.2: canonical `meta-llama/Llama-3.2-1B-Instruct` and 3B repositories are manual-license-gated; `config.json` download returned HTTP 403 for this account. Per instruction, Llama 3.2 was noted and skipped rather than substituted with a mirror. Consequently the behavioral pilot has four models and the originally expected 1,160 generations; the tokenizer scan was not extended with Llama 3.2 in this branch.
- Unquantized bfloat16 safetensors were used. Qwen2.5 uses the 1.5B rather than 3B variant and Qwen3 uses 1.7B rather than 4B under the pilot compute/memory budget; exact repository revision SHAs are pinned.
- Qwen2.5-1.5B and Qwen3-1.7B tokenizer files were verified byte-identical to the already-scanned Qwen2.5-0.5B and Qwen3-0.6B tokenizer artifacts respectively, so the existing tokenizer-specific count bundles and scan covariates apply directly.
- Qwen3 generation used `enable_thinking=False`, a switch in its own chat template. No explicit system message was supplied; each model's own template was allowed to inject its default wrapper text (notably Qwen2.5's default system text). User message content was byte-identical across models.

## Harness-validity gates

| model | ASCII exact | ASCII rate | ASCII gate | wrapper failures | wrapper rate | wrapper gate |
|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0-bf16 | 0/24 | 0.0% | False | 0/182 | 0.0% | True |
| qwen2.5-1.5b-instruct-bf16 | 5/24 | 20.8% | False | 3/182 | 1.6% | True |
| qwen3-1.7b-bf16 | 14/24 | 58.3% | False | 0/182 | 0.0% | True |
| gemma3-1b-it-bf16-mirror | 0/24 | 0.0% | False | 180/182 | 98.9% | False |

Mutation controls: ASCII-gate mutation detected=True; wrapper-gate mutation detected=True; duplicate-wrapper parser mutation detected=True. Completeness: tinyllama-1.1b-chat-v1.0-bf16 290/290; qwen2.5-1.5b-instruct-bf16 290/290; qwen3-1.7b-bf16 290/290; gemma3-1b-it-bf16-mirror 290/290.

## Copy exact-match rates by cohort

| model | ASCII | legacy emoji | legacy non-emoji | E15.1 emoji | E16.0 emoji | E17.0 emoji | E18.0 emoji | new non-emoji |
|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0-bf16 | 0/24 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 0/28 (0.0%) | 0/7 (0.0%) | 0/18 (0.0%) | 0/9 (0.0%) | 0/32 (0.0%) |
| qwen2.5-1.5b-instruct-bf16 | 5/24 (20.8%) | 23/32 (71.9%) | 24/32 (75.0%) | 15/28 (53.6%) | 4/7 (57.1%) | 8/18 (44.4%) | 8/9 (88.9%) | 32/32 (100.0%) |
| qwen3-1.7b-bf16 | 14/24 (58.3%) | 14/32 (43.8%) | 0/32 (0.0%) | 15/28 (53.6%) | 7/7 (100.0%) | 12/18 (66.7%) | 6/9 (66.7%) | 0/32 (0.0%) |
| gemma3-1b-it-bf16-mirror | 0/24 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 0/28 (0.0%) | 0/7 (0.0%) | 0/18 (0.0%) | 0/9 (0.0%) | 1/32 (3.1%) |

## Newest-bin versus old-copy controls

| model | newest exact | newest rate | old exact | old rate | failure gap | copy hard-kill prong | rule-1 signal |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0-bf16 | 0/94 | 0.0% | 0/64 | 0.0% | 0.0 pp | True | False |
| qwen2.5-1.5b-instruct-bf16 | 67/94 | 71.3% | 47/64 | 73.4% | 2.2 pp | False | False |
| qwen3-1.7b-bf16 | 40/94 | 42.6% | 14/64 | 21.9% | -20.7 pp | False | False |
| gemma3-1b-it-bf16-mirror | 1/94 | 1.1% | 0/64 | 0.0% | -1.1 pp | True | False |

## Cohort-trend check

| model | E15.1/E16/E17/E18 exact rates | new-only monotone non-increasing | legacy+new monotone | failure slope pp/step | ASCII exact |
|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0-bf16 | 0.0%, 0.0%, 0.0%, 0.0% | True | True | 0.0 | 0.0% |
| qwen2.5-1.5b-instruct-bf16 | 53.6%, 57.1%, 44.4%, 88.9% | False | False | -9.3 | 20.8% |
| qwen3-1.7b-bf16 | 53.6%, 100.0%, 66.7%, 66.7% | False | False | -0.6 | 58.3% |
| gemma3-1b-it-bf16-mirror | 0.0%, 0.0%, 0.0%, 0.0% | True | True | 0.0 | 0.0% |

Interpretation: positive slope means failure rises from E15.1 to E18. These raw cohort contrasts are descriptive and item-type-confounded because E15.1/E17 are mostly ZWJ families while E16/E18 are atomic; they are not the adjusted confirmatory regression.

## Structural probes

| model | newest decomp | decomp rate | newest count | count rate | high-low count failure gap | structural hard-kill | rule-2 signal |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0-bf16 | 0/24 | 0.0% | 0/18 | 0.0% | 0.0 pp | True | False |
| qwen2.5-1.5b-instruct-bf16 | 0/24 | 0.0% | 8/18 | 44.4% | 22.2 pp | False | True |
| qwen3-1.7b-bf16 | 0/24 | 0.0% | 2/18 | 11.1% | 0.0 pp | True | False |
| gemma3-1b-it-bf16-mirror | 0/24 | 0.0% | 4/18 | 22.2% | 0.0 pp | True | False |

| model | probe | era | fragmentation | exact | rate | format-failure rate |
|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0-bf16 | decomposition | legacy | all | 0/24 | 0.0% | 100.0% |
| tinyllama-1.1b-chat-v1.0-bf16 | decomposition | newest | all | 0/24 | 0.0% | 100.0% |
| tinyllama-1.1b-chat-v1.0-bf16 | count | legacy | high | 0/9 | 0.0% | 100.0% |
| tinyllama-1.1b-chat-v1.0-bf16 | count | legacy | low | 0/9 | 0.0% | 100.0% |
| tinyllama-1.1b-chat-v1.0-bf16 | count | newest | high | 0/9 | 0.0% | 100.0% |
| tinyllama-1.1b-chat-v1.0-bf16 | count | newest | low | 0/9 | 0.0% | 100.0% |
| tinyllama-1.1b-chat-v1.0-bf16 | comparison | legacy | all | 0/12 | 0.0% | 100.0% |
| tinyllama-1.1b-chat-v1.0-bf16 | comparison | newest | all | 0/12 | 0.0% | 100.0% |
| qwen2.5-1.5b-instruct-bf16 | decomposition | legacy | all | 0/24 | 0.0% | 0.0% |
| qwen2.5-1.5b-instruct-bf16 | decomposition | newest | all | 0/24 | 0.0% | 0.0% |
| qwen2.5-1.5b-instruct-bf16 | count | legacy | high | 2/9 | 22.2% | 0.0% |
| qwen2.5-1.5b-instruct-bf16 | count | legacy | low | 6/9 | 66.7% | 0.0% |
| qwen2.5-1.5b-instruct-bf16 | count | newest | high | 4/9 | 44.4% | 0.0% |
| qwen2.5-1.5b-instruct-bf16 | count | newest | low | 4/9 | 44.4% | 0.0% |
| qwen2.5-1.5b-instruct-bf16 | comparison | legacy | all | 8/12 | 66.7% | 0.0% |
| qwen2.5-1.5b-instruct-bf16 | comparison | newest | all | 8/12 | 66.7% | 0.0% |
| qwen3-1.7b-bf16 | decomposition | legacy | all | 0/24 | 0.0% | 12.5% |
| qwen3-1.7b-bf16 | decomposition | newest | all | 0/24 | 0.0% | 4.2% |
| qwen3-1.7b-bf16 | count | legacy | high | 1/9 | 11.1% | 0.0% |
| qwen3-1.7b-bf16 | count | legacy | low | 1/9 | 11.1% | 0.0% |
| qwen3-1.7b-bf16 | count | newest | high | 1/9 | 11.1% | 0.0% |
| qwen3-1.7b-bf16 | count | newest | low | 1/9 | 11.1% | 0.0% |
| qwen3-1.7b-bf16 | comparison | legacy | all | 8/12 | 66.7% | 0.0% |
| qwen3-1.7b-bf16 | comparison | newest | all | 7/12 | 58.3% | 0.0% |
| gemma3-1b-it-bf16-mirror | decomposition | legacy | all | 0/24 | 0.0% | 50.0% |
| gemma3-1b-it-bf16-mirror | decomposition | newest | all | 0/24 | 0.0% | 33.3% |
| gemma3-1b-it-bf16-mirror | count | legacy | high | 3/9 | 33.3% | 0.0% |
| gemma3-1b-it-bf16-mirror | count | legacy | low | 3/9 | 33.3% | 0.0% |
| gemma3-1b-it-bf16-mirror | count | newest | high | 2/9 | 22.2% | 0.0% |
| gemma3-1b-it-bf16-mirror | count | newest | low | 2/9 | 22.2% | 0.0% |
| gemma3-1b-it-bf16-mirror | comparison | legacy | all | 8/12 | 66.7% | 0.0% |
| gemma3-1b-it-bf16-mirror | comparison | newest | all | 3/12 | 25.0% | 8.3% |

## Systematic copy-error classes in newest items

| model | missing_zwj | missing_vs16 | normalization_only_mismatch | replacement_character_emitted | decomposed_rgi_sequence | rule-3 signal |
|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0-bf16 | 38 | 25 | 0 | 0 | 0 | True |
| qwen2.5-1.5b-instruct-bf16 | 1 | 0 | 0 | 0 | 0 | False |
| qwen3-1.7b-bf16 | 0 | 0 | 0 | 0 | 0 | False |
| gemma3-1b-it-bf16-mirror | 0 | 0 | 0 | 0 | 0 | False |

## Manual failure inspection

- TinyLlama ASCII echo: target="(x.O,~^4z7/Kd~cw"; extracted/parsed="` and `"; raw_prefix="Copy the text between `<IN>` and `</IN>` exactly. Preserve every Unicode code point, including invis".
- Qwen2.5 ASCII punctuation loss: target="(x.O,~^4z7/Kd~cw"; extracted/parsed="\nx.O,~^4z7/Kd~cw\n"; raw_prefix="<OUT>\nx.O,~^4z7/Kd~cw\n</OUT>".
- Qwen3 non-emoji newline padding: target="𐀀"; extracted/parsed="\n𐀀\n"; raw_prefix="<OUT>\n𐀀\n</OUT>".
- Gemma missing copy wrapper: target="⛓️‍💥"; extracted/parsed=null; raw_prefix="⛓️‍💥\n".
- Qwen2.5 decomposition hallucination: target="🏃‍♂️‍➡️"; extracted/parsed=["U+1F600", "U+200D"]; raw_prefix="U+1F600 U+200D".

## Proceed/kill verdict under spec 8.3

- Behavioral pilot conclusive: **False** (0 gate-valid models; at least two required). Final behavioral interpretation: **inconclusive_validity_gate_failure**.
- Operational hard kill among gate-valid models: **False**. Structural hard kill among gate-valid models: **False**. Whole-project hard kill: **False**.
- Behavioral proceed rule 1 (newest-minus-old copy failure >=20 pp in >=2 gate-valid models): **False**; models: none.
- Behavioral proceed rule 2 (high-minus-low fragmentation structural failure >=15 pp in >=2 models): **False**; models: none.
- Behavioral proceed rule 3 (systematic edit class in >=5 distinct newest items in any model): **False**; models: none.
- Tokenizer-scan proceed rule 4: **True** (already established). Overall proceed-rule status: **True**.
- Regression-based stop rule: not needed for pilot decision because tokenizer-scan rule 4 already fires.

## Interpretation caveats

- ASCII and wrapper gates determine whether cohort behavior is interpretable as Unicode-specific representation/content failure. Models failing either gate are reported but flagged rather than treated as clean evidence. TinyLlama's apparent wrapper pass is an extraction artifact: it often echoed the instruction, whose literal wrapper mention was then parsed as a wrapper; its ASCII exact score was nevertheless 0/24.
- Copy attention can succeed even when structural operations fail; the copy and structural arms are therefore reported separately.
- Small-model instruction following and batch execution can raise absolute failure rates but are uniform across cohorts. Format-conditional rates are in the CSV artifacts.
- The regression-adjusted full-study analysis is not replaced by these pilot tables.

Machine-readable outputs: `behavioral_manifest.json`, `probe_plan.jsonl`, `behavioral_raw.jsonl`, `behavioral_scored.jsonl`, `behavioral_verdict.json`, `copy_cohort_summary.csv`, `copy_newest_old_summary.csv`, `copy_error_taxonomy_summary.csv`, `structural_summary.csv`, `count_fragmentation_summary.csv`, and `runtime_summary.csv`.
