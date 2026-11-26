# Unicode Release Debt — Deep Research Extractions (2026-11-25)

## Novelty verdict: YES — CLAIM NOVELTY.
No 2022-2026 work measures LLM byte-exact copying or UAX#29 grapheme counting/decomposition stratified by Unicode/emoji release version x tokenizer vintage. The intersection is empty. Even the weaker claim (grapheme behavioral probes WITHOUT version stratification) is unoccupied in academic literature.

## Must-cite closest hits (positioning)
- Jang et al., Improbable Bigrams, EMNLP 2025 (arXiv 2410.23684): incomplete-token bigrams -> 33-77% exact-copy failure; alt tokenization cuts failures 35-98%. STRONGEST mechanism support for us.
- Moon et al., ByteCanary, ICML 2026 (arXiv 2606.14122): strict UTF-8 validity lags partial by 9.9-65.4pp across 10 public LLMs.
- Zhang et al., When Copying Is Hard, NeurIPS 2026 (arXiv 2610.28393): base LLMs 51.85% EM on verbatim copying even for English/code. Copying is brittle BEFORE unicode enters.
- Land & Bartolo, Fishing for Magikarp, EMNLP 2024 outstanding paper: >1400 manually-added emoji tokens UNREACHABLE/untrained in Command-R/R+. Direct evidence emoji vocab entries can be dead.
- UBE (NeurIPS 2026, arXiv 2610.01984): codec-only U17 audit, not LLM behavior. Contrast explicitly.
- TokTier/Delta Tokenizer: pre-tokenizer Unicode-version skew in serving. Engineering layer, not behavior.
- Zhou et al. NLP+CSS 2026: post-cutoff emoji OOV hurts RoBERTa sentiment — the ONLY temporal-release precedent, but semantic classification, not copying.

## Corrections to spec
- DROP the Hanay Zenodo seed (10.5281/zenodo.22064524) — record does not exist. Unverified.
- Di Cristofaro now published IJCL 31(3) 2026, doi:10.1075/ijcl.24116.dic.
- Watch terms for final pre-submission check: "post-freeze characters", "zero-shot codepoints", "DerivedAge", "vocabulary staleness", UTW workshop talks (Foster "Tokenizer Comedy" UTW 2026, unarchived, 33 LLMs).

## Mechanistic gold for the paper (from byte-mechanics review)
- o200k_base regex INADVERTENTLY splits VS16 (U+FE0F) off from ZWJ: heart-on-fire and rainbow-flag split at VS16|ZWJ boundary before BPE even runs. cl100k/Llama3/Qwen2.5 keep them in one pretoken. Concrete, checkable, publication-worthy mechanism.
- SentencePiece nmt_nfkc still maps ZWNJ (U+200C) -> ASCII space (fixed for ZWJ in v0.1.96 only).
- Byte-fallback pieces (<0xNN>) are NEVER merged into multi-byte tokens in SPM — unseen emoji decompose to 1-byte pieces, 3-4 per scalar.
- XLM-R (bge-m3) returns EMPTY token sequences for 97.9% of 382 invisible chars — silent deletion, measurement must separate from fragmentation.

## Implementation pitfalls (bake into harness NOW)
1. NEVER per-token decode([tid]) for byte analysis — use raw byte extraction (decode_single_token_bytes / sp.is_byte / GPT-2 unicode inversion). U+FFFD conflation bug.
2. Pin UCD versions everywhere: Python unicodedata is HOST-DEPENDENT (3.13=UCD15.1, 3.14=16, 3.15=17) and Qwen1 calls host unicodedata.normalize directly. Record unidata_version in manifest. Use pinned unicode normalization lib (UCD 18 available).
3. transformers#49066: Qwen2Tokenizer silently overrides Qwen3.6/3.8 regex. Load tokenizer.json RAW, SHA-256 hash the artifact files, verify against native backends. Do not trust AutoTokenizer.
4. Strip synthetic dummy-prefix (SPM add_dummy_prefix) before tokens_per_cluster counts.
5. explicit_byte_fallback_count (SPM <0xNN> only) vs byte-level BPE single-byte atoms — KEEP SEPARATE. Naive <0xNN> regex misreports.
6. nfc/nfd divergence inversion trap: NFC-normalizing tokenizers return SAME ids for NFC/NFD input precisely because they DESTROY NFD fidelity. Report token_id_diverges + nfc_roundtrip_exact + nfd_roundtrip_exact together.
7. grapheme segmenter must pin UAX#29 revision (GB9c changed in 15.1 AND again in U18 draft). U+1FAEA joined Extended_Pictographic only in Unicode 17.

## Status
- Week-1 kill test build: restart needed (first agent hit rate limit).
- 2 Gemini reports still in flight per user.

## Hosted/serving attribution intel (preprocessing report)
1. Frontier observability has CONTRACTED: GPT-6.2 Astra/Sol disallow logprobs+temperature, Anthropic never exposes logprobs/token-ids, Bedrock none. VALIDATES the local-first design — hosted arms can only ever be a bounded supplement.
2. count_tokens pairing is a normalization fingerprint requiring NO logprobs: identical prompt token counts on NFC vs NFD input pairs => pre-model normalization. Cheap, add to harness.
3. vLLM serving bugs if we use it: #68324 trailing U+FFFD silently dropped at finish (stream AND non-stream), #67755 Tekken corruption, logprobs bytes re-encoded from decoded string (useless for byte forensics). If serving locally, prefer HF transformers generate() directly or TGI; document choice.
4. Gemini uniquely exposes integer tokenIds in logprobs (GA since June 2025) — best hosted endpoint for any supplementary arm.
5. 20-probe black-box audit battery exists in the report (T01-T20): NFC/NFD pairs, Hangul jamo, Vietnamese multi-mark (Precompiled shortest-prefix bug fingerprint), NFKC-only fi/H2/NBSP, Angstrom singleton, noncanonical ccc ordering, ZWJ emoji, VS15/VS16, invisible Cf, C0 controls, noncharacters+PUA, literal U+FFFD trailing-drop detector, punctuation-space cleanup detector, special-token literals, JSON escapes. When building the hosted supplement, reuse these probes wholesale — already byte-verified.
6. Attribution discipline: stream vs non-stream diff = serving layer; equal token counts on canonical pairs = input normalizer; content vs raw bytes logprobs divergence = detokenizer. Bake the three-way separation into analysis.
