#!/bin/sh
set -eu
cd "$(dirname "$0")"
.venv/bin/python src/behavioral_probes.py
.venv/bin/python src/build_tokenizer_linkage.py
.venv/bin/python src/run_behavioral.py --model tinyllama-1.1b-chat-v1.0-bf16 --batch-size 32
.venv/bin/python src/run_behavioral.py --model qwen2.5-1.5b-instruct-bf16 --batch-size 32
.venv/bin/python src/run_behavioral.py --model qwen3-1.7b-bf16 --batch-size 32
.venv/bin/python src/run_behavioral.py --model gemma3-1b-it-bf16-mirror --batch-size 32
.venv/bin/python src/score_behavioral.py
.venv/bin/python src/analyze_behavioral.py
