#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3.12 -m venv .venv
fi
.venv/bin/pip install -r requirements.txt >/dev/null
.venv/bin/python src/ucd_build.py
.venv/bin/python src/tokscan.py inventory
.venv/bin/python src/testset.py base
.venv/bin/python src/tokscan.py scan results/count_candidates.jsonl results/count_candidate_scan.jsonl
.venv/bin/python src/testset.py final
.venv/bin/python src/validate.py
.venv/bin/python src/tokscan.py scan results/scan_units.jsonl results/tokenizer_scan.jsonl
.venv/bin/python src/analyze_scan.py
