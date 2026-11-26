#!/bin/sh
set -eu
cd "$(dirname "$0")"
./run.sh
.venv/bin/python src/behavioral_probes.py
.venv/bin/python src/build_round2_plan.py --copy-protocol fewshot_copy_v5_multiturn_cueless
.venv/bin/python src/score_behavioral_round2.py
.venv/bin/python src/analyze_wrapper_mimicry.py
.venv/bin/python src/analyze_behavioral_round2.py
.venv/bin/python src/make_figures.py
.venv/bin/python src/check_artifact.py
