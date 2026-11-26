#!/usr/bin/env bash
# Regenerates the results and the figures of the article that depend on this
# package.  Run from the repository root:   bash analysis/run_all.sh
# Single core; about 80 minutes in total (see README.md for the cost of each step).
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python3}

# 1. Postprocessing of the archived data (read-only).
$PY analysis/momentum.py --root .
$PY analysis/returns.py --root .
$PY analysis/zero_structure.py --root .

# 2. Short new computations.
$PY analysis/initial_layer.py --N 2048,4096,8192 --p 1.75,2.0,2.25,3.0
$PY analysis/lengths.py --nstar 3.6,5.4,9.0,10.8
$PY analysis/gain_ladder.py --cutoffs 21,35,51,61
$PY analysis/rpo.py --cutoffs 21,23,25 --method newton
$PY analysis/rpo.py --cutoffs 27,29 --method lm --start analysis/results/rpo_orbits.json --append
$PY analysis/rpo_linear_check.py

# 3. Figures (one PDF per panel) and the numbers quoted in the text.
$PY analysis/make_figures.py --root . --out figures_reproduced
$PY analysis/paper_numbers.py --root . > analysis/results/paper_numbers.txt
