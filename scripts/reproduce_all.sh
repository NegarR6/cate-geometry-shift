#!/usr/bin/env bash
# Reproducibility entrypoint for all 10-seed tables and figures.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1] Unit tests (leakage & seed determinism)"
python -m pytest -q tests/test_leakage_and_seed.py

echo "[2] Table 4 & Table 7 target-only baselines (10 seeds, aligned target split)"
python experiments/exp4_target_baselines.py --n_seeds 10 --run_budget_matched

echo "[3] Regenerate paper figures from canonical CSVs"
python scripts/plot_synthetic_regimes.py
python scripts/make_fig3_information_ladders.py
python scripts/plot_e3_shift.py results/table7_router_transfer.csv figures/paper

echo "Done. Full 10-seed neural suite commands:"
echo "  python experiments/exp1_ranking_reversals.py --seeds 0-9"
echo "  python experiments/exp2_two_expert_indist.py --seeds 0-9"
echo "  python experiments/exp3_continuum_crossover.py --seeds 0-9"
echo "  python experiments/exp5_synthetic_ladder.py --n_seeds 10"
echo "  python experiments/exp6_ihdp_x_ladder.py --reps 1-10"
echo "  python experiments/exp7_router_transfer.py --seeds 0-9"
