#!/usr/bin/env bash
# Reproducibility entrypoint.
#
# Tier 1 (fast, always run): unit tests + figure regeneration from existing
#   canonical CSVs.  Completes in seconds and does NOT claim to re-run the
#   full 10-seed neural suite.
# Tier 2 (heavy, explicit): full 10-seed experiment drivers.  These are listed
#   but not executed by default; run them manually or pass --full.
#
# Usage:
#   bash scripts/reproduce_all.sh          # Tier 1 only
#   bash scripts/reproduce_all.sh --full   # Tier 1 + all 10-seed experiments
set -euo pipefail

# Resolve repo root whether this script lives in scripts/ or at repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -d "$SCRIPT_DIR/experiments" && -d "$SCRIPT_DIR/src" ]]; then
  ROOT="$SCRIPT_DIR"
else
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT"

FULL=0
if [[ "${1:-}" == "--full" ]]; then
  FULL=1
fi

echo "============================================================"
echo "Repo root: $ROOT"
echo "Tier 1: unit tests + figure regeneration from canonical CSVs"
echo "============================================================"

echo "[1/3] Unit tests (leakage & seed determinism)"
python -m pytest -q tests/test_leakage_and_seed.py

echo "[2/3] Target-only baselines (Table 4 / budget-matched; 10 seeds)"
python experiments/exp4_target_baselines.py --n_seeds 10 --run_budget_matched

echo "[3/3] Regenerate paper figures from canonical CSVs"
python scripts/plot_synthetic_regimes.py
python scripts/make_fig3_information_ladders.py
if [[ -f results/table8_router_transfer.csv ]]; then
  python scripts/plot_e3_shift.py results/table8_router_transfer.csv figures/paper
elif [[ -f results/table7_router_transfer.csv ]]; then
  python scripts/plot_e3_shift.py results/table7_router_transfer.csv figures/paper
else
  echo "WARN: no router-transfer CSV found; skipping fig_e3_shift"
fi

echo ""
echo "Tier 1 complete."
echo ""

if [[ "$FULL" -eq 0 ]]; then
  echo "--------------------------------------------------------------"
  echo "Heavy 10-seed experiments were NOT run (use --full to execute)."
  echo "Commands for full regeneration from a clean environment:"
  echo "  python experiments/exp1_ranking_reversals.py --seeds 0-9"
  echo "  python experiments/exp2_two_expert_indist.py --seeds 0-9"
  echo "  python experiments/exp3_continuum_crossover.py --seeds 0-9"
  echo "  python experiments/exp5_synthetic_ladder.py --n_seeds 10"
  echo "  python experiments/exp6_ihdp_x_ladder.py --reps 1-10"
  echo "  python experiments/exp7_router_transfer.py --seeds 0-9"
  echo "  python experiments/appA_hard_design.py"
  echo "  python experiments/appB_surrogate_selection.py"
  echo "  python experiments/appC_acic_benchmark.py"
  echo "--------------------------------------------------------------"
  exit 0
fi

echo "============================================================"
echo "Tier 2: full 10-seed suite (may take many GPU-hours)"
echo "============================================================"

python experiments/exp1_ranking_reversals.py --seeds 0-9
python experiments/exp2_two_expert_indist.py --seeds 0-9
python experiments/exp3_continuum_crossover.py --seeds 0-9
python experiments/exp5_synthetic_ladder.py --n_seeds 10
python experiments/exp6_ihdp_x_ladder.py --reps 1-10
python experiments/exp7_router_transfer.py --seeds 0-9
python experiments/appA_hard_design.py || true
python experiments/appB_surrogate_selection.py || true
python experiments/appC_acic_benchmark.py || true

echo "Full reproducibility suite finished."