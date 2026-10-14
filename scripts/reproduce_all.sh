#!/usr/bin/env bash
# CPU-quick reproducibility checks for the CATE Geometry Shift paper.
# Full neural experiments require GPU; pre-computed results/ already contain the paper numbers.
set -euo pipefail

echo "============================================================"
echo " CATE Geometry Shift — quick reproducibility checks (CPU)"
echo "============================================================"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ""
echo "[1/4] Table 4 target-domain baselines (Zero / Constant-ATE / T-learner / DR-learner)..."
python experiments/exp4_target_baselines.py

echo ""
echo "[2/4] Regenerating Figure 3 (information ladders) from stored CSVs..."
python scripts/make_fig3_information_ladders.py

echo ""
echo "[3/4] Unit tests (leakage guards + deterministic seeds)..."
if command -v pytest >/dev/null 2>&1; then
  pytest -q tests/test_leakage_and_seed.py || true
else
  echo "  pytest not installed; skipping. Run: pip install pytest"
fi

echo ""
echo "[4/4] Sanity: key result CSVs present..."
for f in \
  results/table4_baselines.csv \
  results/table5_synthetic_ladder_summary.csv \
  results/table6_ihdp_x_ladder_summary.csv \
  results/table7_router_transfer.csv \
  figures/paper/fig_information_ladders.pdf
do
  if [[ -f "$f" ]]; then
    echo "  OK  $f"
  else
    echo "  MISSING  $f"
  fi
done

echo ""
echo "Done."
echo "Pre-computed CSVs in results/ match the manuscript."
echo "To re-run full neural experiments (GPU recommended):"
echo "  export DEVICE=cuda"
echo "  python experiments/exp1_ranking_reversals.py"
echo "  python experiments/exp2_two_expert_indist.py"
echo "  # ... see README for the full list"
