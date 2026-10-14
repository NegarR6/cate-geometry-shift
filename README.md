# Geometry-Dependent Rankings of CATE Estimators and the Information Cost of Performance Recovery under Pure Effect-Surface Shift

Official reproducibility repository for the manuscript:

> **Geometry-Dependent Rankings of CATE Estimators and the Information Cost of Performance Recovery under Pure Effect-Surface Shift**  
> Negar Rezaei and Ehsan Moeini  
> School of Mathematics & Computer Science, Iran University of Science and Technology, Tehran, Iran

---

## Overview

This repository provides the controlled synthetic and semi-synthetic benchmarks (IHDP-X), the two-expert measurement architecture (`GeometryAdaptiveCATE`), baseline CATE learners, pre-computed results, and the experiment scripts required to reproduce all tables and figures in the paper and appendices:

1. **Controlled ranking reversals across effect geometries (Section 5.1)**  
   Isolating smooth (A), axis-aligned threshold (B), and multiplicative interaction (C) effect surfaces \(\tau(x)\) while holding \(P(X)\), \(e(x)\), \(\mu_0(x)\), and noise fixed.

2. **Performance collapse under pure geometry shift (Section 5.2)**  
   Evaluating source-trained models transferred across geometries \(\{A,B\}\to C\) against zero-effect, constant-ATE, and standalone target-only learners (T-learner, DR-learner).

3. **Target recovery vs. supervision budget \(m\) (Section 5.3)**  
   Comparing factual outcome fine-tuning, transductive doubly robust (DR) pseudo-labels, and oracle \(\tau\)-supervision across budgets \(m\in\{10,20,30,50,100\}\) on the synthetic shift and on IHDP-X.

4. **Limits of observable routers, geometry scores, and active selection (Section 5.4)**  
   Evaluating source-trained observable routers, train-only geometry scores on the synthetic continuum and ACIC 2016, and disagreement-based active query selection.

---

## Repository structure

```text
cate-geometry-shift/
├── README.md
├── LICENSE                          # MIT
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── paper/
│   └── sn-article.tex               # Manuscript source (Springer Nature template)
├── src/                             # Core library
│   ├── synthetic_regimes.py         # Synthetic DGP (regimes A, B, C + continuum)
│   ├── semi_synthetic_geometry.py   # IHDP-X generator
│   ├── models_geometry_adaptive.py  # Two-expert mixture + observable router
│   ├── train_geometry_adaptive.py
│   ├── baselines.py
│   ├── evaluation.py
│   ├── tree_signals.py
│   └── ...
├── experiments/                     # One script per table / figure panel
│   ├── exp1_ranking_reversals.py
│   ├── exp2_two_expert_indist.py
│   ├── exp3_continuum_crossover.py
│   ├── exp4_target_baselines.py
│   ├── exp5_synthetic_ladder.py
│   ├── exp6_ihdp_x_ladder.py
│   ├── exp7_router_transfer.py
│   ├── exp7b_oracle_headroom.py
│   ├── exp7c_indist_router.py
│   ├── exp_geometry_score_*.py
│   ├── appA_hard_design.py
│   ├── appA_pehe_decomposition.py
│   ├── appB_surrogate_selection.py
│   └── appC_acic_benchmark.py
├── results/                         # Pre-computed CSV tables (used by the paper)
├── figures/paper/                   # Final figures included in the manuscript
├── configs/                         # Hyper-parameter YAML files
├── scripts/
│   ├── reproduce_all.sh             # CPU-quick verification + figure regeneration
│   ├── download_data.py             # Downloads IHDP npz files into data/
│   └── make_fig3_information_ladders.py
├── data/
│   └── README.md                    # How to obtain IHDP covariates
└── tests/
    └── test_leakage_and_seed.py
```

---

## Table / figure mapping

| Paper item | Description | Script | Result file |
|------------|-------------|--------|-------------|
| **Fig. 1 / regimes** | PEHE by regime and signal | `experiments/exp1_ranking_reversals.py` | `results/table_fig1_regimes_summary.csv`, `figures/paper/fig_pehe_by_regime_signal.pdf` |
| **Fig. 2** | Winner frequency | same | `results/table_fig2_winner_freq.csv`, `figures/paper/fig_winner_freq.png` |
| **Table 2** | In-distribution two-expert instrument | `experiments/exp2_two_expert_indist.py` | `results/table2_two_expert_indist.csv` |
| **Table 3** | Continuum crossover | `experiments/exp3_continuum_crossover.py` | `results/table3_continuum_summary.csv` |
| **Table 4** | Target baselines (Zero, Constant-ATE, T-learner, DR-learner) | `experiments/exp4_target_baselines.py` | `results/table4_baselines.csv` |
| **Table 5 & Fig. 3 (left)** | Synthetic few-shot ladder `{A,B}->C` | `experiments/exp5_synthetic_ladder.py` | `results/table5_synthetic_ladder_summary.csv` |
| **Table 6 & Fig. 3 (right)** | IHDP-X few-shot ladder | `experiments/exp6_ihdp_x_ladder.py` | `results/table6_ihdp_x_ladder_summary.csv` |
| **Table 7, Figs. 4-5** | Router transfer & oracle headroom | `experiments/exp7_*.py` | `results/table7_router_transfer.csv`, `figures/paper/fig_e3_shift.pdf` |
| **Appendix A** | Hard design | `experiments/appA_*.py` | `results/tableA1_hard_design.csv` |
| **Appendix B** | Surrogate selection by geometry | `experiments/appB_surrogate_selection.py` | `results/tableB1_surrogate_selection.csv` |
| **Appendix C** | ACIC 2016 | `experiments/appC_acic_benchmark.py` | `results/tableC1_acic_by_setting.csv` |

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**External data (IHDP only).**  
Synthetic experiments need no external files. For Table 6 / Fig. 3 (right):

```bash
python scripts/download_data.py
```

This places the IHDP npz files under `data/`.

---

## What is fully reproducible on CPU (quick check)

The following run in well under a minute on CPU and verify the key reference numbers reported in the paper:

```bash
# Table 4 baselines (Zero ~ 1.103, Constant-ATE ~ 1.006, T-learner ~ 0.471, DR-learner ~ 0.794)
python experiments/exp4_target_baselines.py

# Regenerate Figure 3 from the stored summary CSVs
python scripts/make_fig3_information_ladders.py

# Leakage / seed unit tests
pytest tests/test_leakage_and_seed.py
```

Or simply:

```bash
bash scripts/reproduce_all.sh
```

---

## Full experiment suite (GPU recommended)

Neural experiments (Tables 2, 3, 5, 6, 7 and appendices) are computationally heavier. Pre-computed CSVs in `results/` already contain the numbers used in the manuscript. To re-run from scratch:

```bash
export DEVICE=cuda   # or cpu
python experiments/exp1_ranking_reversals.py
python experiments/exp2_two_expert_indist.py
# ... likewise for the other experiment scripts
```

All random seeds used in the paper are fixed inside the corresponding scripts; see also `tests/test_leakage_and_seed.py`.

---

## Key claims and reference numbers (sanity check)

| Claim | Reference number |
|-------|------------------|
| Frozen model on `{A,B}->C` | PEHE 1.259 +/- 0.036 |
| Constant-ATE predictor | 1.006 |
| Null (zero-effect) predictor | 1.103 |
| Target-only T-learner (1400 factual units) | 0.471 |
| Oracle tau at m=50 (synthetic) | 0.814 +/- 0.097 |
| Oracle tau at m=100 (synthetic) | 0.658 +/- 0.069 |

These values are recovered by `experiments/exp4_target_baselines.py` and the ladder scripts.

---

## Citation

If you use this code or the associated findings, please cite the manuscript (see `CITATION.cff`).

---

## License

MIT License — see [LICENSE](LICENSE).
