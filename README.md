# Geometry-Dependent CATE Rankings & Supervision Cost under Pure Effect-Surface Shift

Reproducibility repository for the manuscript submitted to *SN Computer Science*.

**Paper:** under review at *SN Computer Science*  
**Code:** this repository

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import torch, xgboost, econml; print('OK', torch.__version__)"
python scripts/download_data.py    # downloads IHDP
# ACIC 2016: see Data section below
pytest -q tests/
```

## Reproducibility

```bash
# Tier 1 (fast): unit tests + figure regeneration from canonical CSVs
bash scripts/reproduce_all.sh

# Tier 2 (heavy, GPU recommended): full 10-seed suite
bash scripts/reproduce_all.sh --full
```

Individual drivers (same as Tier 2):

```bash
python experiments/exp1_ranking_reversals.py --seeds 0-9
python experiments/exp2_two_expert_indist.py --seeds 0-9
python experiments/exp3_continuum_crossover.py --seeds 0-9
python experiments/exp4_target_baselines.py --n_seeds 10 --run_budget_matched
python experiments/exp5_synthetic_ladder.py --n_seeds 10
python experiments/exp6_ihdp_x_ladder.py --reps 1-10
python experiments/exp7_router_transfer.py --seeds 0-9
python experiments/appA_hard_design.py
python experiments/appB_surrogate_selection.py
python experiments/appC_acic_benchmark.py
```

Canonical table CSVs live under `results/` (e.g. `table2_two_expert_indist.csv`, `table3_continuum_summary.csv`, `table5_synthetic_ladder_summary.csv`, `table6_ihdp_x_ladder_summary.csv`, `table8_router_transfer.csv`, `tableA1_hard_design.csv`, `tableC1_acic_by_setting.csv`). Paper figures are regenerated into `figures/paper/`.

## Protocol notes (P0)

- DR pseudo-labels: adaptation **pool only**, `smooth=False`
- Adaptation: fixed `n_updates` (not growing with `m`)
- Neural early stopping: validation slice, not train loss
- Tree signals frozen on source under shift (no target outcome leakage)
- Propensity fed to models is estimated (`ê`), not oracle DGP `e` where applicable
- IHDP-X smooth geometry uses **leave-one-out** distance-weighted kNN (self-neighbour excluded)

## Data

| Dataset   | Path                                | How |
|-----------|-------------------------------------|-----|
| IHDP      | `data/ihdp_npci_1-100.*.npz`    | `python scripts/download_data.py` |
| ACIC 2016 | `data/acic2016/x.csv`, `zymu_*.csv` | Manual copy from official release (see below) |

### ACIC 2016

We use the 2016 Atlantic Causal Inference Conference challenge data (`n=4802`, `p=58`, 20 simulation settings). Files are **not redistributed** in this repository; obtain them from the public mirror shipped with IBM's causallib:

- https://github.com/BiomedSciAI/causallib/tree/master/causallib/datasets/data/acic_challenge_2016

Place `x.csv` and the relevant `zymu_##.csv` files under `data/acic2016/`. Our Appendix C evaluates settings 1, 2, 3, 4, 5, and 8.

If used for academic purposes, cite the competition organizers:

```
@article{dorie2019automated,
  title={Automated versus do-it-yourself methods for causal inference:
         Lessons learned from a data analysis competition},
  author={Dorie, Vincent and Hill, Jennifer and Shalit, Uri and Scott, Marc and Cervone, Dan},
  journal={Statistical Science},
  volume={34},
  number={1},
  pages={43--68},
  year={2019}
}
```

## License

MIT (code and synthetic results). Third-party datasets (IHDP, ACIC 2016) remain under their original terms; we do not claim ownership or redistribute them.
