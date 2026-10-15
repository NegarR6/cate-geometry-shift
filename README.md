# Geometry-Dependent CATE Rankings & Supervision Cost under Pure Effect-Surface Shift

Reproducibility repository for the manuscript (Springer Nature / SN Computer Science).

## Quick start (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install econml xgboost
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/download_data.py
# ACIC: copy official files to data/acic2016/{x.csv,zymu_*.csv}
pytest -q tests/test_leakage_and_seed.py
```

## Main experiments (GPU recommended)

```powershell
$env:DEVICE = "cuda"
python experiments/exp1_ranking_reversals.py --seeds 0-9
python experiments/exp5_synthetic_ladder.py --device cuda --n_seeds 10
python experiments/exp6_ihdp_x_ladder.py --device cuda --reps 1-10
python experiments/exp4_target_baselines.py --n_seeds 10 --run_budget_matched
python experiments/exp_control_geometry.py --n_seeds 10
```

## Protocol notes (P0)

- DR pseudo-labels: adaptation **pool only**, `smooth=False`
- Adaptation: fixed `n_updates` (not growing with m)
- Neural early stopping: validation slice, not train loss
- Tree signals frozen on source under shift (no target outcome leakage)
- Propensity fed to models is estimated (`ê`), not oracle DGP `e` where applicable

## Data

| Dataset | Path | How |
|---------|------|-----|
| IHDP | `data/ihdp_npci_1-100.*.npz` | `python scripts/download_data.py` |
| ACIC 2016 | `data/acic2016/x.csv`, `zymu_*.csv` | Manual (official release) |

## License

MIT
