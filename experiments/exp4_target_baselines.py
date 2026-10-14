"""Target-domain baselines on Regime C (Table 4 of the paper) and budget-matched factual learners.

Run from the project root:
    python experiments/exp4_target_baselines.py
    python experiments/exp4_target_baselines.py --run_budget_matched

Uses the repository's synthetic generator with target seeds = seed + 99, n = 2000, p = 12, signal = 1.0
(matching the target-generation convention in experiments/exp5_synthetic_ladder.py).
"""
from __future__ import annotations
import zlib
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from synthetic_regimes import SyntheticConfig, compute_all_norm_constants, generate_synthetic


def rms(tau_true: np.ndarray, tau_pred: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean((tau_true - tau_pred) ** 2)))


def make_gbr(seed: int, n_estimators: int = 200) -> GBR:
    return GBR(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )


def fit_t_learner(X_tr: np.ndarray, T_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, seed: int) -> np.ndarray:
    g0 = make_gbr(seed, 200).fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
    g1 = make_gbr(seed, 200).fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
    return g1.predict(X_te) - g0.predict(X_te)


def fit_x_learner(X_tr: np.ndarray, T_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, seed: int) -> np.ndarray:
    g0 = make_gbr(seed, 150).fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
    g1 = make_gbr(seed, 150).fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
    d1 = Y_tr[T_tr == 1] - g0.predict(X_tr[T_tr == 1])
    d0 = g1.predict(X_tr[T_tr == 0]) - Y_tr[T_tr == 0]
    tau1 = make_gbr(seed, 150).fit(X_tr[T_tr == 1], d1).predict(X_te)
    tau0 = make_gbr(seed, 150).fit(X_tr[T_tr == 0], d0).predict(X_te)
    e_te = np.clip(
        LogisticRegression(max_iter=500, random_state=seed).fit(X_tr, T_tr).predict_proba(X_te)[:, 1],
        0.05,
        0.95,
    )
    return e_te * tau0 + (1.0 - e_te) * tau1


def fit_dr_learner(X_tr: np.ndarray, T_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, seed: int) -> np.ndarray:
    n = len(Y_tr)
    e = np.zeros(n)
    m0 = np.zeros(n)
    m1 = np.zeros(n)
    n_splits = 3 if n >= 30 else 2
    for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(X_tr):
        Xa, Ta, Ya = X_tr[a], T_tr[a], Y_tr[a]
        e[b] = np.clip(
            LogisticRegression(max_iter=500, random_state=seed).fit(Xa, Ta).predict_proba(X_tr[b])[:, 1],
            0.05,
            0.95,
        )
        m0[b] = GBR(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed).fit(
            Xa[Ta == 0], Ya[Ta == 0]
        ).predict(X_tr[b])
        m1[b] = GBR(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed).fit(
            Xa[Ta == 1], Ya[Ta == 1]
        ).predict(X_tr[b])
    phi = (m1 - m0) + T_tr * (Y_tr - m1) / e - (1.0 - T_tr) * (Y_tr - m0) / (1.0 - e)
    return GBR(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed).fit(X_tr, phi).predict(X_te)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--run_budget_matched", action="store_true", help="Also evaluate target-only learners at m in {20,50,100,200,400,1400}")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm = compute_all_norm_constants(p=12)
    rows = []
    budget_rows = []
    budgets = [20, 50, 100, 200, 400, 1400]

    for seed in range(5):
        cfg = SyntheticConfig(n=2000, p=12, seed=seed + 99, regime="C", signal=1.0)
        d, _ = generate_synthetic(cfg, norm_constants=norm)
        X, T, Y, tau = d["X"], d["T"], d["Y"], d["tau"]
        tr, te = np.asarray(d["train_idx"]), np.asarray(d["test_idx"])

        tl = fit_t_learner(X[tr], T[tr], Y[tr], X[te], seed)
        dr = fit_dr_learner(X[tr], T[tr], Y[tr], X[te], seed)

        rows.append({
            "seed": seed,
            "zero_effect": rms(tau[te], 0.0),
            "constant_oracle_ate": rms(tau[te], float(tau[tr].mean())),
            "t_learner_gbm_n1400": rms(tau[te], tl),
            "dr_learner_gbm_n1400": rms(tau[te], dr),
        })

        if args.run_budget_matched:
            for m in budgets:
                if m >= len(tr):
                    sel = tr
                else:
                    # Deterministic subset selection
                    tag_crc = zlib.crc32(b"factual_target_only") & 0xFFFFFFFF
                    rng_m = np.random.default_rng(seed * 10007 + m * 17 + (tag_crc % 89))
                    sel = rng_m.choice(tr, size=m, replace=False)
                # Ensure both treatment arms have at least 3 units
                if np.sum(T[sel] == 0) < 3 or np.sum(T[sel] == 1) < 3:
                    continue
                tl_m = fit_t_learner(X[sel], T[sel], Y[sel], X[te], seed)
                xl_m = fit_x_learner(X[sel], T[sel], Y[sel], X[te], seed)
                dr_m = fit_dr_learner(X[sel], T[sel], Y[sel], X[te], seed)
                budget_rows.append({
                    "seed": seed,
                    "m": m,
                    "t_learner_pehe": rms(tau[te], tl_m),
                    "x_learner_pehe": rms(tau[te], xl_m),
                    "dr_learner_pehe": rms(tau[te], dr_m),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table4_baselines_raw.csv", index=False)

    summary = pd.DataFrame({
        "predictor": ["zero_effect", "constant_oracle_ate", "t_learner_gbm_n1400", "dr_learner_gbm_n1400"],
        "pehe_mean": [df[c].mean() for c in ["zero_effect", "constant_oracle_ate", "t_learner_gbm_n1400", "dr_learner_gbm_n1400"]],
        "pehe_std": [df[c].std(ddof=1) for c in ["zero_effect", "constant_oracle_ate", "t_learner_gbm_n1400", "dr_learner_gbm_n1400"]],
    })
    summary.to_csv(out_dir / "table4_baselines.csv", index=False)

    print("=== Table 4 Reference Predictors (5 seeds, sample std ddof=1) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.run_budget_matched and budget_rows:
        bdf = pd.DataFrame(budget_rows)
        bsum = bdf.groupby("m")[["t_learner_pehe", "x_learner_pehe", "dr_learner_pehe"]].agg(["mean", "std"])
        bsum.columns = ["_".join(c) for c in bsum.columns]
        bsum = bsum.reset_index()
        bsum.to_csv(out_dir / "table4_budget_matched_factual.csv", index=False)
        print("\n=== Budget-Matched Target-Only Factual Learners ===")
        print(bsum.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
