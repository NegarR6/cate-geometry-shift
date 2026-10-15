"""
Table 4 and Table 7 budget-matched target-only baselines on Regime C under {A,B}->C shift.
Evaluated on the exact same 10 source and target splits as exp5_synthetic_ladder.py.
"""
from __future__ import annotations

import sys
import pathlib
import argparse

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.synthetic_regimes import generate_synthetic, SyntheticConfig, compute_all_norm_constants
from src.budgeted_ladder_utils import dr_pseudo_tau, source_ate_estimate


def pehe(tau_true, tau_pred):
    return float(np.sqrt(np.mean((np.asarray(tau_true) - np.asarray(tau_pred)) ** 2)))


def t_learner_predict(Xtr, Ttr, Ytr, Xte, seed=0):
    m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed)
    m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed)
    ctrl = Ttr == 0
    treated = Ttr == 1
    m0.fit(Xtr[ctrl], Ytr[ctrl])
    m1.fit(Xtr[treated], Ytr[treated])
    return m1.predict(Xte) - m0.predict(Xte)


def run_seed(seed: int, n: int = 2000, p: int = 12):
    norms = compute_all_norm_constants(p=p)
    # Target regime C (exact same seed+99 split as exp5_synthetic_ladder.py)
    cfg = SyntheticConfig(n=n, p=p, regime="C", signal=1.0, seed=seed + 99)
    data, _ = generate_synthetic(cfg, norm_constants=norms)
    X, T, Y, tau = data["X"], data["T"].astype(int), data["Y"], data["tau"]
    pool, test = data["train_idx"], data["test_idx"]

    # Source data for deployable ATE (exact same seed+17*(i+1) split as exp5_synthetic_ladder.py)
    Xs, Ts, Ys = [], [], []
    offset = 0
    src_train_parts = []
    for i, reg in enumerate(["A", "B"]):
        cfg_s = SyntheticConfig(n=n, p=p, seed=seed + 17 * (i + 1), regime=reg, signal=1.0)
        d_s, _ = generate_synthetic(cfg_s, norm_constants=norms)
        Xs.append(d_s["X"])
        Ts.append(d_s["T"])
        Ys.append(d_s["Y"])
        src_train_parts.append(np.asarray(d_s["train_idx"]) + offset)
        offset += len(d_s["Y"])
    X_all = np.concatenate(Xs, axis=0).astype(np.float32)
    T_all = np.concatenate(Ts, axis=0).astype(np.float32)
    Y_all = np.concatenate(Ys, axis=0).astype(np.float32)
    src_train_idx = np.concatenate(src_train_parts)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(src_train_idx)
    n_val = max(80, int(0.12 * len(perm)))
    tr_idx = perm[n_val:]

    rows = []
    rows.append({"seed": seed, "estimator": "zero", "pehe": pehe(tau[test], np.zeros(len(test)))})
    rows.append({"seed": seed, "estimator": "constant_oracle_ate", "pehe": pehe(tau[test], np.full(len(test), float(np.mean(tau[pool]))))})
    ate_src = source_ate_estimate(X_all[tr_idx], T_all[tr_idx], Y_all[tr_idx], seed=seed)
    rows.append({"seed": seed, "estimator": "source_ate", "pehe": pehe(tau[test], np.full(len(test), ate_src))})
    pred_t = t_learner_predict(X[pool], T[pool], Y[pool], X[test], seed=seed)
    rows.append({"seed": seed, "estimator": "target_T_learner_full", "pehe": pehe(tau[test], pred_t)})
    tau_dr = dr_pseudo_tau(X[pool], T[pool], Y[pool], n_splits=3, seed=seed, smooth=False)
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed)
    reg.fit(X[pool], tau_dr)
    rows.append({"seed": seed, "estimator": "target_DR_learner_full", "pehe": pehe(tau[test], reg.predict(X[test]))})
    return rows


def run_budget(seed: int, m: int, n: int = 2000, p: int = 12):
    norms = compute_all_norm_constants(p=p)
    cfg = SyntheticConfig(n=n, p=p, regime="C", signal=1.0, seed=seed + 99)
    data, _ = generate_synthetic(cfg, norm_constants=norms)
    X, T, Y, tau = data["X"], data["T"].astype(int), data["Y"], data["tau"]
    pool, test = data["train_idx"], data["test_idx"]
    if m >= len(pool):
        sel = pool
    else:
        rng = np.random.default_rng(seed)
        sel = rng.choice(pool, size=m, replace=False)
    pred = t_learner_predict(X[sel], T[sel], Y[sel], X[test], seed=seed)
    return {"seed": seed, "m": m, "estimator": "target_T_learner", "pehe": pehe(tau[test], pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--run_budget_matched", action="store_true", default=True)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for s in range(args.n_seeds):
        all_rows.extend(run_seed(s))
    df = pd.DataFrame(all_rows)
    summary = df.groupby("estimator")["pehe"].agg(["mean", "std"]).reset_index()
    summary.to_csv(out / "table4_baselines.csv", index=False)
    df.to_csv(out / "table4_baselines_raw.csv", index=False)
    print("--- Table 4 Baselines ---")
    print(summary.to_string(index=False))

    if args.run_budget_matched:
        brows = []
        for m in [20, 50, 100, 200, 400, 1400]:
            for s in range(args.n_seeds):
                brows.append(run_budget(s, m))
        bdf = pd.DataFrame(brows)
        bsum = bdf.groupby("m")["pehe"].agg(["mean", "std"]).reset_index()
        bsum.to_csv(out / "table4_budget_matched_factual.csv", index=False)
        print("\n--- Table 7 Budget-Matched T-Learner ---")
        print(bsum.to_string(index=False))


if __name__ == "__main__":
    main()
