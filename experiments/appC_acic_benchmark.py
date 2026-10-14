"""
Priority 2B — ACIC multi-setting ranking table (NO proxy selector claim).

For each ACIC setting × seed:
  train NeuralExpert, TreeExpert, GeoAdaptive joint
  report PEHE: adaptive, neural, tree, average mixture, optional X-Learner

Usage:
  python -m src.run_acic_ranking --settings 1,2,3,4,5,8 --n_seeds 3 \\
    --pretrain_epochs 25 --joint_epochs 30 --device cuda \\
    --out_dir results/acic_ranking
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from models_geometry_adaptive import build_geo_adaptive_from_hparams
from train_geometry_adaptive import GeometryAdaptiveTrainer, set_seed, build_loaders
from tree_signals import CrossFittedTreeSignals

try:
    from budgeted_ladder_utils import pehe
except Exception:
    def pehe(true, pred):
        true = np.asarray(true).ravel()
        pred = np.asarray(pred).ravel()
        return float(np.sqrt(np.mean((true - pred) ** 2)))


def load_acic_setting(setting_id: int, seed: int, data_dir: Path, test_size: float = 0.2):
    try:
        from run_geometry_proxy_acic import load_acic_setting as _load
        return _load(setting_id, seed, data_dir, test_size=test_size)
    except Exception:
        pass
    x_path = data_dir / "x.csv"
    z_path = data_dir / f"zymu_{setting_id}.csv"
    if not x_path.exists():
        # try acic2016 subfolder
        data_dir = data_dir / "acic2016" if (data_dir / "acic2016").exists() else data_dir
        x_path = data_dir / "x.csv"
        z_path = data_dir / f"zymu_{setting_id}.csv"
    Xdf = pd.read_csv(x_path)
    Zdf = pd.read_csv(z_path)
    X = pd.get_dummies(Xdf, drop_first=True).astype(np.float64).values
    T = Zdf["z"].values.astype(np.float64)
    Y = np.where(T > 0.5, Zdf["y1"].values, Zdf["y0"].values).astype(np.float64)
    tau = (Zdf["mu1"].values - Zdf["mu0"].values).astype(np.float64)
    n = len(Y)
    tr, te = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    tr, te = np.asarray(tr), np.asarray(te)
    scaler = StandardScaler()
    X = X.copy()
    X[tr] = scaler.fit_transform(X[tr])
    X[te] = scaler.transform(X[te])
    y_mean, y_std = float(Y[tr].mean()), float(Y[tr].std() + 1e-8)
    Y = (Y - y_mean) / y_std
    tau = tau / y_std
    e = np.full(n, float(np.clip(T.mean(), 0.05, 0.95)))
    return {
        "X": X.astype(np.float32),
        "T": T.astype(np.float32),
        "Y": Y.astype(np.float32),
        "e": e.astype(np.float32),
        "tau": tau.astype(np.float32),
        "train_idx": tr,
        "test_idx": te,
    }


def x_learner_pehe(X, T, Y, tau, tr, te):
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except Exception:
        return float("nan")
    X, T, Y = np.asarray(X), np.asarray(T).ravel(), np.asarray(Y).ravel()
    tr0 = tr[T[tr] == 0]
    tr1 = tr[T[tr] == 1]
    if len(tr0) < 20 or len(tr1) < 20:
        return float("nan")
    m0 = GradientBoostingRegressor(random_state=0, max_depth=3, n_estimators=50)
    m1 = GradientBoostingRegressor(random_state=0, max_depth=3, n_estimators=50)
    m0.fit(X[tr0], Y[tr0])
    m1.fit(X[tr1], Y[tr1])
    d1 = Y[tr1] - m0.predict(X[tr1])
    d0 = m1.predict(X[tr0]) - Y[tr0]
    t1 = GradientBoostingRegressor(random_state=1, max_depth=3, n_estimators=50).fit(X[tr1], d1)
    t0 = GradientBoostingRegressor(random_state=2, max_depth=3, n_estimators=50).fit(X[tr0], d0)
    pred = 0.5 * (t0.predict(X[te]) + t1.predict(X[te]))
    return pehe(tau[te], pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--settings", default="1,2,3,4,5,8")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--pretrain_epochs", type=int, default=25)
    ap.add_argument("--joint_epochs", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="results/acic_ranking")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    settings = [int(x) for x in args.settings.split(",") if x.strip()]
    data_dir = Path(args.data_dir)
    if (data_dir / "acic2016").exists():
        data_root = data_dir / "acic2016"
    else:
        data_root = data_dir

    rows: List[Dict[str, Any]] = []
    verbose = not args.quiet

    for setting in settings:
        for seed in range(args.n_seeds):
            print(f"\n===== ACIC setting={setting} seed={seed} =====")
            set_seed(seed)
            try:
                data = load_acic_setting(setting, seed, data_root)
            except Exception as ex:
                print(f"  SKIP load error: {ex}")
                continue
            X, T, Y, e, tau = data["X"], data["T"], data["Y"], data["e"], data["tau"]
            tr, te = data["train_idx"], data["test_idx"]
            rng = np.random.default_rng(seed)
            perm = rng.permutation(tr)
            n_va = max(50, len(tr) // 10)
            val_idx, train_idx = perm[:n_va], perm[n_va:]

            n, p = X.shape
            tb = CrossFittedTreeSignals(n_splits=3, seed=seed)
            ts = np.zeros((n, 4), np.float32)
            ts[train_idx] = tb.fit_transform(X[train_idx], T[train_idx], Y[train_idx])
            ts[val_idx] = tb.transform(X[val_idx])
            ts[te] = tb.transform(X[te])

            model = build_geo_adaptive_from_hparams(num_features=p)
            trainer = GeometryAdaptiveTrainer(
                model, device=args.device,
                pretrain_epochs=args.pretrain_epochs,
                joint_epochs=args.joint_epochs,
                freeze_experts_epochs=min(8, max(1, args.joint_epochs // 4)),
                route_weight_start=0.15, route_weight_end=0.45,
            )
            tr_loader, va_loader = build_loaders(X, T, Y, e, ts, None, train_idx, val_idx)
            trainer.pretrain_experts(tr_loader, va_loader, verbose=verbose)
            if args.joint_epochs > 0:
                try:
                    pred = trainer.predict_experts(X[train_idx], e[train_idx], ts[train_idx])
                    tn, tt = pred[1], pred[2]
                    # soft oracle routing labels from train tau if available
                    oracle = np.full(n, -1, np.int64)
                    oracle[train_idx] = (
                        np.abs(tt - tau[train_idx]) < np.abs(tn - tau[train_idx])
                    ).astype(np.int64)
                    tr_loader, va_loader = build_loaders(
                        X, T, Y, e, ts, oracle, train_idx, val_idx
                    )
                    if hasattr(trainer, "fit_joint_only"):
                        trainer.fit_joint_only(tr_loader, va_loader, verbose=verbose)
                except Exception as ex:
                    print(f"  [warn] joint: {ex}")

            pred = trainer.predict_experts(X[te], e[te], ts[te])
            tau_a, tau_n, tau_t = pred[0], pred[1], pred[2]
            w = pred[3] if len(pred) > 3 else None
            pn = pehe(tau[te], tau_n)
            pt = pehe(tau[te], tau_t)
            pa = pehe(tau[te], tau_a)
            pavg = pehe(tau[te], 0.5 * (tau_n + tau_t))
            px = x_learner_pehe(X, T, Y, tau, train_idx, te)
            w_n = float(np.mean(w[:, 0])) if w is not None and getattr(w, "ndim", 0) == 2 else float("nan")
            print(
                f"  PEHE adapt={pa:.4f} neural={pn:.4f} tree={pt:.4f} "
                f"avg={pavg:.4f} xlearn={px:.4f} w_n={w_n:.2f}"
            )
            rows.append({
                "setting": setting, "seed": seed,
                "pehe_adaptive": pa, "pehe_neural": pn, "pehe_tree": pt,
                "pehe_average": pavg, "pehe_xlearner": px, "mean_weight_neural": w_n,
                "n_train": len(train_idx), "n_test": len(te), "p": p,
            })

    if not rows:
        print("No rows; check ACIC data paths.")
        return
    df = pd.DataFrame(rows)
    raw = Path(args.out_dir) / "acic_ranking_raw.csv"
    df.to_csv(raw, index=False)
    by = df.groupby("setting")[
        ["pehe_adaptive", "pehe_neural", "pehe_tree", "pehe_average", "pehe_xlearner"]
    ].agg(["mean", "std"])
    by.columns = ["_".join(c) for c in by.columns]
    by = by.reset_index()
    by_path = Path(args.out_dir) / "acic_ranking_by_setting.csv"
    by.to_csv(by_path, index=False)
    overall = df[["pehe_adaptive", "pehe_neural", "pehe_tree", "pehe_average", "pehe_xlearner"]].agg(["mean", "std"])
    overall.to_csv(Path(args.out_dir) / "acic_ranking_overall.csv")
    print("\n=== By setting ===")
    print(by.to_string(index=False))
    print("\n=== Overall ===")
    print(overall)
    print(f"Saved {raw}")


if __name__ == "__main__":
    main()
