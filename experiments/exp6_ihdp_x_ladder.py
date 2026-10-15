"""Semi-synthetic geometry shift ladder on IHDP-X (Table 6 / Fig 3 right).

P0 FIXES:
- DR computed only on adaptation pool.
- Fixed n_updates for adaptation (independent of m).
- Report zero / constant baselines alongside frozen.
- Source-only tree signals (never re-fit on target).
- Estimated propensity from tree signals when available.
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from adapt_geometry_shift import adapt_fewshot
from budgeted_ladder_utils import dr_pseudo_tau as dr_pseudo_tau_util
from budgeted_ladder_utils import source_ate_estimate
from models_geometry_adaptive import build_geo_adaptive_from_hparams
from semi_synthetic_geometry import make_geometry_pair_from_real
from train_geometry_adaptive import GeometryAdaptiveTrainer, build_loaders, set_seed
from tree_signals import CrossFittedTreeSignals


def pehe(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_ihdp_rep(rep: int, data_dir: Path) -> Dict[str, np.ndarray]:
    train_p = Path(data_dir) / "ihdp_npci_1-100.train.npz"
    if not train_p.exists():
        raise FileNotFoundError(
            f"IHDP dataset not found at {train_p}. Run `python scripts/download_data.py` first."
        )
    tr = np.load(train_p)
    i = int(rep) - 1
    X_tr = StandardScaler().fit_transform(tr["x"][:, :, i].astype(np.float64))
    T_tr = tr["t"][:, i].astype(np.float64)
    Y_tr = tr["yf"][:, i].astype(np.float64)
    Y_tr = (Y_tr - float(Y_tr.mean())) / (float(Y_tr.std()) + 1e-8)
    return {
        "X_tr": X_tr.astype(np.float32),
        "T_tr": T_tr.astype(np.float32),
        "Y_tr": Y_tr.astype(np.float32),
    }


def split_idx(n: int, seed: int, val_frac: float = 0.1, test_frac: float = 0.2) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_te = max(50, int(test_frac * n))
    n_va = max(30, int(val_frac * n))
    return {
        "test_idx": perm[:n_te],
        "val_idx": perm[n_te : n_te + n_va],
        "train_idx": perm[n_te + n_va :],
    }


def train_on_source_world(
    X, T, Y, e, tau, train_idx, val_idx, device, pretrain_epochs, joint_epochs, seed, verbose,
):
    n, p = X.shape
    tb = CrossFittedTreeSignals(n_splits=3, seed=seed)
    ts = np.zeros((n, 4), np.float32)
    ts[train_idx] = tb.fit_transform(X[train_idx], T[train_idx], Y[train_idx])
    ts[val_idx] = tb.transform(X[val_idx])

    e_hat = ts[:, 3].copy()
    assert (e_hat[train_idx] > 0).all() and (e_hat[val_idx] > 0).all(), "propensity not filled"

    model = build_geo_adaptive_from_hparams(num_features=p)
    trainer = GeometryAdaptiveTrainer(
        model,
        device=device,
        pretrain_epochs=pretrain_epochs,
        joint_epochs=joint_epochs,
        freeze_experts_epochs=min(8, max(1, joint_epochs // 4)),
        route_weight_start=0.15,
        route_weight_end=0.45,
    )
    tr_loader, va_loader = build_loaders(X, T, Y, e_hat, ts, None, train_idx, val_idx)
    trainer.pretrain_experts(tr_loader, va_loader, verbose=verbose)
    if joint_epochs > 0 and hasattr(trainer, "fit_joint_only"):
        pred = trainer.predict_experts(X[train_idx], e_hat[train_idx], ts[train_idx])
        tn, tt = pred[1], pred[2]
        oracle = np.full(n, -1, np.int64)
        oracle[train_idx] = (
            np.abs(tt - tau[train_idx]) < np.abs(tn - tau[train_idx])
        ).astype(np.int64)
        tr_loader, va_loader = build_loaders(X, T, Y, e_hat, ts, oracle, train_idx, val_idx)
        trainer.fit_joint_only(tr_loader, va_loader, verbose=verbose)
    return trainer, tb


def predict_tau(trainer, X, e, ts, idx):
    pred = trainer.predict_experts(X[idx], e[idx], ts[idx])
    return pred[0]


def deterministic_sup_offset(sup: str, mod: int = 89) -> int:
    return int(zlib.crc32(sup.encode("utf-8")) & 0xFFFFFFFF) % mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--reps", default="1-10")
    ap.add_argument("--ms", default="10,20,50,100")
    ap.add_argument("--supervisions", default="factual,dr_pseudo,oracle_tau")
    ap.add_argument("--pretrain_epochs", type=int, default=25)
    ap.add_argument("--joint_epochs", type=int, default=30)
    ap.add_argument("--n_updates", type=int, default=200)
    ap.add_argument("--noise_std", type=float, default=0.4)
    ap.add_argument("--signal_scale", type=float, default=2.5)
    ap.add_argument("--level_gap", type=float, default=3.0)
    ap.add_argument("--cal_frac", type=float, default=0.15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ms = [int(x) for x in args.ms.split(",") if x.strip()]
    supervisions = [s.strip() for s in args.supervisions.split(",") if s.strip()]
    reps: List[int] = []
    for part in args.reps.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            reps.extend(range(int(a), int(b) + 1))
        else:
            reps.append(int(part))

    rows: List[Dict[str, Any]] = []
    verbose = not args.quiet
    data_dir = Path(args.data_dir)

    for rep in reps:
        set_seed(rep)
        raw = load_ihdp_rep(rep, data_dir)
        X = np.asarray(raw["X_tr"], np.float32)
        T_base = np.asarray(raw["T_tr"], np.float32).ravel()
        Y_obs = np.asarray(raw["Y_tr"], np.float32).ravel()

        pair = make_geometry_pair_from_real(
            X, T_base, Y_obs, seed=rep,
            noise_std=args.noise_std,
            signal_scale=args.signal_scale,
            level_gap=args.level_gap,
        )
        Xs, Ts, Ys, es, taus = X, pair["T"], pair["Y_smooth"], pair["e"], pair["tau_smooth"]
        Xt, Tt, Yt, et, taut = X, pair["T"], pair["Y_threshold"], pair["e"], pair["tau_threshold"]

        idx_s = split_idx(len(Ys), rep)
        trainer, tb = train_on_source_world(
            Xs, Ts, Ys, es, taus,
            idx_s["train_idx"], idx_s["val_idx"],
            args.device, args.pretrain_epochs, args.joint_epochs, rep, verbose,
        )

        rng = np.random.default_rng(rep + 99)
        perm = rng.permutation(len(Yt))
        n_cal = max(30, int(args.cal_frac * len(Yt)))
        n_te = max(80, len(Yt) // 5)
        test_idx = perm[n_cal : n_cal + n_te]
        pool = perm[n_cal + n_te :]

        # Strict source-only tree signal transform
        ts_t = tb.transform(Xt)
        e_hat = ts_t[:, 3].copy()
        assert ((e_hat > 0) & (e_hat < 1)).all(), "estimated propensity out of (0,1)"

        tau_f_all = predict_tau(trainer, Xt, e_hat, ts_t, np.arange(len(Yt)))
        pehe_f_te = pehe(taut[test_idx], tau_f_all[test_idx])
        pehe_zero = pehe(taut[test_idx], np.zeros(len(test_idx)))
        # NOT deployable (uses true target tau); kept only as an upper reference.
        pehe_const = pehe(taut[test_idx], np.full(len(test_idx), float(np.mean(taut[pool]))))
        # Deployable constant: DR-estimated ATE in the SOURCE world, no target labels.
        s_tr = idx_s["train_idx"]
        src_ate = source_ate_estimate(Xs[s_tr], Ts[s_tr], Ys[s_tr], seed=rep)
        pehe_src_ate = pehe(taut[test_idx], np.full(len(test_idx), src_ate))

        print(
            f"rep={rep}  frozen={pehe_f_te:.4f}  zero={pehe_zero:.4f}  "
            f"const_oracle={pehe_const:.4f}  source_ate={pehe_src_ate:.4f}"
        )

        # DR only on pool
        tau_dr = np.zeros(len(Yt), dtype=np.float32)
        tau_dr[pool] = dr_pseudo_tau_util(
            Xt[pool], Tt[pool], Yt[pool], n_splits=3, seed=rep, smooth=False
        )

        for m in ms:
            for sup in supervisions:
                rng_m = np.random.default_rng(
                    rep * 10007 + m * 17 + deterministic_sup_offset(sup, 89)
                )
                sel = rng_m.choice(pool, size=min(m, len(pool)), replace=False)

                if sup == "oracle_tau":
                    tau_lab = taut[sel]
                    mode = "full_adapt"
                elif sup == "dr_pseudo":
                    tau_lab = tau_dr[sel]
                    mode = "full_adapt"
                else:
                    tau_lab = None
                    mode = "supervised_factual"

                model_ad = adapt_fewshot(
                    trainer.model,
                    Xt[sel], Tt[sel], Yt[sel], e_hat[sel], ts_t[sel],
                    tau_k=tau_lab,
                    mode=mode,
                    device=args.device,
                    n_updates=args.n_updates,
                    entropy_weight=0.0,
                    seed=int(rng_m.integers(0, 2**31 - 1)),
                )
                t_ad = GeometryAdaptiveTrainer(
                    model_ad, device=args.device, pretrain_epochs=0, joint_epochs=0
                )
                t_ad.device = args.device
                pehe_always = pehe(
                    taut[test_idx],
                    predict_tau(t_ad, Xt, e_hat, ts_t, test_idx),
                )

                rows.append({
                    "rep": rep,
                    "m": m,
                    "supervision": sup,
                    "pehe_frozen": pehe_f_te,
                    "pehe_always_adapt": pehe_always,
                    "delta_always": pehe_f_te - pehe_always,
                    "pehe_zero": pehe_zero,
                    "pehe_const_oracle": pehe_const,
                    "pehe_source_ate": pehe_src_ate,
                    "n_updates": args.n_updates,
                })

    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    df.to_csv(out / "table6_ihdp_x_ladder_raw.csv", index=False)
    g = (
        df.groupby(["m", "supervision"])[
            ["pehe_frozen", "pehe_always_adapt", "delta_always", "pehe_zero", "pehe_const_oracle", "pehe_source_ate"]
        ]
        .agg(["mean", "std"])
    )
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    g.to_csv(out / "table6_ihdp_x_ladder_summary.csv", index=False)
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
