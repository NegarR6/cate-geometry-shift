"""
Path β Stage B1 — Predict neural vs tree winner on ACIC from train-only D_G.

Protocol (locked):
  1) Fit NeuralExpert + TreeExpert on TRAIN only
  2) Compute D_G on TRAIN only (geometry_proxy v2)
  3) Predict: D_G > t => tree wins, else neural wins
  4) t from synthetic continuum (default) — NOT from test PEHE
  5) True label = argmin PEHE on TEST between neural and tree
  6) Baselines: always_neural, always_tree, random

Usage:
  python -m src.run_geometry_proxy_acic --settings 1,4,5,8 --n_seeds 3 --device cuda \\
      --threshold -0.33 --out_dir results/geometry_proxy_acic
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from geometry_proxy import compute_D_G, fit_norm_from_component_rows, norm_from_synthetic_csv, _z
from models_geometry_adaptive import build_geo_adaptive_from_hparams
from train_geometry_adaptive import GeometryAdaptiveTrainer, set_seed, build_loaders
from tree_signals import CrossFittedTreeSignals

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_acic_setting(
    setting_id: int,
    seed: int,
    data_dir: Path,
    test_size: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Try canonical loader; else local ACIC2016 CSV fallback."""
    try:
        from canonical_data_loader import load_acic_canonical
        data = load_acic_canonical(setting_id=setting_id, seed=seed, scale_features=True)
        # expected keys vary; normalize
        X = np.asarray(data["X"], dtype=np.float64)
        T = np.asarray(data.get("T", data.get("w", data.get("t"))), dtype=np.float64).ravel()
        Y = np.asarray(data.get("Y", data.get("y")), dtype=np.float64).ravel()
        if "tau" in data:
            tau = np.asarray(data["tau"], dtype=np.float64).ravel()
        elif "mu0" in data and "mu1" in data:
            tau = np.asarray(data["mu1"] - data["mu0"], dtype=np.float64).ravel()
        else:
            raise KeyError("no tau/mu in canonical acic")
        if "train_idx" in data and "test_idx" in data:
            tr, te = np.asarray(data["train_idx"]), np.asarray(data["test_idx"])
        else:
            n = len(Y)
            tr, te = train_test_split(
                np.arange(n), test_size=test_size, random_state=seed
            )
            tr, te = np.asarray(tr), np.asarray(te)
        e = np.full(len(T), float(np.clip(T.mean(), 0.05, 0.95)), dtype=np.float64)
        return {
            "X": X.astype(np.float32),
            "T": T.astype(np.float32),
            "Y": Y.astype(np.float32),
            "e": e.astype(np.float32),
            "tau": tau.astype(np.float32),
            "train_idx": tr,
            "test_idx": te,
        }
    except Exception as ex:
        print(f"  [info] canonical ACIC unavailable ({ex}); using local CSV fallback")

    x_path = data_dir / "x.csv"
    z_path = data_dir / f"zymu_{setting_id}.csv"
    if not x_path.exists() or not z_path.exists():
        raise FileNotFoundError(
            f"ACIC files not found under {data_dir}. "
            f"Need x.csv and zymu_{setting_id}.csv"
        )

    Xdf = pd.read_csv(x_path)
    Zdf = pd.read_csv(z_path)
    # one-hot categoricals
    X = pd.get_dummies(Xdf, drop_first=True).astype(np.float64).values
    T = Zdf["z"].values.astype(np.float64)
    # factual outcome
    Y = np.where(T > 0.5, Zdf["y1"].values, Zdf["y0"].values).astype(np.float64)
    tau = (Zdf["mu1"].values - Zdf["mu0"].values).astype(np.float64)

    n = len(Y)
    tr, te = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    tr, te = np.asarray(tr), np.asarray(te)

    scaler = StandardScaler()
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


def pehe(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def run_one_setting_seed(
    setting_id: int,
    seed: int,
    data_dir: Path,
    device: str,
    pretrain_epochs: int,
    joint_epochs: int,
    threshold: float,
    verbose: bool,
    norm=None,
) -> Dict[str, Any]:
    set_seed(seed + 1000 * setting_id)
    data = load_acic_setting(setting_id, seed, data_dir)
    X, T, Y, e, tau = data["X"], data["T"], data["Y"], data["e"], data["tau"]
    tr, te = data["train_idx"], data["test_idx"]
    n, p = X.shape

    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr)
    n_val = max(80, int(0.15 * len(perm)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tb = CrossFittedTreeSignals(n_splits=3, seed=seed)
    ts = np.zeros((n, 4), dtype=np.float32)
    ts[tr_idx] = tb.fit_transform(X[tr_idx], T[tr_idx], Y[tr_idx])
    ts[val_idx] = tb.transform(X[val_idx])
    ts[te] = tb.transform(X[te])

    model = build_geo_adaptive_from_hparams(num_features=p)
    trainer = GeometryAdaptiveTrainer(
        model,
        device=device,
        pretrain_epochs=pretrain_epochs,
        joint_epochs=joint_epochs,
        freeze_experts_epochs=6,
        route_weight_start=0.15,
        route_weight_end=0.4,
    )
    train_loader, val_loader = build_loaders(X, T, Y, e, ts, None, tr_idx, val_idx)
    trainer.pretrain_experts(train_loader, val_loader, verbose=verbose)

    # train expert preds for D_G
    _, tau_n_tr, tau_t_tr, _ = trainer.predict_experts(X[tr_idx], e[tr_idx], ts[tr_idx])
    proxy = compute_D_G(
        X[tr_idx], T[tr_idx], Y[tr_idx], tau_n_tr, tau_t_tr, norm=norm
    )
    dis = proxy.components["disagreement"]
    stump = proxy.components["stump_gain"]
    rough = proxy.components["roughness"]
    D_G = float(proxy.D_G)

    # test PEHE for true winner
    _, tau_n_te, tau_t_te, _ = trainer.predict_experts(X[te], e[te], ts[te])
    pn, pt = pehe(tau[te], tau_n_te), pehe(tau[te], tau_t_te)
    true_winner = "tree" if pt < pn else "neural"
    pred_winner = "tree" if D_G > threshold else "neural"
    correct = int(pred_winner == true_winner)

    return {
        "setting": setting_id,
        "seed": seed,
        "pehe_neural": pn,
        "pehe_tree": pt,
        "margin_tree_minus_neural": pt - pn,
        "D_G": D_G,
        "c_disagreement": dis,
        "c_stump_gain": stump,
        "c_roughness": rough,
        "threshold": threshold,
        "true_winner": true_winner,
        "pred_winner": pred_winner,
        "correct": correct,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te)),
        "p": int(p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=str, default="1,4,5,8")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--pretrain_epochs", type=int, default=30)
    ap.add_argument("--joint_epochs", type=int, default=0,
                    help="0 = experts only (enough for ranking); >0 optional joint")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--threshold",
        type=float,
        default=-0.33,
        help="Synthetic median D_G at lambda~0.5 from B0 v2 (z-scored runs used -0.33). "
             "For RAW score (stump+dis-rough) you may need a different scale — "
             "see --auto_threshold_from_rows.",
    )
    ap.add_argument(
        "--auto_threshold",
        action="store_true",
        help="Set t = median(D_G) over all setting×seed runs (still no test PEHE). "
             "Use only as sensitivity; primary is fixed synthetic threshold.",
    )
    ap.add_argument("--data_dir", type=str, default="data/acic2016")
    ap.add_argument("--out_dir", type=str, default="results/geometry_proxy_acic")
    ap.add_argument(
        "--synthetic_csv",
        type=str,
        default="results/geometry_proxy_synthetic_v2_s5/proxy_AB_raw.csv",
        help="B0 CSV to fit the same z-norm as synthetic continuum",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    settings = [int(s) for s in args.settings.split(",") if s.strip()]
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        alt = ROOT / "data" / "acic2016"
        if alt.exists():
            data_dir = alt

    verbose = not args.quiet
    rows: List[Dict[str, Any]] = []

    norm = None
    syn = Path(args.synthetic_csv)
    if not syn.is_absolute():
        syn = ROOT / syn
    if syn.exists():
        norm = norm_from_synthetic_csv(str(syn))
        print(f"[norm] loaded z-norm from {syn}")
        print(f"       stump={norm['stump_gain']} dis={norm['disagreement']} rough={norm['roughness']}")
    else:
        print(f"[warn] synthetic CSV not found at {syn}; using RAW D_G (threshold scale may be off)")

    for setting in settings:
        for seed in range(args.n_seeds):
            print(f"\n===== ACIC setting={setting} seed={seed} =====")
            row = run_one_setting_seed(
                setting, seed, data_dir, args.device,
                args.pretrain_epochs, args.joint_epochs,
                threshold=args.threshold, verbose=verbose,
                norm=norm,
            )
            rows.append(row)
            print(
                f"  D_G={row['D_G']:.4f}  pn={row['pehe_neural']:.4f}  pt={row['pehe_tree']:.4f}  "
                f"true={row['true_winner']} pred={row['pred_winner']} ok={row['correct']}"
            )

    df = pd.DataFrame(rows)

    # Optional: re-threshold at median D_G across runs (no test leakage)
    if args.auto_threshold:
        t_auto = float(df["D_G"].median())
        df["threshold_auto"] = t_auto
        df["pred_winner_auto"] = np.where(df["D_G"] > t_auto, "tree", "neural")
        df["correct_auto"] = (df["pred_winner_auto"] == df["true_winner"]).astype(int)
        print(f"\n[auto_threshold] median D_G = {t_auto:.4f}")

    raw_path = os.path.join(args.out_dir, "proxy_acic_raw.csv")
    df.to_csv(raw_path, index=False)

    # Metrics for fixed threshold
    acc = float(df["correct"].mean())
    # baselines
    always_neural = float((df["true_winner"] == "neural").mean())
    always_tree = float((df["true_winner"] == "tree").mean())
    majority = max(always_neural, always_tree)

    summary = {
        "n_rows": len(df),
        "settings": str(settings),
        "threshold": args.threshold,
        "accuracy": acc,
        "frac_true_neural": always_neural,
        "frac_true_tree": always_tree,
        "majority_baseline": majority,
        "beats_majority": acc > majority + 1e-9,
        "beats_random": acc > 0.5 + 1e-9,
        "go_acc_ge_0.70": acc >= 0.70,
        "go_beats_majority": acc > majority + 1e-9,
    }
    if args.auto_threshold:
        acc_a = float(df["correct_auto"].mean())
        summary["accuracy_auto"] = acc_a
        summary["threshold_auto"] = float(df["threshold_auto"].iloc[0])
        summary["go_auto_acc_ge_0.70"] = acc_a >= 0.70

    # per-setting accuracy
    by = df.groupby("setting")["correct"].mean()
    print("\n=== Per-setting accuracy (fixed threshold) ===")
    print(by.to_string())
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    pd.DataFrame([summary]).to_csv(
        os.path.join(args.out_dir, "proxy_acic_summary.csv"), index=False
    )
    by.reset_index().to_csv(
        os.path.join(args.out_dir, "proxy_acic_by_setting.csv"), index=False
    )
    print(f"\nSaved {raw_path}")
    print(
        "GO B1? accuracy>=0.70 and beats majority:",
        summary["go_acc_ge_0.70"] and summary["go_beats_majority"],
    )


if __name__ == "__main__":
    main()
