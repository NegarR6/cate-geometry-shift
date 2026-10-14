"""
Attack Plan α — Geometry continuum with REAL NeuralExpert + TreeExpert.

For each λ in continuum A→B (or A→C):
  Stage1 pretrain experts independently on τ_λ data
  Stage2 joint soft mixture
  Report PEHE: neural, tree, adaptive, uniform avg, xlearner (if available)
  Track margin tree-neural and win rates

Smoke (GPU recommended):
  python -m src.run_geometry_continuum_experts --n_seeds 1 --lambdas 0,0.5,1 --n 2000 --pretrain_epochs 25 --joint_epochs 30 --device cuda --out_dir results/geometry_continuum_experts_smoke

Full:
  python -m src.run_geometry_continuum_experts --n_seeds 5 --pair A-B --n 3000 --device cuda --out_dir results/geometry_continuum_experts
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from synthetic_regimes import compute_all_norm_constants, generate_continuum
from models_geometry_adaptive import build_geo_adaptive_from_hparams
from train_geometry_adaptive import GeometryAdaptiveTrainer, set_seed, build_loaders
from tree_signals import CrossFittedTreeSignals

try:
    from baselines import XLearnerXGBoost
    _HAS_XL = True
except Exception:
    _HAS_XL = False


def pehe(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_one_lam(
    lam: float,
    seed: int,
    n: int,
    p: int,
    signal: float,
    noise: float,
    pair: str,
    device: str,
    pretrain_epochs: int,
    joint_epochs: int,
    batch_size: int,
    verbose: bool,
) -> Dict[str, Any]:
    set_seed(seed)
    ra, rb = pair.split("-")
    norms = compute_all_norm_constants(p=p)
    data, meta = generate_continuum(
        n=n, p=p, lam=lam, seed=seed, noise_std=noise, signal=signal,
        regime_a=ra, regime_b=rb, norm_constants=norms,
    )
    X, T, Y, e, tau = data["X"], data["T"], data["Y"], data["e"], data["tau"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]

    rng = np.random.default_rng(seed + 123)
    perm = rng.permutation(train_idx)
    n_val = max(80, int(0.15 * len(perm)))
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])

    tree_builder = CrossFittedTreeSignals(n_splits=3, seed=seed)
    tree_sig = np.zeros((len(X), 4), dtype=np.float32)
    tree_sig[tr_idx] = tree_builder.fit_transform(X[tr_idx], T[tr_idx], Y[tr_idx])
    tree_sig[val_idx] = tree_builder.transform(X[val_idx])
    tree_sig[test_idx] = tree_builder.transform(X[test_idx])

    model = build_geo_adaptive_from_hparams(num_features=p)
    trainer = GeometryAdaptiveTrainer(
        model,
        device=device,
        lr=3.3e-4,
        weight_decay=1e-5,
        ipm_weight=0.19,
        ipm_type="mean_diff",
        route_weight_start=0.15,
        route_weight_end=0.55,
        route_warmup_epochs=10,
        entropy_weight=0.015,
        pretrain_epochs=pretrain_epochs,
        pretrain_patience=10,
        joint_epochs=joint_epochs,
        joint_patience=12,
        freeze_experts_epochs=6,
        expert_ft_lr_mult=0.3,
    )

    tl, vl = build_loaders(X, T, Y, e, tree_sig, None, tr_idx, val_idx, batch_size=batch_size)
    trainer.pretrain_experts(tl, vl, verbose=verbose)

    oracle_full = np.full(len(X), -1, dtype=np.int64)
    oracle_idx = np.concatenate([tr_idx, val_idx])
    oracle_full[oracle_idx] = trainer.build_oracle_from_experts(
        X[oracle_idx], e[oracle_idx], tree_sig[oracle_idx], tau[oracle_idx]
    )
    if verbose:
        pref_tree = float((oracle_full[oracle_idx] == 1).mean())
        print(f"  λ={lam:.2f} oracle_pref_tree={pref_tree:.3f}")

    tl, vl = build_loaders(X, T, Y, e, tree_sig, oracle_full, tr_idx, val_idx, batch_size=batch_size)
    trainer.fit_joint_only(tl, vl, verbose=verbose)

    tau_ad, tau_n, tau_t, weights = trainer.predict_experts(
        X[test_idx], e[test_idx], tree_sig[test_idx]
    )
    tau_true = tau[test_idx]
    pe_n = pehe(tau_true, tau_n)
    pe_t = pehe(tau_true, tau_t)
    pe_ad = pehe(tau_true, tau_ad)
    pe_avg = pehe(tau_true, 0.5 * (tau_n + tau_t))

    pe_x = float("nan")
    if _HAS_XL:
        try:
            xl = XLearnerXGBoost(random_state=seed)
            xl.fit(X[tr_idx], T[tr_idx], Y[tr_idx])
            pe_x = pehe(tau_true, xl.predict_cate(X[test_idx]))
        except Exception:
            pass

    pehes = {
        "neural_expert": pe_n,
        "tree_expert": pe_t,
        "adaptive": pe_ad,
        "uniform_avg": pe_avg,
    }
    if not np.isnan(pe_x):
        pehes["xlearner"] = pe_x
    winner = min(pehes, key=pehes.get)

    return {
        "pair": pair,
        "lam": lam,
        "seed": seed,
        "winner": winner,
        "pehe_neural": pe_n,
        "pehe_tree": pe_t,
        "pehe_adaptive": pe_ad,
        "pehe_uniform_avg": pe_avg,
        "pehe_xlearner": pe_x,
        "margin_tree_minus_neural": pe_t - pe_n,
        "mean_w_neural": float(weights[:, 0].mean()),
        "mean_w_tree": float(weights[:, 1].mean()),
        "oracle_pref_tree": float((oracle_full[oracle_idx] == 1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=str, default="A-B")
    ap.add_argument("--lambdas", type=str, default="")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--p", type=int, default=12)
    ap.add_argument("--signal", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--pretrain_epochs", type=int, default=35)
    ap.add_argument("--joint_epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_dir", type=str, default="results/geometry_continuum_experts")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.lambdas.strip():
        lams = [float(x) for x in args.lambdas.split(",") if x.strip()]
    else:
        lams = [round(i * 0.25, 2) for i in range(5)]  # 0,0.25,0.5,0.75,1 default full grid finer optional

    os.makedirs(args.out_dir, exist_ok=True)
    verbose = not args.quiet
    print("=" * 70)
    print(f"CONTINUUM EXPERTS pair={args.pair} λ={lams} seeds={args.n_seeds} device={args.device}")
    print("=" * 70)

    rows: List[Dict] = []
    for seed in range(args.n_seeds):
        for lam in lams:
            print(f"\n----- seed={seed} λ={lam:.2f} -----")
            r = run_one_lam(
                lam, seed, args.n, args.p, args.signal, args.noise, args.pair,
                args.device, args.pretrain_epochs, args.joint_epochs, args.batch_size, verbose,
            )
            rows.append(r)
            print(
                f"  RESULT n={r['pehe_neural']:.4f} t={r['pehe_tree']:.4f} "
                f"ad={r['pehe_adaptive']:.4f} avg={r['pehe_uniform_avg']:.4f} "
                f"xl={r['pehe_xlearner']:.4f} margin={r['margin_tree_minus_neural']:+.4f} "
                f"w_n={r['mean_w_neural']:.2f} win={r['winner']}"
            )

    df = pd.DataFrame(rows)
    tag = args.pair.replace("-", "")
    raw = os.path.join(args.out_dir, f"continuum_experts_{tag}_raw.csv")
    df.to_csv(raw, index=False)

    metric_cols = [
        "pehe_neural", "pehe_tree", "pehe_adaptive", "pehe_uniform_avg",
        "pehe_xlearner", "margin_tree_minus_neural", "mean_w_neural", "oracle_pref_tree",
    ]
    summary = df.groupby("lam")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    sum_path = os.path.join(args.out_dir, f"continuum_experts_{tag}_summary.csv")
    summary.to_csv(sum_path, index=False)

    win = df.groupby(["lam", "winner"]).size().unstack(fill_value=0)
    win = win.div(win.sum(axis=1), axis=0)
    win_path = os.path.join(args.out_dir, f"continuum_experts_{tag}_winrate.csv")
    win.to_csv(win_path)

    print("\n=== Mean by λ ===")
    show = [c for c in summary.columns if c == "lam" or c.endswith("_mean")]
    print(summary[show].round(4).to_string(index=False))
    print("\n=== Win rates ===")
    print(win.round(3).to_string())
    print(f"\nSaved → {args.out_dir}")


if __name__ == "__main__":
    main()
