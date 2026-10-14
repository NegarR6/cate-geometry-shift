"""
Experiment runner for Geometry-Adaptive CATE — TWO-STAGE PROTOCOL

Stage 1: pre-train NeuralExpert & TreeExpert independently
Stage 2: rebuild oracle from pre-trained experts → joint training with curriculum

PowerShell (single line):
  python -m src.run_geometry_adaptive_regimes --n_seeds 3 --signal 1.0 --pretrain_epochs 35 --joint_epochs 50 --device cuda --out_dir results/geometry_adaptive_v2
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from synthetic_regimes import SyntheticConfig, generate_synthetic, compute_all_norm_constants
from models_geometry_adaptive import build_geo_adaptive_from_hparams
from train_geometry_adaptive import GeometryAdaptiveTrainer, set_seed, build_loaders
from tree_signals import CrossFittedTreeSignals
try:
    from adapt_geometry_shift import evaluate_shift_with_adaptation
except ImportError:
    from src.adapt_geometry_shift import evaluate_shift_with_adaptation

try:
    from baselines import XLearnerXGBoost
    _HAS_XLEARNER = True
except Exception:
    _HAS_XLEARNER = False


def pehe(tau_true: np.ndarray, tau_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau_true - tau_pred) ** 2)))


def fit_xlearner(X, T, Y, seed: int):
    if not _HAS_XLEARNER:
        return None
    m = XLearnerXGBoost(random_state=seed)
    m.fit(X, T, Y)
    return m


def run_single_seed(
    regime: str,
    signal: float,
    seed: int,
    n: int = 3000,
    p: int = 12,
    noise_std: float = 0.5,
    device: str = "cpu",
    pretrain_epochs: int = 35,
    joint_epochs: int = 50,
    batch_size: int = 64,
    use_oracle_route: bool = True,
    hard_inference: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    set_seed(seed)
    norm = compute_all_norm_constants(p=p)
    cfg = SyntheticConfig(n=n, p=p, noise_std=noise_std, seed=seed, regime=regime, signal=signal)
    data, _ = generate_synthetic(cfg, norm_constants=norm)

    X, T, Y, e, tau = data["X"], data["T"], data["Y"], data["e"], data["tau"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]

    rng = np.random.default_rng(seed + 123)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(0.2 * len(perm)))
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])

    # OOF tree signals
    tree_builder = CrossFittedTreeSignals(n_splits=5, seed=seed)
    tree_sig_full = np.zeros((len(X), 4), dtype=np.float32)
    tree_sig_full[tr_idx] = tree_builder.fit_transform(X[tr_idx], T[tr_idx], Y[tr_idx])
    tree_sig_full[val_idx] = tree_builder.transform(X[val_idx])
    tree_sig_full[test_idx] = tree_builder.transform(X[test_idx])

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
        route_warmup_epochs=12,
        entropy_weight=0.015,
        pretrain_epochs=pretrain_epochs,
        pretrain_patience=12,
        joint_epochs=joint_epochs,
        joint_patience=15,
        freeze_experts_epochs=8,
        expert_ft_lr_mult=0.3,
    )

    # Stage 1 loaders (no oracle)
    train_loader, val_loader = build_loaders(
        X, T, Y, e, tree_sig_full, None, tr_idx, val_idx, batch_size=batch_size
    )
    trainer.pretrain_experts(train_loader, val_loader, verbose=verbose)

    # Rebuild oracle from pre-trained experts
    if use_oracle_route:
        oracle_idx = np.concatenate([tr_idx, val_idx])
        oracle_full = np.full(len(X), -1, dtype=np.int64)
        oracle_part = trainer.build_oracle_from_experts(
            X[oracle_idx], e[oracle_idx], tree_sig_full[oracle_idx], tau[oracle_idx]
        )
        oracle_full[oracle_idx] = oracle_part
        if verbose:
            print(f"  Oracle fraction preferring tree: {float((oracle_part == 1).mean()):.3f}")
    else:
        oracle_full = None

    train_loader, val_loader = build_loaders(
        X, T, Y, e, tree_sig_full, oracle_full, tr_idx, val_idx, batch_size=batch_size
    )
    trainer.fit_joint_only(train_loader, val_loader, verbose=verbose)

    # Test
    tau_adapt, tau_n, tau_t, weights = trainer.predict_experts(
        X[test_idx], e[test_idx], tree_sig_full[test_idx]
    )
    if hard_inference:
        tau_adapt = trainer.predict_cate(
            X[test_idx], e[test_idx], tree_sig_full[test_idx], hard=True
        )

    tau_x = np.full(len(test_idx), np.nan)
    if _HAS_XLEARNER:
        xl = fit_xlearner(X[tr_idx], T[tr_idx], Y[tr_idx], seed)
        if xl is not None:
            tau_x = xl.predict_cate(X[test_idx])

    tau_true = tau[test_idx]
    return {
        "regime": regime,
        "signal": signal,
        "seed": seed,
        "pehe_adaptive": pehe(tau_true, tau_adapt),
        "pehe_neural_expert": pehe(tau_true, tau_n),
        "pehe_tree_expert": pehe(tau_true, tau_t),
        "pehe_xlearner": pehe(tau_true, tau_x) if not np.isnan(tau_x).any() else None,
        "pehe_average": pehe(tau_true, 0.5 * (tau_n + tau_t)),
        "mean_weight_neural": float(weights[:, 0].mean()),
        "mean_weight_tree": float(weights[:, 1].mean()),
        "router_entropy": float(
            -(weights * np.log(np.clip(weights, 1e-8, 1.0))).sum(1).mean()
        ),
    }


def run_shift_experiment(
    train_regimes: List[str],
    test_regime: str,
    signal: float,
    seed: int,
    n: int = 3000,
    p: int = 12,
    device: str = "cpu",
    pretrain_epochs: int = 30,
    joint_epochs: int = 40,
    verbose: bool = True,
    adapt_k: int = 0,
    adapt_mode: str = "supervised_oracle",
    adapt_steps: int = 80,
) -> Dict[str, Any]:
    set_seed(seed)
    norm = compute_all_norm_constants(p=p)

    Xs, Ts, Ys, es, taus, tree_sigs = [], [], [], [], [], []
    for r in train_regimes:
        cfg = SyntheticConfig(n=n // len(train_regimes), p=p, seed=seed, regime=r, signal=signal)
        data, _ = generate_synthetic(cfg, norm_constants=norm)
        Xs.append(data["X"]); Ts.append(data["T"]); Ys.append(data["Y"])
        es.append(data["e"]); taus.append(data["tau"])
        tb = CrossFittedTreeSignals(seed=seed)
        tree_sigs.append(tb.fit_transform(data["X"], data["T"], data["Y"]))

    X = np.vstack(Xs)
    T = np.concatenate(Ts)
    Y = np.concatenate(Ys)
    e = np.concatenate(es)
    tau = np.concatenate(taus)
    tree_sig = np.vstack(tree_sigs)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(0.15 * len(idx)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = build_geo_adaptive_from_hparams(num_features=p)
    trainer = GeometryAdaptiveTrainer(
        model, device=device,
        pretrain_epochs=pretrain_epochs, joint_epochs=joint_epochs,
        freeze_experts_epochs=6, route_weight_start=0.15, route_weight_end=0.5,
    )

    train_loader, val_loader = build_loaders(X, T, Y, e, tree_sig, None, tr_idx, val_idx)
    trainer.pretrain_experts(train_loader, val_loader, verbose=verbose)

    oracle_full = np.full(len(X), -1, dtype=np.int64)
    oracle_full[tr_idx] = trainer.build_oracle_from_experts(
        X[tr_idx], e[tr_idx], tree_sig[tr_idx], tau[tr_idx]
    )
    train_loader, val_loader = build_loaders(X, T, Y, e, tree_sig, oracle_full, tr_idx, val_idx)
    trainer.fit_joint_only(train_loader, val_loader, verbose=verbose)

    cfg_te = SyntheticConfig(n=n, p=p, seed=seed + 999, regime=test_regime, signal=signal)
    data_te, _ = generate_synthetic(cfg_te, norm_constants=norm)
    tb_te = CrossFittedTreeSignals(seed=seed + 1)
    n_te = len(data_te["X"])
    te_fit = np.arange(min(500, n_te // 3))
    ts_te = np.zeros((n_te, 4), dtype=np.float32)
    ts_te[te_fit] = tb_te.fit_transform(
        data_te["X"][te_fit], data_te["T"][te_fit], data_te["Y"][te_fit]
    )
    rest = np.setdiff1d(np.arange(n_te), te_fit)
    if len(rest):
        ts_te[rest] = tb_te.transform(data_te["X"][rest])

    # Full baseline suite under shift
    tau_adapt, tau_n, tau_t, weights = trainer.predict_experts(
        data_te["X"], data_te["e"], ts_te
    )
    tau_true = data_te["tau"]

    # X-Learner trained only on source regimes
    tau_x = np.full(len(tau_true), np.nan)
    if _HAS_XLEARNER:
        xl = fit_xlearner(X[tr_idx], T[tr_idx], Y[tr_idx], seed)
        if xl is not None:
            tau_x = xl.predict_cate(data_te["X"])

    base = {
        "train_regimes": "".join(train_regimes),
        "test_regime": test_regime,
        "signal": signal,
        "seed": seed,
        "pehe_adaptive_shift": pehe(tau_true, tau_adapt),
        "pehe_neural_shift": pehe(tau_true, tau_n),
        "pehe_tree_shift": pehe(tau_true, tau_t),
        "pehe_average_shift": pehe(tau_true, 0.5 * (tau_n + tau_t)),
        "pehe_xlearner_shift": pehe(tau_true, tau_x) if not np.isnan(tau_x).any() else None,
        "mean_weight_neural_shift": float(weights[:, 0].mean()),
    }
    if adapt_k and adapt_k > 0:
        adapt_stats = evaluate_shift_with_adaptation(
            model,
            data_te["X"], data_te["T"], data_te["Y"], data_te["e"], ts_te, data_te["tau"],
            k=adapt_k, adapt_mode=adapt_mode, adapt_steps=adapt_steps,
            seed=seed, device=device, verbose=verbose,
        )
        base.update({
            "pehe_before_adapt": adapt_stats["pehe_before"],
            "pehe_after_adapt": adapt_stats["pehe_after"],
            "delta_pehe_adapt": adapt_stats["delta_pehe"],
            "w_neural_after_adapt": adapt_stats["w_neural_after"],
            "adapt_k": adapt_stats["k"],
            "adapt_mode": adapt_stats["adapt_mode"],
        })
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--signal", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--p", type=int, default=12)
    parser.add_argument("--pretrain_epochs", type=int, default=35)
    parser.add_argument("--joint_epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="results/geometry_adaptive_v2")
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--train_regimes", nargs="+", default=["A", "B"])
    parser.add_argument("--test_regime", type=str, default="C")
    parser.add_argument("--no_oracle_route", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--adapt_k", type=int, default=0,
                        help="If >0 under --shift, run few-shot router adaptation with k target samples")
    parser.add_argument("--adapt_mode", type=str, default="full_adapt",
                        choices=["full_adapt", "router_only", "supervised_oracle",
                                 "supervised_factual", "unsupervised_tta"],
                        help="full_adapt = router + expert FT (strong); router_only = freeze experts")
    parser.add_argument("--adapt_steps", type=int, default=150)
    parser.add_argument("--adapt_ks", type=str, default="",
                        help="Comma-separated k values for sweep, e.g. 25,50,100,200 (overrides --adapt_k)")
    parser.add_argument("--adapt_modes", type=str, default="",
                        help="Comma-separated modes for ablation, e.g. full_adapt,router_only,supervised_factual")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> cpu")
        device = "cpu"
    verbose = not args.quiet

    if args.shift:
        ks = []
        if getattr(args, "adapt_ks", "") and str(args.adapt_ks).strip():
            ks = [int(x) for x in str(args.adapt_ks).split(",") if x.strip()]
        elif args.adapt_k and args.adapt_k > 0:
            ks = [args.adapt_k]
        else:
            ks = [0]

        modes = []
        if getattr(args, "adapt_modes", "") and str(args.adapt_modes).strip():
            modes = [m.strip() for m in str(args.adapt_modes).split(",") if m.strip()]
        else:
            modes = [args.adapt_mode]

        rows = []
        for seed in range(args.n_seeds):
            print(f"\n[SHIFT] seed={seed} train={args.train_regimes} -> test={args.test_regime}")
            base_row = run_shift_experiment(
                args.train_regimes, args.test_regime, args.signal, seed,
                n=args.n, p=args.p, device=device,
                pretrain_epochs=args.pretrain_epochs, joint_epochs=args.joint_epochs,
                verbose=verbose,
                adapt_k=0, adapt_mode=modes[0], adapt_steps=args.adapt_steps,
            )
            print(
                f"  BASE  adapt={base_row['pehe_adaptive_shift']:.4f}  "
                f"neural={base_row['pehe_neural_shift']:.4f}  "
                f"tree={base_row['pehe_tree_shift']:.4f}  "
                f"w_n={base_row['mean_weight_neural_shift']:.2f}"
            )
            for mode in modes:
                for k in ks:
                    if k <= 0:
                        r = dict(base_row)
                        r["adapt_k"] = 0
                        r["adapt_mode"] = mode
                        rows.append(r)
                        continue
                    print(f"  -> adapt k={k} mode={mode}")
                    row = run_shift_experiment(
                        args.train_regimes, args.test_regime, args.signal, seed,
                        n=args.n, p=args.p, device=device,
                        pretrain_epochs=args.pretrain_epochs, joint_epochs=args.joint_epochs,
                        verbose=verbose,
                        adapt_k=k, adapt_mode=mode, adapt_steps=args.adapt_steps,
                    )
                    rows.append(row)
                    if "pehe_after_adapt" in row:
                        print(
                            f"     [{mode}] k={k} BEFORE={row['pehe_before_adapt']:.4f}  "
                            f"AFTER={row['pehe_after_adapt']:.4f}  "
                            f"DELTA={row['delta_pehe_adapt']:.4f}"
                        )
        df = pd.DataFrame(rows)
        path = os.path.join(
            args.out_dir,
            f"shift_{''.join(args.train_regimes)}_to_{args.test_regime}.csv",
        )
        df.to_csv(path, index=False)
        print(f"Saved {path}")
        if "pehe_after_adapt" in df.columns:
            keys = [c for c in ["adapt_mode", "adapt_k"] if c in df.columns]
            if keys:
                print("\n=== ablation / k-sweep summary ===")
                g = df.groupby(keys)[
                    ["pehe_before_adapt", "pehe_after_adapt", "delta_pehe_adapt"]
                ].agg(["mean", "std"])
                print(g.round(4).to_string())
        else:
            cols = [c for c in df.columns if c.startswith("pehe_")]
            print(df[cols].describe().round(4))
        return

    all_rows = []
    for regime in ["A", "B", "C"]:
        for seed in range(args.n_seeds):
            print(f"\n===== regime={regime} seed={seed} signal={args.signal} =====")
            row = run_single_seed(
                regime=regime, signal=args.signal, seed=seed, n=args.n, p=args.p,
                device=device, pretrain_epochs=args.pretrain_epochs,
                joint_epochs=args.joint_epochs,
                use_oracle_route=not args.no_oracle_route, verbose=verbose,
            )
            all_rows.append(row)
            print(
                f"  RESULT  adapt={row['pehe_adaptive']:.4f}  "
                f"neural={row['pehe_neural_expert']:.4f}  "
                f"tree={row['pehe_tree_expert']:.4f}  "
                f"xlearn={row['pehe_xlearner']}  "
                f"w_n={row['mean_weight_neural']:.2f}"
            )

    df = pd.DataFrame(all_rows)
    path = os.path.join(args.out_dir, f"regimes_signal{args.signal}.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved {path}")

    summary = (
        df.groupby("regime")[
            ["pehe_adaptive", "pehe_neural_expert", "pehe_tree_expert",
             "pehe_average", "mean_weight_neural"]
        ].agg(["mean", "std"]).round(4)
    )
    print("\n=== Summary (mean +/- std) ===")
    print(summary)
    summary.to_csv(os.path.join(args.out_dir, f"summary_signal{args.signal}.csv"))
    print("Summary saved.")


if __name__ == "__main__":
    main()
