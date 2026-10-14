"""
Priority 1 runner — FIXED for project API:
  - model.forward → dict with tau, tau_n, tau_t, weights [B,2]
  - trainer.predict_experts → (tau_a, tau_n, tau_t, weights)
  - adapt via adapt_geometry_shift.adapt_fewshot
"""
from __future__ import annotations
import zlib
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from budgeted_ladder_utils import (
    active_scores,
    dr_pseudo_tau,
    pehe,
    select_indices,
)

from synthetic_regimes import SyntheticConfig, compute_all_norm_constants, generate_synthetic
from models_geometry_adaptive import build_geo_adaptive_from_hparams
from train_geometry_adaptive import GeometryAdaptiveTrainer, set_seed, build_loaders
from tree_signals import CrossFittedTreeSignals
from adapt_geometry_shift import adapt_fewshot


def _parse_list(s: str, cast=str) -> List:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def make_shift_data(
    train_regimes: Sequence[str],
    test_regime: str,
    n: int,
    p: int,
    seed: int,
    signal: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    norm = compute_all_norm_constants(p=p)
    Xs, Ts, Ys, es, taus = [], [], [], [], []
    offset = 0
    src_train_parts = []
    for i, reg in enumerate(train_regimes):
        cfg = SyntheticConfig(n=n, p=p, seed=seed + 17 * (i + 1), regime=reg, signal=signal)
        d, _ = generate_synthetic(cfg, norm_constants=norm)
        Xs.append(d["X"])
        Ts.append(d["T"])
        Ys.append(d["Y"])
        es.append(d["e"])
        taus.append(d["tau"])
        src_train_parts.append(np.asarray(d["train_idx"]) + offset)
        offset += len(d["Y"])

    source = {
        "X": np.concatenate(Xs, axis=0).astype(np.float32),
        "T": np.concatenate(Ts, axis=0).astype(np.float32),
        "Y": np.concatenate(Ys, axis=0).astype(np.float32),
        "e": np.concatenate(es, axis=0).astype(np.float32),
        "tau": np.concatenate(taus, axis=0).astype(np.float32),
    }
    source["train_idx"] = np.concatenate(src_train_parts)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(source["train_idx"])
    n_val = max(80, int(0.12 * len(perm)))
    source["val_idx"] = perm[:n_val]
    source["tr_idx"] = perm[n_val:]

    cfg_t = SyntheticConfig(n=n, p=p, seed=seed + 99, regime=test_regime, signal=signal)
    target, _ = generate_synthetic(cfg_t, norm_constants=norm)
    return source, target


def train_source_model(
    source: Dict[str, np.ndarray],
    device: str,
    pretrain_epochs: int,
    joint_epochs: int,
    seed: int,
    verbose: bool,
) -> GeometryAdaptiveTrainer:
    X, T, Y, e = source["X"], source["T"], source["Y"], source["e"]
    tr_idx, val_idx = source["tr_idx"], source["val_idx"]
    p = X.shape[1]
    n = len(Y)

    tb = CrossFittedTreeSignals(n_splits=3, seed=seed)
    ts = np.zeros((n, 4), dtype=np.float32)
    ts[tr_idx] = tb.fit_transform(X[tr_idx], T[tr_idx], Y[tr_idx])
    ts[val_idx] = tb.transform(X[val_idx])
    source["tree_signals"] = ts
    source["tree_builder"] = tb

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
    train_loader, val_loader = build_loaders(X, T, Y, e, ts, None, tr_idx, val_idx)
    trainer.pretrain_experts(train_loader, val_loader, verbose=verbose)

    if joint_epochs > 0:
        try:
            oracle = np.full(n, -1, dtype=np.int64)
            pred = trainer.predict_experts(X[tr_idx], e[tr_idx], ts[tr_idx])
            tn, tt = pred[1], pred[2]
            tau_tr = source["tau"][tr_idx]
            prefer_tree = np.abs(tt - tau_tr) < np.abs(tn - tau_tr)
            oracle[tr_idx] = prefer_tree.astype(np.int64)
            train_loader, val_loader = build_loaders(X, T, Y, e, ts, oracle, tr_idx, val_idx)
            if hasattr(trainer, "fit_joint_only"):
                trainer.fit_joint_only(train_loader, val_loader, verbose=verbose)
            elif hasattr(trainer, "train_joint"):
                trainer.train_joint(train_loader, val_loader, verbose=verbose)
        except Exception as ex:
            print(f"  [warn] joint stage skipped: {ex}")
    return trainer


def unpack_predict(pred) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if len(pred) == 4:
        tau_a, tau_n, tau_t, ws = pred
        ws = np.asarray(ws)
        if ws.ndim == 2 and ws.shape[1] >= 1:
            w_n = ws[:, 0]
        else:
            w_n = ws.ravel()
        return (
            np.asarray(tau_a).ravel(),
            np.asarray(tau_n).ravel(),
            np.asarray(tau_t).ravel(),
            np.asarray(w_n).ravel(),
        )
    tau_a, tau_n, tau_t = pred[0], pred[1], pred[2]
    return (
        np.asarray(tau_a).ravel(),
        np.asarray(tau_n).ravel(),
        np.asarray(tau_t).ravel(),
        None,
    )


def run_cell(
    trainer_src: GeometryAdaptiveTrainer,
    target: Dict[str, np.ndarray],
    tb_transform,
    m: int,
    supervision: str,
    policy: str,
    seed: int,
    adapt_steps: int,
    device: str,
) -> Dict[str, Any]:
    X = np.asarray(target["X"], dtype=np.float32)
    T = np.asarray(target["T"], dtype=np.float32)
    Y = np.asarray(target["Y"], dtype=np.float32)
    e = np.asarray(target["e"], dtype=np.float32)
    tau = np.asarray(target["tau"], dtype=np.float32)
    pool = np.asarray(target["train_idx"])
    test = np.asarray(target["test_idx"])
    n = len(Y)

    ts = np.zeros((n, 4), dtype=np.float32)
    ts[pool] = tb_transform(X[pool])
    ts[test] = tb_transform(X[test])
    rest = np.setdiff1d(np.arange(n), np.unique(np.concatenate([pool, test])))
    if len(rest):
        ts[rest] = tb_transform(X[rest])

    tau_a, tau_n, tau_t, _ = unpack_predict(
        trainer_src.predict_experts(X[test], e[test], ts[test])
    )
    pehe_frozen = pehe(tau[test], tau_a)

    if m == 0 or supervision == "none":
        return {
            "m": 0,
            "supervision": "none",
            "policy": "none",
            "pehe_after": pehe_frozen,
            "pehe_frozen": pehe_frozen,
            "pehe_neural": pehe(tau[test], tau_n),
            "pehe_tree": pehe(tau[test], tau_t),
            "delta_vs_frozen": 0.0,
            "n_selected": 0,
        }

    tau_a_p, tau_n_p, tau_t_p, w_p = unpack_predict(
        trainer_src.predict_experts(X[pool], e[pool], ts[pool])
    )
    if len(tau_n_p) != len(pool):
        raise RuntimeError(f"score len {len(tau_n_p)} != pool {len(pool)}")

    if policy == "random":
        scores = np.zeros(len(pool), dtype=np.float64)
    else:
        mode = policy if policy in (
            "active_disagreement", "active_entropy", "active_hybrid"
        ) else "active_hybrid"
        scores = np.asarray(active_scores(tau_n_p, tau_t_p, w_p, mode=mode)).ravel()
        if len(scores) != len(pool):
            raise RuntimeError(f"scores {len(scores)} != pool {len(pool)}")

    rng = np.random.default_rng(seed + 13 * int(m) + 17 * ((zlib.crc32(str(policy).encode('utf-8')) & 0xFFFFFFFF) % 1000))
    sel = np.asarray(select_indices(pool, scores, m, policy, rng), dtype=int)

    tau_k = None
    mode = "supervised_factual"
    if supervision == "oracle_tau":
        tau_k = tau[sel]
        mode = "full_adapt"
    elif supervision == "dr_pseudo":
        tau_dr_full = dr_pseudo_tau(X, T, Y, n_splits=3, seed=seed, smooth=True)
        tau_k = np.asarray(tau_dr_full[sel], dtype=np.float32)
        mode = "full_adapt"
    elif supervision == "factual":
        tau_k = None
        mode = "supervised_factual"
    else:
        raise ValueError(supervision)

    model_adapted = adapt_fewshot(
        trainer_src.model,
        X[sel],
        T[sel],
        Y[sel],
        e[sel],
        ts[sel],
        tau_k=tau_k,
        mode=mode,
        device=device,
        steps=adapt_steps,
        lr_router=1e-3,
        lr_expert=3e-4,
    )

    t2 = GeometryAdaptiveTrainer(
        model_adapted,
        device=device,
        pretrain_epochs=0,
        joint_epochs=0,
    )
    if hasattr(trainer_src, "device"):
        t2.device = trainer_src.device

    tau_a2, tau_n2, tau_t2, _ = unpack_predict(
        t2.predict_experts(X[test], e[test], ts[test])
    )
    pehe_after = pehe(tau[test], tau_a2)

    return {
        "m": m,
        "supervision": supervision,
        "policy": policy,
        "pehe_after": pehe_after,
        "pehe_frozen": pehe_frozen,
        "pehe_neural": pehe(tau[test], tau_n2),
        "pehe_tree": pehe(tau[test], tau_t2),
        "delta_vs_frozen": float(pehe_frozen - pehe_after),
        "n_selected": int(len(sel)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_regimes", nargs="+", default=["A", "B"])
    ap.add_argument("--test_regime", default="C")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--p", type=int, default=12)
    ap.add_argument("--signal", type=float, default=1.0)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--ms", default="0,20,50,100")
    ap.add_argument("--supervisions", default="factual,dr_pseudo,oracle_tau")
    ap.add_argument("--policies", default="random,active_hybrid")
    ap.add_argument("--pretrain_epochs", type=int, default=30)
    ap.add_argument("--joint_epochs", type=int, default=40)
    ap.add_argument("--adapt_steps", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="results/budgeted_ladder")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ms = _parse_list(args.ms, int)
    supervisions = _parse_list(args.supervisions)
    policies = _parse_list(args.policies)
    verbose = not args.quiet

    rows: List[Dict[str, Any]] = []

    for seed in range(args.n_seeds):
        print(f"\n======== seed={seed} shift={args.train_regimes}->{args.test_regime} ========")
        set_seed(seed)
        source, target = make_shift_data(
            args.train_regimes, args.test_regime, args.n, args.p, seed, args.signal
        )
        trainer = train_source_model(
            source, args.device, args.pretrain_epochs, args.joint_epochs, seed, verbose
        )
        tb = source["tree_builder"]

        for m in ms:
            if m == 0:
                print("  cell m=0 sup=none pol=none")
                cell = run_cell(
                    trainer, target, tb.transform, 0, "none", "none",
                    seed, args.adapt_steps, args.device,
                )
                cell.update({
                    "seed": seed,
                    "train_regimes": ",".join(args.train_regimes),
                    "test_regime": args.test_regime,
                    "signal": args.signal,
                })
                rows.append(cell)
                print(f"    frozen PEHE={cell['pehe_after']:.4f}")
                continue

            for sup in supervisions:
                for pol in policies:
                    print(f"  cell m={m} sup={sup} pol={pol}")
                    try:
                        cell = run_cell(
                            trainer, target, tb.transform, m, sup, pol,
                            seed, args.adapt_steps, args.device,
                        )
                        print(
                            f"    after={cell['pehe_after']:.4f} "
                            f"frozen={cell['pehe_frozen']:.4f} "
                            f"delta={cell['delta_vs_frozen']:.4f}"
                        )
                    except Exception as ex:
                        import traceback
                        traceback.print_exc()
                        print(f"  [error] {ex}")
                        cell = {
                            "m": m,
                            "supervision": sup,
                            "policy": pol,
                            "pehe_after": np.nan,
                            "pehe_frozen": np.nan,
                            "delta_vs_frozen": np.nan,
                            "error": str(ex),
                        }
                    cell.update({
                        "seed": seed,
                        "train_regimes": ",".join(args.train_regimes),
                        "test_regime": args.test_regime,
                        "signal": args.signal,
                    })
                    rows.append(cell)

    df = pd.DataFrame(rows)
    raw_path = os.path.join(args.out_dir, "budgeted_ladder_raw.csv")
    df.to_csv(raw_path, index=False)

    keys = ["m", "supervision", "policy"]
    if len(df):
        summ = (
            df.groupby(keys, dropna=False)[["pehe_after", "pehe_frozen", "delta_vs_frozen"]]
            .agg(["mean", "std"])
        )
        summ.columns = ["_".join(c) for c in summ.columns]
        summ = summ.reset_index()
        summ_path = os.path.join(args.out_dir, "budgeted_ladder_summary.csv")
        summ.to_csv(summ_path, index=False)
        print("\n=== Summary (mean pehe_after) ===")
        print(summ.to_string(index=False))
        print(f"\nSaved {raw_path}\nSaved {summ_path}")


if __name__ == "__main__":
    main()
