"""Path β Stage B0: D_G vs ranking margin on continuum A–B (and A–C)."""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry_proxy import compute_D_G, fit_norm_from_component_rows
from models_geometry_adaptive import build_geo_adaptive_from_hparams
from train_geometry_adaptive import (
    GeometryAdaptiveTrainer,
    set_seed,
    build_loaders,
)
from tree_signals import CrossFittedTreeSignals

# continuum data — adapt import to your synthetic_regimes API
from synthetic_regimes import SyntheticConfig, generate_synthetic, compute_all_norm_constants

try:
    from synthetic_regimes import generate_continuum
    HAS_CONT = True
except Exception:
    HAS_CONT = False


def pehe(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_continuum_data(lam, n, p, seed, pair="A-B", signal=1.0):
    """Fallback: blend tau of two regimes if generate_continuum missing."""
    norm = compute_all_norm_constants(p=p)
    if HAS_CONT and hasattr(generate_continuum, "__call__"):
        try:
            return generate_continuum(lam=lam, n=n, p=p, seed=seed, pair=pair, signal=signal, norm_constants=norm)
        except TypeError:
            pass
    # manual blend of potential outcomes via two regime draws with same X noise seed
    r0, r1 = pair.split("-")
    cfg0 = SyntheticConfig(n=n, p=p, seed=seed, regime=r0, signal=signal)
    cfg1 = SyntheticConfig(n=n, p=p, seed=seed, regime=r1, signal=signal)
    d0, _ = generate_synthetic(cfg0, norm_constants=norm)
    d1, _ = generate_synthetic(cfg1, norm_constants=norm)
    # same X,T indices: re-draw with fixed X from d0, blend tau
    data = dict(d0)
    data["tau"] = (1.0 - lam) * d0["tau"] + lam * d1["tau"]
    # rebuild Y roughly: Y = baseline + T*tau + noise already in d0 — scale tau effect
    # safer: keep Y from d0 when lam=0 else approximate
    if lam <= 0:
        return d0, None
    if lam >= 1:
        return d1, None
    # use d0 structure but replace tau for evaluation; train still uses factual Y of blend proxy
    y0 = d0["Y"] - d0["T"] * d0["tau"]  # crude residual
    data["Y"] = y0 + data["T"] * data["tau"]
    data["tau"] = data["tau"].astype(np.float32)
    data["Y"] = data["Y"].astype(np.float32)
    return data, None


def run_one(lam, seed, n, p, device, pretrain, joint, pair, signal):
    set_seed(seed)
    data, _ = make_continuum_data(lam, n, p, seed, pair=pair, signal=signal)
    X, T, Y, e, tau = data["X"], data["T"], data["Y"], data["e"], data["tau"]
    tr, te = data["train_idx"], data["test_idx"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr)
    n_val = max(50, int(0.12 * len(perm)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tb = CrossFittedTreeSignals(n_splits=3, seed=seed)
    ts = np.zeros((len(X), 4), np.float32)
    ts[tr_idx] = tb.fit_transform(X[tr_idx], T[tr_idx], Y[tr_idx])
    ts[val_idx] = tb.transform(X[val_idx])
    ts[te] = tb.transform(X[te])

    model = build_geo_adaptive_from_hparams(num_features=p)
    trainer = GeometryAdaptiveTrainer(
        model, device=device,
        pretrain_epochs=pretrain, joint_epochs=joint,
        freeze_experts_epochs=6, route_weight_start=0.15, route_weight_end=0.45,
    )
    train_loader, val_loader = build_loaders(X, T, Y, e, ts, None, tr_idx, val_idx)
    trainer.pretrain_experts(train_loader, val_loader, verbose=False)

    # expert predictions on train/test (no need full joint for margin ranking)
    tau_a_tr, tau_n_tr, tau_t_tr, _ = trainer.predict_experts(X[tr_idx], e[tr_idx], ts[tr_idx])
    tau_a_te, tau_n_te, tau_t_te, _ = trainer.predict_experts(X[te], e[te], ts[te])

    # optional short joint for adaptive pehe
    oracle = np.full(len(X), -1, dtype=np.int64)
    try:
        oracle[tr_idx] = trainer.build_oracle_from_experts(
            X[tr_idx], e[tr_idx], ts[tr_idx], tau[tr_idx]
        )
        train_loader, val_loader = build_loaders(X, T, Y, e, ts, oracle, tr_idx, val_idx)
        trainer.fit_joint_only(train_loader, val_loader, verbose=False)
        tau_a_te, tau_n_te, tau_t_te, _ = trainer.predict_experts(X[te], e[te], ts[te])
    except Exception:
        pass

    pn, pt = pehe(tau[te], tau_n_te), pehe(tau[te], tau_t_te)
    margin = pt - pn  # >0 neural better
    proxy = compute_D_G(X[tr_idx], T[tr_idx], Y[tr_idx], tau_n_tr, tau_t_tr, norm=None)

    return {
        "pair": pair, "lam": lam, "seed": seed,
        "pehe_neural": pn, "pehe_tree": pt, "margin_tree_minus_neural": margin,
        "pehe_adaptive": pehe(tau[te], tau_a_te),
        "D_G_raw": proxy.D_G,
        **{f"c_{k}": v for k, v in proxy.components.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="A-B")
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--p", type=int, default=12)
    ap.add_argument("--signal", type=float, default=1.0)
    ap.add_argument("--pretrain_epochs", type=int, default=25)
    ap.add_argument("--joint_epochs", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="results/geometry_proxy_synthetic")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lams = [float(x) for x in args.lambdas.split(",")]
    rows = []
    for seed in range(args.n_seeds):
        for lam in lams:
            print(f"seed={seed} lam={lam}")
            rows.append(run_one(
                lam, seed, args.n, args.p, args.device,
                args.pretrain_epochs, args.joint_epochs, args.pair, args.signal,
            ))
    df = pd.DataFrame(rows)

    # normalize D_G using component means over this run
    norm = fit_norm_from_component_rows(
        [{"disagreement": r["c_disagreement"], "stump_gain": r["c_stump_gain"], "roughness": r["c_roughness"]}
         for r in rows]
    )
    def norm_row(r):
        from geometry_proxy import _z
        return (
        _z(r["c_stump_gain"], *norm["stump_gain"])
        + _z(r["c_disagreement"], *norm["disagreement"])
        - _z(r["c_roughness"], *norm["roughness"])
        )
    df["D_G"] = df.apply(norm_row, axis=1)

    path = os.path.join(args.out_dir, f"proxy_{args.pair.replace('-', '')}_raw.csv")
    df.to_csv(path, index=False)

    corr = float(df["D_G"].corr(df["margin_tree_minus_neural"]))
    # expected: higher D_G (threshold-like) => tree better => margin = pehe_tree - pehe_neural smaller/negative
    summary = {
        "pair": args.pair,
        "n_rows": len(df),
        "corr_DG_margin": corr,
        "go_abs_corr_ge_0.7": abs(corr) >= 0.7,
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.out_dir, f"proxy_{args.pair.replace('-', '')}_corr.csv"), index=False
    )
    print(df.groupby("lam")[["pehe_neural", "pehe_tree", "margin_tree_minus_neural", "D_G"]].mean())
    print("CORR(D_G, margin_tree_minus_neural) =", corr)
    print("GO B0 (|corr|>=0.7)?", abs(corr) >= 0.7)
    print("Saved", path)


if __name__ == "__main__":
    main()