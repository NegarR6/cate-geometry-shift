"""
Harder synthetic ranking-reversal stress test (appendix only).

Same tau geometries A/B/C as SPEC v1, but p=50, noise_std=1.0, n=5000.
Does NOT overwrite results/synthetic_regimes/ SPEC tables.

Usage:
  # Smoke
  python -m src.run_synthetic_hard --regimes A,B --signals 1.0 --seeds 0-2 --epochs 40

  # Appendix
  python -m src.run_synthetic_hard --regimes A,B,C --signals 0.5,1.0,2.0 --seeds 0-9 --epochs 80 --patience 15

Output → results/synthetic_regimes_hard/
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT_DIR = ROOT / "results" / "synthetic_regimes_hard"

from synthetic_regimes import (  # noqa: E402
    SyntheticConfigHard,
    compute_all_norm_constants,
    generate_synthetic_hard,
    split_train_test,
    overlap_metrics,
)
from models_hybrid_v2 import ResidualAttentionCATE  # noqa: E402
from train_neural import set_global_seed, CausalLoss  # noqa: E402
from evaluation import evaluate_cate  # noqa: E402


def _parse_seeds(s: str) -> List[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def fit_xlearner(X_tr, T_tr, Y_tr, X_te, seed: int) -> np.ndarray:
    from econml.metalearners import XLearner
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

    xl = XLearner(
        models=GradientBoostingRegressor(random_state=seed),
        cate_models=GradientBoostingRegressor(random_state=seed),
        propensity_model=RandomForestClassifier(
            n_estimators=100, min_samples_leaf=10, random_state=seed
        ),
    )
    xl.fit(Y_tr, T_tr, X=X_tr)
    return xl.effect(X_te).reshape(-1)


def fit_residual(
    X_tr, T_tr, Y_tr, X_te, seed: int, epochs: int, patience: int, device: str
) -> np.ndarray:
    set_global_seed(seed)
    idx = np.arange(len(Y_tr))
    strat = T_tr.astype(int) if len(np.unique(T_tr)) > 1 else None
    fit_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=strat
    )
    model = ResidualAttentionCATE(
        num_features=X_tr.shape[1],
        d_token=32,
        num_layers=3,
        nhead=4,
        dim_feedforward=64,
        dropout=0.1,
        head_hidden_dim=64,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = CausalLoss(ipm_weight=0.1, ipm_type="mean_diff")

    def loader(ix, shuffle):
        ds = TensorDataset(
            torch.as_tensor(X_tr[ix], dtype=torch.float32),
            torch.as_tensor(T_tr[ix], dtype=torch.float32),
            torch.as_tensor(Y_tr[ix], dtype=torch.float32),
        )
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=64, shuffle=shuffle, generator=g)

    tr_loader, va_loader = loader(fit_idx, True), loader(val_idx, False)
    best, state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for X, w, y in tr_loader:
            X, w, y = X.to(device), w.to(device), y.to(device)
            y0, y1, z = model(X, return_representation=True)
            loss, _, _ = loss_fn(y0, y1, y, w, representation=z)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        v, n = 0.0, 0
        with torch.no_grad():
            for X, w, y in va_loader:
                X, w, y = X.to(device), w.to(device), y.to(device)
                y0, y1, z = model(X, return_representation=True)
                loss, _, _ = loss_fn(y0, y1, y, w, representation=z)
                v += float(loss.item()) * len(y)
                n += len(y)
        avg = v / max(n, 1)
        if avg < best - 1e-6:
            best, stale = avg, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if state:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        Xq = torch.as_tensor(X_te, dtype=torch.float32, device=device)
        y0, y1 = model(Xq)
        return (y1 - y0).reshape(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="A,B")
    ap.add_argument("--signals", default="1.0")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--p", type=int, default=50)
    ap.add_argument("--noise", type=float, default=1.0)
    args = ap.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    signals = _parse_floats(args.signals)
    seeds = _parse_seeds(args.seeds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norms = compute_all_norm_constants(p=args.p)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for regime in regimes:
        for signal in signals:
            for seed in seeds:
                cfg = SyntheticConfigHard(
                    n=args.n,
                    p=args.p,
                    noise_std=args.noise,
                    seed=seed,
                    regime=regime,  # type: ignore
                    signal=signal,
                )
                data, meta = generate_synthetic_hard(cfg, norm_constants=norms)
                tr, te = split_train_test(data)
                # residual
                tau_r = fit_residual(
                    tr["X"], tr["T"], tr["Y"], te["X"], seed, args.epochs, args.patience, device
                )
                pehe_r = float(np.sqrt(np.mean((tau_r - te["tau"]) ** 2)))
                # xlearner
                try:
                    tau_x = fit_xlearner(tr["X"], tr["T"], tr["Y"], te["X"], seed)
                    pehe_x = float(np.sqrt(np.mean((tau_x - te["tau"]) ** 2)))
                except Exception as e:
                    pehe_x = float("nan")
                    print("  XLearner failed:", e)
                ov = overlap_metrics(data["e"])
                row = {
                    "regime": regime,
                    "signal": signal,
                    "seed": seed,
                    "PEHE_ripo_no_tree": pehe_r,
                    "PEHE_xlearner": pehe_x,
                    "delta_x_minus_ripo": pehe_x - pehe_r,  # >0 → residual better
                    **{f"overlap_{k}": v for k, v in ov.items()},
                }
                rows.append(row)
                print(
                    f"  {regime} sig={signal} seed={seed} "
                    f"PEHE_ripo={pehe_r:.4f} PEHE_x={pehe_x:.4f}"
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "raw.csv", index=False)
    summary = (
        raw.groupby(["regime", "signal"], as_index=False)
        .agg(
            n=("seed", "count"),
            PEHE_ripo_mean=("PEHE_ripo_no_tree", "mean"),
            PEHE_ripo_std=("PEHE_ripo_no_tree", "std"),
            PEHE_x_mean=("PEHE_xlearner", "mean"),
            PEHE_x_std=("PEHE_xlearner", "std"),
            delta_mean=("delta_x_minus_ripo", "mean"),
        )
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    meta = {
        "spec": "SYNTHETIC_HARD_v1",
        "n": args.n,
        "p": args.p,
        "noise_std": args.noise,
        "regimes": regimes,
        "signals": signals,
        "seeds": seeds,
        "elapsed_sec": time.time() - t0,
        "note": "Appendix stress-test only. Do not replace SPEC v1 tables.",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("Saved →", OUT_DIR)


if __name__ == "__main__":
    main()
