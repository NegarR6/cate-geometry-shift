"""
E1 — Oracle gap for synthetic regimes (see E1_ORACLE_GAP_SPEC.md).

Re-fits models on the locked SYNTHETIC_SPEC_v1 DGPs, stores test τ̂ for all
models, and computes pointwise-oracle PEHE vs fixed estimators.

Smoke:
  python -m src.run_e1_oracle_gap --regimes A,B --signals 0.5,2.0 --seeds 0-2 --epochs 40 --patience 10

Full:
  python -m src.run_e1_oracle_gap --regimes A,B,C --signals 0.5,1.0,2.0 --seeds 0-9 --epochs 80 --patience 15
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "synthetic_regimes" / "e1_oracle_gap"

from src.synthetic_regimes import (  # noqa: E402
    SyntheticConfig,
    compute_all_norm_constants,
    generate_synthetic,
    split_train_test,
)
from src.run_synthetic_regimes import (  # noqa: E402
    _parse_floats,
    _parse_seeds,
    fit_predict_causalforest,
    fit_predict_neural,
    fit_predict_xlearner,
    pehe,
)

MODELS = ["ripo_no_tree", "ft_tarnet", "xlearner", "causalforest"]


def fit_tau_hat(
    model: str,
    tr: Dict[str, np.ndarray],
    te: Dict[str, np.ndarray],
    seed: int,
    epochs: int,
    patience: int,
) -> np.ndarray:
    X_tr, T_tr, Y_tr = tr["X"], tr["T"], tr["Y"]
    X_te = te["X"]
    if model == "xlearner":
        return fit_predict_xlearner(X_tr, T_tr, Y_tr, X_te, seed)
    if model == "causalforest":
        return fit_predict_causalforest(X_tr, T_tr, Y_tr, X_te, seed)
    if model in ("ripo_no_tree", "ft_tarnet"):
        return fit_predict_neural(model, X_tr, T_tr, Y_tr, X_te, seed, epochs, patience)
    raise ValueError(model)


def oracle_from_predictions(
    preds: Dict[str, np.ndarray], tau_true: np.ndarray
) -> Dict[str, Any]:
    """Pointwise oracle over models in preds."""
    names = list(preds.keys())
    stack = np.column_stack([preds[m] for m in names])  # (n, M)
    abs_err = np.abs(stack - tau_true.reshape(-1, 1))
    pick = np.argmin(abs_err, axis=1)
    tau_oracle = stack[np.arange(len(tau_true)), pick]
    pehe_oracle = pehe(tau_oracle, tau_true)

    share = {m: float(np.mean(pick == j)) for j, m in enumerate(names)}
    pehe_m = {m: pehe(preds[m], tau_true) for m in names}
    best_global = min(pehe_m, key=pehe_m.get)
    pehe_global = pehe_m[best_global]

    rows = []
    for m in names:
        rows.append(
            {
                "model": m,
                "PEHE": pehe_m[m],
                "PEHE_oracle": pehe_oracle,
                "PEHE_global_best": pehe_global,
                "global_best_model": best_global,
                "excess_vs_oracle": pehe_m[m] - pehe_oracle,
                "excess_global_vs_oracle": pehe_global - pehe_oracle,
                "oracle_share": share[m],
            }
        )
    return {
        "rows": rows,
        "pehe_oracle": pehe_oracle,
        "pehe_global": pehe_global,
        "best_global": best_global,
        "share": share,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", type=str, default="A,B,C")
    ap.add_argument("--signals", type=str, default="0.5,1.0,2.0")
    ap.add_argument("--seeds", type=str, default="0-9")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--p", type=int, default=12)
    args = ap.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",")]
    signals = _parse_floats(args.signals)
    seeds = _parse_seeds(args.seeds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Prefer locked constants from previous full run if present
    meta_path = ROOT / "results" / "synthetic_regimes" / "synthetic_meta.json"
    if meta_path.exists():
        locked = json.loads(meta_path.read_text())
        norm_constants = locked.get("norm_constants") or compute_all_norm_constants(p=args.p)
    else:
        norm_constants = compute_all_norm_constants(p=args.p)

    print("Norm constants:", norm_constants)
    print(
        f"E1 grid regimes={regimes} signals={signals} seeds={seeds} "
        f"models={MODELS} total_cells={len(regimes)*len(signals)*len(seeds)}"
    )

    raw_rows: List[Dict[str, Any]] = []
    share_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for regime in regimes:
        for signal in signals:
            for seed in seeds:
                cfg = SyntheticConfig(
                    n=args.n, p=args.p, seed=seed, regime=regime, signal=signal  # type: ignore
                )
                data, meta = generate_synthetic(cfg, norm_constants)
                tr, te = split_train_test(data)
                tau_true = te["tau"]

                preds: Dict[str, np.ndarray] = {}
                print(f"=== E1 regime={regime} signal={signal} seed={seed} ===")
                for model in MODELS:
                    try:
                        tau_hat = fit_tau_hat(
                            model, tr, te, seed, args.epochs, args.patience
                        )
                        assert len(tau_hat) == len(tau_true)
                        preds[model] = np.asarray(tau_hat, dtype=np.float64).reshape(-1)
                        print(f"  {model}: PEHE={pehe(preds[model], tau_true):.4f}")
                    except Exception as e:
                        print(f"  {model} FAILED: {e}")
                        # skip cell model; oracle over remaining
                if len(preds) < 2:
                    print("  skip cell: fewer than 2 models succeeded")
                    continue

                out = oracle_from_predictions(preds, tau_true)
                print(
                    f"  ORACLE PEHE={out['pehe_oracle']:.4f}  "
                    f"GLOBAL({out['best_global']})={out['pehe_global']:.4f}  "
                    f"excess_global={out['pehe_global']-out['pehe_oracle']:.4f}"
                )
                for r in out["rows"]:
                    raw_rows.append(
                        {
                            "regime": regime,
                            "signal": signal,
                            "seed": seed,
                            "n_test": meta["n_test"],
                            **r,
                        }
                    )
                for m, sh in out["share"].items():
                    share_rows.append(
                        {
                            "regime": regime,
                            "signal": signal,
                            "seed": seed,
                            "model": m,
                            "oracle_share": sh,
                            "pehe_oracle": out["pehe_oracle"],
                            "pehe_global": out["pehe_global"],
                            "global_best_model": out["best_global"],
                        }
                    )

                # incremental save
                pd.DataFrame(raw_rows).to_csv(OUT_DIR / "e1_raw.csv", index=False)
                pd.DataFrame(share_rows).to_csv(OUT_DIR / "e1_oracle_share_by_seed.csv", index=False)

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(OUT_DIR / "e1_raw.csv", index=False)

    # Cell summary: one row per regime × signal × model
    if len(raw):
        summary = (
            raw.groupby(["regime", "signal", "model"])
            .agg(
                n=("PEHE", "count"),
                PEHE_mean=("PEHE", "mean"),
                PEHE_std=("PEHE", "std"),
                PEHE_oracle_mean=("PEHE_oracle", "mean"),
                PEHE_oracle_std=("PEHE_oracle", "std"),
                PEHE_global_mean=("PEHE_global_best", "mean"),
                excess_mean=("excess_vs_oracle", "mean"),
                excess_std=("excess_vs_oracle", "std"),
                excess_global_mean=("excess_global_vs_oracle", "mean"),
                excess_global_std=("excess_global_vs_oracle", "std"),
                oracle_share_mean=("oracle_share", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(OUT_DIR / "e1_cell_summary.csv", index=False)

        # Compact cell-level gap (unique oracle/global per seed then average)
        gap = (
            raw.drop_duplicates(["regime", "signal", "seed"])[
                [
                    "regime",
                    "signal",
                    "seed",
                    "PEHE_oracle",
                    "PEHE_global_best",
                    "excess_global_vs_oracle",
                    "global_best_model",
                ]
            ]
        )
        gap_summary = (
            gap.groupby(["regime", "signal"])
            .agg(
                n=("PEHE_oracle", "count"),
                PEHE_oracle_mean=("PEHE_oracle", "mean"),
                PEHE_oracle_std=("PEHE_oracle", "std"),
                PEHE_global_mean=("PEHE_global_best", "mean"),
                PEHE_global_std=("PEHE_global_best", "std"),
                excess_global_mean=("excess_global_vs_oracle", "mean"),
                excess_global_std=("excess_global_vs_oracle", "std"),
            )
            .reset_index()
        )
        gap_summary.to_csv(OUT_DIR / "e1_gap_by_cell.csv", index=False)

        share_df = pd.DataFrame(share_rows)
        share_sum = (
            share_df.groupby(["regime", "signal", "model"])["oracle_share"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        share_sum.to_csv(OUT_DIR / "e1_oracle_share.csv", index=False)

        print("\n=== Gap by cell (global-best − oracle) ===")
        print(gap_summary.to_string(index=False))
        print("\n=== Mean oracle share ===")
        print(share_sum.to_string(index=False))
    else:
        print("No successful rows.")

    meta_out = {
        "spec": "E1_ORACLE_GAP_SPEC",
        "parent_spec": "SYNTHETIC_SPEC_v1",
        "norm_constants": norm_constants,
        "regimes": regimes,
        "signals": signals,
        "seeds": seeds,
        "models": MODELS,
        "n": args.n,
        "p": args.p,
        "epochs": args.epochs,
        "patience": args.patience,
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / "e1_meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
