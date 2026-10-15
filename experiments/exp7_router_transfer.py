"""
E3 — Cross-regime shift for observable router.

Train router on source regime(s); evaluate on target regime (different geometry).
Primary question: do observable reliability signals transfer across τ-geometries?

Smoke:
  python -m src.run_e3_shift_router --pairs B>A,A>B --signals 0.5,2.0 --seeds 0-2 --epochs 40 --patience 10

Full:
  python -m src.run_e3_shift_router --pairs B>A,A>B,B>C,A>C --signals 0.5,1.0,2.0 --seeds 0-9 --epochs 80 --patience 15
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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "synthetic_regimes" / "e3_shift"

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
from src.e2_router_features import (  # noqa: E402
    MODEL_ORDER,
    build_features,
    oracle_labels,
    split_router_train_eval,
)


def fit_all_preds(tr, X_query, seed, epochs, patience) -> Dict[str, np.ndarray]:
    X_tr, T_tr, Y_tr = tr["X"], tr["T"], tr["Y"]
    preds = {}
    for model in MODEL_ORDER:
        if model == "xlearner":
            preds[model] = fit_predict_xlearner(X_tr, T_tr, Y_tr, X_query, seed)
        elif model == "causalforest":
            preds[model] = fit_predict_causalforest(X_tr, T_tr, Y_tr, X_query, seed)
        else:
            preds[model] = fit_predict_neural(
                model, X_tr, T_tr, Y_tr, X_query, seed, epochs, patience
            )
        preds[model] = np.asarray(preds[model], dtype=np.float64).reshape(-1)
    return preds


def pehe_from_choice(choice, preds, names, tau) -> float:
    stack = np.column_stack([preds[m] for m in names])
    tau_hat = stack[np.arange(len(choice)), np.clip(choice, 0, len(names) - 1)]
    return pehe(tau_hat, tau)


def prepare_regime(
    regime: str,
    signal: float,
    seed: int,
    n: int,
    p: int,
    norm_constants: Dict[str, float],
    epochs: int,
    patience: int,
) -> Dict[str, Any]:
    """Generate data, fit models on train, predict on test, build labels/features."""
    cfg = SyntheticConfig(n=n, p=p, seed=seed, regime=regime, signal=signal)  # type: ignore
    data, meta = generate_synthetic(cfg, norm_constants)
    tr, te = split_train_test(data)
    preds = fit_all_preds(tr, te["X"], seed, epochs, patience)
    y, names = oracle_labels(preds, te["tau"])
    pehe_m = {m: pehe(preds[m], te["tau"]) for m in names}
    m_g = min(pehe_m, key=pehe_m.get)
    return {
        "regime": regime,
        "signal": signal,
        "seed": seed,
        "te": te,
        "preds": preds,
        "y": y,
        "names": names,
        "pehe_m": pehe_m,
        "global_model": m_g,
        "pehe_global": pehe_m[m_g],
        "pehe_oracle": pehe_from_choice(y, preds, names, te["tau"]),
    }


def run_pair(
    src: str,
    tgt: str,
    signal: float,
    seed: int,
    epochs: int,
    patience: int,
    n: int,
    p: int,
    norm_constants: Dict[str, float],
    version: str,
) -> Dict[str, Any]:
    print(f"  source={src} → target={tgt} signal={signal} seed={seed}")
    S = prepare_regime(src, signal, seed, n, p, norm_constants, epochs, patience)
    T = prepare_regime(tgt, signal, seed + 10_000, n, p, norm_constants, epochs, patience)
    # Note: different seed offset on target data gen so X/tau not identical; models still
    # trained independently. For fair shift we use same signal strength.

    # Router train: half of SOURCE test
    r_tr, _ = split_router_train_eval(len(S["y"]), seed)
    X_src, _ = build_features(S["preds"], S["te"]["e"], version, X_cov=S["te"]["X"])
    y_src = S["y"]
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X_src[r_tr], y_src[r_tr])

    # Eval on FULL target test
    X_tgt, _ = build_features(T["preds"], T["te"]["e"], version, X_cov=T["te"]["X"])
    choice = np.asarray(clf.predict(X_tgt), dtype=np.int64)
    choice = np.clip(choice, 0, len(T["names"]) - 1)

    pehe_router = pehe_from_choice(choice, T["preds"], T["names"], T["te"]["tau"])
    pehe_oracle = T["pehe_oracle"]
    pehe_global = T["pehe_global"]
    acc = float(np.mean(choice == T["y"]))

    # In-distribution control: router trained on src, eval on held-out src
    _, r_ev_s = split_router_train_eval(len(S["y"]), seed)
    choice_id = np.asarray(clf.predict(X_src[r_ev_s]), dtype=np.int64)
    choice_id = np.clip(choice_id, 0, len(S["names"]) - 1)
    pehe_id = pehe_from_choice(
        choice_id, {m: S["preds"][m][r_ev_s] for m in S["names"]}, S["names"], S["te"]["tau"][r_ev_s]
    )
    pehe_id_global = min(
        pehe(S["preds"][m][r_ev_s], S["te"]["tau"][r_ev_s]) for m in S["names"]
    )
    pehe_id_oracle = pehe_from_choice(
        S["y"][r_ev_s],
        {m: S["preds"][m][r_ev_s] for m in S["names"]},
        S["names"],
        S["te"]["tau"][r_ev_s],
    )

    return {
        "src": src,
        "tgt": tgt,
        "signal": signal,
        "seed": seed,
        "version": version,
        "PEHE_router_tgt": pehe_router,
        "PEHE_global_tgt": pehe_global,
        "PEHE_oracle_tgt": pehe_oracle,
        "excess_tgt": pehe_router - pehe_oracle,
        "delta_vs_global_tgt": pehe_global - pehe_router,
        "accuracy_tgt": acc,
        "global_model_tgt": T["global_model"],
        "PEHE_router_src_id": pehe_id,
        "PEHE_global_src_id": pehe_id_global,
        "PEHE_oracle_src_id": pehe_id_oracle,
        "delta_vs_global_src_id": pehe_id_global - pehe_id,
    }


def parse_pairs(s: str) -> List[Tuple[str, str]]:
    pairs = []
    for part in s.split(","):
        part = part.strip()
        if ">" not in part:
            raise ValueError(f"pair must be SRC>TGT, got {part}")
        a, b = part.split(">", 1)
        pairs.append((a.strip().upper(), b.strip().upper()))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default="A>B,A>C,B>A,B>C")
    ap.add_argument("--signals", type=str, default="0.5,1.0,2.0")
    ap.add_argument("--seeds", type=str, default="0-9")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--p", type=int, default=12)
    ap.add_argument("--version", type=str, default="V2")
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs)
    signals = _parse_floats(args.signals)
    seeds = _parse_seeds(args.seeds)
    version = args.version.upper()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = ROOT / "results" / "synthetic_regimes" / "synthetic_meta.json"
    if meta_path.exists():
        norm_constants = json.loads(meta_path.read_text()).get("norm_constants") or compute_all_norm_constants(p=args.p)
    else:
        norm_constants = compute_all_norm_constants(p=args.p)

    print("E3 Cross-regime shift")
    print("pairs", pairs, "signals", signals, "seeds", seeds, "version", version)

    rows: List[Dict] = []
    t0 = time.time()
    for src, tgt in pairs:
        for signal in signals:
            for seed in seeds:
                print(f"=== {src}>{tgt} signal={signal} seed={seed} ===")
                try:
                    row = run_pair(
                        src, tgt, signal, seed, args.epochs, args.patience,
                        args.n, args.p, norm_constants, version,
                    )
                    rows.append(row)
                    print(
                        f"  tgt: router={row['PEHE_router_tgt']:.4f} "
                        f"global={row['PEHE_global_tgt']:.4f} "
                        f"Δ={row['delta_vs_global_tgt']:.4f} "
                        f"| src_id Δ={row['delta_vs_global_src_id']:.4f}"
                    )
                except Exception as e:
                    print(f"  FAILED: {e}")
                if rows:
                    pd.DataFrame(rows).to_csv(OUT_DIR / "e3_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "e3_raw.csv", index=False)
    if len(df):
        summary = (
            df.groupby(["src", "tgt", "signal"])
            .agg(
                n=("PEHE_router_tgt", "count"),
                PEHE_router_mean=("PEHE_router_tgt", "mean"),
                PEHE_global_mean=("PEHE_global_tgt", "mean"),
                PEHE_oracle_mean=("PEHE_oracle_tgt", "mean"),
                delta_tgt_mean=("delta_vs_global_tgt", "mean"),
                delta_tgt_std=("delta_vs_global_tgt", "std"),
                excess_tgt_mean=("excess_tgt", "mean"),
                acc_mean=("accuracy_tgt", "mean"),
                delta_src_id_mean=("delta_vs_global_src_id", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(OUT_DIR / "e3_summary.csv", index=False)
        print("\n=== E3 Summary (delta_tgt > 0 means router beats target global) ===")
        print(summary.to_string(index=False))

    meta = {
        "spec": "E3_SHIFT",
        "pairs": [f"{a}>{b}" for a, b in pairs],
        "signals": signals,
        "seeds": seeds,
        "version": version,
        "epochs": args.epochs,
        "elapsed_sec": time.time() - t0,
        "norm_constants": norm_constants,
    }
    (OUT_DIR / "e3_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
