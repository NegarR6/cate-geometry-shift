"""
E2 — Observable router (excess PEHE + risk-coverage). Spec: E2 locked in chat.

Smoke:
  python -m src.run_e2_observable_router --regimes A,B --signals 0.5,2.0 --seeds 0-2 --epochs 40 --patience 10

Full:
  python -m src.run_e2_observable_router --regimes A,B,C --signals 0.5,1.0,2.0 --seeds 0-9 --epochs 80 --patience 15
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
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "synthetic_regimes" / "e2_router"

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

COVERAGES = [1.0, 0.8, 0.6, 0.4, 0.2]


def fit_all_preds(
    tr: Dict[str, np.ndarray],
    X_query: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
) -> Dict[str, np.ndarray]:
    """Fit on train; predict tau on X_query (e.g. full test X)."""
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


def pehe_from_choice(
    choice: np.ndarray, preds: Dict[str, np.ndarray], names: List[str], tau: np.ndarray
) -> float:
    stack = np.column_stack([preds[m] for m in names])
    tau_hat = stack[np.arange(len(choice)), choice]
    return pehe(tau_hat, tau)


def risk_coverage_curve(
    conf: np.ndarray,
    choice: np.ndarray,
    preds: Dict[str, np.ndarray],
    names: List[str],
    tau: np.ndarray,
    pehe_oracle: float,
) -> List[Dict[str, float]]:
    n = len(conf)
    order = np.argsort(-conf)  # high confidence first
    rows = []
    for c in COVERAGES:
        k = max(1, int(np.ceil(c * n)))
        sel = order[:k]
        stack = np.column_stack([preds[m] for m in names])
        tau_hat = stack[sel, choice[sel]]
        p = pehe(tau_hat, tau[sel])
        rows.append(
            {
                "coverage": c,
                "n_kept": int(k),
                "PEHE": p,
                "excess_vs_oracle": p - pehe_oracle,
            }
        )
    return rows


def train_router(X_tr, y_tr, seed: int, kind: str):
    if kind == "gbm":
        clf = GradientBoostingClassifier(random_state=seed)
    elif kind == "logistic":
        clf = LogisticRegression(max_iter=2000, multi_class="auto", random_state=seed)
    else:
        raise ValueError(kind)
    clf.fit(X_tr, y_tr)
    return clf


def run_cell(
    regime: str,
    signal: float,
    seed: int,
    epochs: int,
    patience: int,
    n: int,
    p: int,
    norm_constants: Dict[str, float],
    versions: List[str],
    router_kinds: List[str],
) -> Tuple[List[Dict], List[Dict]]:
    cfg = SyntheticConfig(n=n, p=p, seed=seed, regime=regime, signal=signal)  # type: ignore
    data, meta = generate_synthetic(cfg, norm_constants)
    tr, te = split_train_test(data)
    tau_te = te["tau"]
    e_te = te["e"]

    print(f"  fitting 4 models on train → predict test (n_test={len(tau_te)})")
    preds = fit_all_preds(tr, te["X"], seed, epochs, patience)

    y_all, names = oracle_labels(preds, tau_te)
    # PEHE singles + oracle + global on FULL test (for reference)
    pehe_m = {m: pehe(preds[m], tau_te) for m in names}
    m_global = min(pehe_m, key=pehe_m.get)
    pehe_global_full = pehe_m[m_global]
    pehe_oracle_full = pehe_from_choice(y_all, preds, names, tau_te)

    r_tr_idx, r_ev_idx = split_router_train_eval(len(tau_te), seed)
    tau_ev = tau_te[r_ev_idx]
    y_tr = y_all[r_tr_idx]
    y_ev = y_all[r_ev_idx]
    e_tr, e_ev = e_te[r_tr_idx], e_te[r_ev_idx]
    preds_tr = {m: preds[m][r_tr_idx] for m in names}
    preds_ev = {m: preds[m][r_ev_idx] for m in names}

    pehe_oracle_ev = pehe_from_choice(y_ev, preds_ev, names, tau_ev)
    pehe_m_ev = {m: pehe(preds_ev[m], tau_ev) for m in names}
    m_g_ev = min(pehe_m_ev, key=pehe_m_ev.get)
    pehe_global_ev = pehe_m_ev[m_g_ev]

    raw_rows: List[Dict[str, Any]] = []
    rc_rows: List[Dict[str, Any]] = []

    def add_method(method: str, choice: np.ndarray, conf: np.ndarray | None = None):
        p = pehe_from_choice(choice, preds_ev, names, tau_ev)
        raw_rows.append(
            {
                "regime": regime,
                "signal": signal,
                "seed": seed,
                "method": method,
                "PEHE": p,
                "PEHE_oracle": pehe_oracle_ev,
                "PEHE_global": pehe_global_ev,
                "global_model": m_g_ev,
                "excess_vs_oracle": p - pehe_oracle_ev,
                "delta_vs_global": pehe_global_ev - p,  # >0 router better
                "accuracy": float(np.mean(choice == y_ev)) if choice is not None else np.nan,
                "n_eval": len(tau_ev),
                "pehe_oracle_full_test": pehe_oracle_full,
                "pehe_global_full_test": pehe_global_full,
            }
        )
        if conf is not None:
            for row in risk_coverage_curve(
                conf, choice, preds_ev, names, tau_ev, pehe_oracle_ev
            ):
                rc_rows.append(
                    {
                        "regime": regime,
                        "signal": signal,
                        "seed": seed,
                        "method": method,
                        **row,
                    }
                )

    # Baselines on eval
    for m in names:
        choice = np.full(len(tau_ev), names.index(m), dtype=np.int64)
        add_method(f"single_{m}", choice)
    add_method("global_best", np.full(len(tau_ev), names.index(m_g_ev), dtype=np.int64))
    add_method("oracle", y_ev)
    rng = np.random.default_rng(seed + 123)
    add_method("random", rng.integers(0, len(names), size=len(tau_ev)))

    # Routers
    for ver in versions:
        X_all, _ = build_features(preds, e_te, ver, X_cov=te["X"])
        X_tr = X_all[r_tr_idx]
        X_ev = X_all[r_ev_idx]
        for kind in router_kinds:
            try:
                clf = train_router(X_tr, y_tr, seed, kind)
            except Exception as e:
                print(f"  router {ver}/{kind} failed: {e}")
                continue
            choice = clf.predict(X_ev)
            # confidence
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_ev)
                conf = proba.max(axis=1)
            else:
                conf = np.ones(len(X_ev))
            # map choice to names index if sklearn compresses classes
            classes = list(clf.classes_)
            choice_idx = np.array([classes.index(c) if c in classes else 0 for c in choice])
            # If classes are a subset, map to full names index
            mapped = np.array([names.index(names[c]) if c < len(names) else 0 for c in choice])
            # sklearn y was integer indices into names at train time
            mapped = np.asarray(choice, dtype=np.int64)
            # ensure in range
            mapped = np.clip(mapped, 0, len(names) - 1)
            add_method(f"router_{ver}_{kind}", mapped, conf)

    return raw_rows, rc_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", type=str, default="A,B,C")
    ap.add_argument("--signals", type=str, default="0.5,1.0,2.0")
    ap.add_argument("--seeds", type=str, default="0-9")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--p", type=int, default=12)
    ap.add_argument("--versions", type=str, default="V0,V1,V2,V3")
    ap.add_argument("--routers", type=str, default="gbm")
    args = ap.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",")]
    signals = _parse_floats(args.signals)
    seeds = _parse_seeds(args.seeds)
    versions = [v.strip().upper() for v in args.versions.split(",")]
    router_kinds = [k.strip().lower() for k in args.routers.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = ROOT / "results" / "synthetic_regimes" / "synthetic_meta.json"
    if meta_path.exists():
        norm_constants = json.loads(meta_path.read_text()).get("norm_constants")
        if not norm_constants:
            norm_constants = compute_all_norm_constants(p=args.p)
    else:
        norm_constants = compute_all_norm_constants(p=args.p)

    print("E2 Observable Router")
    print("norms", norm_constants)
    print(f"grid {regimes} {signals} {seeds} versions={versions} routers={router_kinds}")

    all_raw: List[Dict] = []
    all_rc: List[Dict] = []
    t0 = time.time()

    for regime in regimes:
        for signal in signals:
            for seed in seeds:
                print(f"=== E2 regime={regime} signal={signal} seed={seed} ===")
                try:
                    raw, rc = run_cell(
                        regime,
                        signal,
                        seed,
                        args.epochs,
                        args.patience,
                        args.n,
                        args.p,
                        norm_constants,
                        versions,
                        router_kinds,
                    )
                    all_raw.extend(raw)
                    all_rc.extend(rc)
                    # quick print routers
                    for r in raw:
                        if str(r["method"]).startswith("router") or r["method"] in (
                            "global_best",
                            "oracle",
                        ):
                            print(
                                f"  {r['method']}: PEHE={r['PEHE']:.4f} "
                                f"excess={r['excess_vs_oracle']:.4f} "
                                f"Δvs_global={r['delta_vs_global']:.4f}"
                            )
                except Exception as e:
                    print(f"  CELL FAILED: {e}")
                pd.DataFrame(all_raw).to_csv(OUT_DIR / "e2_raw.csv", index=False)
                if all_rc:
                    pd.DataFrame(all_rc).to_csv(OUT_DIR / "e2_risk_coverage.csv", index=False)

    raw_df = pd.DataFrame(all_raw)
    raw_df.to_csv(OUT_DIR / "e2_raw.csv", index=False)
    if all_rc:
        pd.DataFrame(all_rc).to_csv(OUT_DIR / "e2_risk_coverage.csv", index=False)

    if len(raw_df):
        summary = (
            raw_df.groupby(["regime", "signal", "method"])
            .agg(
                n=("PEHE", "count"),
                PEHE_mean=("PEHE", "mean"),
                PEHE_std=("PEHE", "std"),
                excess_mean=("excess_vs_oracle", "mean"),
                excess_std=("excess_vs_oracle", "std"),
                delta_vs_global_mean=("delta_vs_global", "mean"),
                delta_vs_global_std=("delta_vs_global", "std"),
                accuracy_mean=("accuracy", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(OUT_DIR / "e2_summary_by_cell.csv", index=False)
        print("\n=== Summary (delta_vs_global > 0 means router beats global) ===")
        show = summary[
            summary.method.str.startswith("router")
            | summary.method.isin(["global_best", "oracle", "random"])
        ]
        print(show.to_string(index=False))

        # Wilcoxon router vs global where possible
        try:
            from scipy.stats import wilcoxon

            wrows = []
            for (regime, signal), sub in raw_df.groupby(["regime", "signal"]):
                g = sub[sub.method == "global_best"].set_index("seed")["PEHE"]
                for method in sub.method.unique():
                    if not str(method).startswith("router"):
                        continue
                    r = sub[sub.method == method].set_index("seed")["PEHE"]
                    common = g.index.intersection(r.index)
                    if len(common) < 5:
                        continue
                    stat, pval = wilcoxon(r.loc[common], g.loc[common])
                    wrows.append(
                        {
                            "regime": regime,
                            "signal": signal,
                            "method": method,
                            "n": len(common),
                            "mean_router": float(r.loc[common].mean()),
                            "mean_global": float(g.loc[common].mean()),
                            "wilcoxon_p": pval,
                        }
                    )
            if wrows:
                pd.DataFrame(wrows).to_csv(OUT_DIR / "e2_wilcoxon.csv", index=False)
        except Exception as e:
            print("Wilcoxon skip:", e)

    meta = {
        "spec": "E2_OBSERVABLE_ROUTER",
        "norm_constants": norm_constants,
        "regimes": regimes,
        "signals": signals,
        "seeds": seeds,
        "versions": versions,
        "routers": router_kinds,
        "epochs": args.epochs,
        "patience": args.patience,
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / "e2_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
