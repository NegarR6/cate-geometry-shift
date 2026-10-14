"""
Surrogate CATE model selection across τ-geometries (Direction 1).

Smoke:
  python -m src.run_surrogate_selection --regimes A,B,C --signals 0.5,2.0 --seeds 0-2 --epochs 40 --patience 10

Full:
  python -m src.run_surrogate_selection --regimes A,B,C --signals 0.5,1.0,2.0 --seeds 0-9 --epochs 80 --patience 15
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "synthetic_regimes" / "surrogate_selection"

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

MODELS = ("ripo_no_tree", "ft_tarnet", "xlearner", "causalforest")


def _fit_predict_all(tr_X, tr_T, tr_Y, X_q, seed, epochs, patience) -> Dict[str, np.ndarray]:
    out = {}
    for m in MODELS:
        if m == "xlearner":
            out[m] = fit_predict_xlearner(tr_X, tr_T, tr_Y, X_q, seed)
        elif m == "causalforest":
            out[m] = fit_predict_causalforest(tr_X, tr_T, tr_Y, X_q, seed)
        else:
            out[m] = fit_predict_neural(m, tr_X, tr_T, tr_Y, X_q, seed, epochs, patience)
        out[m] = np.asarray(out[m], dtype=np.float64).reshape(-1)
    return out


def _split_fit_score(n: int, seed: int, fit_frac: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed + 17)
    idx = rng.permutation(n)
    k = max(1, int(fit_frac * n))
    return idx[:k], idx[k:]


def _fit_nuisance(X, T, Y, seed):
    """Propensity + outcome models on fit fold only."""
    T = np.asarray(T).astype(int).reshape(-1)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    # propensity
    try:
        e_clf = LogisticRegression(max_iter=500, random_state=seed)
        e_clf.fit(X, T)
        def e_hat(Xq):
            p = e_clf.predict_proba(np.asarray(Xq, dtype=np.float64))[:, 1]
            return np.clip(p, 0.05, 0.95)
    except Exception:
        def e_hat(Xq):
            return np.full(len(Xq), 0.5)

    mu = {}
    for t_val in (0, 1):
        mask = T == t_val
        if mask.sum() < 5:
            mu[t_val] = lambda Xq, _y=Y.mean(): np.full(len(Xq), _y)
            continue
        reg = GradientBoostingRegressor(random_state=seed, max_depth=3, n_estimators=80)
        reg.fit(X[mask], Y[mask])
        mu[t_val] = (lambda model: (lambda Xq: model.predict(np.asarray(Xq, dtype=np.float64))))(reg)
    return e_hat, mu[0], mu[1]


def _factual_mu(tau_hat, mu0, mu1, T):
    """If model only returns τ̂, reconstruct μ_T ≈ μ0 + T*τ̂ with shared nuisance μ0/μ1.
    For scoring we use nuisance μ and τ̂ to form predicted factual."""
    T = np.asarray(T).astype(int).reshape(-1)
    return mu0 + T * tau_hat


def score_surrogates(
    preds_score: Dict[str, np.ndarray],
    X_s, T_s, Y_s,
    e_hat, mu0_fn, mu1_fn,
) -> Dict[str, Dict[str, float]]:
    """Return surrogate_name -> {model: score} (lower is better for all)."""
    T_s = np.asarray(T_s).astype(int).reshape(-1)
    Y_s = np.asarray(Y_s, dtype=np.float64).reshape(-1)
    e = np.asarray(e_hat(X_s), dtype=np.float64).reshape(-1)
    mu0 = np.asarray(mu0_fn(X_s), dtype=np.float64).reshape(-1)
    mu1 = np.asarray(mu1_fn(X_s), dtype=np.float64).reshape(-1)
    phi_dr = (mu1 - mu0) + T_s * (Y_s - mu1) / e - (1 - T_s) * (Y_s - mu0) / (1 - e)

    scores: Dict[str, Dict[str, float]] = {s: {} for s in ("S1", "S2", "S3", "S4", "S5")}
    for m, tau in preds_score.items():
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        # S1 factual mse using μ_T = mu0 + T*tau
        y_hat = mu0 + T_s * tau
        scores["S1"][m] = float(np.mean((Y_s - y_hat) ** 2))
        # S2: prefer higher Var(tau) → score = -var
        scores["S2"][m] = float(-np.var(tau))
        # S3 IPW residual energy
        w = T_s / e - (1 - T_s) / (1 - e)
        scores["S3"][m] = float(np.mean((w * (Y_s - y_hat)) ** 2))
        # S4 pseudo DR
        scores["S4"][m] = float(np.mean((phi_dr - tau) ** 2))
        # S5 policy value proxy (higher better) → score = -value
        treat = (tau > 0).astype(np.float64)
        scores["S5"][m] = float(-np.mean(treat * phi_dr))
    return scores


def run_cell(regime, signal, seed, epochs, patience, n, p, norms):
    cfg = SyntheticConfig(n=n, p=p, seed=seed, regime=regime, signal=signal)  # type: ignore
    data, _ = generate_synthetic(cfg, norms)
    tr, te = split_train_test(data)
    fit_idx, score_idx = _split_fit_score(len(tr["Y"]), seed)

    X_fit, T_fit, Y_fit = tr["X"][fit_idx], tr["T"][fit_idx], tr["Y"][fit_idx]
    X_sc, T_sc, Y_sc = tr["X"][score_idx], tr["T"][score_idx], tr["Y"][score_idx]

    # models predict on score + test
    preds_sc = _fit_predict_all(X_fit, T_fit, Y_fit, X_sc, seed, epochs, patience)
    preds_te = _fit_predict_all(X_fit, T_fit, Y_fit, te["X"], seed, epochs, patience)

    pehe_te = {m: pehe(preds_te[m], te["tau"]) for m in MODELS}
    oracle_m = min(pehe_te, key=pehe_te.get)
    pehe_oracle = pehe_te[oracle_m]

    e_hat, mu0_fn, mu1_fn = _fit_nuisance(X_fit, T_fit, Y_fit, seed)
    sur = score_surrogates(preds_sc, X_sc, T_sc, Y_sc, e_hat, mu0_fn, mu1_fn)

    rows = []
    # random baseline
    rng = np.random.RandomState(seed + 99)
    rand_m = MODELS[int(rng.randint(0, len(MODELS)))]
    rows.append({
        "regime": regime, "signal": signal, "seed": seed,
        "surrogate": "S0_random", "selected": rand_m,
        "PEHE_selected": pehe_te[rand_m], "PEHE_oracle": pehe_oracle,
        "regret": pehe_te[rand_m] - pehe_oracle,
        "selection_acc": int(rand_m == oracle_m),
        "spearman": np.nan, "oracle_model": oracle_m,
    })

    for s_name, s_dict in sur.items():
        selected = min(s_dict, key=s_dict.get)
        score_vec = np.array([s_dict[m] for m in MODELS])
        pehe_vec = np.array([pehe_te[m] for m in MODELS])
        sp = spearmanr(score_vec, pehe_vec).correlation
        rows.append({
            "regime": regime, "signal": signal, "seed": seed,
            "surrogate": s_name, "selected": selected,
            "PEHE_selected": pehe_te[selected], "PEHE_oracle": pehe_oracle,
            "regret": pehe_te[selected] - pehe_oracle,
            "selection_acc": int(selected == oracle_m),
            "spearman": float(sp) if sp == sp else np.nan,
            "oracle_model": oracle_m,
        })
    # also store per-model PEHE for analysis
    for m in MODELS:
        rows.append({
            "regime": regime, "signal": signal, "seed": seed,
            "surrogate": f"PEHE_{m}", "selected": m,
            "PEHE_selected": pehe_te[m], "PEHE_oracle": pehe_oracle,
            "regret": pehe_te[m] - pehe_oracle,
            "selection_acc": int(m == oracle_m),
            "spearman": np.nan, "oracle_model": oracle_m,
        })
    return rows


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

    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = ROOT / "results" / "synthetic_regimes" / "synthetic_meta.json"
    if meta_path.exists():
        norms = json.loads(meta_path.read_text()).get("norm_constants") or compute_all_norm_constants(p=args.p)
    else:
        norms = compute_all_norm_constants(p=args.p)

    print("Surrogate selection × geometry")
    print("norms", {k: round(v, 4) for k, v in norms.items()})
    raw: List[dict] = []
    t0 = time.time()
    for regime in regimes:
        for signal in signals:
            for seed in seeds:
                print(f"=== {regime} signal={signal} seed={seed} ===")
                try:
                    rows = run_cell(regime, signal, seed, args.epochs, args.patience, args.n, args.p, norms)
                    raw.extend(rows)
                    # print compact
                    sub = [r for r in rows if r["surrogate"] in ("S1", "S4", "S0_random")]
                    for r in sub:
                        print(
                            f"  {r['surrogate']}: pick={r['selected']} "
                            f"regret={r['regret']:.4f} acc={r['selection_acc']}"
                        )
                except Exception as e:
                    print(f"  FAILED: {e}")
                if raw:
                    pd.DataFrame(raw).to_csv(OUT / "surrogate_raw.csv", index=False)

    df = pd.DataFrame(raw)
    df.to_csv(OUT / "surrogate_raw.csv", index=False)

    # summary only on real surrogates + random
    mask = df["surrogate"].isin(["S0_random", "S1", "S2", "S3", "S4", "S5"])
    sdf = df.loc[mask]
    if len(sdf):
        summary = (
            sdf.groupby(["regime", "signal", "surrogate"])
            .agg(
                n=("regret", "count"),
                regret_mean=("regret", "mean"),
                regret_std=("regret", "std"),
                acc_mean=("selection_acc", "mean"),
                spearman_mean=("spearman", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(OUT / "surrogate_summary.csv", index=False)
        by_reg = (
            sdf.groupby(["regime", "surrogate"])
            .agg(regret_mean=("regret", "mean"), acc_mean=("selection_acc", "mean"), n=("regret", "count"))
            .reset_index()
        )
        by_reg.to_csv(OUT / "surrogate_by_regime.csv", index=False)
        print("\n=== By regime (acc / regret) ===")
        print(by_reg.to_string(index=False))
        print("\n=== By regime × signal ===")
        print(summary.to_string(index=False))

    meta = {
        "spec": "SURROGATE_SELECTION_GEOMETRY_v1",
        "regimes": regimes,
        "signals": signals,
        "seeds": seeds,
        "epochs": args.epochs,
        "elapsed_sec": time.time() - t0,
        "norm_constants": norms,
        "surrogates": ["S0_random", "S1_factual_mse", "S2_neg_tau_var", "S3_ipw", "S4_pseudo_dr", "S5_policy"],
    }
    (OUT / "surrogate_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
