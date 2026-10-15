"""
Run synthetic-spec experiment.

Usage:
  python -m src.run_synthetic_regimes --regimes A,B,C --signals 0.5,1.0,2.0 --seeds 0-9

Optional quick smoke:
  python -m src.run_synthetic_regimes --regimes A,B --signals 0.5,2.0 --seeds 0-2 --epochs 40

Outputs under results/synthetic_regimes/
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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "synthetic_regimes"

from src.synthetic_regimes import (  # noqa: E402
    SIGNAL_LEVELS,
    SyntheticConfig,
    compute_all_norm_constants,
    generate_synthetic,
    split_train_test,
)


def _parse_seeds(s: str) -> List[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def ate_error(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(abs(np.mean(tau_hat) - np.mean(tau_true)))


def policy_value_and_regret(
    tau_hat: np.ndarray, tau_true: np.ndarray, treat_threshold: float = 0.0
) -> Tuple[float, float]:
    """Simple decision: treat if tau_hat > threshold. Value under true tau."""
    a_hat = (tau_hat > treat_threshold).astype(np.float64)
    a_star = (tau_true > treat_threshold).astype(np.float64)
    # potential outcomes relative: treating yields tau_true if treated vs 0 control scale
    v_hat = float(np.mean(a_hat * tau_true))
    v_star = float(np.mean(a_star * tau_true))
    return v_hat, float(v_star - v_hat)


# ---------------------------------------------------------------------------
# Estimator backends
# ---------------------------------------------------------------------------

def fit_predict_xlearner(X_tr, T_tr, Y_tr, X_te, seed: int) -> np.ndarray:
    try:
        from econml.metalearners import XLearner
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    except ImportError as e:
        raise ImportError("econml + sklearn required for X-Learner") from e

    models = GradientBoostingRegressor(random_state=seed)
    propensity = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)
    xl = XLearner(models=models, propensity_model=propensity)
    xl.fit(Y_tr, T_tr, X=X_tr)
    return np.asarray(xl.effect(X_te)).reshape(-1)


def fit_predict_causalforest(X_tr, T_tr, Y_tr, X_te, seed: int) -> np.ndarray:
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as e:
        raise ImportError("econml required for CausalForest") from e

    est = CausalForestDML(
        model_y=GradientBoostingRegressor(random_state=seed),
        model_t=GradientBoostingRegressor(random_state=seed),
        n_estimators=200,
        min_samples_leaf=20,
        random_state=seed,
    )
    est.fit(Y_tr, T_tr, X=X_tr, W=None)
    return np.asarray(est.effect(X_te)).reshape(-1)


def fit_predict_neural(
    name: str,
    X_tr, T_tr, Y_tr, X_te,
    seed: int,
    epochs: int,
    patience: int,
    allow_fallback: bool = False,
) -> np.ndarray:
    """
    Project models only by default (RIPOAttention / TARNet_FTTransformer).
    Fallback Mini-TARNet only if allow_fallback=True (NOT for paper numbers).
    """
    try:
        return _fit_predict_project_neural(name, X_tr, T_tr, Y_tr, X_te, seed, epochs, patience)
    except Exception as e:
        if not allow_fallback:
            raise RuntimeError(
                "Project neural path failed for %s: %s. "
                "Fix model wiring; do not use fallback for paper synthetic runs."
                % (name, e)
            ) from e
        print(f"  [warn] project neural path failed ({e}); using minimal fallback")
        return _fit_predict_minimal_tarnet(X_tr, T_tr, Y_tr, X_te, seed, epochs, patience)


def _fit_predict_project_neural(name, X_tr, T_tr, Y_tr, X_te, seed, epochs, patience):
    """Wire to real project classes: RIPOAttention, TARNet_FTTransformer."""
    import inspect
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = int(X_tr.shape[1])

    def _construct(cls, extra_kwargs=None):
        """Match __init__ parameters that exist on the class."""
        extra_kwargs = dict(extra_kwargs or {})
        sig = inspect.signature(cls.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        candidates = {
            "n_features": n_features,
            "input_dim": n_features,
            "d_in": n_features,
            "n_cont_features": n_features,
            "num_features": n_features,
            "num_continuous": n_features,
            "num_categories": 0,
            "cat_cardinalities": None,
            "in_dim": n_features,
            "d_token": 32,
            "num_layers": 3,
            "n_layers": 3,
            "nhead": 4,
            "n_heads": 4,
            "dim_feedforward": 64,
            "d_ffn": 64,
            "dropout": 0.1,
            "head_hidden_dim": 64,
            "hidden_dim": 64,
            "ipm_weight": 0.1,
            "residual_alpha": 0.25,
            # IMPORTANT: tree_signal_dim=0 creates Linear(0,*) and crashes.
            # Paper no-tree path keeps dim=4 and feeds zeros (see locked HP).
            "tree_signal_dim": 4,
            "use_tree_signals": False,
            "use_tree": False,
            "with_tree": False,
            "n_tree_features": 4,
        }
        candidates.update(extra_kwargs)
        kwargs = {k: v for k, v in candidates.items() if k in params}
        # required params without default must be filled
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if param.default is inspect.Parameter.empty and pname not in kwargs:
                # last resort guesses
                if "feature" in pname or pname in ("d_in", "input_dim", "in_dim"):
                    kwargs[pname] = n_features
                elif pname in ("d_token", "hidden_dim", "dim"):
                    kwargs[pname] = 32
        print(f"  constructing {cls.__name__}({kwargs})")
        return cls(**kwargs)

    if name == "ripo_no_tree":
        from src.models_hybrid_v2 import RIPOAttention

        model = _construct(
            RIPOAttention,
            {
                "use_tree_signals": False,
                "use_tree": False,
                "with_tree": False,
                "tree_signal_dim": 4,
                "n_tree_features": 4,
            },
        ).to(device)
    elif name == "ft_tarnet":
        from src.models_neural import TARNet_FTTransformer

        model = _construct(TARNet_FTTransformer).to(device)
    else:
        raise ValueError(name)


    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    T_t = torch.tensor(T_tr, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    # Zero tree signals for ripo_no_tree (dim must match tree_signal_dim=4)
    tree_dim = 4
    tree_tr = torch.zeros(len(X_tr), tree_dim, dtype=torch.float32, device=device)
    tree_te = torch.zeros(len(X_te), tree_dim, dtype=torch.float32, device=device)

    def _forward_y0y1(model, x, t=None, tree=None):
        """Try common forward signatures; return (y0, y1) tensors."""
        trials = []
        if tree is not None and t is not None:
            trials.extend([
                lambda: model(x, t, tree),
                lambda: model(x, tree, t),
                lambda: model(x, t, tree_signals=tree),
                lambda: model(x, tree_signals=tree),
            ])
        if t is not None:
            trials.append(lambda: model(x, t))
        trials.append(lambda: model(x))
        if hasattr(model, "predict_ite"):
            trials.append(lambda: model.predict_ite(x))
        last_err = None
        for fn in trials:
            try:
                out = fn()
            except Exception as e:
                last_err = e
                continue
            if isinstance(out, dict):
                if "y0" in out and "y1" in out:
                    return out["y0"].squeeze(), out["y1"].squeeze()
                if "mu0" in out and "mu1" in out:
                    return out["mu0"].squeeze(), out["mu1"].squeeze()
                if "tau" in out:
                    # fabricate y0=0, y1=tau for loss only if needed
                    tau = out["tau"].squeeze()
                    return torch.zeros_like(tau), tau
            if isinstance(out, (tuple, list)):
                if len(out) >= 2:
                    a, b = out[0], out[1]
                    if torch.is_tensor(a) and torch.is_tensor(b):
                        return a.squeeze(), b.squeeze()
                if len(out) == 1 and torch.is_tensor(out[0]):
                    # single tau
                    tau = out[0].squeeze()
                    return torch.zeros_like(tau), tau
            if torch.is_tensor(out):
                tau = out.squeeze()
                return torch.zeros_like(tau), tau
        raise RuntimeError(f"Could not parse model forward output; last_err={last_err}")

    ds = TensorDataset(X_t, T_t, Y_t, tree_tr)
    loader = DataLoader(ds, batch_size=min(128, len(X_tr)), shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_loss = float("inf")
    bad = 0
    best_state = None
    model.train()
    for epoch in range(epochs):
        total = 0.0
        nobs = 0
        for xb, tb, yb, treeb in loader:
            opt.zero_grad()
            y0, y1 = _forward_y0y1(model, xb, tb, treeb if name == "ripo_no_tree" else None)
            pred = tb * y1 + (1.0 - tb) * y0
            loss = torch.mean((pred - yb) ** 2)
            if hasattr(model, "ipm_loss"):
                try:
                    loss = loss + 0.1 * model.ipm_loss(xb, tb)
                except Exception:
                    pass
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            nobs += len(xb)
        epoch_loss = total / max(nobs, 1)
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xte = torch.tensor(X_te, dtype=torch.float32, device=device)
        T0 = torch.zeros(len(X_te), dtype=torch.float32, device=device)
        y0, y1 = _forward_y0y1(
            model, Xte, T0, tree_te if name == "ripo_no_tree" else None
        )
        tau_hat = (y1 - y0).cpu().numpy()
    return np.asarray(tau_hat, dtype=np.float64).reshape(-1)



def _nargs(model) -> int:
    import inspect
    try:
        return len(inspect.signature(model.forward).parameters)
    except Exception:
        return 2


def _fit_predict_minimal_tarnet(X_tr, T_tr, Y_tr, X_te, seed, epochs, patience):
    """Minimal shared-representation TARNet fallback (PyTorch)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = X_tr.shape[1]

    class MiniTARNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(
                nn.Linear(p, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
            )
            self.h0 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
            self.h1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            z = self.phi(x)
            return self.h0(z).squeeze(-1), self.h1(z).squeeze(-1)

    model = MiniTARNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    T_t = torch.tensor(T_tr, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(X_t, T_t, Y_t), batch_size=128, shuffle=True)

    best, bad, state = float("inf"), 0, None
    for _ in range(epochs):
        model.train()
        total, nobs = 0.0, 0
        for xb, tb, yb in loader:
            opt.zero_grad()
            y0, y1 = model(xb)
            pred = tb * y1 + (1 - tb) * y0
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            nobs += len(xb)
        el = total / max(nobs, 1)
        if el < best - 1e-5:
            best, bad = el, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        y0, y1 = model(torch.tensor(X_te, dtype=torch.float32, device=device))
        tau_hat = (y1 - y0).cpu().numpy()
    return np.asarray(tau_hat, dtype=np.float64).reshape(-1)


def run_one(
    model: str,
    data_tr: Dict[str, np.ndarray],
    data_te: Dict[str, np.ndarray],
    seed: int,
    epochs: int,
    patience: int,
) -> Dict[str, float]:
    X_tr, T_tr, Y_tr = data_tr["X"], data_tr["T"], data_tr["Y"]
    X_te, tau_te = data_te["X"], data_te["tau"]

    if model == "xlearner":
        tau_hat = fit_predict_xlearner(X_tr, T_tr, Y_tr, X_te, seed)
    elif model == "causalforest":
        tau_hat = fit_predict_causalforest(X_tr, T_tr, Y_tr, X_te, seed)
    elif model in ("ripo_no_tree", "ft_tarnet"):
        tau_hat = fit_predict_neural(model, X_tr, T_tr, Y_tr, X_te, seed, epochs, patience)
    else:
        raise ValueError(model)

    p_val, regret = policy_value_and_regret(tau_hat, tau_te)
    return {
        "PEHE": pehe(tau_hat, tau_te),
        "ATE_Error": ate_error(tau_hat, tau_te),
        "Policy_Value": p_val,
        "Regret": regret,
    }


def summarize(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gcols = ["regime", "signal", "model"]
    summary = (
        raw.groupby(gcols)
        .agg(
            n=("PEHE", "count"),
            PEHE_mean=("PEHE", "mean"),
            PEHE_std=("PEHE", "std"),
            ATE_Error_mean=("ATE_Error", "mean"),
            ATE_Error_std=("ATE_Error", "std"),
            Regret_mean=("Regret", "mean"),
        )
        .reset_index()
    )

    # winner frequency
    winners = []
    for (regime, signal), sub in raw.groupby(["regime", "signal"]):
        for seed, ssub in sub.groupby("seed"):
            i = ssub["PEHE"].idxmin()
            winners.append(
                {"regime": regime, "signal": signal, "seed": seed, "winner": ssub.loc[i, "model"]}
            )
    wdf = pd.DataFrame(winners)
    freq = (
        wdf.groupby(["regime", "signal", "winner"])
        .size()
        .reset_index(name="n_wins")
    )
    return summary, freq


def wilcoxon_key_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    pairs = [
        ("ripo_no_tree", "xlearner"),
        ("ripo_no_tree", "causalforest"),
        ("ripo_no_tree", "ft_tarnet"),
        ("xlearner", "causalforest"),
    ]
    rows = []
    for (regime, signal), sub in raw.groupby(["regime", "signal"]):
        for a, b in pairs:
            da = sub[sub.model == a].set_index("seed")["PEHE"]
            db = sub[sub.model == b].set_index("seed")["PEHE"]
            common = da.index.intersection(db.index)
            if len(common) < 5:
                continue
            x, y = da.loc[common].values, db.loc[common].values
            try:
                stat, p = wilcoxon(x, y)
            except Exception:
                stat, p = np.nan, np.nan
            rows.append(
                {
                    "regime": regime,
                    "signal": signal,
                    "model_a": a,
                    "model_b": b,
                    "n": len(common),
                    "mean_a": float(np.mean(x)),
                    "mean_b": float(np.mean(y)),
                    "diff_a_minus_b": float(np.mean(x) - np.mean(y)),
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": p,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", type=str, default="A,B,C")
    parser.add_argument("--signals", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--models", type=str, default="ripo_no_tree,ft_tarnet,xlearner,causalforest")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--p", type=int, default=12)
    args = parser.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",")]
    signals = _parse_floats(args.signals)
    seeds = _parse_seeds(args.seeds)
    models = [m.strip() for m in args.models.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    norm_constants = compute_all_norm_constants(p=args.p)
    print("Norm constants:", norm_constants)
    print(f"Grid: regimes={regimes} signals={signals} seeds={seeds} models={models}")
    print(f"Total runs: {len(regimes)*len(signals)*len(seeds)*len(models)}")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for regime in regimes:
        for signal in signals:
            for seed in seeds:
                cfg = SyntheticConfig(
                    n=args.n, p=args.p, seed=seed, regime=regime, signal=signal  # type: ignore
                )
                data, meta = generate_synthetic(cfg, norm_constants)
                tr, te = split_train_test(data)
                for model in models:
                    print(f"=== regime={regime} signal={signal} seed={seed} model={model} ===")
                    try:
                        metrics = run_one(model, tr, te, seed, args.epochs, args.patience)
                        print(f"  PEHE={metrics['PEHE']:.4f} ATE_Err={metrics['ATE_Error']:.4f}")
                        rows.append(
                            {
                                "regime": regime,
                                "signal": signal,
                                "seed": seed,
                                "model": model,
                                **metrics,
                                "n_train": meta["n_train"],
                                "n_test": meta["n_test"],
                            }
                        )
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        rows.append(
                            {
                                "regime": regime,
                                "signal": signal,
                                "seed": seed,
                                "model": model,
                                "PEHE": np.nan,
                                "ATE_Error": np.nan,
                                "Policy_Value": np.nan,
                                "Regret": np.nan,
                                "n_train": meta["n_train"],
                                "n_test": meta["n_test"],
                                "error": str(e),
                            }
                        )
                    # incremental save
                    pd.DataFrame(rows).to_csv(OUT_DIR / "synthetic_raw.csv", index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "synthetic_raw.csv", index=False)
    summary, freq = summarize(raw.dropna(subset=["PEHE"]))
    summary.to_csv(OUT_DIR / "synthetic_summary_by_cell.csv", index=False)
    freq.to_csv(OUT_DIR / "synthetic_winner_freq.csv", index=False)
    try:
        w = wilcoxon_key_pairs(raw.dropna(subset=["PEHE"]))
        w.to_csv(OUT_DIR / "synthetic_wilcoxon.csv", index=False)
    except Exception as e:
        print("Wilcoxon failed:", e)

    meta_out = {
        "spec": "synthetic-spec",
        "norm_constants": norm_constants,
        "regimes": regimes,
        "signals": signals,
        "seeds": seeds,
        "models": models,
        "n": args.n,
        "p": args.p,
        "epochs": args.epochs,
        "patience": args.patience,
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / "synthetic_meta.json").write_text(json.dumps(meta_out, indent=2))

    print("\n=== Summary (PEHE mean) ===")
    print(summary.to_string(index=False))
    print("\n=== Winner frequency ===")
    print(freq.to_string(index=False))
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
