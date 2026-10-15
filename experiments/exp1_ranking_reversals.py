"""
Ranking reversals across effect-surface geometries (Q1 / Fig 1–2).

P0 FIXES vs original:
- Early stopping uses a held-out validation slice (20% of train), NOT training loss.
- IPM (mean-diff) is applied through CausalLoss on the shared representation; it is
  mandatory (raises if it is identically zero). No try/except-pass anywhere.
- ripo_no_tree uses tree_signals=None (pure FT path), not zero-filled tree tokens.
- TARNet_FTTransformer is built with num_continuous= (the previous num_features=
  raised TypeError and the model silently disappeared from the results).
- Any model failure aborts the run (use --continue_on_error only for debugging).
- Hyperparameters are explicit CLI/DEFAULT_NEURAL_HP and written to synthetic_meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "results" / "synthetic_regimes"

from src.synthetic_regimes import (  # noqa: E402
    SyntheticConfig,
    compute_all_norm_constants,
    generate_synthetic,
    split_train_test,
)


DEFAULT_NEURAL_HP: Dict[str, Any] = dict(
    d_token=32, num_layers=3, nhead=4, dim_feedforward=64, dropout=0.1,
    head_hidden_dim=64, lr=1e-3, weight_decay=1e-5, batch_size=128,
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
    a_hat = (tau_hat > treat_threshold).astype(np.float64)
    a_star = (tau_true > treat_threshold).astype(np.float64)
    v_hat = float(np.mean(a_hat * tau_true))
    v_star = float(np.mean(a_star * tau_true))
    return v_hat, float(v_star - v_hat)


# ---------------------------------------------------------------------------
# Estimator backends
# ---------------------------------------------------------------------------

def fit_predict_xlearner(X_tr, T_tr, Y_tr, X_te, seed: int) -> np.ndarray:
    from econml.metalearners import XLearner
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

    models = GradientBoostingRegressor(random_state=seed)
    propensity = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)
    xl = XLearner(models=models, propensity_model=propensity)
    xl.fit(Y_tr, T_tr, X=X_tr)
    return np.asarray(xl.effect(X_te)).reshape(-1)


def fit_predict_causalforest(X_tr, T_tr, Y_tr, X_te, seed: int) -> np.ndarray:
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

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
    ipm_weight: float = 0.1,
    val_frac: float = 0.2,
    hp: Dict[str, Any] | None = None,
) -> np.ndarray:
    """
    RIPO (pure path, tree_signals=None) or FT-TARNet.

    - Early stopping on a held-out validation slice (factual MSE only).
    - Training loss = CausalLoss = factual MSE + ipm_weight * mean-diff IPM on the
      shared representation z. IPM is REQUIRED (no silent skipping): if
      ipm_weight > 0 and the IPM term is identically zero over training, we raise.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.train_neural import CausalLoss

    hp = {**DEFAULT_NEURAL_HP, **(hp or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = int(X_tr.shape[1])

    if name == "ripo_no_tree":
        from src.models_hybrid_v2 import RIPOAttention
        model = RIPOAttention(
            num_features=n_features,
            tree_signal_dim=4,
            d_token=hp["d_token"],
            num_layers=hp["num_layers"],
            nhead=hp["nhead"],
            dim_feedforward=hp["dim_feedforward"],
            dropout=hp["dropout"],
            head_hidden_dim=hp["head_hidden_dim"],
        ).to(device)
    elif name == "ft_tarnet":
        from src.models_neural import TARNet_FTTransformer
        model = TARNet_FTTransformer(
            num_continuous=n_features,          # NOTE: not `num_features`
            d_token=hp["d_token"],
            num_layers=hp["num_layers"],
            nhead=hp["nhead"],
            dim_feedforward=hp["dim_feedforward"],
            dropout=hp["dropout"],
            head_hidden_dim=hp["head_hidden_dim"],
        ).to(device)
    else:
        raise ValueError(name)

    def _forward(x):
        """Return (y0, y1, z) with y0,y1 of shape [B] and z of shape [B, D]."""
        if name == "ripo_no_tree":
            y0, y1, z, _ = model(x, tree_signals=None)       # pure FT path
        else:
            y0, y1, z = model(x, return_representation=True)
        return y0.reshape(-1), y1.reshape(-1), z

    # ---- validation split (early stopping never sees the training loss) ----
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr))
    n_val = max(32, int(val_frac * len(X_tr)))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]

    def _t(a, idx):
        return torch.tensor(np.asarray(a)[idx], dtype=torch.float32, device=device)

    X_t, T_t, Y_t = _t(X_tr, tr_idx), _t(T_tr, tr_idx), _t(Y_tr, tr_idx)
    X_va, T_va, Y_va = _t(X_tr, va_idx), _t(T_tr, va_idx), _t(Y_tr, va_idx)

    causal = CausalLoss(ipm_weight=ipm_weight, ipm_type="mean_diff")
    loader = DataLoader(
        TensorDataset(X_t, T_t, Y_t), batch_size=min(hp["batch_size"], len(tr_idx)), shuffle=True
    )
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    best_val, bad, best_state = float("inf"), 0, None
    ipm_total = 0.0
    for _epoch in range(epochs):
        model.train()
        for xb, tb, yb in loader:
            opt.zero_grad()
            y0, y1, z = _forward(xb)
            loss, _outcome, ipm_term = causal(y0, y1, yb, tb, z)
            ipm_total += float(ipm_term.detach())
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            y0v, y1v, _ = _forward(X_va)
            pred_va = T_va * y1v + (1.0 - T_va) * y0v
            val_loss = float(torch.mean((pred_va - Y_va) ** 2).item())

        if val_loss < best_val - 1e-5:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if ipm_weight > 0 and ipm_total == 0.0:
        raise RuntimeError(
            "IPM term was identically zero during training although ipm_weight > 0; "
            "check that both treatment arms appear in mini-batches."
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y0, y1, _ = _forward(torch.tensor(np.asarray(X_te), dtype=torch.float32, device=device))
        tau_hat = (y1 - y0).cpu().numpy()
    return np.asarray(tau_hat, dtype=np.float64).reshape(-1)


def run_one(
    model: str,
    data_tr: Dict[str, np.ndarray],
    data_te: Dict[str, np.ndarray],
    seed: int,
    epochs: int,
    patience: int,
    ipm_weight: float = 0.1,
) -> Dict[str, float]:
    X_tr, T_tr, Y_tr = data_tr["X"], data_tr["T"], data_tr["Y"]
    X_te, tau_te = data_te["X"], data_te["tau"]

    if model == "xlearner":
        tau_hat = fit_predict_xlearner(X_tr, T_tr, Y_tr, X_te, seed)
    elif model == "causalforest":
        tau_hat = fit_predict_causalforest(X_tr, T_tr, Y_tr, X_te, seed)
    elif model in ("ripo_no_tree", "ft_tarnet"):
        tau_hat = fit_predict_neural(
            model, X_tr, T_tr, Y_tr, X_te, seed, epochs, patience, ipm_weight=ipm_weight
        )
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
    parser.add_argument("--ipm_weight", type=float, default=0.1)
    parser.add_argument("--continue_on_error", action="store_true",
                        help="debug only: record NaN rows instead of aborting")
    args = parser.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",")]
    signals = _parse_floats(args.signals)
    seeds = _parse_seeds(args.seeds)
    models = [m.strip() for m in args.models.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    norm_constants = compute_all_norm_constants(p=args.p)
    print("Norm constants:", norm_constants)
    print(f"Grid: regimes={regimes} signals={signals} seeds={seeds} models={models}")

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
                        metrics = run_one(model, tr, te, seed, args.epochs, args.patience, args.ipm_weight)
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
                        if not args.continue_on_error:
                            raise
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
                    pd.DataFrame(rows).to_csv(OUT_DIR / "synthetic_raw.csv", index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "synthetic_raw.csv", index=False)
    n_failed = int(raw["PEHE"].isna().sum())
    expected = len(regimes) * len(signals) * len(seeds) * len(models)
    if n_failed or len(raw) != expected:
        raise SystemExit(f"{n_failed} failed runs / {len(raw)} of {expected} expected; "
                         "refusing to summarise an incomplete grid")
    summary, freq = summarize(raw)
    summary.to_csv(OUT_DIR / "synthetic_summary_by_cell.csv", index=False)
    freq.to_csv(OUT_DIR / "synthetic_winner_freq.csv", index=False)

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
        "early_stopping": "validation_20pct",
        "ipm_weight": args.ipm_weight,
        "ipm_type": "mean_diff",
        "neural_hparams": DEFAULT_NEURAL_HP,
        "continue_on_error": bool(args.continue_on_error),
        "ripo_tree_path": "None (pure FT)",
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
