"""
Level 2 — PEHE decomposition under mean-diff balancing via CausalLoss.

CRITICAL: RIPOAttention has NO model.ipm_loss. Balancing lives in
src.train_neural.CausalLoss and needs `representation` from the model.

  PEHE^2 ≈ MSE(mu0) + MSE(mu1) + cross
  cross := PEHE^2 - MSE0 - MSE1

Smoke:
  python -m src.run_pehe_decomposition --regimes A,B --signals 1.0 --seeds 0-2 --epochs 40 --patience 10

Full:
  python -m src.run_pehe_decomposition --regimes A,B,C --signals 0.5,1.0,2.0 --seeds 0-9 --epochs 80 --patience 15
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "synthetic_regimes" / "pehe_decomposition"

from src.synthetic_regimes import (  # noqa: E402
    SyntheticConfig,
    compute_all_norm_constants,
    generate_synthetic,
    split_train_test,
)
from src.run_synthetic_regimes import _parse_floats, _parse_seeds  # noqa: E402



def _ess(e: np.ndarray, t: np.ndarray) -> dict:
    """Effective sample size style overlap diagnostics from propensity."""
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    e = np.clip(e, 1e-3, 1 - 1e-3)
    w1 = t / e
    w0 = (1 - t) / (1 - e)
    def ess(w):
        s = w.sum()
        return float((s * s) / (np.sum(w * w) + 1e-12))
    n = len(t)
    return {
        "ESS_treated": ess(w1),
        "ESS_control": ess(w0),
        "ESS_ratio": ess(w1) / (ess(w0) + 1e-12),
        "mean_abs_e_half": float(np.mean(np.abs(e - 0.5))),
        "frac_extreme_e": float(np.mean((e < 0.05) | (e > 0.95))),
        "n": n,
    }

def _mse(a, b) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d ** 2))


def _forward_mu_rep(model, x, tree, device):
    """Return (mu0, mu1, representation_or_None)."""
    t0 = torch.zeros(len(x), device=device)
    trials = [
        lambda: model(x, return_representation=True),
        lambda: model(x, t0, tree, return_representation=True),
        lambda: model(x, tree_signals=tree, return_representation=True),
        lambda: model(x, t0, tree),
        lambda: model(x, tree_signals=tree),
        lambda: model(x),
    ]
    last = None
    for fn in trials:
        try:
            out = fn()
        except Exception as e:
            last = e
            continue
        if isinstance(out, dict):
            mu0 = out.get("mu0", out.get("y0"))
            mu1 = out.get("mu1", out.get("y1"))
            rep = out.get("representation", out.get("rep", out.get("phi")))
            if mu0 is not None and mu1 is not None:
                return mu0.squeeze(), mu1.squeeze(), rep
        if isinstance(out, (tuple, list)):
            if len(out) >= 3 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                return out[0].squeeze(), out[1].squeeze(), out[2]
            if len(out) >= 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                return out[0].squeeze(), out[1].squeeze(), None
        if torch.is_tensor(out):
            tau = out.squeeze()
            return torch.zeros_like(tau), tau, None
    raise RuntimeError(f"cannot parse forward for mu/rep; last={last}")


def train_ripo_mu(
    X_tr, T_tr, Y_tr, X_te,
    seed: int,
    epochs: int,
    patience: int,
    ipm_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    from src.models_hybrid_v2 import RIPOAttention
    from src.train_neural import CausalLoss

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = int(X_tr.shape[1])

    sig = inspect.signature(RIPOAttention.__init__)
    params = set(sig.parameters.keys()) - {"self"}
    cand = {
        "num_features": n_features,
        "n_features": n_features,
        "d_token": 32,
        "num_layers": 3,
        "nhead": 4,
        "dim_feedforward": 64,
        "dropout": 0.1,
        "head_hidden_dim": 64,
        "residual_alpha": 0.25,
        "tree_signal_dim": 4,
        "use_tree_signals": False,
        "use_tree": False,
    }
    kwargs = {k: v for k, v in cand.items() if k in params}
    model = RIPOAttention(**kwargs).to(device)
    loss_fn = CausalLoss(ipm_weight=float(ipm_weight), ipm_type="mean_diff")

    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    T_t = torch.tensor(T_tr, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    tree_tr = torch.zeros(len(X_tr), 4, dtype=torch.float32, device=device)
    tree_te = torch.zeros(len(X_te), 4, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=device)

    ds = TensorDataset(X_t, T_t, Y_t, tree_tr)
    loader = DataLoader(ds, batch_size=min(128, len(X_tr)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best, bad, state = float("inf"), 0, None
    last_ipm = 0.0
    model.train()
    for _ in range(epochs):
        total, nobs, ipm_sum, nb = 0.0, 0, 0.0, 0
        for xb, tb, yb, treeb in loader:
            opt.zero_grad()
            mu0, mu1, rep = _forward_mu_rep(model, xb, treeb, device)
            if rep is None and ipm_weight > 0:
                # Fallback representation so IPM is not silently zero
                rep = torch.stack([mu0, mu1], dim=-1)
            loss, outcome, ipm_l = loss_fn(mu0, mu1, yb, tb, representation=rep)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            nobs += len(xb)
            ipm_sum += float(ipm_l.item())
            nb += 1
        last_ipm = ipm_sum / max(nb, 1)
        el = total / max(nobs, 1)
        if el < best - 1e-5:
            best, bad = el, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if state is not None:
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    with torch.no_grad():
        mu0, mu1, _ = _forward_mu_rep(model, Xte, tree_te, device)
    mu0 = mu0.cpu().numpy().reshape(-1)
    mu1 = mu1.cpu().numpy().reshape(-1)
    return mu0, mu1, mu1 - mu0, last_ipm


def run_cell(regime, signal, seed, epochs, patience, n, p, norms, ipm_weights):
    cfg = SyntheticConfig(n=n, p=p, seed=seed, regime=regime, signal=signal)  # type: ignore
    data, _ = generate_synthetic(cfg, norms)
    tr, te = split_train_test(data)
    # Overlap from true propensity on train (DGP-known; diagnostic only)
    if "e" in tr:
        ov = _ess(tr["e"], tr["T"])
    else:
        ov = {"ESS_treated": np.nan, "ESS_control": np.nan, "ESS_ratio": np.nan,
              "mean_abs_e_half": np.nan, "frac_extreme_e": np.nan, "n": len(tr["T"])}
    rows = []
    for w in ipm_weights:
        mu0_h, mu1_h, tau_h, last_ipm = train_ripo_mu(
            tr["X"], tr["T"], tr["Y"], te["X"], seed, epochs, patience, w
        )
        mse0 = _mse(mu0_h, te["mu0"])
        mse1 = _mse(mu1_h, te["mu1"])
        p2 = _mse(tau_h, te["tau"])
        cross = p2 - mse0 - mse1
        row = {
                "regime": regime,
                "signal": signal,
                "seed": seed,
                "ipm_weight": w,
                "PEHE": float(np.sqrt(max(p2, 0.0))),
                "PEHE2": p2,
                "MSE_mu0": mse0,
                "MSE_mu1": mse1,
                "cross_term": cross,
                "frac_cross": cross / p2 if p2 > 1e-12 else np.nan,
                "last_epoch_ipm": last_ipm,
            }
        row.update(ov)
        rows.append(row)
        print(
            f"  ipm_w={w}: PEHE={np.sqrt(max(p2,0)):.4f} mse0={mse0:.4f} mse1={mse1:.4f} "
            f"cross={cross:.4f} last_ipm={last_ipm:.6f}"
        )
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
    ap.add_argument("--ipm_weights", type=str, default="0.0,0.1")
    args = ap.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",")]
    signals = _parse_floats(args.signals)
    seeds = _parse_seeds(args.seeds)
    ipm_weights = _parse_floats(args.ipm_weights)

    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = ROOT / "results" / "synthetic_regimes" / "synthetic_meta.json"
    if meta_path.exists():
        norms = json.loads(meta_path.read_text()).get("norm_constants") or compute_all_norm_constants(p=args.p)
    else:
        norms = compute_all_norm_constants(p=args.p)

    print("PEHE decomposition via CausalLoss(mean_diff)")
    print("SANITY: last_epoch_ipm must be > 0 when ipm_weight > 0")
    all_rows: List[Dict] = []
    t0 = time.time()
    for regime in regimes:
        for signal in signals:
            for seed in seeds:
                print(f"=== {regime} signal={signal} seed={seed} ===")
                try:
                    all_rows.extend(
                        run_cell(
                            regime, signal, seed, args.epochs, args.patience,
                            args.n, args.p, norms, ipm_weights,
                        )
                    )
                except Exception as e:
                    print(f"  FAILED: {e}")
                pd.DataFrame(all_rows).to_csv(OUT / "decomp_raw.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "decomp_raw.csv", index=False)
    if len(df):
        summary = (
            df.groupby(["regime", "signal", "ipm_weight"])
            .agg(
                n=("PEHE", "count"),
                PEHE_mean=("PEHE", "mean"),
                PEHE_std=("PEHE", "std"),
                MSE0_mean=("MSE_mu0", "mean"),
                MSE1_mean=("MSE_mu1", "mean"),
                cross_mean=("cross_term", "mean"),
                frac_cross_mean=("frac_cross", "mean"),
                ipm_monitor_mean=("last_epoch_ipm", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(OUT / "decomp_summary.csv", index=False)
        print("\n=== Summary ===")
        print(summary.to_string(index=False))

        if 0.0 in ipm_weights and any(w > 0 for w in ipm_weights):
            w_on = [w for w in ipm_weights if w > 0][0]
            deltas = []
            for (regime, signal, seed), sub in df.groupby(["regime", "signal", "seed"]):
                off = sub[sub.ipm_weight == 0.0]
                on = sub[sub.ipm_weight == w_on]
                if len(off) and len(on):
                    deltas.append(
                        {
                            "regime": regime,
                            "signal": signal,
                            "seed": int(seed),
                            "d_PEHE": float(on.PEHE.values[0] - off.PEHE.values[0]),
                            "d_MSE0": float(on.MSE_mu0.values[0] - off.MSE_mu0.values[0]),
                            "d_MSE1": float(on.MSE_mu1.values[0] - off.MSE_mu1.values[0]),
                            "d_cross": float(on.cross_term.values[0] - off.cross_term.values[0]),
                            "ipm_on_monitor": float(on.last_epoch_ipm.values[0]),
                        }
                    )
            if deltas:
                ddf = pd.DataFrame(deltas)
                ddf.to_csv(OUT / "decomp_delta_ipm.csv", index=False)
                dsum = (
                    ddf.groupby(["regime", "signal"])
                    .agg(
                        n=("d_PEHE", "count"),
                        d_PEHE_mean=("d_PEHE", "mean"),
                        d_cross_mean=("d_cross", "mean"),
                        d_MSE0_mean=("d_MSE0", "mean"),
                        d_MSE1_mean=("d_MSE1", "mean"),
                        ipm_monitor_mean=("ipm_on_monitor", "mean"),
                    )
                    .reset_index()
                )
                dsum.to_csv(OUT / "decomp_delta_summary.csv", index=False)
                print("\n=== Delta (IPM on - off) ===")
                print(dsum.to_string(index=False))
                if ddf["ipm_on_monitor"].abs().max() < 1e-12:
                    print("\n[CRITICAL] last_epoch_ipm still ~0 with weight>0 → representation not wired")

    meta = {
        "spec": "PEHE_DECOMPOSITION_L2_v2_CausalLoss",
        "ipm_weights": ipm_weights,
        "regimes": regimes,
        "signals": signals,
        "seeds": seeds,
        "epochs": args.epochs,
        "elapsed_sec": time.time() - t0,
        "norm_constants": norms,
        "note": "Uses CausalLoss mean_diff; monitors last_epoch_ipm",
    }
    (OUT / "decomp_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
