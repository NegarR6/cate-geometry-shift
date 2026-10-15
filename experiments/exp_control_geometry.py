"""P0 control: geometry-within-class shifts A->A2 and C->C2.

If PEHE under A->A2 is also near/above constant baselines, failure is not
unique to cross-geometry shift (identity/function change also hurts).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from synthetic_regimes import SyntheticConfig, compute_all_norm_constants, generate_synthetic
from budgeted_ladder_utils import source_ate_estimate, pehe


def split_tr_te(data):
    tr, te = data["train_idx"], data["test_idx"]
    return {k: (v[tr] if hasattr(v, "__getitem__") and k not in ("train_idx","test_idx","val_idx") else v) for k,v in data.items()}, \
           {k: (v[te] if hasattr(v, "__getitem__") and k not in ("train_idx","test_idx","val_idx") else v) for k,v in data.items()}


def eval_transfer(src_regimes, tgt_regime, n=2000, p=12, seed=0, signal=1.0):
    norms = compute_all_norm_constants(p=p)
    # pool source train
    Xs, Ts, Ys, taus = [], [], [], []
    for i, r in enumerate(src_regimes):
        cfg = SyntheticConfig(n=n, p=p, seed=seed+17*(i+1), regime=r, signal=signal)
        d, _ = generate_synthetic(cfg, norm_constants=norms)
        tr = d["train_idx"]
        Xs.append(d["X"][tr]); Ts.append(d["T"][tr].astype(int)); Ys.append(d["Y"][tr]); taus.append(d["tau"][tr])
    Xs = np.vstack(Xs); Ts = np.concatenate(Ts); Ys = np.concatenate(Ys)

    cfg_t = SyntheticConfig(n=n, p=p, seed=seed+99, regime=tgt_regime, signal=signal)
    tgt, _ = generate_synthetic(cfg_t, norm_constants=norms)
    te = tgt["test_idx"]
    tau_te = tgt["tau"][te]
    pool = tgt["train_idx"]

    # baselines on target
    zero = pehe(tau_te, np.zeros(len(te)))
    const_oracle = pehe(tau_te, np.full(len(te), float(np.mean(tgt["tau"][pool]))))
    ate_src = source_ate_estimate(Xs, Ts, Ys, seed=seed)
    src_const = pehe(tau_te, np.full(len(te), ate_src))

    # simple target-only ridge on full pool factual via T-learner-style two ridge
    from sklearn.ensemble import GradientBoostingRegressor
    Xpool, Tpool, Ypool = tgt["X"][pool], tgt["T"][pool].astype(int), tgt["Y"][pool]
    m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
    m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
    m0.fit(Xpool[Tpool==0], Ypool[Tpool==0])
    m1.fit(Xpool[Tpool==1], Ypool[Tpool==1])
    t_full = pehe(tau_te, m1.predict(tgt["X"][te]) - m0.predict(tgt["X"][te]))

    return {
        "src": ",".join(src_regimes),
        "tgt": tgt_regime,
        "seed": seed,
        "pehe_zero": zero,
        "pehe_const_oracle": const_oracle,
        "pehe_source_ate": src_const,
        "pehe_target_T_full": t_full,
        "corr_src_mean_tau": float(np.mean([np.corrcoef(np.concatenate(taus), np.tile(tgt["tau"][pool][:len(taus[0])], len(src_regimes)))[0,1]]) if False else np.nan),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--out", default="results/table_control_geometry.csv")
    args = ap.parse_args()
    rows = []
    pairs = [
        (["A", "B"], "C"),   # cross-geometry (paper main)
        (["A"], "A2"),       # within smooth class
        (["C"], "C2"),       # within interaction class
        (["A", "B"], "A2"),
    ]
    for seed in range(args.n_seeds):
        for src, tgt in pairs:
            print(f"seed={seed} {src}->{tgt}")
            rows.append(eval_transfer(src, tgt, seed=seed))
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summ = df.groupby(["src", "tgt"])[["pehe_zero","pehe_const_oracle","pehe_source_ate","pehe_target_T_full"]].agg(["mean","std"])
    print(summ)
    print("saved", args.out)

if __name__ == "__main__":
    main()
