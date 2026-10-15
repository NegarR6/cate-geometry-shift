"""Generate Figure 3 (fig_information_ladders.pdf) from Table 5 and Table 6 summary CSVs."""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn_csv", default="results/budgeted_ladder/budgeted_ladder_summary.csv")
    ap.add_argument("--ihdp_csv", default="results/table6_ihdp_x_ladder_summary.csv")
    ap.add_argument("--out", default="figures/paper/fig_information_ladders.pdf")
    args = ap.parse_args()

    df_t5 = pd.read_csv(args.syn_csv)
    df_t6 = pd.read_csv(args.ihdp_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    styles = {
        "factual": ("-o", "#1f77b4", "Factual"),
        "dr_pseudo": ("-s", "#ff7f0e", "DR pseudo-$\\tau$ (transductive)"),
        "oracle_tau": ("-^", "#2ca02c", "Oracle $\\tau$"),
    }

    # Left panel: Synthetic {A, B} -> C
    sub5 = df_t5[(df_t5["policy"] == "random") & (df_t5["m"] > 0)] if "policy" in df_t5.columns else df_t5[df_t5["m"] > 0]
    frozen_syn = float(df_t5["pehe_frozen_mean"].iloc[0])
    for sup, (fmt, color, label) in styles.items():
        s = sub5[sub5["supervision"] == sup].sort_values("m")
        if s.empty:
            continue
        ax1.errorbar(
            s["m"], s["pehe_after_mean"], yerr=s["pehe_after_std"],
            fmt=fmt, color=color, capsize=3, label=label, linewidth=1.8, markersize=6
        )
    ax1.axhline(frozen_syn, color="#555555", linestyle="--", linewidth=1.4, label=f"Frozen mixture ({frozen_syn:.2f})")
    ax1.axhline(1.010, color="#8c564b", linestyle=":", linewidth=1.4, label="Constant ATE (oracle, 1.01)")
    ax1.axhline(0.544, color="#9467bd", linestyle="-.", linewidth=1.4, label="Target T-learner (n=1400, 0.54)")
    ax1.set_title("Synthetic geometry shift $\\{A, B\\} \\to C$")
    ax1.set_xlabel("Target supervision budget $m$")
    ax1.set_ylabel("Target root PEHE")
    ax1.set_xticks([10, 20, 30, 50, 100])
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, loc="upper right", fontsize=8)

    # Right panel: Semi-synthetic IHDP-X (Frozen = 1.96)
    frozen_ihdp = float(df_t6["pehe_frozen_mean"].iloc[0])
    mean_col = "pehe_always_adapt_mean" if "pehe_always_adapt_mean" in df_t6.columns else "pehe_after_mean"
    std_col = "pehe_always_adapt_std" if "pehe_always_adapt_std" in df_t6.columns else "pehe_after_std"

    for sup, (fmt, color, label) in styles.items():
        s = df_t6[df_t6["supervision"] == sup].sort_values("m")
        if s.empty:
            continue
        ax2.errorbar(
            s["m"], s[mean_col], yerr=s[std_col],
            fmt=fmt, color=color, capsize=3, label=label, linewidth=1.8, markersize=6
        )
    ax2.axhline(frozen_ihdp, color="#555555", linestyle="--", linewidth=1.4, label=f"Frozen mixture ({frozen_ihdp:.2f})")
    ax2.set_title("Semi-synthetic IHDP-X (smooth $\\to$ threshold)")
    ax2.set_xlabel("Target supervision budget $m$")
    ax2.set_ylabel("Target root PEHE")
    ax2.set_xticks([10, 20, 50, 100])
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False, loc="upper right", fontsize=8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved vector PDF figure to {out_path}")


if __name__ == "__main__":
    main()
