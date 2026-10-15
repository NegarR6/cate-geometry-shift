"""Plot synthetic-spec PEHE curves for the paper figure."""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "synthetic_regimes" / "synthetic_summary_by_cell.csv"
OUT = ROOT / "results" / "synthetic_regimes" / "fig_pehe_by_regime_signal.png"


def main():
    df = pd.read_csv(CSV)
    regimes = ["A", "B", "C"]
    titles = {
        "A": "A: Smooth",
        "B": "B: Threshold / tree-like",
        "C": "C: Interaction",
    }
    models = ["ripo_no_tree", "ft_tarnet", "xlearner", "causalforest"]
    labels = {
        "ripo_no_tree": "RIPO (no tree)",
        "ft_tarnet": "FT-TARNet",
        "xlearner": "X-Learner",
        "causalforest": "Causal Forest",
    }
    colors = {
        "ripo_no_tree": "#1f77b4",
        "ft_tarnet": "#17becf",
        "xlearner": "#d62728",
        "causalforest": "#ff7f0e",
    }
    markers = {
        "ripo_no_tree": "o",
        "ft_tarnet": "s",
        "xlearner": "^",
        "causalforest": "D",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=False)
    for ax, regime in zip(axes, regimes):
        sub = df[df.regime == regime]
        for m in models:
            s = sub[sub.model == m].sort_values("signal")
            ax.errorbar(
                s["signal"],
                s["PEHE_mean"],
                yerr=s["PEHE_std"],
                label=labels[m],
                color=colors[m],
                marker=markers[m],
                linewidth=2,
                capsize=3,
            )
        ax.set_title(titles[regime], fontsize=11)
        ax.set_xlabel("Signal strength $s$")
        ax.set_xticks([0.5, 1.0, 2.0])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PEHE (mean ± std over 10 seeds)")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    paper_out = ROOT / "figures" / "paper" / "fig_pehe_by_regime_signal.pdf"
    paper_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_out, bbox_inches="tight")
    print(f"Saved {OUT}")
    print(f"Saved {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
