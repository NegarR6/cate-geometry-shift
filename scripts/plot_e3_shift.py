"""
Plot E3 shift results (F5) from e3_summary.csv.
Usage:
  python artifacts/paper_pack/plot_e3_shift.py path/to/e3_summary.csv [outdir]
Default outdir: same folder as the CSV.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(csv_path: Path, outdir: Path) -> None:
    df = pd.read_csv(csv_path)
    required = {"src", "tgt", "signal", "delta_tgt_mean", "delta_tgt_std"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    pairs = [("A", "B"), ("A", "C"), ("B", "A"), ("B", "C")]
    signals = sorted(df["signal"].unique())
    x = np.arange(len(signals))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    # --- Left: delta_tgt (all should be negative) ---
    ax = axes[0]
    for i, (s, t) in enumerate(pairs):
        sub = df[(df["src"] == s) & (df["tgt"] == t)].sort_values("signal")
        if sub.empty:
            continue
        vals = sub["delta_tgt_mean"].to_numpy()
        err = sub["delta_tgt_std"].to_numpy()
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            yerr=err,
            capsize=2,
            label=f"{s}→{t}",
        )
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in signals])
    ax.set_xlabel("Signal s")
    ax.set_ylabel(r"$\Delta_{\mathrm{tgt}}$ (global $-$ router)")
    ax.set_title("Shift: router vs target global (negative = fail)")
    ax.legend(fontsize=8, ncol=2)

    # --- Right: in-distribution control delta_src_id ---
    ax = axes[1]
    if "delta_src_id_mean" in df.columns:
        # one value per (src, signal); average over targets sharing same src
        ctrl = (
            df.groupby(["src", "signal"], as_index=False)["delta_src_id_mean"]
            .mean()
            .sort_values(["src", "signal"])
        )
        for i, src in enumerate(sorted(ctrl["src"].unique())):
            sub = ctrl[ctrl["src"] == src]
            ax.plot(
                sub["signal"],
                sub["delta_src_id_mean"],
                marker="o",
                label=f"src={src} (ID)",
            )
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Signal s")
        ax.set_ylabel(r"$\Delta_{\mathrm{src\,ID}}$")
        ax.set_title("In-distribution control (should match E2)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no delta_src_id_mean column", ha="center")
        ax.axis("off")

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_e3_shift.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved → {outdir / 'fig_e3_shift.png'}")
    print(f"Saved → {outdir / 'fig_e3_shift.pdf'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    csv = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else csv.parent
    main(csv, out)
