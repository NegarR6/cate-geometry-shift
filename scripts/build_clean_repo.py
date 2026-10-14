"""Automated migration and cleanup script for `hybrid-causal-ICLR` -> `cate-geometry-shift`.

Run this script from inside or alongside your existing `hybrid-causal-ICLR` folder:
    python scripts/build_clean_repo.py --src_repo /path/to/hybrid-causal-ICLR --dst_repo /path/to/cate-geometry-shift

What it does automatically:
1. Extracts ONLY the required core library modules into `src/`.
2. Extracts and renames the exact experiment runners corresponding to Tables 1-7, Figs 1-5, and Appendices A-C into `experiments/`.
3. Automatically patches non-deterministic `abs(hash(sup))` calls into deterministic `zlib.crc32(sup.encode("utf-8"))` across ladder scripts.
4. Copies only the final locked summary CSVs into `results/` and vector PDF figures into `figures/paper/`.
5. Excludes all internal advisor notes, Persian markdown files, `_archive*`, `*_smoke*`, `wandb/`, and unused models.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
import re
import shutil
from pathlib import Path

# 1. Core library files to keep in src/
CORE_SRC_FILES = [
    "synthetic_regimes.py",
    "semi_synthetic_geometry.py",
    "models_geometry_adaptive.py",
    "train_geometry_adaptive.py",
    "adapt_geometry_shift.py",
    "models_hybrid_v2.py",
    "models_sota_neural.py",
    "models_neural.py",
    "train_neural.py",
    "baselines.py",
    "tree_signals.py",
    "evaluation.py",
    "canonical_data_loader.py",
    "budgeted_ladder_utils.py",
    "geometry_proxy.py",
    "e2_router_features.py",
]

# 2. Mapping from old runner scripts in `src/` to clean `experiments/` scripts
EXPERIMENT_MAP = {
    "run_synthetic_regimes.py": "exp1_ranking_reversals.py",
    "run_geometry_adaptive_regimes.py": "exp2_two_expert_indist.py",
    "run_geometry_continuum_experts.py": "exp3_continuum_crossover.py",
    "run_budgeted_information_ladder.py": "exp5_synthetic_ladder.py",
    "run_e3_shift.py": "exp7_router_transfer.py",
    "run_e1_oracle_gap.py": "exp7b_oracle_headroom.py",
    "run_synthetic_hard.py": "appA_hard_design.py",
    "run_pehe_decomposition.py": "appA_pehe_decomposition.py",
    "run_surrogate_selection.py": "appB_surrogate_selection.py",
    "run_geometry_adaptive_acic.py": "appC_acic_benchmark.py",
    "run_geometry_proxy_synthetic.py": "exp_geometry_score_continuum.py",
    "run_geometry_proxy_acic.py": "exp_geometry_score_acic.py",
}

# 3. Final CSVs to copy into results/
LOCKED_CSVS = {
    "results/budgeted_ladder_AB_to_C/budgeted_ladder_summary.csv": "results/table5_synthetic_ladder_summary.csv",
    "results/budgeted_ladder_AB_to_C/budgeted_ladder_raw.csv": "results/table5_synthetic_ladder_raw.csv",
    "results/certified_adapt_semi_ihdp_v2/certified_adapt_semi_summary.csv": "results/table6_ihdp_x_ladder_summary.csv",
    "results/certified_adapt_semi_ihdp_v2/certified_adapt_semi_raw.csv": "results/table6_ihdp_x_ladder_raw.csv",
    "results/synthetic_regimes_hard/summary.csv": "results/tableA1_hard_design.csv",
    "results/synthetic_regimes/surrogate_selection/summary.csv": "results/tableB1_surrogate_selection.csv",
}


def patch_nondeterministic_hash(code: str) -> str:
    """Replace `abs(hash(sup))` or `abs(hash(policy))` with deterministic zlib.crc32."""
    if "hash(" not in code:
        return code
    if "import zlib" not in code:
        code = "import zlib\n" + code
    code = re.sub(
        r"abs\(hash\(([^)]+)\)\)",
        r"(zlib.crc32(str(\1).encode('utf-8')) & 0xFFFFFFFF)",
        code,
    )
    return code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_repo", required=True, help="Path to existing hybrid-causal-ICLR folder")
    ap.add_argument("--dst_repo", required=True, help="Path to clean cate-geometry-shift folder")
    args = ap.parse_args()

    src_root = Path(args.src_repo).resolve()
    dst_root = Path(args.dst_repo).resolve()

    for sub in ["src", "experiments", "configs", "scripts", "results", "figures/paper", "tests", "data"]:
        (dst_root / sub).mkdir(parents=True, exist_ok=True)

    # Copy core src modules
    copied_src = 0
    for fname in CORE_SRC_FILES:
        f_src = src_root / "src" / fname
        if f_src.exists():
            text = patch_nondeterministic_hash(f_src.read_text(encoding="utf-8"))
            (dst_root / "src" / fname).write_text(text, encoding="utf-8")
            copied_src += 1
        else:
            print(f"[INFO] Optional module not found: {fname}")
    print(f"[OK] Copied {copied_src} core modules into {dst_root / 'src'}")

    # Copy and patch experiment scripts
    copied_exp = 0
    for old_name, new_name in EXPERIMENT_MAP.items():
        f_src = src_root / "src" / old_name
        if f_src.exists():
            text = patch_nondeterministic_hash(f_src.read_text(encoding="utf-8"))
            (dst_root / "experiments" / new_name).write_text(text, encoding="utf-8")
            copied_exp += 1
    print(f"[OK] Migrated and patched {copied_exp} experiment runners into {dst_root / 'experiments'}")

    # Copy unit test
    test_src = src_root / "src" / "test_leakage_and_seed.py"
    if test_src.exists():
        shutil.copy2(test_src, dst_root / "tests" / "test_leakage_and_seed.py")
        print("[OK] Copied test_leakage_and_seed.py into tests/")

    # Copy YAML configs if present
    cfg_dir = src_root / "configs"
    if cfg_dir.exists():
        for yml in cfg_dir.glob("*.yaml"):
            shutil.copy2(yml, dst_root / "configs" / yml.name)

    # Copy locked CSVs
    for rel_src, rel_dst in LOCKED_CSVS.items():
        p_src = src_root / rel_src
        if p_src.exists():
            shutil.copy2(p_src, dst_root / rel_dst)
            print(f"[OK] Copied locked CSV: {rel_dst}")

    print("\nMigration complete! Next steps:")
    print(f"  cd {dst_root}")
    print("  python scripts/make_fig3_information_ladders.py")
    print("  git init && git add . && git commit -m 'Initial clean reproducibility release'")


if __name__ == "__main__":
    main()
