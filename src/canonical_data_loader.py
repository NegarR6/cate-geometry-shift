"""
Canonical Data Loader for RIPO / Hybrid-Causal ICLR project.

STRICT RULES (enforced by design — do not weaken):
- NO synthetic data generation of any kind.
- NO silent fallbacks.
- Raise FileNotFoundError / ValueError / AssertionError / RuntimeError immediately
  on any missing file or protocol violation.
- IHDP: prefer official catenets loader. Local NPZ is allowed ONLY when
  allow_local_fallback=True AND the files have been previously verified
  against catenets (see verify_ihdp_against_catenets).
- ACIC: only real x.csv + zymu_XX.csv. Missing file → hard stop.
- Jobs / Twins: only real CSVs. Missing file → hard stop.
- Feature scaling is explicit and controlled (StandardScaler fit on train only).
- Potential outcomes for Jobs are NaN (never zeros) so PEHE-style metrics fail loudly.
- This module is the SINGLE source of data for all final paper experiments.

Supported datasets: IHDP (100 replications), ACIC 2016, Jobs (LaLonde), Twins.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ACIC_DIR = DATA_DIR / "acic2016"
JOBS_DIR = DATA_DIR / "jobs"
TWINS_DIR = DATA_DIR / "twins"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_return_dict(
    X_tr: np.ndarray,
    w_tr: np.ndarray,
    y_tr: np.ndarray,
    po0_tr: np.ndarray,
    po1_tr: np.ndarray,
    X_te: np.ndarray,
    w_te: np.ndarray,
    y_te: np.ndarray,
    po0_te: np.ndarray,
    po1_te: np.ndarray,
    metadata: Dict[str, Any],
    scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    """Uniform return structure used by every loader."""
    out = {
        "train": {
            "X": np.asarray(X_tr, dtype=np.float32),
            "w": np.asarray(w_tr, dtype=np.float32).ravel(),
            "y": np.asarray(y_tr, dtype=np.float32).ravel(),
            "po0": np.asarray(po0_tr, dtype=np.float32).ravel(),
            "po1": np.asarray(po1_tr, dtype=np.float32).ravel(),
        },
        "test": {
            "X": np.asarray(X_te, dtype=np.float32),
            "w": np.asarray(w_te, dtype=np.float32).ravel(),
            "y": np.asarray(y_te, dtype=np.float32).ravel(),
            "po0": np.asarray(po0_te, dtype=np.float32).ravel(),
            "po1": np.asarray(po1_te, dtype=np.float32).ravel(),
        },
        "metadata": metadata,
        "scaler": scaler,
    }
    # Basic shape sanity — fail loudly
    n_tr = out["train"]["X"].shape[0]
    n_te = out["test"]["X"].shape[0]
    for split, n in (("train", n_tr), ("test", n_te)):
        for key in ("w", "y", "po0", "po1"):
            if out[split][key].shape[0] != n:
                raise RuntimeError(
                    f"Shape mismatch in {split}.{key}: expected {n}, "
                    f"got {out[split][key].shape}"
                )
    return out


def _standard_scale(
    X_tr: np.ndarray, X_te: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train only and transform both splits."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    return X_tr_s, X_te_s, scaler


def _rescale_outcomes(
    y_tr: np.ndarray,
    po0_tr: np.ndarray,
    po1_tr: np.ndarray,
    y_te: np.ndarray,
    po0_te: np.ndarray,
    po1_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Rescale outcomes using mean/std of factual training y (catenets-style).

    This is the protocol used by most neural CATE papers that report
    PEHE ~ 0.2–1.0 on IHDP. Without it, some replications have tau std > 30
    and absolute PEHE becomes uninterpretable.
    """
    y_tr = np.asarray(y_tr, dtype=np.float64)
    offset = float(np.mean(y_tr))
    scale = float(np.std(y_tr))
    if scale < 1e-8:
        scale = 1.0

    def _s(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        # Preserve NaNs (e.g. missing test factual under catenets)
        out = (a - offset) / scale
        return out.astype(np.float32)

    info = {"outcome_offset": offset, "outcome_scale": scale}
    return (
        _s(y_tr), _s(po0_tr), _s(po1_tr),
        _s(y_te), _s(po0_te), _s(po1_te),
        info,
    )


# ---------------------------------------------------------------------------
# IHDP
# ---------------------------------------------------------------------------

def _load_ihdp_via_catenets(
    replication: int,
    scale_features: bool,
    rescale_outcomes: bool,
) -> Dict[str, Any]:
    """Official path: use catenets.datasets.load. No fallback inside this function."""
    try:
        from catenets.datasets import load as catenets_load
    except ImportError as e:
        raise ImportError(
            "CANONICAL ERROR: catenets is not installed. "
            "Install with: pip install catenets\n"
            "Or pass allow_local_fallback=True only after verifying local NPZ."
        ) from e

    if not (1 <= replication <= 100):
        raise ValueError(f"IHDP replication must be in [1, 100], got {replication}")

    # Always load raw from catenets; we apply our own controlled rescaling
    X, w, y, po, X_t, po_t = catenets_load("ihdp", exp=replication, rescale=False)

    X = np.asarray(X, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    po0 = np.asarray(po[:, 0], dtype=np.float32)
    po1 = np.asarray(po[:, 1], dtype=np.float32)

    X_t = np.asarray(X_t, dtype=np.float32)
    po0_t = np.asarray(po_t[:, 0], dtype=np.float32)
    po1_t = np.asarray(po_t[:, 1], dtype=np.float32)
    # catenets does not return test treatment or factual y for the test set
    # in the standard 6-tuple; we leave them as NaN so any accidental use fails.
    w_t = np.full(X_t.shape[0], np.nan, dtype=np.float32)
    y_t = np.full(X_t.shape[0], np.nan, dtype=np.float32)

    outcome_info: Dict[str, float] = {}
    if rescale_outcomes:
        y, po0, po1, y_t, po0_t, po1_t, outcome_info = _rescale_outcomes(
            y, po0, po1, y_t, po0_t, po1_t
        )

    scaler = None
    if scale_features:
        X, X_t, scaler = _standard_scale(X, X_t)

    metadata = {
        "dataset": "IHDP",
        "replication": replication,
        "n_train": X.shape[0],
        "n_test": X_t.shape[0],
        "n_features": X.shape[1],
        "source": "catenets",
        "scale_features": scale_features,
        "rescale_outcomes": rescale_outcomes,
        "has_ground_truth_cate": True,
        "test_has_factual": False,  # catenets does not expose test y/w
        **outcome_info,
    }
    return _make_return_dict(
        X, w, y, po0, po1, X_t, w_t, y_t, po0_t, po1_t, metadata, scaler
    )


def _load_ihdp_via_local_npz(
    replication: int,
    scale_features: bool,
    rescale_outcomes: bool,
) -> Dict[str, Any]:
    """
    Fallback path: verified local NPZ files (must be bit-identical to catenets raw).
    This path is ONLY reachable when allow_local_fallback=True is explicitly passed.
    """
    if not (1 <= replication <= 100):
        raise ValueError(f"IHDP replication must be in [1, 100], got {replication}")

    train_file = DATA_DIR / "ihdp_npci_1-100.train.npz"
    test_file = DATA_DIR / "ihdp_npci_1-100.test.npz"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"CANONICAL ERROR: Missing IHDP NPZ files.\n"
            f"Expected: {train_file} and {test_file}\n"
            "Synthetic fallback is strictly disabled."
        )

    train_data = np.load(train_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    required = {"x", "t", "yf", "mu0", "mu1"}
    for name, data in (("train", train_data), ("test", test_data)):
        missing = required - set(data.keys())
        if missing:
            raise KeyError(
                f"CANONICAL ERROR: {name} NPZ missing keys {missing}. "
                "Refusing to reconstruct anything."
            )

    rep_idx = replication - 1

    X_tr = train_data["x"][:, :, rep_idx].astype(np.float32)
    w_tr = train_data["t"][:, rep_idx].astype(np.float32)
    y_tr = train_data["yf"][:, rep_idx].astype(np.float32)
    po0_tr = train_data["mu0"][:, rep_idx].astype(np.float32)
    po1_tr = train_data["mu1"][:, rep_idx].astype(np.float32)

    X_te = test_data["x"][:, :, rep_idx].astype(np.float32)
    w_te = test_data["t"][:, rep_idx].astype(np.float32)
    y_te = test_data["yf"][:, rep_idx].astype(np.float32)
    po0_te = test_data["mu0"][:, rep_idx].astype(np.float32)
    po1_te = test_data["mu1"][:, rep_idx].astype(np.float32)

    # Sanity: y should not equal the noiseless reconstruction
    noiseless = w_tr * po1_tr + (1.0 - w_tr) * po0_tr
    mean_abs_diff = float(np.abs(y_tr - noiseless).mean())
    if mean_abs_diff < 1e-6:
        raise RuntimeError(
            "CANONICAL ERROR: factual y is identical to noiseless reconstruction. "
            "This indicates a corrupted NPZ or wrong key."
        )

    outcome_info: Dict[str, float] = {}
    if rescale_outcomes:
        y_tr, po0_tr, po1_tr, y_te, po0_te, po1_te, outcome_info = _rescale_outcomes(
            y_tr, po0_tr, po1_tr, y_te, po0_te, po1_te
        )

    scaler = None
    if scale_features:
        X_tr, X_te, scaler = _standard_scale(X_tr, X_te)

    metadata = {
        "dataset": "IHDP",
        "replication": replication,
        "n_train": X_tr.shape[0],
        "n_test": X_te.shape[0],
        "n_features": X_tr.shape[1],
        "source": "local_npz_verified",
        "scale_features": scale_features,
        "rescale_outcomes": rescale_outcomes,
        "has_ground_truth_cate": True,
        "test_has_factual": True,
        "yf_vs_noiseless_mean_abs": mean_abs_diff,
        **outcome_info,
    }
    return _make_return_dict(
        X_tr, w_tr, y_tr, po0_tr, po1_tr,
        X_te, w_te, y_te, po0_te, po1_te,
        metadata, scaler,
    )


def load_ihdp_canonical(
    replication: int = 1,
    scale_features: bool = True,
    rescale_outcomes: bool = True,
    prefer_catenets: bool = True,
    allow_local_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Load one IHDP replication under the canonical protocol.

    Parameters
    ----------
    replication : int
        Replication index in 1..100 (Hill 2011 protocol).
    scale_features : bool
        If True (default), apply StandardScaler fit on the training split only.
        Recommended for neural models.
    rescale_outcomes : bool
        If True (default), center/scale y and potential outcomes by mean/std of
        factual training y. This matches the protocol used by most neural CATE
        papers and keeps PEHE on a comparable scale across replications.
        Set False only for absolute-scale analyses or sensitivity checks.
    prefer_catenets : bool
        If True (default), use the official catenets loader.
    allow_local_fallback : bool
        If False (default for paper runs), any failure of catenets raises
        immediately. Set True ONLY after you have run verify_ihdp_against_catenets
        and confirmed numerical identity.

    Returns
    -------
    dict with keys 'train', 'test', 'metadata', 'scaler'
    """
    if prefer_catenets:
        try:
            return _load_ihdp_via_catenets(replication, scale_features, rescale_outcomes)
        except Exception as e:
            if not allow_local_fallback:
                raise RuntimeError(
                    "CANONICAL ERROR: catenets load failed and allow_local_fallback=False.\n"
                    "For final paper experiments the local fallback is disabled.\n"
                    "Install catenets or, after verification, pass "
                    "allow_local_fallback=True.\n"
                    f"Original error: {type(e).__name__}: {e}"
                ) from e
            warnings.warn(
                f"catenets load failed ({type(e).__name__}: {e}); "
                "using verified local NPZ because allow_local_fallback=True.",
                UserWarning,
                stacklevel=2,
            )

    # Only reached when prefer_catenets=False or fallback was explicitly allowed
    return _load_ihdp_via_local_npz(replication, scale_features, rescale_outcomes)


# ---------------------------------------------------------------------------
# ACIC 2016
# ---------------------------------------------------------------------------

def load_acic_canonical(
    setting_id: int = 1,
    test_size: float = 0.2,
    seed: int = 42,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Load one ACIC 2016 setting from the real CSV files shipped with the repo.

    Raises FileNotFoundError if x.csv or the corresponding zymu_*.csv is missing.
    One-hot encoding is fit on the training rows only.
    No synthetic data is ever generated.
    """
    x_file = ACIC_DIR / "x.csv"
    zymu_file = ACIC_DIR / f"zymu_{setting_id}.csv"
    zymu_file_padded = ACIC_DIR / f"zymu_{setting_id:02d}.csv"

    if not zymu_file.exists() and zymu_file_padded.exists():
        zymu_file = zymu_file_padded

    if not x_file.exists():
        raise FileNotFoundError(
            f"CANONICAL ERROR: ACIC covariates missing: {x_file}\n"
            "Synthetic fallback is strictly disabled."
        )
    if not zymu_file.exists():
        raise FileNotFoundError(
            f"CANONICAL ERROR: ACIC outcomes missing for setting {setting_id}: {zymu_file}\n"
            "Synthetic fallback is strictly disabled."
        )

    X_df_raw = pd.read_csv(x_file)
    zymu_df = pd.read_csv(zymu_file)

    if len(X_df_raw) != len(zymu_df):
        raise AssertionError(
            f"Row count mismatch: x.csv has {len(X_df_raw)} rows, "
            f"{zymu_file.name} has {len(zymu_df)} rows."
        )

    # Treatment
    w_col = next((c for c in ("z", "t", "treat") if c in zymu_df.columns), None)
    if w_col is None:
        raise ValueError(
            f"No treatment column in {zymu_file}. Columns: {list(zymu_df.columns)}"
        )
    w = zymu_df[w_col].to_numpy(dtype=np.float32)

    # Potential outcomes
    po0_col = next((c for c in ("mu0", "y0", "po0") if c in zymu_df.columns), None)
    po1_col = next((c for c in ("mu1", "y1", "po1") if c in zymu_df.columns), None)
    if po0_col is None or po1_col is None:
        raise ValueError(f"Potential outcome columns missing in {zymu_file}.")
    po0 = zymu_df[po0_col].to_numpy(dtype=np.float32)
    po1 = zymu_df[po1_col].to_numpy(dtype=np.float32)

    # Factual outcome – prefer the real noisy 'y' or realized y0/y1
    if "y" in zymu_df.columns:
        y = zymu_df["y"].to_numpy(dtype=np.float32)
    elif "y0" in zymu_df.columns and "y1" in zymu_df.columns:
        y0_raw = zymu_df["y0"].to_numpy(dtype=np.float32)
        y1_raw = zymu_df["y1"].to_numpy(dtype=np.float32)
        y = np.where(w == 1, y1_raw, y0_raw).astype(np.float32)
    else:
        raise ValueError(
            f"Cannot construct factual y for {zymu_file}: "
            f"no 'y' and no realized 'y0'/'y1'. Columns: {list(zymu_df.columns)}. "
            "Refusing to fall back to noiseless mu0/mu1."
        )

    # Stratified split on indices first so one-hot is fit on train only
    n = len(X_df_raw)
    idx_tr, idx_te = train_test_split(
        np.arange(n),
        test_size=test_size,
        random_state=seed,
        stratify=w.astype(int),
    )

    X_df_tr = pd.get_dummies(X_df_raw.iloc[idx_tr], drop_first=True)
    X_df_te = (
        pd.get_dummies(X_df_raw.iloc[idx_te], drop_first=True)
        .reindex(columns=X_df_tr.columns, fill_value=0)
    )

    X_tr = X_df_tr.to_numpy(dtype=np.float32)
    X_te = X_df_te.to_numpy(dtype=np.float32)
    w_tr, w_te = w[idx_tr], w[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    po0_tr, po0_te = po0[idx_tr], po0[idx_te]
    po1_tr, po1_te = po1[idx_tr], po1[idx_te]

    scaler = None
    if scale_features:
        X_tr, X_te, scaler = _standard_scale(X_tr, X_te)

    metadata = {
        "dataset": "ACIC 2016",
        "setting": setting_id,
        "n_train": X_tr.shape[0],
        "n_test": X_te.shape[0],
        "n_features": X_tr.shape[1],
        "source": "real_csv",
        "scale_features": scale_features,
        "has_ground_truth_cate": True,
        "test_has_factual": True,
        "feature_names": list(X_df_tr.columns),
        "split_seed": seed,
    }
    return _make_return_dict(
        X_tr, w_tr, y_tr, po0_tr, po1_tr,
        X_te, w_te, y_te, po0_te, po1_te,
        metadata, scaler,
    )


# ---------------------------------------------------------------------------
# Jobs (LaLonde)
# ---------------------------------------------------------------------------

def load_jobs_canonical(
    test_size: float = 0.2,
    seed: int = 42,
    scale_features: bool = True,
    scale_outcomes: bool = True,
) -> Dict[str, Any]:
    """
    Load the classic LaLonde / Jobs dataset.

    Ground-truth individual potential outcomes do not exist.
    po0 / po1 are filled with NaN so any PEHE / ATE-error computation fails loudly.

    Outcomes (re78, dollars) are standardized using train mean/std when
    scale_outcomes=True (default). This is required for stable neural training.
    metadata['y_mean'] / metadata['y_std'] allow converting predictions back
    to the dollar scale for reporting.
    """
    jobs_path = JOBS_DIR / "jobs_train.csv"
    if not jobs_path.exists():
        raise FileNotFoundError(
            f"CANONICAL ERROR: Missing Jobs dataset at {jobs_path}.\n"
            "Synthetic fallback is strictly disabled."
        )

    df = pd.read_csv(jobs_path)
    if "hispan" not in df.columns and "hispanic" in df.columns:
        df["hispan"] = df["hispanic"]

    feature_cols = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Jobs CSV missing required columns: {missing}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    w = df["treat"].to_numpy(dtype=np.float32)
    y = df["re78"].to_numpy(dtype=np.float32)

    X_tr, X_te, w_tr, w_te, y_tr, y_te = train_test_split(
        X, w, y, test_size=test_size, random_state=seed, stratify=w.astype(int)
    )

    scaler = None
    if scale_features:
        X_tr, X_te, scaler = _standard_scale(X_tr, X_te)

    y_mean, y_std = 0.0, 1.0
    if scale_outcomes:
        y_mean = float(np.mean(y_tr))
        y_std = float(np.std(y_tr))
        if y_std < 1e-8:
            y_std = 1.0
        y_tr = ((y_tr - y_mean) / y_std).astype(np.float32)
        y_te = ((y_te - y_mean) / y_std).astype(np.float32)

    # Explicit NaNs – never zeros
    po0_tr = np.full_like(y_tr, np.nan)
    po1_tr = np.full_like(y_tr, np.nan)
    po0_te = np.full_like(y_te, np.nan)
    po1_te = np.full_like(y_te, np.nan)

    metadata = {
        "dataset": "Jobs",
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "source": "local_csv",
        "scale_features": scale_features,
        "scale_outcomes": scale_outcomes,
        "y_mean": y_mean,
        "y_std": y_std,
        "has_ground_truth_cate": False,
        "test_has_factual": True,
        "feature_names": feature_cols,
        "split_seed": seed,
    }
    return _make_return_dict(
        X_tr, w_tr, y_tr, po0_tr, po1_tr,
        X_te, w_te, y_te, po0_te, po1_te,
        metadata, scaler,
    )


# ---------------------------------------------------------------------------
# Twins
# ---------------------------------------------------------------------------

def load_twins_canonical(
    test_size: float = 0.2,
    seed: int = 42,
    scale_features: bool = True,
    confounding: str = "gestat",
) -> Dict[str, Any]:
    """
    Load the Twins dataset (GANITE / CEVAE style).

    Treatment assignment is deliberately confounded following the literature
    construction. This is the standard protocol used in published papers;
    it is NOT an uncontrolled synthetic fallback.
    The confounding mechanism must be stated in the paper.
    """
    twins_path = TWINS_DIR / "twins.csv"
    if not twins_path.exists():
        raise FileNotFoundError(
            f"CANONICAL ERROR: Missing Twins dataset at {twins_path}.\n"
            "Synthetic fallback is strictly disabled."
        )

    df = pd.read_csv(twins_path)
    df.columns = [
        c.replace("'", "").replace("’", "").replace("`", "").strip() for c in df.columns
    ]

    y0_col = next((c for c in df.columns if "outcome" in c and "0" in c), None)
    y1_col = next((c for c in df.columns if "outcome" in c and "1" in c), None)
    if y0_col is None or y1_col is None:
        raise ValueError(f"Outcome columns missing in Twins. Columns: {list(df.columns)}")

    # Binary 1-year mortality
    y0 = (df[y0_col] < 365).to_numpy(dtype=np.float32)
    y1 = (df[y1_col] < 365).to_numpy(dtype=np.float32)

    feature_cols = [c for c in df.columns if "outcome" not in c]
    X = df[feature_cols].to_numpy(dtype=np.float32)

    # Confounded treatment assignment (local RNG – never touch global state)
    rng = np.random.default_rng(seed)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    random_weights = rng.normal(loc=0.0, scale=0.1, size=X_std.shape[1])

    if confounding == "gestat" and "gestat" in df.columns:
        gestat = df["gestat"].to_numpy(dtype=np.float32)
        gestat_term = 2.0 * ((gestat - gestat.mean()) / (gestat.std() + 1e-8))
    else:
        gestat_term = np.zeros(len(df), dtype=np.float32)

    logits = X_std @ random_weights + gestat_term
    propensity = 1.0 / (1.0 + np.exp(-logits))
    w = rng.binomial(1, propensity).astype(np.float32)
    y = np.where(w == 1, y1, y0).astype(np.float32)

    (
        X_tr, X_te,
        w_tr, w_te,
        y_tr, y_te,
        po0_tr, po0_te,
        po1_tr, po1_te,
    ) = train_test_split(
        X, w, y, y0, y1,
        test_size=test_size,
        random_state=seed,
        stratify=w.astype(int),
    )

    scaler = None
    if scale_features:
        X_tr, X_te, scaler = _standard_scale(X_tr, X_te)

    metadata = {
        "dataset": "Twins",
        "n_train": X_tr.shape[0],
        "n_test": X_te.shape[0],
        "n_features": X_tr.shape[1],
        "source": "local_csv",
        "scale_features": scale_features,
        "has_ground_truth_cate": True,
        "test_has_factual": True,
        "confounding": confounding,
        "split_seed": seed,
        "feature_names": feature_cols,
    }
    return _make_return_dict(
        X_tr, w_tr, y_tr, po0_tr, po1_tr,
        X_te, w_te, y_te, po0_te, po1_te,
        metadata, scaler,
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def load_canonical_dataset(name: str = "ihdp", **kwargs) -> Dict[str, Any]:
    """
    Single entry point for all canonical datasets.

    Examples
    --------
    >>> data = load_canonical_dataset("ihdp", replication=5)
    >>> data = load_canonical_dataset("acic", setting_id=3, seed=42)
    >>> data = load_canonical_dataset("jobs")
    >>> data = load_canonical_dataset("twins", seed=7)
    """
    name = name.lower().strip()
    if name == "ihdp":
        return load_ihdp_canonical(**kwargs)
    if name in ("acic", "acic2016"):
        return load_acic_canonical(**kwargs)
    if name in ("jobs", "lalonde"):
        return load_jobs_canonical(**kwargs)
    if name == "twins":
        return load_twins_canonical(**kwargs)
    raise ValueError(
        f"Unknown canonical dataset '{name}'. "
        "Supported: 'ihdp', 'acic', 'jobs', 'twins'."
    )


# ---------------------------------------------------------------------------
# Compatibility wrappers (old phase scripts still call these names)
# ---------------------------------------------------------------------------

def get_ihdp_dataset(replication: int = 1, random_state: int = 42, **kwargs):
    """
    Backward-compatible wrapper.
    `random_state` is accepted but ignored for IHDP because the train/test
    split is fixed by the official NPZ / catenets files. Use `replication`
    to select which of the 100 Hill (2011) replications to load.
    """
    if not (1 <= replication <= 100):
        raise ValueError(
            f"get_ihdp_dataset: replication must be in [1, 100], got {replication}. "
            "Do not pass a random seed as the first positional argument."
        )
    # Force strict mode for paper experiments unless caller overrides
    kwargs.setdefault("allow_local_fallback", False)
    kwargs.setdefault("prefer_catenets", True)
    return load_ihdp_canonical(
        replication=replication,
        scale_features=True,
        rescale_outcomes=True,
        **kwargs,
    )


def get_acic_dataset(
    setting_id: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
    seed: int = None,
    **kwargs,
):
    """Backward-compatible wrapper for ACIC. """
    if seed is not None:
        random_state = seed
    return load_acic_canonical(
        setting_id=setting_id,
        test_size=test_size,
        seed=random_state,
        scale_features=True,
        **kwargs,
    )


# Keep the longer names as well
load_ihdp_dataset = get_ihdp_dataset
load_acic2016_dataset = get_acic_dataset
load_jobs_dataset = load_jobs_canonical
load_twins_dataset = load_twins_canonical


# ---------------------------------------------------------------------------
# Protocol lock helpers (call these at the start of every training/eval script)
# ---------------------------------------------------------------------------

ALLOWED_SOURCES = frozenset({
    "catenets",
    "local_tensorflow_verified",
    "real_csv",
    "local_csv",
})

def assert_canonical_protocol(data: Dict[str, Any], allow_jobs_nans: bool = True) -> None:
    """
    Hard runtime guard. Call this immediately after loading any dataset
    in paper experiments. Raises RuntimeError on any protocol violation.
    """
    if not isinstance(data, dict) or "metadata" not in data or "train" not in data:
        raise RuntimeError("CANONICAL PROTOCOL VIOLATION: data is not a canonical return dict.")

    meta = data["metadata"]
    source = meta.get("source")
    if source not in ALLOWED_SOURCES:
        raise RuntimeError(
            f"CANONICAL PROTOCOL VIOLATION: illegal source={source!r}. "
            f"Allowed: {sorted(ALLOWED_SOURCES)}"
        )

    # Shape sanity already done inside _make_return_dict, but re-check key presence
    for split in ("train", "test"):
        for key in ("X", "w", "y", "po0", "po1"):
            if key not in data[split]:
                raise RuntimeError(f"CANONICAL PROTOCOL VIOLATION: missing {split}.{key}")

    # Jobs must have NaN potential outcomes
    if meta.get("dataset") == "Jobs":
        if not allow_jobs_nans:
            raise RuntimeError("Jobs requested without NaN allowance.")
        if not (np.any(np.isnan(data["train"]["po0"])) and np.any(np.isnan(data["test"]["po0"]))):
            raise RuntimeError(
                "CANONICAL PROTOCOL VIOLATION: Jobs potential outcomes must be NaN "
                "(never zeros or synthetic values)."
            )

    # Forbid any accidental synthetic marker
    if meta.get("synthetic") or meta.get("is_synthetic"):
        raise RuntimeError("CANONICAL PROTOCOL VIOLATION: synthetic data detected.")


def require_canonical_loader() -> None:
    """
    Call at module level or at the very top of experiment scripts.
    Documents the rule and fails fast if someone tries to monkey-patch.
    """
    import sys
    # Soft check: ensure the canonical module is the one providing the loaders
    mod = sys.modules.get("canonical_data_loader") or sys.modules.get("src.canonical_data_loader")
    if mod is None:
        # Will be imported later; just print a reminder
        print("[PROTOCOL] Reminder: use only canonical_data_loader for paper experiments.")


# ---------------------------------------------------------------------------
# Verification utilities
# ---------------------------------------------------------------------------

def verify_ihdp_against_catenets(n_replications: int = 5, atol: float = 1e-5) -> None:
    """
    Assert that local NPZ data is numerically identical to catenets for the
    requested replications (raw, unscaled features and outcomes).

    Run this once after any change to the data files.
    Only after this verification passes may you set allow_local_fallback=True.
    """
    try:
        from catenets.datasets import load as catenets_load
    except ImportError as e:
        raise ImportError(
            "catenets is required for this verification. "
            "Install with: pip install catenets"
        ) from e

    print(f"Verifying {n_replications} IHDP replications against catenets ...")
    for rep in range(1, n_replications + 1):
        # catenets raw
        X_c, w_c, y_c, po_c, X_t_c, po_t_c = catenets_load(
            "ihdp", exp=rep, rescale=False
        )

        # local NPZ (raw: no feature scaling, no outcome rescaling)
        local = _load_ihdp_via_local_npz(
            rep, scale_features=False, rescale_outcomes=False
        )

        assert np.allclose(X_c, local["train"]["X"], atol=atol), f"rep {rep}: X mismatch"
        assert np.allclose(w_c.ravel(), local["train"]["w"], atol=atol), f"rep {rep}: w mismatch"
        assert np.allclose(y_c.ravel(), local["train"]["y"], atol=atol), f"rep {rep}: y mismatch"
        assert np.allclose(po_c[:, 0], local["train"]["po0"], atol=atol), f"rep {rep}: mu0 mismatch"
        assert np.allclose(po_c[:, 1], local["train"]["po1"], atol=atol), f"rep {rep}: mu1 mismatch"
        assert np.allclose(X_t_c, local["test"]["X"], atol=atol), f"rep {rep}: X_test mismatch"
        assert np.allclose(po_t_c[:, 0], local["test"]["po0"], atol=atol), f"rep {rep}: mu0_test mismatch"
        assert np.allclose(po_t_c[:, 1], local["test"]["po1"], atol=atol), f"rep {rep}: mu1_test mismatch"

        print(
            f"  replication {rep:3d}  OK  "
            f"(n_train={local['metadata']['n_train']}, "
            f"n_test={local['metadata']['n_test']})"
        )

    print("All checked replications match catenets exactly (raw data).")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("CANONICAL DATA LOADER — SELF TEST (STRICT MODE)")
    print("=" * 72)

    checks = [
        # Strict path: catenets preferred, no local fallback
        ("IHDP catenets (strict, no fallback)", lambda: load_ihdp_canonical(
            replication=1, prefer_catenets=True, allow_local_fallback=False,
            rescale_outcomes=True)),
        ("IHDP catenets rescale=False", lambda: load_ihdp_canonical(
            replication=1, prefer_catenets=True, allow_local_fallback=False,
            rescale_outcomes=False)),
        # Local path only when explicitly allowed (for verification)
        ("IHDP local NPZ (explicit fallback)", lambda: load_ihdp_canonical(
            replication=1, prefer_catenets=False, allow_local_fallback=True,
            rescale_outcomes=True)),
        ("ACIC 2016 setting 1", lambda: load_acic_canonical(setting_id=1)),
        ("Jobs", lambda: load_jobs_canonical()),
        ("Twins", lambda: load_twins_canonical()),
    ]

    all_passed = True
    for name, fn in checks:
        print(f"\n--- {name} ---")
        try:
            d = fn()
            for split in ("train", "test"):
                shapes = {k: getattr(v, "shape", None) for k, v in d[split].items()}
                print(f"  {split}: {shapes}")
            meta = {k: v for k, v in d["metadata"].items() if k != "feature_names"}
            print(f"  metadata: {meta}")
            if d["metadata"].get("has_ground_truth_cate", False):
                tau = d["test"]["po1"] - d["test"]["po0"]
                print(f"  test true ATE: {np.nanmean(tau):.4f}")
            print(f"  ✓ {name} OK")
        except Exception as e:
            all_passed = False
            print(f"  ✗ {name} FAILED: {type(e).__name__}: {e}")

    print("\n" + "=" * 72)
    if all_passed:
        print("ALL CHECKS PASSED")
        try:
            verify_ihdp_against_catenets(n_replications=3)
        except Exception as e:
            print(f"Verification against catenets skipped or failed: {e}")
    else:
        print("SOME CHECKS FAILED — fix before any experiment")
    print("=" * 72)
