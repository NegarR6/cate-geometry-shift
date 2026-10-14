"""Feature matrices and oracle labels for E2 observable router."""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from typing import Dict, List, Tuple

import numpy as np

MODEL_ORDER = ["ripo_no_tree", "ft_tarnet", "xlearner", "causalforest"]


def oracle_labels(preds: Dict[str, np.ndarray], tau_true: np.ndarray) -> np.ndarray:
    """y_i* = argmin_m |tau_hat_m - tau| as integer index into MODEL_ORDER (present models)."""
    names = [m for m in MODEL_ORDER if m in preds]
    stack = np.column_stack([preds[m] for m in names])
    pick = np.argmin(np.abs(stack - tau_true.reshape(-1, 1)), axis=1)
    return pick.astype(np.int64), names


def build_features(
    preds: Dict[str, np.ndarray],
    e: np.ndarray | None,
    version: str,
    X_cov: np.ndarray | None = None,
    max_x_dims: int = 12,
) -> Tuple[np.ndarray, List[str]]:
    """
    version in {V0, V1, V2, V3}
      V0: pairwise |tau_a - tau_b|
      V1: V0 + tau levels
      V2: V1 + propensity features
      V3: V2 + raw covariates X (first max_x_dims columns)
    """
    version = version.upper()
    names = [m for m in MODEL_ORDER if m in preds]
    if len(names) < 2:
        raise ValueError("need at least 2 models for disagreement features")

    cols: List[np.ndarray] = []
    col_names: List[str] = []

    # V0 disagreement
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            cols.append(np.abs(preds[a] - preds[b]))
            col_names.append(f"abs_{a}_minus_{b}")

    if version in ("V1", "V2", "V3"):
        for m in names:
            cols.append(preds[m])
            col_names.append(f"tau_{m}")

    if version in ("V2", "V3"):
        if e is None:
            raise ValueError(f"{version} requires propensity e(x)")
        e = np.asarray(e, dtype=np.float64).reshape(-1)
        cols.extend([e, 1.0 - e, e * (1.0 - e)])
        col_names.extend(["e", "one_minus_e", "e_one_minus_e"])

    if version == "V3":
        if X_cov is None:
            raise ValueError("V3 requires covariate matrix X_cov")
        Xc = np.asarray(X_cov, dtype=np.float64)
        if Xc.ndim != 2:
            raise ValueError("X_cov must be 2D")
        d = min(max_x_dims, Xc.shape[1])
        for j in range(d):
            cols.append(Xc[:, j])
            col_names.append(f"x{j}")

    feat = np.column_stack(cols).astype(np.float64)
    return feat, col_names


def split_router_train_eval(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic half split of test indices (shuffled with seed).
    First half -> router train, second -> router eval.
    """
    rng = np.random.default_rng(seed + 10_007)
    idx = rng.permutation(n)
    mid = n // 2
    return np.sort(idx[:mid]), np.sort(idx[mid:])
