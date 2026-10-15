"""
Priority-1 utilities: DR pseudo-τ (pool-only, no test leakage),
active selection scores, and simple label baselines.

CRITICAL FIXES (P0):
- dr_pseudo_tau is NEVER called on the full target array that includes
  held-out test units. Callers must pass only the adaptation pool.
- linear Ridge smoothing is OFF by default (was the source of the
  artificial "plateau" finding). Use smooth=True only as an explicitly
  named "linear-smoothed DR" ablation arm.
"""
from __future__ import annotations

import sys as _sys
import pathlib as _pl

_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor


def pehe(tau_true, tau_pred) -> float:
    """Root PEHE."""
    a = np.asarray(tau_true).ravel()
    b = np.asarray(tau_pred).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))



def clip_propensity(e: np.ndarray, delta: float = 0.05) -> np.ndarray:
    return np.clip(np.asarray(e, dtype=np.float64), delta, 1.0 - delta)


def crossfit_propensity(
    X: np.ndarray,
    T: np.ndarray,
    n_splits: int = 3,
    seed: int = 0,
    delta: float = 0.05,
) -> np.ndarray:
    """Out-of-fold propensity P(T=1|X)."""
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).ravel().astype(int)
    n = len(T)
    e_hat = np.zeros(n, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        clf = LogisticRegression(max_iter=500, random_state=seed)
        clf.fit(X[tr], T[tr])
        if hasattr(clf, "predict_proba"):
            e_hat[te] = clf.predict_proba(X[te])[:, 1]
        else:
            e_hat[te] = clf.predict(X[te])
    return clip_propensity(e_hat, delta)


def crossfit_mu(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 3,
    seed: int = 0,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold μ0(x), μ1(x) via Ridge on each arm."""
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(np.float64)
    n = len(Y)
    mu0 = np.zeros(n, dtype=np.float64)
    mu1 = np.zeros(n, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr, te in kf.split(X):
        tr0 = tr[T[tr] == 0]
        tr1 = tr[T[tr] == 1]
        if len(tr0) < 5 or len(tr1) < 5:
            # fallback: use whatever arm samples exist in fold
            pass
        if len(tr0) >= 2:
            m0 = Ridge(alpha=alpha).fit(X[tr0], Y[tr0])
            mu0[te] = m0.predict(X[te])
        else:
            mu0[te] = float(np.mean(Y[T == 0])) if np.any(T == 0) else 0.0
        if len(tr1) >= 2:
            m1 = Ridge(alpha=alpha).fit(X[tr1], Y[tr1])
            mu1[te] = m1.predict(X[te])
        else:
            mu1[te] = float(np.mean(Y[T == 1])) if np.any(T == 1) else 0.0
    return mu0, mu1


def dr_pseudo_tau(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 3,
    seed: int = 0,
    delta: float = 0.05,
    smooth: bool = False,
    smooth_alpha: float = 1.0,
) -> np.ndarray:
    """
    Doubly robust pseudo-outcome for CATE (Kennedy-style).

    IMPORTANT: Call this ONLY on the adaptation pool indices.
    Never pass the full target array that includes held-out test units.

    smooth=False (default) after P0 fix. The previous default smooth=True
    produced a linear function of X and artificially plateaued near the
    constant-ATE baseline; that finding was an artifact.
    """
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(np.float64)
    X = np.asarray(X, dtype=np.float64)
    assert len(X) == len(T) == len(Y), "X, T, Y must have same length (pool only)"

    e = crossfit_propensity(X, T, n_splits=n_splits, seed=seed, delta=delta)
    mu0, mu1 = crossfit_mu(X, T, Y, n_splits=n_splits, seed=seed)
    # DR transformation
    tau = (mu1 - mu0) + (T / e) * (Y - mu1) - ((1 - T) / (1 - e)) * (Y - mu0)
    tau = np.asarray(tau, dtype=np.float64)
    tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth:
        reg = Ridge(alpha=smooth_alpha).fit(X, tau)
        tau = reg.predict(X).astype(np.float64)
    return tau


def source_ate_estimate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 3,
    seed: int = 0,
) -> float:
    """Deployable constant predictor: mean of DR pseudo-outcomes on source data."""
    tau_dr = dr_pseudo_tau(X, T, Y, n_splits=n_splits, seed=seed, smooth=False)
    return float(np.mean(tau_dr))


def simple_label_baseline(
    X_train: np.ndarray,
    tau_train: np.ndarray,
    X_test: np.ndarray,
    method: str = "ridge",
    seed: int = 0,
) -> np.ndarray:
    """
    Simple supervised baselines that use the same (X, tau-label) pairs
    available to the oracle / DR adaptation arms. Used for fair comparison
    against neural fine-tuning under small m.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    tau_train = np.asarray(tau_train).ravel()
    X_test = np.asarray(X_test, dtype=np.float64)
    method = method.lower()
    if method == "ridge":
        reg = Ridge(alpha=1.0).fit(X_train, tau_train)
        return reg.predict(X_test)
    if method == "gbr":
        reg = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed
        )
        reg.fit(X_train, tau_train)
        return reg.predict(X_test)
    if method == "knn":
        k = min(15, max(1, len(X_train) // 3))
        reg = KNeighborsRegressor(n_neighbors=k)
        reg.fit(X_train, tau_train)
        return reg.predict(X_test)
    raise ValueError(f"Unknown simple baseline method: {method}")


def expert_disagreement_score(tau_n: np.ndarray, tau_t: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(tau_n).ravel() - np.asarray(tau_t).ravel())


def router_entropy_score(w_neural: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Binary mixture entropy from neural weight in (0,1)."""
    w = np.clip(np.asarray(w_neural).ravel(), eps, 1.0 - eps)
    return -(w * np.log(w) + (1 - w) * np.log(1 - w))


def active_scores(
    tau_n: np.ndarray,
    tau_t: np.ndarray,
    w_neural: Optional[np.ndarray] = None,
    mode: str = "active_hybrid",
) -> np.ndarray:
    """Higher score => more valuable to label under geometry uncertainty."""
    d = expert_disagreement_score(tau_n, tau_t)
    if mode == "active_disagreement":
        return d
    if mode == "active_entropy":
        if w_neural is None:
            raise ValueError("active_entropy requires w_neural")
        return router_entropy_score(w_neural)
    if mode == "active_hybrid":
        if w_neural is None:
            return d
        h = router_entropy_score(w_neural)

        def z(v):
            s = float(np.std(v))
            if s < 1e-12:
                return np.zeros_like(v)
            return (v - np.mean(v)) / s

        return z(d) + z(h)
    raise ValueError(f"Unknown active mode: {mode}")


def select_indices(
    pool_idx: np.ndarray,
    scores: np.ndarray,
    m: int,
    policy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select m indices from pool_idx."""
    pool_idx = np.asarray(pool_idx)
    scores = np.asarray(scores).ravel()
    assert len(pool_idx) == len(scores)
    m = min(m, len(pool_idx))
    if m <= 0:
        return np.array([], dtype=int)
    if policy == "random":
        return rng.choice(pool_idx, size=m, replace=False)
    if policy.startswith("active"):
        order = np.argsort(-scores)  # descending
        return pool_idx[order[:m]]
    raise ValueError(f"Unknown selection policy: {policy}")
