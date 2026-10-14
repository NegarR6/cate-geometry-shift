"""
Priority 1 utilities: DR pseudo-τ, active selection scores, adapt targets.
Drop into project as src/budgeted_ladder_utils.py (or merge into adapt_geometry_shift).
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold


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
            # fallback: global fit on all tr
            tr0 = tr[T[tr] == 0]
            tr1 = tr[T[tr] == 1]
        m0 = Ridge(alpha=alpha).fit(X[tr0], Y[tr0])
        m1 = Ridge(alpha=alpha).fit(X[tr1], Y[tr1])
        mu0[te] = m0.predict(X[te])
        mu1[te] = m1.predict(X[te])
    return mu0, mu1


def dr_pseudo_tau(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 3,
    seed: int = 0,
    delta: float = 0.05,
    smooth: bool = True,
    smooth_alpha: float = 1.0,
) -> np.ndarray:
    """
    Doubly robust pseudo-outcome for CATE (Kennedy-style).
    Optionally Ridge-smooth τ_DR on X for stabler adaptation targets.
    """
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(np.float64)
    e = crossfit_propensity(X, T, n_splits=n_splits, seed=seed, delta=delta)
    mu0, mu1 = crossfit_mu(X, T, Y, n_splits=n_splits, seed=seed)
    # DR transformation
    tau = (mu1 - mu0) + (T / e) * (Y - mu1) - ((1 - T) / (1 - e)) * (Y - mu0)
    tau = np.asarray(tau, dtype=np.float64)
    # sanitize
    tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth:
        reg = Ridge(alpha=smooth_alpha).fit(X, tau)
        tau = reg.predict(X).astype(np.float64)
    return tau


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
            # degrade to disagreement
            return d
        h = router_entropy_score(w_neural)
        # z-score within pool
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
    if m <= 0:
        return np.array([], dtype=int)
    m = min(m, len(pool_idx))
    if policy == "random":
        return rng.choice(pool_idx, size=m, replace=False)
    if policy.startswith("active"):
        # scores aligned with pool order: caller must pass scores for pool only
        order = np.argsort(-np.asarray(scores).ravel())
        return pool_idx[order[:m]]
    raise ValueError(f"Unknown policy: {policy}")


def pehe(true_tau: np.ndarray, pred: np.ndarray) -> float:
    a = np.asarray(true_tau).ravel()
    b = np.asarray(pred).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))
