"""
Semi-synthetic geometry morphing — STRONG contrast version.

Design goals for journal external validity:
  1) Real covariates (IHDP / ACIC-style arrays)
  2) Two τ geometries that *force* ranking / adaptation gap:
       - smooth: leave-one-out distance-weighted kNN of base tau, then ridge-smooth
         (self-neighbour exclusion prevents identity reconstruction)
       - threshold: sharp multi-feature piecewise constant with LARGE level gap
  3) Shared propensity / baseline mu0 fitted on the full paired cohort;
     only τ geometry changes (intentional design for pure effect-surface shift)
  4) Optional signal amplification so geometry dominates noise

Note: full-cohort nuisance fitting and feature selection are intentional so that
source and target worlds share identical P(X), e(x) and mu0(x).  The resulting
benchmark is a controlled paired stress test, not an independent two-cohort
external validation.  kNN smoothing uses strict leave-one-out predictions.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


def fit_nuisances(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(np.float64)
    clf = LogisticRegression(max_iter=800, C=0.5)
    clf.fit(X, T)
    e = np.clip(clf.predict_proba(X)[:, 1], 0.05, 0.95)
    m0 = Ridge(alpha=1.0).fit(X[T == 0], Y[T == 0])
    m1 = Ridge(alpha=1.0).fit(X[T == 1], Y[T == 1])
    return e.astype(np.float32), m0.predict(X).astype(np.float32), m1.predict(X).astype(np.float32)


def base_tau_proxy(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    e, mu0, mu1 = fit_nuisances(X, T, Y)
    return (mu1 - mu0).astype(np.float32)


def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    s = float(v.std())
    if s < 1e-8:
        return np.zeros_like(v)
    return (v - v.mean()) / s


def smooth_tau(tau: np.ndarray, X: np.ndarray, k: int = 40) -> np.ndarray:
    """
    Strong smooth geometry: leave-one-out distance-weighted kNN + Ridge projection.

    Critical design choice: kNN predictions are strictly leave-one-out so that
    no point uses itself as a neighbor (distance-0 self-reconstruction is forbidden).
    This yields a genuine low-frequency surface rather than an identity map.
    """
    from sklearn.metrics import pairwise_distances

    X = np.asarray(X, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64).ravel()
    n = len(tau)
    k = int(min(max(15, k), max(10, n // 5)))
    # LOO distances: set self-distance to +inf so a point never appears in its own
    # neighbour set (prevents the identity contamination of in-sample kNN).
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn_idx = np.argpartition(D, kth=k, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    d_nn = D[rows, nn_idx]
    w = 1.0 / (d_nn + 1e-8)
    w /= w.sum(axis=1, keepdims=True)
    t1 = (w * tau[nn_idx]).sum(axis=1)
    # second smooth via ridge on the LOO-kNN surface
    t2 = Ridge(alpha=5.0).fit(X, t1).predict(X)
    return _zscore(0.5 * t1 + 0.5 * t2).astype(np.float32)


def threshold_tau(
    tau: np.ndarray,
    X: np.ndarray,
    n_features: int = 3,
    depth: int = 3,
    level_gap: float = 2.5,
) -> np.ndarray:
    """
    Strong threshold geometry:
      - shallow decision tree on X predicting base tau → piecewise regions
      - replace leaf predictions by *amplified* two-level or leaf-mean gaps
    This creates sharp discontinuities unlike smooth_tau.
    """
    X = np.asarray(X, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    # pick top coordinates by |corr| with tau
    corrs = []
    for j in range(X.shape[1]):
        c = np.corrcoef(X[:, j], tau)[0, 1]
        corrs.append(0.0 if not np.isfinite(c) else abs(c))
    top = np.argsort(corrs)[::-1][: max(1, n_features)]
    X_sub = X[:, top]

    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(20, len(tau) // 30))
    tree.fit(X_sub, tau)
    leaf_id = tree.apply(X_sub)
    out = np.zeros(len(tau), dtype=np.float64)
    # amplify separation across leaves
    global_med = np.median(tau)
    for lid in np.unique(leaf_id):
        mask = leaf_id == lid
        m = float(np.mean(tau[mask]))
        # push leaves away from global median
        direction = 1.0 if m >= global_med else -1.0
        out[mask] = direction * level_gap + 0.25 * _zscore(tau[mask])
    return _zscore(out).astype(np.float32)


def mix_tau(tau_a: np.ndarray, tau_b: np.ndarray, lam: float) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    return _zscore((1.0 - lam) * tau_a + lam * tau_b).astype(np.float32)


def resimulate_outcomes(
    X: np.ndarray,
    T: np.ndarray,
    tau: np.ndarray,
    mu0: np.ndarray,
    noise_std: float = 0.5,
    signal_scale: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Y = mu0 + T * (signal_scale * tau) + eps; tau assumed z-scored."""
    rng = np.random.default_rng(seed)
    T = np.asarray(T).ravel().astype(np.float64)
    tau = np.asarray(tau).ravel().astype(np.float64)
    mu0 = np.asarray(mu0).ravel().astype(np.float64)
    y = mu0 + T * (signal_scale * tau) + rng.normal(0.0, noise_std, size=len(T))
    return y.astype(np.float32)


def make_geometry_pair_from_real(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    seed: int = 0,
    noise_std: float = 0.5,
    signal_scale: float = 2.0,
    level_gap: float = 2.5,
) -> Dict[str, np.ndarray]:
    """
    Build high-contrast smooth vs threshold semi-synthetic worlds.
    Returns tau_* already scaled for PEHE against signal_scale*tau in outcomes:
    we store tau_effect = signal_scale * zscored_geometry so PEHE matches Y process.
    """
    e, mu0, mu1 = fit_nuisances(X, T, Y)
    tau0 = (mu1 - mu0).astype(np.float32)

    tau_s = smooth_tau(tau0, X)
    tau_t = threshold_tau(tau0, X, level_gap=level_gap)

    # effect used in outcome = signal_scale * geometry unit
    tau_s_eff = (signal_scale * tau_s).astype(np.float32)
    tau_t_eff = (signal_scale * tau_t).astype(np.float32)

    Y_s = resimulate_outcomes(
        X, T, tau_s, mu0, noise_std=noise_std, signal_scale=signal_scale, seed=seed + 1
    )
    Y_t = resimulate_outcomes(
        X, T, tau_t, mu0, noise_std=noise_std, signal_scale=signal_scale, seed=seed + 2
    )

    # geometry distance diagnostic
    gap = float(np.sqrt(np.mean((tau_s_eff - tau_t_eff) ** 2)))
    return {
        "X": np.asarray(X, dtype=np.float32),
        "T": np.asarray(T, dtype=np.float32),
        "e": e,
        "mu0": mu0.astype(np.float32),
        "tau_smooth": tau_s_eff,
        "tau_threshold": tau_t_eff,
        "Y_smooth": Y_s,
        "Y_threshold": Y_t,
        "geometry_gap_rmse": gap,
    }


def dr_pseudo_tau_stable(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    delta: float = 0.08,
    smooth: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    More stable DR for small-n real covariates.
    Returns (tau_dr, quality) where quality ≈ -MAE of outcome nuisances on OOF (higher better).
    """
    from sklearn.model_selection import KFold

    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(np.float64)
    n = len(Y)
    e_hat = np.zeros(n)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_mae = []
    for tr, te in kf.split(X):
        clf = LogisticRegression(max_iter=800, C=0.5)
        clf.fit(X[tr], T[tr])
        e_hat[te] = np.clip(clf.predict_proba(X[te])[:, 1], delta, 1 - delta)
        tr0, tr1 = tr[T[tr] == 0], tr[T[tr] == 1]
        if len(tr0) < 10 or len(tr1) < 10:
            continue
        m0 = Ridge(alpha=2.0).fit(X[tr0], Y[tr0])
        m1 = Ridge(alpha=2.0).fit(X[tr1], Y[tr1])
        mu0[te] = m0.predict(X[te])
        mu1[te] = m1.predict(X[te])
        y_hat = np.where(T[te] == 1, mu1[te], mu0[te])
        oof_mae.append(float(np.mean(np.abs(Y[te] - y_hat))))
    e_hat = np.clip(e_hat, delta, 1 - delta)
    tau = (mu1 - mu0) + (T / e_hat) * (Y - mu1) - ((1 - T) / (1 - e_hat)) * (Y - mu0)
    tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth:
        tau = Ridge(alpha=2.0).fit(X, tau).predict(X)
    quality = -float(np.mean(oof_mae)) if oof_mae else -1e9
    return tau.astype(np.float32), quality
