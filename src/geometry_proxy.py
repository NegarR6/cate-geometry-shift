"""Geometry proxy D_G from train-only observables (Path β v2)."""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor


def _z(x: float, mu: float, sd: float) -> float:
    sd = sd if sd > 1e-8 else 1.0
    return (x - mu) / sd


@dataclass
class GeometryProxyResult:
    D_G: float
    components: Dict[str, float]


def expert_disagreement_score(tau_n: np.ndarray, tau_t: np.ndarray) -> float:
    d = np.abs(np.asarray(tau_n).ravel() - np.asarray(tau_t).ravel())
    return float(np.mean(d))


def stump_gain_score(X: np.ndarray, tau_proxy: np.ndarray, max_depth: int = 2) -> float:
    y = np.asarray(tau_proxy).ravel()
    var0 = float(np.var(y) + 1e-12)
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    tree.fit(X, y)
    pred = tree.predict(X)
    var1 = float(np.var(y - pred) + 1e-12)
    return float(max(0.0, 1.0 - var1 / var0))


def roughness_score(X: np.ndarray, tau_proxy: np.ndarray, k: int = 15) -> float:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(tau_proxy).ravel()
    n = len(y)
    k = min(k, max(2, n // 10))
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n, 400), replace=False)
    Xs, ys = X[idx], y[idx]
    d2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, kth=k, axis=1)[:, :k]
    local_std = np.std(ys[nn], axis=1)
    return float(np.mean(local_std))


def ridge_cate_proxy(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel()
    m0 = Ridge(alpha=1.0).fit(X[T == 0], Y[T == 0])
    m1 = Ridge(alpha=1.0).fit(X[T == 1], Y[T == 1])
    return (m1.predict(X) - m0.predict(X)).astype(np.float64)


def compute_D_G(
    X_train: np.ndarray,
    T_train: np.ndarray,
    Y_train: np.ndarray,
    tau_n_train: np.ndarray,
    tau_t_train: np.ndarray,
    norm: Optional[Dict[str, Tuple[float, float]]] = None,
) -> GeometryProxyResult:
    """
    Higher D_G => more threshold-like / tree-favoring (v2).
    D_G = z(stump) + z(dis) - z(rough)   if norm given
        = stump + dis - rough            if norm is None
    """
    tau_proxy = ridge_cate_proxy(X_train, T_train, Y_train)
    tau_mid = 0.5 * (
        np.asarray(tau_n_train).ravel() + np.asarray(tau_t_train).ravel()
    )
    tau_use = 0.5 * tau_proxy + 0.5 * tau_mid

    dis = expert_disagreement_score(tau_n_train, tau_t_train)
    stump = stump_gain_score(X_train, tau_use)
    rough = roughness_score(X_train, tau_use)
    comps = {"disagreement": dis, "stump_gain": stump, "roughness": rough}

    if norm is None:
        D = stump + dis - rough
    else:
        D = (
            _z(stump, *norm["stump_gain"])
            + _z(dis, *norm["disagreement"])
            - _z(rough, *norm["roughness"])
        )
    return GeometryProxyResult(D_G=float(D), components=comps)


def fit_norm_from_component_rows(rows: list) -> Dict[str, Tuple[float, float]]:
    keys = ["disagreement", "stump_gain", "roughness"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = (float(vals.mean()), float(vals.std() + 1e-12))
    return out


def norm_from_synthetic_csv(path: str) -> Dict[str, Tuple[float, float]]:
    """Build z-norm from B0 proxy_AB_raw.csv columns c_* ."""
    import pandas as pd
    df = pd.read_csv(path)
    rows = [
        {
            "disagreement": float(r["c_disagreement"]),
            "stump_gain": float(r["c_stump_gain"]),
            "roughness": float(r["c_roughness"]),
        }
        for _, r in df.iterrows()
    ]
    return fit_norm_from_component_rows(rows)
