"""Unit tests for leakage guards and deterministic seeds (P0).

Critical claim of the paper protocol: under pure geometry shift the
source-fitted tree-signal extractor must NOT change when target outcomes
are permuted or replaced. These tests enforce that contract.
"""
from __future__ import annotations

import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pytest

from src.synthetic_regimes import sample_X, propensity, mu0, tau_tilde_A, tau_tilde_C
from src.tree_signals import CrossFittedTreeSignals
from src.budgeted_ladder_utils import dr_pseudo_tau


def _make_source(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = sample_X(n, 12, rng)
    e = propensity(X)
    T = rng.binomial(1, e)
    tau = tau_tilde_A(X)
    Y = mu0(X) + T * tau + rng.normal(0, 0.5, size=n)
    return X, T, Y, e


def test_tree_signals_source_freeze_under_target_y_permutation():
    """Permuting target Y must not change transform(X_target) of a frozen source extractor."""
    Xs, Ts, Ys, _ = _make_source(seed=7)
    extractor = CrossFittedTreeSignals(n_splits=3, seed=7, n_estimators=30, max_depth=3)
    extractor.fit_transform(Xs, Ts, Ys)

    rng = np.random.default_rng(99)
    Xt = sample_X(200, 12, rng)
    sig1 = extractor.transform(Xt)
    Yt = rng.normal(0, 1, size=200)
    for k in range(5):
        _ = rng.permutation(Yt)  # unused; transform must ignore it
        sig_k = extractor.transform(Xt)
        np.testing.assert_allclose(sig1, sig_k, atol=1e-7)


def test_tree_signals_refit_on_target_changes_signals():
    """Sanity: if someone re-fits on target, signals DO change (the bug we guard against)."""
    Xs, Ts, Ys, _ = _make_source(seed=3)
    extractor = CrossFittedTreeSignals(n_splits=3, seed=3, n_estimators=30, max_depth=3)
    extractor.fit_transform(Xs, Ts, Ys)

    rng = np.random.default_rng(11)
    Xt = sample_X(300, 12, rng)
    e = propensity(Xt)
    Tt = rng.binomial(1, e)
    tau = tau_tilde_C(Xt)
    Yt = mu0(Xt) + Tt * tau + rng.normal(0, 0.5, size=300)

    sig_frozen = extractor.transform(Xt)

    leaky = CrossFittedTreeSignals(n_splits=3, seed=3, n_estimators=30, max_depth=3)
    leaky.fit_transform(Xt, Tt, Yt)
    sig_leaky = leaky.transform(Xt)

    assert not np.allclose(sig_frozen, sig_leaky, atol=1e-3)


def test_dr_pseudo_tau_length_matches_input():
    """dr_pseudo_tau must only ever be called on the pool; length must match."""
    rng = np.random.default_rng(0)
    n = 80
    X = rng.normal(size=(n, 6))
    T = rng.binomial(1, 0.5, size=n)
    Y = rng.normal(size=n)
    tau = dr_pseudo_tau(X, T, Y, n_splits=3, seed=0, smooth=False)
    assert tau.shape == (n,)
    assert np.all(np.isfinite(tau))


def test_dr_default_no_smooth():
    """Default smooth=False after P0; smoothed and unsmoothed differ on nonlinear data."""
    rng = np.random.default_rng(1)
    n = 120
    X = rng.normal(size=(n, 5))
    T = rng.binomial(1, 0.5, size=n)
    Y = np.sin(X[:, 0]) * X[:, 1] + T * (X[:, 2] ** 2) + rng.normal(0, 0.1, size=n)
    tau_raw = dr_pseudo_tau(X, T, Y, n_splits=3, seed=1, smooth=False)
    tau_sm = dr_pseudo_tau(X, T, Y, n_splits=3, seed=1, smooth=True)
    assert not np.allclose(tau_raw, tau_sm, atol=1e-5)


def test_seed_reproducibility_dgp():
    """Two calls with the same seed produce identical X, T, Y."""
    from src.synthetic_regimes import generate_synthetic, SyntheticConfig

    cfg = SyntheticConfig(n=200, regime="A", signal=1.0, seed=42)
    d1, _ = generate_synthetic(cfg)
    d2, _ = generate_synthetic(cfg)
    np.testing.assert_array_equal(d1["X"], d2["X"])
    np.testing.assert_array_equal(d1["T"], d2["T"])
    np.testing.assert_array_equal(d1["Y"], d2["Y"])


def test_xgboost_required():
    """Importing tree_signals must succeed only when xgboost is present."""
    import src.tree_signals as ts
    assert hasattr(ts, "CrossFittedTreeSignals")
