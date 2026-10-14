"""
Unit tests: tree-signal leakage safety + seed reproducibility smoke.

Run from repo root:
  python -m src.test_leakage_and_seed
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import numpy as np
import torch

from src.tree_signals import CrossFittedTreeSignals
from src.train_neural import set_global_seed, CausalLoss
from src.models_hybrid_v2 import ResidualAttentionCATE, RIPOAttention


def test_tree_signals_oof_no_full_arm_fallback():
    """Held-out rows must not appear in outcome nuisance fits (leakage check)."""
    rng = np.random.default_rng(0)
    n, p = 80, 6
    X = rng.normal(size=(n, p))
    w = rng.integers(0, 2, size=n)
    # Force both arms present
    w[:20] = 0
    w[20:40] = 1
    y = rng.normal(size=n)

    ext = CrossFittedTreeSignals(n_splits=4, seed=0, n_estimators=20, max_depth=2)
    signals = ext.fit_transform(X, w, y)
    assert signals.shape == (n, 4)
    # tau column = mu1 - mu0
    np.testing.assert_allclose(signals[:, 2], signals[:, 1] - signals[:, 0], atol=1e-5)
    # propensity clipped
    assert signals[:, 3].min() >= 0.025 - 1e-6
    assert signals[:, 3].max() <= 0.975 + 1e-6
    print("  OK tree OOF shape + pack")


def test_seed_reproducibility_forward():
    # eval() disables dropout so forward is deterministic given same weights+input
    set_global_seed(123)
    m1 = ResidualAttentionCATE(num_features=8, d_token=16, num_layers=1, nhead=2)
    m1.eval()
    set_global_seed(123)
    x = torch.randn(16, 8)
    with torch.no_grad():
        y0a, y1a = m1(x)

    set_global_seed(123)
    m2 = ResidualAttentionCATE(num_features=8, d_token=16, num_layers=1, nhead=2)
    m2.eval()
    set_global_seed(123)
    x2 = torch.randn(16, 8)
    with torch.no_grad():
        y0b, y1b = m2(x2)

    assert torch.allclose(x, x2), "input draw not reproducible"
    assert torch.allclose(y0a, y0b), "seed mismatch y0"
    assert torch.allclose(y1a, y1b), "seed mismatch y1"
    print("  OK seed-reproducible init + forward")


def test_no_tree_path_equals_mean_tokens():
    set_global_seed(0)
    m = RIPOAttention(num_features=5, d_token=16, num_layers=1, nhead=2)
    x = torch.randn(8, 5)
    y0, y1, z, extra = m(x, tree_signals=None)
    assert extra is None
    assert z.shape == (8, 16)
    print("  OK no-tree path returns pure neural mean")


def test_causal_loss_ipm_types():
    set_global_seed(0)
    rep = torch.randn(20, 8)
    w = torch.tensor([0, 1] * 10).float()
    y0 = torch.randn(20)
    y1 = torch.randn(20)
    y = torch.randn(20)
    for t in ("mean_diff", "mmd", "sliced", "sinkhorn"):
        loss_fn = CausalLoss(ipm_weight=0.5, ipm_type=t)
        total, out, ipm = loss_fn(y0, y1, y, w, representation=rep)
        assert total.ndim == 0
        assert torch.isfinite(total)
    print("  OK CausalLoss all ipm_types")


if __name__ == "__main__":
    print("Running leakage / seed tests...")
    test_tree_signals_oof_no_full_arm_fallback()
    test_seed_reproducibility_forward()
    test_no_tree_path_equals_mean_tokens()
    test_causal_loss_ipm_types()
    print("ALL PASSED")
