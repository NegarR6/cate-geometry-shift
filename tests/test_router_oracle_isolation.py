"""Regression tests for isolating factual experts from oracle router labels."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from src.models_geometry_adaptive import GeometryAdaptiveCATE
from src.train_geometry_adaptive import GeometryAdaptiveTrainer, build_loaders


def _small_model(n_features: int = 6) -> GeometryAdaptiveCATE:
    return GeometryAdaptiveCATE(
        num_features=n_features,
        neural_d_token=8,
        neural_layers=1,
        neural_nhead=2,
        neural_ff=16,
        neural_dropout=0.0,
        neural_head_dim=8,
        tree_d_token=8,
        tree_layers=1,
        tree_nhead=2,
        tree_ff=16,
        tree_dropout=0.0,
        tree_head_dim=8,
        router_hidden=8,
        router_dropout=0.0,
    )


def test_router_logits_have_no_gradient_path_to_experts():
    """Router-only loss must update the router, not expert parameters."""
    torch.manual_seed(4)
    model = _small_model()
    x = torch.randn(12, 6)
    e = torch.full((12,), 0.5)
    tree_signals = torch.randn(12, 4)

    out = model(x, e, tree_signals=tree_signals)
    # A route-label loss is a function of logits only.
    route_only_loss = out["logits"][:, 0].sum()
    route_only_loss.backward()

    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.router.parameters())
    for expert in (model.neural_expert, model.tree_expert):
        assert all(p.grad is None for p in expert.parameters())


def test_paper_stage2_keeps_pretrained_experts_fixed():
    """Even after multiple Stage-2 epochs, expert parameters must not change."""
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    n, p = 20, 6
    x = rng.normal(size=(n, p)).astype(np.float32)
    t = rng.binomial(1, 0.5, size=n).astype(np.float32)
    e = np.full(n, 0.5, dtype=np.float32)
    y = (x[:, 0] + t * (x[:, 1] * x[:, 2])).astype(np.float32)
    tree_signals = rng.normal(size=(n, 4)).astype(np.float32)
    oracle = np.tile(np.array([0, 1], dtype=np.int64), n // 2)
    tr_idx = np.arange(14)
    val_idx = np.arange(14, n)
    train_loader, val_loader = build_loaders(
        x, t, y, e, tree_signals, oracle, tr_idx, val_idx, batch_size=7
    )

    model = _small_model(p)
    trainer = GeometryAdaptiveTrainer(
        model,
        joint_epochs=2,
        joint_patience=2,
        freeze_experts_epochs=1,
        route_weight_start=0.5,
        route_weight_end=0.5,
        route_warmup_epochs=1,
        freeze_experts_entire_stage2=True,
    )
    before = {
        name: param.detach().clone()
        for name, param in list(model.neural_expert.named_parameters())
    }
    before_tree = {
        name: param.detach().clone()
        for name, param in list(model.tree_expert.named_parameters())
    }

    trainer.joint_train(train_loader, val_loader, verbose=False)

    for name, param in model.neural_expert.named_parameters():
        torch.testing.assert_close(param.detach(), before[name], rtol=0.0, atol=0.0)
        assert not param.requires_grad
    for name, param in model.tree_expert.named_parameters():
        torch.testing.assert_close(param.detach(), before_tree[name], rtol=0.0, atol=0.0)
        assert not param.requires_grad
