"""
SOTA-style neural CATE baselines (pure PyTorch, no catenets dependency).

DragonNet (Shi et al.): shared representation + two outcome heads + propensity head.
CFR-style: TARNet backbone with explicit IPM (mean-diff / MMD) — training uses
the existing CausalLoss IPM in train_neural; this file only defines the architecture.

These are the neural baselines required for a credible ICLR comparison table.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DragonNet(nn.Module):
    """
    Dragonnet-style architecture on an MLP backbone.

    Components
    ----------
    - shared representation phi(x)
    - outcome heads mu0, mu1
    - propensity head g(x) = P(W=1|x)

    Training loss (see dragonnet_loss):
      L = L_outcome + alpha * L_propensity  (+ optional targeted reg)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(200, 200, 200),
        head_hidden: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.rep_dim = prev

        def head():
            return nn.Sequential(
                nn.Linear(self.rep_dim, head_hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        self.head_0 = head()
        self.head_1 = head()
        self.propensity = nn.Sequential(
            nn.Linear(self.rep_dim, head_hidden),
            nn.ELU(),
            nn.Linear(head_hidden, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None, return_all: bool = False
    ):
        z = self.encode(x)
        y0 = self.head_0(z)
        y1 = self.head_1(z)
        logit_p = self.propensity(z)
        if return_all:
            return y0, y1, logit_p, z
        if w is not None:
            w = w.unsqueeze(-1).float()
            return w * y1 + (1.0 - w) * y0
        return y0, y1

    def predict_effect(self, x: torch.Tensor) -> torch.Tensor:
        y0, y1 = self.forward(x, w=None)
        return (y1 - y0).squeeze(-1)


class TARNetMLP(nn.Module):
    """Simple MLP TARNet (no propensity) — CFR backbone without IPM in the module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(200, 200, 200),
        head_hidden: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.rep_dim = prev

        def head():
            return nn.Sequential(
                nn.Linear(self.rep_dim, head_hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        self.head_0 = head()
        self.head_1 = head()

    def forward(self, x, w=None, return_representation=False):
        z = self.encoder(x)
        y0, y1 = self.head_0(z), self.head_1(z)
        if return_representation:
            return y0, y1, z
        if w is not None:
            w = w.unsqueeze(-1).float()
            return w * y1 + (1.0 - w) * y0
        return y0, y1

    def predict_effect(self, x):
        y0, y1 = self.forward(x)
        return (y1 - y0).squeeze(-1)


def dragonnet_loss(
    y0: torch.Tensor,
    y1: torch.Tensor,
    logit_p: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Outcome MSE + propensity BCE."""
    w = w.reshape(-1).float()
    y = y.reshape(-1)
    y0 = y0.reshape(-1)
    y1 = y1.reshape(-1)
    logit_p = logit_p.reshape(-1)

    y_pred = w * y1 + (1.0 - w) * y0
    outcome = F.mse_loss(y_pred, y)
    prop = F.binary_cross_entropy_with_logits(logit_p, w)
    total = outcome + alpha * prop
    return total, outcome, prop

# ---------------------------------------------------------------------------
# TransTEE (stable v2) — Zhang et al. arXiv:2202.01336 tabular binary
# ---------------------------------------------------------------------------

class _PreNormAttn(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.ln_q(q)
        kvn = self.ln_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + self.dropout(out)
        q = q + self.ff(self.ln_ff(q))
        return q


class TransTEE(nn.Module):
    """
    Tabular binary TransTEE (S-learner with optional dual readout).

    Covariates embedded feature-wise; treatment embedded; self-attn on X;
    cross-attn with treatment as query; project to outcome.
    """

    def __init__(
        self,
        n_features: int,
        d_emb: int = 16,
        n_layers: int = 1,
        nhead: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_emb % nhead != 0:
            d_emb = int(math.ceil(d_emb / nhead) * nhead)
        self.n_features = n_features
        self.d_emb = d_emb
        self.cov_emb = nn.Linear(1, d_emb)
        self.trt_emb = nn.Embedding(2, d_emb)  # binary treatment tokens
        self.pos = nn.Parameter(torch.zeros(1, n_features, d_emb))
        nn.init.normal_(self.pos, std=0.02)
        self.self_blocks = nn.ModuleList(
            [_PreNormAttn(d_emb, nhead, dropout) for _ in range(n_layers)]
        )
        self.cross_blocks = nn.ModuleList(
            [_PreNormAttn(d_emb, nhead, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_emb),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_cov(self, x: torch.Tensor) -> torch.Tensor:
        # (B, p) -> (B, p, d)
        tok = self.cov_emb(x.unsqueeze(-1)) + self.pos
        for blk in self.self_blocks:
            tok = blk(tok, tok)
        return tok

    def forward_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.view(-1)
        t = t.long().clamp(0, 1)
        cov = self._encode_cov(x)
        h = self.trt_emb(t).unsqueeze(1)  # (B, 1, d)
        for blk in self.cross_blocks:
            h = blk(h, cov)
        return self.head(h.squeeze(1))

    def forward(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None, return_all: bool = False
    ):
        B = x.size(0)
        device = x.device
        t0 = torch.zeros(B, dtype=torch.long, device=device)
        t1 = torch.ones(B, dtype=torch.long, device=device)
        y0 = self.forward_outcome(x, t0)
        y1 = self.forward_outcome(x, t1)
        if return_all:
            return y0, y1, None, None
        if w is not None:
            w = w.view(-1, 1).float()
            return w * y1 + (1.0 - w) * y0
        return y0, y1

    def predict_effect(self, x: torch.Tensor) -> torch.Tensor:
        y0, y1 = self.forward(x, w=None)
        return (y1 - y0).squeeze(-1)


def transtee_loss(y0, y1, w, y) -> torch.Tensor:
    w = w.reshape(-1).float()
    y = y.reshape(-1)
    y_pred = w * y1.reshape(-1) + (1.0 - w) * y0.reshape(-1)
    return F.mse_loss(y_pred, y)
