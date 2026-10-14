"""
RIPO (Residual Intervention-aware Predictive Outcome) models.

LOCKED MAIN ARCHITECTURE FOR THE PAPER (aligned with CLAIMS_LOCKED)
-------------------------------------------------------------------
Primary estimator reported in main tables and abstract:
  **ripo_no_tree**  =  FT-Transformer encoder + two outcome heads
                       + mean-difference representation balancing (IPM-style).

  When tree_signals is None (the default path used for primary results),
  RIPOAttention reduces exactly to a pure residual-attention / FT-TARNet
  backbone. Tree cross-attention and residual fusion are DISABLED in the
  main paper path because ablation showed they are null / non-significant
  for PEHE (IHDP p≈0.64, ACIC pooled p≈0.52).

Tree-fusion variants (RIPOAttention with tree signals, RIPOAdaptive,
RIPODistill, RIPOLateFusion) are retained ONLY for the Ablation section
and for historical completeness. Do not present them as the primary
contribution.

Balancing term (see CausalLoss in train_neural.py):
  Default is L2 mean-difference of representations (linear MMD).
  This is NOT Wasserstein distance. Never claim Wasserstein in the paper.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models_neural import FeatureTokenizer, FTTransformerEncoder
except ImportError:
    from src.models_neural import FeatureTokenizer, FTTransformerEncoder


# =========================================================
# Variant 1: Knowledge Distillation
# =========================================================

class RIPODistill(nn.Module):
    # (بدون تغییر - دقیقاً مشابه کد اصلی شما)
    def __init__(self, num_features, d_token=32, num_layers=2, nhead=4, dim_feedforward=128, dropout=0.1, head_hidden_dim=64):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token=d_token)
        self.encoder = FTTransformerEncoder(num_layers, d_token, nhead, dim_feedforward, dropout)
        def outcome_head():
            return nn.Sequential(nn.Linear(d_token, head_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden_dim, 1))
        self.head_0 = outcome_head()
        self.head_1 = outcome_head()
        self.distill_temp = 2.0
    
    def encode(self, x):
        tokens = self.encoder(self.tokenizer(x))
        return tokens.mean(dim=1)
    
    def forward(self, x, tree_po0=None, tree_po1=None, training=False):
        z = self.encode(x)
        y0 = self.head_0(z)
        y1 = self.head_1(z)
        if training and tree_po0 is not None and tree_po1 is not None:
            # Force matching 1-D shapes so B=1 never triggers broadcasting warning
            distill_0 = F.mse_loss(y0.reshape(-1), tree_po0.reshape(-1), reduction="mean")
            distill_1 = F.mse_loss(y1.reshape(-1), tree_po1.reshape(-1), reduction="mean")
            distill_loss = (distill_0 + distill_1) / 2
            return y0, y1, z, distill_loss
        else:
            return y0, y1, z, y0.new_tensor(0.0)
    
    def predict_effect(self, x):
        z = self.encode(x)
        y0 = self.head_0(z)
        y1 = self.head_1(z)
        return y1 - y0


# =========================================================
# Cross-Attention Fusion block (used by the main model)
# =========================================================

class CrossAttentionFusion(nn.Module):
    """
    Tree signal attends over neural feature tokens.

    Query  = projected tree signal   [B, 1, D]
    Key/Value = neural tokens        [B, F, D]

    Returns a gated residual combination of the neural mean and the
    attention context, together with the attention weights (for analysis).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        initial_temperature: float = 1.0,
        learnable_temp: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.learnable_temp = learnable_temp

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_proj = nn.Linear(embed_dim * 2, 1)

        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        else:
            self.register_buffer("temperature", torch.tensor(initial_temperature))

    def forward(
        self, x_neural: torch.Tensor, x_tree: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_neural: [B, F, D]   x_tree: [B, 1, D]
        Q = self.query_proj(x_tree)
        K = self.key_proj(x_neural)
        V = self.value_proj(x_neural)

        if self.learnable_temp:
            temp = torch.clamp(F.softplus(self.log_temp), min=0.2, max=5.0)
        else:
            temp = self.temperature

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attention_weights = F.softmax(scores / temp, dim=-1)  # [B, 1, F]
        context = torch.matmul(attention_weights, V)          # [B, 1, D]

        x_neural_agg = x_neural.mean(dim=1, keepdim=True)
        gate_input = torch.cat([x_neural_agg, context], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        fused_output = gate * x_neural_agg + (1.0 - gate) * context

        return fused_output, attention_weights, scores, gate


# =========================================================
# MAIN MODEL CLASS (used both for primary no-tree path and tree ablation)
# =========================================================

class RIPOAttention(nn.Module):
    """
    RIPO-Attention.

    PRIMARY PAPER PATH (ripo_no_tree):
      Call with tree_signals=None → pure FT-Transformer + two heads.
      This is the estimator reported in main tables / abstract.
      Balancing is applied externally via CausalLoss (mean-diff IPM-style).

    ABLATION PATH (ripo_main / tree fusion):
      1. FT-Transformer encodes covariates into feature tokens.
      2. Cross-Attention lets the tree signal attend over those tokens.
      3. A residual connection mixes the attended representation with the
         pure neural mean (default residual_alpha=0.7).
      4. Two independent outcome heads produce mu0 and mu1.

    Tree fusion was found non-significant for PEHE (see CLAIMS_LOCKED C2).
    Do not present the tree path as the main contribution.
    """

    def __init__(
        self,
        num_features: int,
        tree_signal_dim: int = 4,
        d_token: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        head_hidden_dim: int = 64,
        residual_alpha: float = 0.7,
    ):
        super().__init__()
        self.residual_alpha = residual_alpha
        self.tokenizer = FeatureTokenizer(num_features, d_token=d_token)
        self.encoder = FTTransformerEncoder(
            num_layers, d_token, nhead, dim_feedforward, dropout
        )
        self.tree_projection = nn.Sequential(
            nn.Linear(tree_signal_dim, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = CrossAttentionFusion(
            embed_dim=d_token, initial_temperature=1.0, learnable_temp=True
        )
        self.feature_importance_: Optional[torch.Tensor] = None

        def outcome_head():
            return nn.Sequential(
                nn.Linear(d_token, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )

        self.head_0 = outcome_head()
        self.head_1 = outcome_head()

    def encode(
        self, x: torch.Tensor, tree_signals: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        tokens = self.encoder(self.tokenizer(x))  # [B, F, D]

        if tree_signals is None:
            return tokens.mean(dim=1), None

        tree_rep = self.tree_projection(tree_signals).unsqueeze(1)  # [B, 1, D]
        fused, attn_weights, scores, gate = self.fusion(tokens, tree_rep)
        fused = fused.mean(dim=1)  # [B, D]

        # Residual connection (advisor-recommended design)
        neural_mean = tokens.mean(dim=1)
        alpha = self.residual_alpha
        fused = alpha * fused + (1.0 - alpha) * neural_mean

        # NOTE: feature_importance_ is stored for analysis only.
        # It must NOT be multiplied into the fused [B, d_token] vector
        # (importance has shape [n_features], not [d_token]).

        return fused, (attn_weights, scores, gate)

    def forward(
        self, x: torch.Tensor, tree_signals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        z, extra = self.encode(x, tree_signals)
        y0 = self.head_0(z)
        y1 = self.head_1(z)
        return y0, y1, z, extra

    def predict_effect(
        self, x: torch.Tensor, tree_signals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y0, y1, _, _ = self.forward(x, tree_signals)
        return y1 - y0

    def get_attention_weights(
        self, x: torch.Tensor, tree_signals: torch.Tensor
    ) -> torch.Tensor:
        """Convenience helper for interpretability analysis."""
        _, extra = self.encode(x, tree_signals)
        if extra is None:
            raise RuntimeError("No tree signals provided; attention is undefined.")
        attn_weights, _, _ = extra
        return attn_weights  # [B, 1, F]



# =========================================================
# MAIN MODEL v2 — Adaptive Tree–Neural Fusion (Option A)
# =========================================================

class HeterogeneityGate(nn.Module):
    """
    Sample-wise residual weight α(x) ∈ (0, 1).

    Motivation (matches tree-signal diagnosis on ACIC):
      - When estimated treatment-effect heterogeneity is high and the tree
        CATE is confident, α → larger  (trust tree-attended path more).
      - When τ̂ is near-constant / propensity is uncertain, α → smaller
        (fall back to pure neural representation).

    Inputs to the gate (per sample):
      1. |τ̂_tree|          — magnitude of OOF tree CATE
      2. propensity entropy — uncertainty of treatment assignment model
      3. ||z_neural||       — scale of neural representation
      4. ||z_tree_attn||    — scale of tree-attended representation

    A tiny MLP maps these 4 scalars → logit → sigmoid → α.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Start near α≈0.3 (favor neural), consistent with ACIC HPO finding
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -0.85)  # sigmoid(-0.85) ≈ 0.30

    def forward(
        self,
        tree_signals: torch.Tensor,
        z_neural: torch.Tensor,
        z_tree: torch.Tensor,
    ) -> torch.Tensor:
        """
        tree_signals : [B, 4]  (mu0, mu1, tau, propensity)
        z_neural     : [B, D]
        z_tree       : [B, D]
        returns α    : [B, 1]  in (0, 1)
        """
        tau = tree_signals[:, 2:3].abs()                          # [B, 1]
        prop = tree_signals[:, 3:4].clamp(1e-4, 1.0 - 1e-4)     # [B, 1]
        # Binary entropy of propensity (high = uncertain assignment)
        ent = -(prop * prop.log() + (1.0 - prop) * (1.0 - prop).log())  # [B, 1]
        nrm_n = z_neural.norm(dim=-1, keepdim=True) / (z_neural.size(-1) ** 0.5)
        nrm_t = z_tree.norm(dim=-1, keepdim=True) / (z_tree.size(-1) ** 0.5)
        features = torch.cat([tau, ent, nrm_n, nrm_t], dim=-1)  # [B, 4]
        return torch.sigmoid(self.net(features))                 # [B, 1]


class RIPOAdaptive(nn.Module):
    """
    RIPO with **Adaptive residual fusion** — ABLATION / ARCHIVE ONLY.

    NOT the primary paper estimator. Primary path is ResidualAttentionCATE /
    ripo_no_tree (no tree signals). Tree fusion was null on IHDP and ACIC n=5.

    Pipeline (ablation completeness only)
    ------------------------------------
    1. FT-Transformer encodes covariates → feature tokens.
    2. Cross-Attention: tree signal attends over neural tokens.
    3. HeterogeneityGate produces sample-wise α(x) ∈ (0,1).
    4. z = α · z_tree_attn + (1−α) · z_neural.
    5. Two outcome heads → μ0, μ1.

    When tree signals are None, reduces to pure neural TARNet-style.
    """

    def __init__(
        self,
        num_features: int,
        tree_signal_dim: int = 4,
        d_token: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
        gate_hidden: int = 16,
        # fallback fixed alpha if gate disabled (ablation)
        fixed_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.fixed_alpha = fixed_alpha
        self.tokenizer = FeatureTokenizer(num_features, d_token=d_token)
        self.encoder = FTTransformerEncoder(
            num_layers, d_token, nhead, dim_feedforward, dropout
        )
        self.tree_projection = nn.Sequential(
            nn.Linear(tree_signal_dim, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = CrossAttentionFusion(
            embed_dim=d_token, initial_temperature=1.0, learnable_temp=True
        )
        self.gate = HeterogeneityGate(hidden=gate_hidden)

        def outcome_head():
            return nn.Sequential(
                nn.Linear(d_token, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )

        self.head_0 = outcome_head()
        self.head_1 = outcome_head()

    def encode(
        self,
        x: torch.Tensor,
        tree_signals: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[torch.Tensor]]:
        """
        Returns
        -------
        z            : [B, D] fused representation
        extra        : (attn_weights, scores, fusion_gate) or None
        alpha        : [B, 1] adaptive residual weights (None if no tree)
        """
        tokens = self.encoder(self.tokenizer(x))       # [B, F, D]
        neural_mean = tokens.mean(dim=1)               # [B, D]

        if tree_signals is None:
            return neural_mean, None, None

        tree_rep = self.tree_projection(tree_signals).unsqueeze(1)  # [B, 1, D]
        fused, attn_weights, scores, fuse_gate = self.fusion(tokens, tree_rep)
        z_tree = fused.mean(dim=1)                     # [B, D]

        if self.fixed_alpha is not None:
            alpha = z_tree.new_full((z_tree.size(0), 1), float(self.fixed_alpha))
        else:
            alpha = self.gate(tree_signals, neural_mean, z_tree)  # [B, 1]

        z = alpha * z_tree + (1.0 - alpha) * neural_mean
        return z, (attn_weights, scores, fuse_gate), alpha

    def forward(
        self,
        x: torch.Tensor,
        tree_signals: Optional[torch.Tensor] = None,
        return_alpha: bool = False,
    ):
        z, extra, alpha = self.encode(x, tree_signals)
        y0 = self.head_0(z)
        y1 = self.head_1(z)
        if return_alpha:
            return y0, y1, z, extra, alpha
        return y0, y1, z, extra

    def predict_effect(
        self, x: torch.Tensor, tree_signals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y0, y1, _, _ = self.forward(x, tree_signals)
        return y1 - y0

    def get_attention_weights(
        self, x: torch.Tensor, tree_signals: torch.Tensor
    ) -> torch.Tensor:
        _, extra, _ = self.encode(x, tree_signals)
        if extra is None:
            raise RuntimeError("No tree signals; attention undefined.")
        return extra[0]  # [B, 1, F]

    def get_alpha(
        self, x: torch.Tensor, tree_signals: torch.Tensor
    ) -> torch.Tensor:
        """Expose α(x) for interpretability / heterogeneity analysis."""
        _, _, alpha = self.encode(x, tree_signals)
        if alpha is None:
            raise RuntimeError("No tree signals; alpha undefined.")
        return alpha



# =========================================================
# Variant 3: Late Fusion
# =========================================================

class RIPOLateFusion(nn.Module):
    """
    RIPO with Late Fusion.
    
    Neural encoder runs independently. Tree signals are only
    concatenated at the final prediction layer.
    """
    
    def __init__(
        self,
        num_features,
        tree_signal_dim=4,
        d_token=32,
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1,
        head_hidden_dim=64,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token=d_token)
        self.encoder = FTTransformerEncoder(
            num_layers, d_token, nhead, dim_feedforward, dropout
        )
        
        # Late fusion: neural + tree signals
        fusion_dim = d_token + tree_signal_dim
        
        def fused_outcome_head():
            return nn.Sequential(
                nn.Linear(fusion_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1)
            )
        
        self.head_0 = fused_outcome_head()
        self.head_1 = fused_outcome_head()
    
    def encode(self, x):
        tokens = self.encoder(self.tokenizer(x))
        return tokens.mean(dim=1)
    
    def forward(self, x, tree_signals):
        z = self.encode(x)
        
        # Late fusion at prediction layer
        if tree_signals is not None:
            fused_0 = torch.cat([z, tree_signals], dim=1)
            fused_1 = torch.cat([z, tree_signals], dim=1)
        else:
            # Pad with zeros if no tree signals
            fused_0 = torch.cat([z, torch.zeros_like(z[:, :4])], dim=1)
            fused_1 = torch.cat([z, torch.zeros_like(z[:, :4])], dim=1)
        
        y0 = self.head_0(fused_0)
        y1 = self.head_1(fused_1)
        return y0, y1, z, None
    
    def predict_effect(self, x, tree_signals):
        y0, y1, _, _ = self.forward(x, tree_signals)
        return y1 - y0


# =========================================================
# Model Factory
# =========================================================

def create_ripo_v2(variant: str, num_features: int, **kwargs) -> nn.Module:
    """
    Factory for all RIPO variants.

    Parameters
    ----------
    variant : str
        One of:
          - 'attention' / 'main' / 'no_tree' → RIPOAttention
            (primary path: call with tree_signals=None)
          - 'distill'                         → RIPODistill (ablation only)
          - 'late_fusion'                     → RIPOLateFusion (ablation only)
    """
    variant = variant.lower().strip()
    if variant in ("attention", "main", "ripo", "ripo_attention", "no_tree", "ripo_no_tree"):
        return RIPOAttention(num_features, **kwargs)
    if variant == "distill":
        kwargs.pop("tree_signal_dim", None)
        return RIPODistill(num_features, **kwargs)
    if variant in ("late_fusion", "late"):
        return RIPOLateFusion(num_features, **kwargs)
    raise ValueError(
        f"Unknown variant '{variant}'. "
        "Use 'attention'/'no_tree' (primary), 'distill', or 'late_fusion'."
    )


def create_main_ripo(num_features: int, **kwargs) -> RIPOAttention:
    """
    Canonical constructor for the primary paper estimator (ripo_no_tree path).
    Returns RIPOAttention; callers must pass tree_signals=None at inference/train
    for the locked main results. Tree fusion is ablation-only.
    """
    return RIPOAttention(num_features, **kwargs)


if __name__ == "__main__":
    # Test all variants
    print("Testing RIPO v2 variants...")
    
    batch_size = 32
    num_features = 58  # ACIC
    tree_signal_dim = 4
    
    x = torch.randn(batch_size, num_features)
    tree_signals = torch.randn(batch_size, tree_signal_dim)
    
    # Test Distill
    print("\n[1] Testing RIPO-Distill...")
    model_distill = RIPODistill(num_features)
    y0, y1, z, distill_loss = model_distill(x, training=False)
    print(f"    ✅ Output shapes: y0={y0.shape}, y1={y1.shape}, z={z.shape}")
    
    # Test Attention
    print("\n[2] Testing RIPO-Attention...")
    model_attn = RIPOAttention(num_features, tree_signal_dim=tree_signal_dim)
    y0, y1, z, _ = model_attn(x, tree_signals)
    print(f"    ✅ Output shapes: y0={y0.shape}, y1={y1.shape}, z={z.shape}")
    
    # Test Late Fusion
    print("\n[3] Testing RIPO-LateFusion...")
    model_late = RIPOLateFusion(num_features, tree_signal_dim=tree_signal_dim)
    y0, y1, z, _ = model_late(x, tree_signals)
    print(f"    ✅ Output shapes: y0={y0.shape}, y1={y1.shape}, z={z.shape}")
    
    print("\n✅ All RIPO v2 variants passed!")


# =========================================================
# PAPER-FACING WRAPPER (primary estimator — no tree path)
# =========================================================

class ResidualAttentionCATE(nn.Module):
    """
    Locked primary estimator for the paper (ripo_no_tree path).

    Thin wrapper around RIPOAttention that never accepts tree signals.
    Use this in main tables / abstract so the no-tree contract cannot be
    violated by accident.
    """

    def __init__(self, num_features: int, **kwargs):
        super().__init__()
        kwargs.pop("tree_signal_dim", None)
        self.backbone = RIPOAttention(
            num_features=num_features,
            tree_signal_dim=4,  # unused; encode() gets None
            **kwargs,
        )

    def forward(self, x: torch.Tensor, w=None, return_representation: bool = False):
        y0, y1, z, _ = self.backbone(x, tree_signals=None)
        if return_representation:
            return y0, y1, z
        if w is not None:
            w = w.reshape(-1, 1).float()
            return w * y1 + (1.0 - w) * y0
        return y0, y1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.backbone.encode(x, tree_signals=None)
        return z

    def predict_effect(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.predict_effect(x, tree_signals=None)
