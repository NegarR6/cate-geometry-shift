"""
Geometry-Aware Adaptive CATE Estimator (GeoAdaptive / GeoRIPO)

LOCKED DESIGN FOR METHOD SECTION (builds on CLAIMS_LOCKED + E2/E3)
-------------------------------------------------------------------
Problem addressed
  Ranking of CATE estimators reverses with τ-geometry.
  Observable routers recover gains only in-distribution and fail under
  geometry shift (E3). Tree–neural fusion was null under fixed residual_alpha.

Solution
  A single end-to-end model that
  1. Maintains two specialised experts:
       - NeuralExpert  = pure residual-attention (ripo_no_tree backbone)
       - TreeExpert    = lightweight tree-biased pathway (or frozen OOF signals)
  2. Computes an *observable* geometry score from disagreement, propensity
     and local statistics (re-uses the V2 feature family from e2_router_features).
  3. Routes / mixes the experts with a small GeometryRouter (soft mixture by
     default; hard routing optional).
  4. Produces a single CATE prediction that is geometry-conditional.

Training modes
  - synthetic  : optional supervised routing loss against oracle geometry label
                 (regime A/B/C or argmin PEHE expert) + factual outcome loss
  - real-world : purely observable (DR / T-score surrogate or factual MSE)
                 + entropy regularisation on the router to avoid collapse

The primary reported estimator is the soft-mixture version with mean-diff
balancing applied to the final shared representation (or to each expert).
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models_neural import FeatureTokenizer, FTTransformerEncoder
except ImportError:
    from src.models_neural import FeatureTokenizer, FTTransformerEncoder


# ---------------------------------------------------------------------------
# Geometry feature builder (torch-native, mirrors e2_router_features V2)
# ---------------------------------------------------------------------------

def build_geometry_features_torch(
    tau_experts: torch.Tensor,
    propensity: torch.Tensor,
    extra: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    tau_experts : [B, E]   CATE predictions of the E experts
    propensity  : [B] or [B, 1]
    extra       : [B, D_extra] optional raw covariates or other diagnostics

    Returns feature matrix [B, F] used by the router.
    Features (V2-style):
      - pairwise |τ_i - τ_j|
      - each τ_i itself
      - e, 1-e, e*(1-e)
      - (optional) extra columns
    """
    if tau_experts.dim() != 2:
        raise ValueError(f"tau_experts must be [B, E], got {tau_experts.shape}")
    B, E = tau_experts.shape
    prop = propensity.reshape(B, 1).clamp(1e-4, 1.0 - 1e-4)

    cols: List[torch.Tensor] = []

    # pairwise absolute disagreement
    for i in range(E):
        for j in range(i + 1, E):
            cols.append((tau_experts[:, i] - tau_experts[:, j]).abs().unsqueeze(1))

    # level of each expert
    cols.append(tau_experts)

    # propensity features
    cols.append(prop)
    cols.append(1.0 - prop)
    cols.append(prop * (1.0 - prop))

    if extra is not None:
        if extra.dim() == 1:
            extra = extra.unsqueeze(1)
        cols.append(extra)

    return torch.cat(cols, dim=1)


def geometry_feature_dim(n_experts: int, extra_dim: int = 0) -> int:
    """Number of features produced by build_geometry_features_torch."""
    n_pair = n_experts * (n_experts - 1) // 2
    return n_pair + n_experts + 3 + extra_dim


# ---------------------------------------------------------------------------
# Geometry Router
# ---------------------------------------------------------------------------

class GeometryRouter(nn.Module):
    """
    Small MLP that maps observable geometry features → mixture weights.

    Output is a probability simplex over experts (soft) or logits for hard routing.
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int = 2,
        hidden: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
        learnable_temp: bool = True,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_experts),
        )
        if learnable_temp:
            # softplus ensures positive temperature; init ≈ 1.0
            self.log_temp = nn.Parameter(torch.tensor(math.log(max(temperature, 1e-3))))
        else:
            self.register_buffer("temperature", torch.tensor(float(temperature)))
            self.log_temp = None

        # Prefer slightly higher weight on neural expert at init (bias)
        nn.init.zeros_(self.net[-1].weight)
        bias_init = torch.zeros(n_experts)
        if n_experts >= 2:
            bias_init[0] = 0.3   # neural
            bias_init[1] = -0.3  # tree
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias.copy_(bias_init)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
          weights : [B, E]  softmax probabilities
          logits  : [B, E]
        """
        logits = self.net(features)
        if self.log_temp is not None:
            temp = F.softplus(self.log_temp).clamp(0.2, 5.0)
        else:
            temp = self.temperature
        weights = F.softmax(logits / temp, dim=-1)
        return weights, logits


# ---------------------------------------------------------------------------
# Expert backbones
# ---------------------------------------------------------------------------

class NeuralExpert(nn.Module):
    """
    Pure residual-attention / FT-TARNet expert (identical inductive bias to ripo_no_tree).
    """

    def __init__(
        self,
        num_features: int,
        d_token: int = 32,
        num_layers: int = 3,
        nhead: int = 8,
        dim_feedforward: int = 64,
        dropout: float = 0.2,
        head_hidden_dim: int = 128,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token=d_token)
        self.encoder = FTTransformerEncoder(
            num_layers, d_token, nhead, dim_feedforward, dropout
        )
        def _head():
            return nn.Sequential(
                nn.Linear(d_token, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        self.head_0 = _head()
        self.head_1 = _head()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(self.tokenizer(x))
        return tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        y0 = self.head_0(z)
        y1 = self.head_1(z)
        return y0, y1, z

    def predict_effect(self, x: torch.Tensor) -> torch.Tensor:
        y0, y1, _ = self.forward(x)
        return (y1 - y0).squeeze(-1)


class TreeBiasedExpert(nn.Module):
    """
    Lightweight expert that is biased toward piecewise / threshold structure.

    Design choices (deliberately simple so capacity stays comparable):
      - Same FT-Transformer backbone but shallower / narrower by default
      - An extra residual path that injects a linear projection of the
        (optional) OOF tree signals (mu0, mu1, tau, e)
      - Final heads remain two independent outcome heads

    When tree_signals is None the expert still runs; the residual path is zero.
    This keeps the interface uniform and allows ablation "neural-only router".
    """

    def __init__(
        self,
        num_features: int,
        tree_signal_dim: int = 4,
        d_token: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 64,
        dropout: float = 0.15,
        head_hidden_dim: int = 64,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token=d_token)
        self.encoder = FTTransformerEncoder(
            num_layers, d_token, nhead, dim_feedforward, dropout
        )
        self.tree_proj = nn.Sequential(
            nn.Linear(tree_signal_dim, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mix = nn.Linear(d_token * 2, d_token)
        def _head():
            return nn.Sequential(
                nn.Linear(d_token, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        self.head_0 = _head()
        self.head_1 = _head()

    def encode(
        self, x: torch.Tensor, tree_signals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tokens = self.encoder(self.tokenizer(x))
        z_neural = tokens.mean(dim=1)
        if tree_signals is None:
            return z_neural
        z_tree = self.tree_proj(tree_signals)
        return self.mix(torch.cat([z_neural, z_tree], dim=-1))

    def forward(
        self, x: torch.Tensor, tree_signals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x, tree_signals)
        y0 = self.head_0(z)
        y1 = self.head_1(z)
        return y0, y1, z

    def predict_effect(
        self, x: torch.Tensor, tree_signals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y0, y1, _ = self.forward(x, tree_signals)
        return (y1 - y0).squeeze(-1)


# ---------------------------------------------------------------------------
# Main adaptive model
# ---------------------------------------------------------------------------

class GeometryAdaptiveCATE(nn.Module):
    """
    End-to-end geometry-aware adaptive CATE estimator.

    Forward returns a rich dictionary so that training can use
    factual loss, IPM, routing supervision, and entropy regularisation.
    """

    def __init__(
        self,
        num_features: int,
        # Neural expert (matches locked ripo_no_tree capacity by default)
        neural_d_token: int = 32,
        neural_layers: int = 3,
        neural_nhead: int = 8,
        neural_ff: int = 64,
        neural_dropout: float = 0.23,
        neural_head_dim: int = 128,
        # Tree-biased expert (slightly lighter)
        tree_d_token: int = 32,
        tree_layers: int = 2,
        tree_nhead: int = 4,
        tree_ff: int = 64,
        tree_dropout: float = 0.15,
        tree_head_dim: int = 64,
        tree_signal_dim: int = 4,
        # Router
        router_hidden: int = 64,
        router_dropout: float = 0.1,
        router_temperature: float = 1.0,
        # Behaviour
        soft_mixture: bool = True,
        extra_feature_dim: int = 0,
        share_final_heads: bool = False,
    ):
        super().__init__()
        self.soft_mixture = soft_mixture
        self.n_experts = 2
        self.tree_signal_dim = tree_signal_dim
        self.extra_feature_dim = extra_feature_dim

        self.neural_expert = NeuralExpert(
            num_features=num_features,
            d_token=neural_d_token,
            num_layers=neural_layers,
            nhead=neural_nhead,
            dim_feedforward=neural_ff,
            dropout=neural_dropout,
            head_hidden_dim=neural_head_dim,
        )
        self.tree_expert = TreeBiasedExpert(
            num_features=num_features,
            tree_signal_dim=tree_signal_dim,
            d_token=tree_d_token,
            num_layers=tree_layers,
            nhead=tree_nhead,
            dim_feedforward=tree_ff,
            dropout=tree_dropout,
            head_hidden_dim=tree_head_dim,
        )

        feat_dim = geometry_feature_dim(self.n_experts, extra_dim=extra_feature_dim)
        self.router = GeometryRouter(
            in_dim=feat_dim,
            n_experts=self.n_experts,
            hidden=router_hidden,
            dropout=router_dropout,
            temperature=router_temperature,
            learnable_temp=True,
        )

        # Optional shared final heads (rarely needed; kept for ablation)
        self.share_final_heads = share_final_heads
        if share_final_heads:
            d = neural_d_token
            self.final_head_0 = nn.Sequential(
                nn.Linear(d, neural_head_dim), nn.GELU(), nn.Linear(neural_head_dim, 1)
            )
            self.final_head_1 = nn.Sequential(
                nn.Linear(d, neural_head_dim), nn.GELU(), nn.Linear(neural_head_dim, 1)
            )

    def _expert_outputs(
        self,
        x: torch.Tensor,
        tree_signals: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        y0_n, y1_n, z_n = self.neural_expert(x)
        y0_t, y1_t, z_t = self.tree_expert(x, tree_signals)

        tau_n = (y1_n - y0_n).squeeze(-1)
        tau_t = (y1_t - y0_t).squeeze(-1)

        return {
            "y0_n": y0_n, "y1_n": y1_n, "z_n": z_n, "tau_n": tau_n,
            "y0_t": y0_t, "y1_t": y1_t, "z_t": z_t, "tau_t": tau_t,
        }

    def forward(
        self,
        x: torch.Tensor,
        propensity: torch.Tensor,
        tree_signals: Optional[torch.Tensor] = None,
        extra_features: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x              : [B, p]
        propensity     : [B] or [B, 1]  – estimated or true e(x)
        tree_signals   : [B, 4] or None – OOF (mu0, mu1, tau, e)
        extra_features : [B, D] optional additional router inputs
        hard           : if True, use argmax routing instead of soft mixture

        Returns dict with keys:
          y0, y1, tau, z, weights, logits, tau_n, tau_t, ...
        """
        out = self._expert_outputs(x, tree_signals)
        tau_stack = torch.stack([out["tau_n"], out["tau_t"]], dim=1)  # [B, 2]

        # Router inputs are diagnostics about the independently fitted experts,
        # not a path for routing-label gradients to rewrite those experts.
        # The experts still receive gradients from the factual mixture objective
        # if a caller explicitly enables Stage-2 expert fine-tuning.
        geom_feat = build_geometry_features_torch(
            tau_stack.detach(), propensity, extra=extra_features
        )
        weights, logits = self.router(geom_feat)  # [B, 2]

        if hard or (not self.soft_mixture):
            # Straight-through hard routing for inference / ablation
            idx = weights.argmax(dim=-1)
            hard_w = F.one_hot(idx, num_classes=self.n_experts).float()
            weights_used = hard_w
        else:
            weights_used = weights

        # Mixture of potential outcomes
        y0 = weights_used[:, 0:1] * out["y0_n"] + weights_used[:, 1:2] * out["y0_t"]
        y1 = weights_used[:, 0:1] * out["y1_n"] + weights_used[:, 1:2] * out["y1_t"]
        tau = (y1 - y0).squeeze(-1)

        # Representation for IPM: weighted average of expert representations
        # (project tree expert to same dim if needed – here both use d_token≈32)
        z = weights_used[:, 0:1] * out["z_n"] + weights_used[:, 1:2] * out["z_t"]

        return {
            "y0": y0,
            "y1": y1,
            "tau": tau,
            "z": z,
            "weights": weights,          # soft weights (always)
            "weights_used": weights_used,
            "logits": logits,
            "tau_n": out["tau_n"],
            "tau_t": out["tau_t"],
            "y0_n": out["y0_n"],
            "y1_n": out["y1_n"],
            "y0_t": out["y0_t"],
            "y1_t": out["y1_t"],
            "z_n": out["z_n"],
            "z_t": out["z_t"],
        }

    def predict_effect(
        self,
        x: torch.Tensor,
        propensity: torch.Tensor,
        tree_signals: Optional[torch.Tensor] = None,
        extra_features: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(
                x, propensity, tree_signals, extra_features, hard=hard
            )
        return out["tau"]

    def router_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Mean entropy of the routing distribution (encourage exploration)."""
        p = weights.clamp(1e-8, 1.0)
        return -(p * p.log()).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Convenience constructor matching locked hparams
# ---------------------------------------------------------------------------

def build_geo_adaptive_from_hparams(
    num_features: int,
    hparams: Optional[Dict[str, Any]] = None,
) -> GeometryAdaptiveCATE:
    """
    Build model with defaults aligned to configs/canonical_hparams.yaml
    (ripo_no_tree block) so capacity is comparable to the locked baseline.
    """
    h = hparams or {}
    return GeometryAdaptiveCATE(
        num_features=num_features,
        neural_d_token=h.get("d_token", 32),
        neural_layers=h.get("num_layers", 3),
        neural_nhead=h.get("nhead", 8),
        neural_ff=h.get("dim_feedforward", 64),
        neural_dropout=h.get("dropout", 0.23),
        neural_head_dim=h.get("head_hidden_dim", 128),
        tree_d_token=h.get("tree_d_token", 32),
        tree_layers=h.get("tree_layers", 2),
        tree_nhead=h.get("tree_nhead", 4),
        tree_ff=h.get("tree_ff", 64),
        tree_dropout=h.get("tree_dropout", 0.15),
        tree_head_dim=h.get("tree_head_dim", 64),
        tree_signal_dim=h.get("tree_signal_dim", 4),
        router_hidden=h.get("router_hidden", 64),
        router_dropout=h.get("router_dropout", 0.1),
        router_temperature=h.get("router_temperature", 1.0),
        soft_mixture=h.get("soft_mixture", True),
        extra_feature_dim=h.get("extra_feature_dim", 0),
    )
