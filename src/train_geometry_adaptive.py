"""
Training utilities for GeometryAdaptiveCATE — PRODUCTION VERSION

Two-stage protocol (critical for strong NeuralExpert performance)
-----------------------------------------------------------------
Stage 1 – Expert Pre-training (independent)
  - NeuralExpert and TreeBiasedExpert are trained separately with
    factual MSE + mean-diff IPM (exactly the locked CausalLoss).
  - No routing loss. Each expert sees the full gradient signal.
  - Early-stopping on factual validation MSE.

Stage 2 – Joint Adaptive Training
  - Experts are loaded from Stage 1 (optionally frozen for the first
    N epochs, then fine-tuned with a smaller LR).
  - Router is trained from scratch.
  - Oracle labels are built from the *pre-trained* experts’ predictions
    on the training set (point-wise argmin |τ̂ − τ| when τ is known,
    or a strong surrogate otherwise).
  - Curriculum on route_weight:
        epochs 1 .. warm:   route_weight = route_weight_start (small)
        then linear ramp to route_weight_end.
  - Entropy regularisation keeps the router from collapsing.

This design directly fixes the failure mode observed in the first
3-seed run: NeuralExpert was under-trained under joint optimisation
and collapsed on regime C.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from train_neural import CausalLoss
except ImportError:
    from src.train_neural import CausalLoss

try:
    from models_geometry_adaptive import (
        GeometryAdaptiveCATE,
        NeuralExpert,
        TreeBiasedExpert,
        build_geo_adaptive_from_hparams,
    )
except ImportError:
    from src.models_geometry_adaptive import (
        GeometryAdaptiveCATE,
        NeuralExpert,
        TreeBiasedExpert,
        build_geo_adaptive_from_hparams,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class GeometryAdaptiveLoss(nn.Module):
    def __init__(
        self,
        ipm_weight: float = 0.1,
        ipm_type: str = "mean_diff",
        route_weight: float = 0.5,
        entropy_weight: float = 0.01,
        route_label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.causal = CausalLoss(ipm_weight=ipm_weight, ipm_type=ipm_type)
        self.route_weight = route_weight
        self.entropy_weight = entropy_weight
        self.label_smoothing = route_label_smoothing

    def set_route_weight(self, w: float) -> None:
        self.route_weight = float(w)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        w: torch.Tensor,
        oracle_expert: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        y0, y1, z, weights = out["y0"], out["y1"], out["z"], out["weights"]

        total_causal, outcome_loss, ipm_loss = self.causal(y0, y1, y_true, w, z)

        route_loss = y0.new_tensor(0.0)
        if (
            oracle_expert is not None
            and self.route_weight > 0
            and (oracle_expert >= 0).any()
        ):
            # ignore sentinel -1
            mask = oracle_expert >= 0
            if mask.any():
                logits = out["logits"][mask]
                target = oracle_expert[mask]
                log_probs = F.log_softmax(logits, dim=-1)
                n_exp = logits.size(1)
                if self.label_smoothing > 0 and n_exp > 1:
                    with torch.no_grad():
                        soft = torch.full_like(log_probs, self.label_smoothing / (n_exp - 1))
                        soft.scatter_(1, target.view(-1, 1), 1.0 - self.label_smoothing)
                    route_loss = -(soft * log_probs).sum(dim=-1).mean()
                else:
                    route_loss = F.nll_loss(log_probs, target)

        entropy = -(weights.clamp(1e-8, 1.0) * weights.clamp(1e-8, 1.0).log()).sum(-1).mean()
        ent_term = -self.entropy_weight * entropy

        total = total_causal + self.route_weight * route_loss + ent_term
        stats = {
            "total": float(total.detach()),
            "outcome": float(outcome_loss.detach()),
            "ipm": float(ipm_loss.detach()),
            "route": float(route_loss.detach()),
            "entropy": float(entropy.detach()),
        }
        return total, stats


# ---------------------------------------------------------------------------
# Single-expert trainer (Stage 1)
# ---------------------------------------------------------------------------

class ExpertTrainer:
    """Train one expert (Neural or Tree) with factual + IPM loss only."""

    def __init__(
        self,
        expert: nn.Module,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        ipm_weight: float = 0.19,
        ipm_type: str = "mean_diff",
        grad_clip: float = 1.0,
        epochs: int = 40,
        patience: int = 12,
        is_tree_expert: bool = False,
    ):
        self.expert = expert.to(device)
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.is_tree_expert = is_tree_expert

        self.optimizer = optim.AdamW(
            expert.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=max(4, patience // 3)
        )
        self.loss_fn = CausalLoss(ipm_weight=ipm_weight, ipm_type=ipm_type)
        self.best_state: Optional[Dict] = None
        self.best_val = float("inf")

    def _step(self, batch, train: bool = True):
        x = batch[0].to(self.device).float()
        w = batch[1].to(self.device).float()
        y = batch[2].to(self.device).float()
        tree_sig = batch[4]
        if tree_sig is not None and tree_sig.numel() > 0 and tree_sig.size(-1) > 0:
            tree_sig = tree_sig.to(self.device).float()
        else:
            tree_sig = None

        if self.is_tree_expert:
            y0, y1, z = self.expert(x, tree_sig)
        else:
            y0, y1, z = self.expert(x)

        loss, outcome, ipm = self.loss_fn(y0, y1, y, w, z)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.expert.parameters(), self.grad_clip)
            self.optimizer.step()
        return float(outcome.detach()), float(ipm.detach())

    def fit(self, train_loader, val_loader, verbose: bool = False, name: str = "expert"):
        patience_left = self.patience
        for epoch in range(1, self.epochs + 1):
            self.expert.train()
            tr_out, n = 0.0, 0
            for batch in train_loader:
                o, _ = self._step(batch, train=True)
                tr_out += o * batch[0].size(0)
                n += batch[0].size(0)
            tr_out /= max(n, 1)

            self.expert.eval()
            va_out, n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    o, _ = self._step(batch, train=False)
                    va_out += o * batch[0].size(0)
                    n += batch[0].size(0)
            va_out /= max(n, 1)
            self.scheduler.step(va_out)

            if va_out < self.best_val - 1e-5:
                self.best_val = va_out
                self.best_state = copy.deepcopy(self.expert.state_dict())
                patience_left = self.patience
            else:
                patience_left -= 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  [{name}] ep={epoch:03d} tr={tr_out:.4f} va={va_out:.4f} best={self.best_val:.4f}")

            if patience_left <= 0:
                if verbose:
                    print(f"  [{name}] early stop at {epoch}")
                break

        if self.best_state is not None:
            self.expert.load_state_dict(self.best_state)
        return self


# ---------------------------------------------------------------------------
# Full two-stage trainer
# ---------------------------------------------------------------------------

class GeometryAdaptiveTrainer:
    """
    Stage-1 pre-train + Stage-2 joint adaptive training with curriculum.
    """

    def __init__(
        self,
        model: GeometryAdaptiveCATE,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        ipm_weight: float = 0.19,
        ipm_type: str = "mean_diff",
        route_weight_start: float = 0.15,
        route_weight_end: float = 0.6,
        route_warmup_epochs: int = 15,
        entropy_weight: float = 0.015,
        grad_clip: float = 1.0,
        # Stage 1
        pretrain_epochs: int = 35,
        pretrain_patience: int = 12,
        # Stage 2
        joint_epochs: int = 50,
        joint_patience: int = 15,
        freeze_experts_epochs: int = 8,
        expert_ft_lr_mult: float = 0.25,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.ipm_weight = ipm_weight
        self.ipm_type = ipm_type
        self.route_weight_start = route_weight_start
        self.route_weight_end = route_weight_end
        self.route_warmup_epochs = route_warmup_epochs
        self.entropy_weight = entropy_weight
        self.grad_clip = grad_clip
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_patience = pretrain_patience
        self.joint_epochs = joint_epochs
        self.joint_patience = joint_patience
        self.freeze_experts_epochs = freeze_experts_epochs
        self.expert_ft_lr_mult = expert_ft_lr_mult

        self.loss_fn = GeometryAdaptiveLoss(
            ipm_weight=ipm_weight,
            ipm_type=ipm_type,
            route_weight=route_weight_start,
            entropy_weight=entropy_weight,
        )
        self.best_state: Optional[Dict] = None
        self.best_val = float("inf")

    # ----- helpers -----

    def _prepare_batch(self, batch: Tuple) -> Dict[str, Any]:
        x = batch[0].to(self.device).float()
        w = batch[1].to(self.device).float()
        y = batch[2].to(self.device).float()
        e = batch[3].to(self.device).float()
        tree_sig = batch[4]
        if tree_sig is not None and tree_sig.numel() > 0 and tree_sig.size(-1) > 0:
            tree_sig = tree_sig.to(self.device).float()
        else:
            tree_sig = None
        oracle = batch[5]
        if oracle is not None:
            oracle = oracle.to(self.device).long()
        extra = batch[6] if len(batch) > 6 else None
        if extra is not None and extra.numel() > 0 and extra.size(-1) > 0:
            extra = extra.to(self.device).float()
        else:
            extra = None
        return {"x": x, "w": w, "y": y, "e": e, "tree_sig": tree_sig, "oracle": oracle, "extra": extra}

    def _route_weight_at(self, epoch: int) -> float:
        if epoch <= self.route_warmup_epochs:
            return self.route_weight_start
        t = (epoch - self.route_warmup_epochs) / max(1, self.joint_epochs - self.route_warmup_epochs)
        t = min(1.0, max(0.0, t))
        return self.route_weight_start + t * (self.route_weight_end - self.route_weight_start)

    # ----- Stage 1 -----

    def pretrain_experts(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> None:
        if verbose:
            print("=== Stage 1: Pre-training experts independently ===")

        # Neural expert
        neural_trainer = ExpertTrainer(
            self.model.neural_expert,
            device=self.device,
            lr=self.lr,
            weight_decay=self.weight_decay,
            ipm_weight=self.ipm_weight,
            ipm_type=self.ipm_type,
            epochs=self.pretrain_epochs,
            patience=self.pretrain_patience,
            is_tree_expert=False,
        )
        neural_trainer.fit(train_loader, val_loader, verbose=verbose, name="NeuralExpert")

        # Tree expert
        tree_trainer = ExpertTrainer(
            self.model.tree_expert,
            device=self.device,
            lr=self.lr,
            weight_decay=self.weight_decay,
            ipm_weight=self.ipm_weight,
            ipm_type=self.ipm_type,
            epochs=self.pretrain_epochs,
            patience=self.pretrain_patience,
            is_tree_expert=True,
        )
        tree_trainer.fit(train_loader, val_loader, verbose=verbose, name="TreeExpert")

        if verbose:
            print("=== Stage 1 done ===\n")

    # ----- Oracle from pre-trained experts -----

    @torch.no_grad()
    def build_oracle_from_experts(
        self,
        X: np.ndarray,
        e: np.ndarray,
        tree_signals: Optional[np.ndarray],
        tau_true: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Point-wise oracle: which pre-trained expert is closer to true τ."""
        self.model.eval()
        n = len(X)
        tau_n = np.zeros(n, dtype=np.float64)
        tau_t = np.zeros(n, dtype=np.float64)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xt = torch.as_tensor(X[start:end], dtype=torch.float32, device=self.device)
            et = torch.as_tensor(e[start:end], dtype=torch.float32, device=self.device)
            tst = (
                torch.as_tensor(tree_signals[start:end], dtype=torch.float32, device=self.device)
                if tree_signals is not None
                else None
            )
            out = self.model(xt, et, tree_signals=tst)
            tau_n[start:end] = out["tau_n"].cpu().numpy()
            tau_t[start:end] = out["tau_t"].cpu().numpy()

        err_n = np.abs(tau_n - tau_true)
        err_t = np.abs(tau_t - tau_true)
        return (err_t < err_n).astype(np.int64)

    # ----- Stage 2 -----

    def _build_optimizer(self, freeze_experts: bool) -> optim.Optimizer:
        if freeze_experts:
            for p in self.model.neural_expert.parameters():
                p.requires_grad = False
            for p in self.model.tree_expert.parameters():
                p.requires_grad = False
            params = list(self.model.router.parameters())
        else:
            for p in self.model.parameters():
                p.requires_grad = True
            params = [
                {"params": self.model.router.parameters(), "lr": self.lr},
                {
                    "params": list(self.model.neural_expert.parameters())
                    + list(self.model.tree_expert.parameters()),
                    "lr": self.lr * self.expert_ft_lr_mult,
                },
            ]
        return optim.AdamW(params, weight_decay=self.weight_decay)

    def joint_train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> None:
        if verbose:
            print("=== Stage 2: Joint adaptive training (curriculum) ===")

        patience_left = self.joint_patience
        self.best_val = float("inf")
        self.best_state = None

        for epoch in range(1, self.joint_epochs + 1):
            freeze = epoch <= self.freeze_experts_epochs
            # rebuild optimizer when freeze status changes
            if epoch == 1 or epoch == self.freeze_experts_epochs + 1:
                self.optimizer = self._build_optimizer(freeze_experts=freeze)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.5, patience=5
                )

            rw = self._route_weight_at(epoch)
            self.loss_fn.set_route_weight(rw)

            # train
            self.model.train()
            # keep experts in eval if frozen (BatchNorm / Dropout behaviour)
            if freeze:
                self.model.neural_expert.eval()
                self.model.tree_expert.eval()

            agg = {"total": 0.0, "outcome": 0.0, "route": 0.0, "entropy": 0.0}
            n = 0
            for batch in train_loader:
                data = self._prepare_batch(batch)
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(
                    data["x"], data["e"],
                    tree_signals=data["tree_sig"],
                    extra_features=data["extra"],
                )
                loss, stats = self.loss_fn(out, data["y"], data["w"], data["oracle"])
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip,
                    )
                self.optimizer.step()
                bs = data["x"].size(0)
                for k in agg:
                    agg[k] += stats.get(k, 0.0) * bs
                n += bs
            tr = {k: v / max(n, 1) for k, v in agg.items()}

            # val
            self.model.eval()
            agg = {"total": 0.0, "outcome": 0.0, "route": 0.0, "entropy": 0.0}
            n = 0
            with torch.no_grad():
                for batch in val_loader:
                    data = self._prepare_batch(batch)
                    out = self.model(
                        data["x"], data["e"],
                        tree_signals=data["tree_sig"],
                        extra_features=data["extra"],
                    )
                    _, stats = self.loss_fn(out, data["y"], data["w"], data["oracle"])
                    bs = data["x"].size(0)
                    for k in agg:
                        agg[k] += stats.get(k, 0.0) * bs
                    n += bs
            va = {k: v / max(n, 1) for k, v in agg.items()}
            self.scheduler.step(va["outcome"])

            if va["outcome"] < self.best_val - 1e-5:
                self.best_val = va["outcome"]
                self.best_state = copy.deepcopy(self.model.state_dict())
                patience_left = self.joint_patience
            else:
                patience_left -= 1

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(
                    f"  [Joint] ep={epoch:03d} freeze={freeze} rw={rw:.2f} "
                    f"tr_out={tr['outcome']:.4f} tr_route={tr['route']:.4f} "
                    f"tr_ent={tr['entropy']:.3f} | va_out={va['outcome']:.4f} "
                    f"best={self.best_val:.4f} pat={patience_left}"
                )

            if patience_left <= 0:
                if verbose:
                    print(f"  [Joint] early stop at {epoch}")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if verbose:
            print("=== Stage 2 done ===\n")

    # ----- Public API -----

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        X_for_oracle: Optional[np.ndarray] = None,
        e_for_oracle: Optional[np.ndarray] = None,
        tree_for_oracle: Optional[np.ndarray] = None,
        tau_true_for_oracle: Optional[np.ndarray] = None,
        oracle_already_set: bool = False,
        verbose: bool = True,
    ) -> "GeometryAdaptiveTrainer":
        """
        Full two-stage pipeline.

        If tau_true_for_oracle is provided, oracle labels are rebuilt
        *after* Stage 1 from the pre-trained experts (recommended).
        The caller must then rebuild the DataLoaders with the new oracle
        before Stage 2, OR pass the arrays and let this method handle it
        via a callback-style rebuild (see run script).
        """
        # Stage 1
        self.pretrain_experts(train_loader, val_loader, verbose=verbose)

        # Note: oracle rebuild + loader rebuild is handled by the runner
        # because DataLoader construction lives outside.  The runner calls
        # build_oracle_from_experts and recreates loaders before joint_train.

        return self

    def fit_joint_only(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> "GeometryAdaptiveTrainer":
        """Call after pretrain + oracle rebuild."""
        self.joint_train(train_loader, val_loader, verbose=verbose)
        return self

    @torch.no_grad()
    def predict_cate(
        self,
        X: np.ndarray,
        e: np.ndarray,
        tree_signals: Optional[np.ndarray] = None,
        extra: Optional[np.ndarray] = None,
        hard: bool = False,
        batch_size: int = 256,
    ) -> np.ndarray:
        self.model.eval()
        preds = []
        n = len(X)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xt = torch.as_tensor(X[start:end], dtype=torch.float32, device=self.device)
            et = torch.as_tensor(e[start:end], dtype=torch.float32, device=self.device)
            tst = (
                torch.as_tensor(tree_signals[start:end], dtype=torch.float32, device=self.device)
                if tree_signals is not None
                else None
            )
            ext = (
                torch.as_tensor(extra[start:end], dtype=torch.float32, device=self.device)
                if extra is not None
                else None
            )
            out = self.model(xt, et, tree_signals=tst, extra_features=ext, hard=hard)
            preds.append(out["tau"].cpu().numpy())
        return np.concatenate(preds, axis=0)

    @torch.no_grad()
    def predict_experts(
        self,
        X: np.ndarray,
        e: np.ndarray,
        tree_signals: Optional[np.ndarray] = None,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (tau_adaptive, tau_neural, tau_tree) and mean weights."""
        self.model.eval()
        tau_a, tau_n, tau_t, ws = [], [], [], []
        n = len(X)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xt = torch.as_tensor(X[start:end], dtype=torch.float32, device=self.device)
            et = torch.as_tensor(e[start:end], dtype=torch.float32, device=self.device)
            tst = (
                torch.as_tensor(tree_signals[start:end], dtype=torch.float32, device=self.device)
                if tree_signals is not None
                else None
            )
            out = self.model(xt, et, tree_signals=tst)
            tau_a.append(out["tau"].cpu().numpy())
            tau_n.append(out["tau_n"].cpu().numpy())
            tau_t.append(out["tau_t"].cpu().numpy())
            ws.append(out["weights"].cpu().numpy())
        return (
            np.concatenate(tau_a),
            np.concatenate(tau_n),
            np.concatenate(tau_t),
            np.concatenate(ws),
        )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_oracle_expert_labels(
    tau_true: np.ndarray,
    tau_neural: np.ndarray,
    tau_tree: np.ndarray,
) -> np.ndarray:
    err_n = np.abs(tau_neural - tau_true)
    err_t = np.abs(tau_tree - tau_true)
    return (err_t < err_n).astype(np.int64)


def build_loaders(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    e: np.ndarray,
    tree_signals: Optional[np.ndarray],
    oracle: Optional[np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 64,
    extra: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader]:
    def _subset(idx):
        n = len(idx)
        ts = (
            torch.as_tensor(tree_signals[idx], dtype=torch.float32)
            if tree_signals is not None
            else torch.zeros(n, 0)
        )
        oc = (
            torch.as_tensor(oracle[idx], dtype=torch.long)
            if oracle is not None
            else torch.full((n,), -1, dtype=torch.long)
        )
        ex = (
            torch.as_tensor(extra[idx], dtype=torch.float32)
            if extra is not None
            else torch.zeros(n, 0)
        )
        return TensorDataset(
            torch.as_tensor(X[idx], dtype=torch.float32),
            torch.as_tensor(T[idx], dtype=torch.float32),
            torch.as_tensor(Y[idx], dtype=torch.float32),
            torch.as_tensor(e[idx], dtype=torch.float32),
            ts,
            oc,
            ex,
        )

    train_loader = DataLoader(
        _subset(train_idx), batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        _subset(val_idx), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader
