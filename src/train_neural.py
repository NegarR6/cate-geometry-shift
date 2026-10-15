import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))
"""
Training Script for Neural Causal Models
Includes: Training Loop, Causal Loss Functions, IPM Regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import copy
import random

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


# =========================================================
# Loss Functions
# =========================================================

class CausalLoss(nn.Module):
    """
    Causal Loss for TARNet-style models.
    Combines outcome prediction MSE with optional representation-balancing
    (IPM-style) regularization.

    Supported ipm_type values
    -------------------------
    - 'mean_diff'  (default): ||E[rep|T=1] − E[rep|T=0]||₂
                   linear MMD / first-order moment matching.
                   Fast, stable, used for all currently locked tables.
    - 'mmd'        : biased RBF-kernel MMD.
    - 'sinkhorn'   : entropic Wasserstein-2 approximation via Sinkhorn
                   (pure PyTorch, no POT dependency).
    - 'sliced'     : Sliced Wasserstein-2 (Monte-Carlo projections).
                   Recommended practical Wasserstein proxy for batches.

    Legacy alias: 'wasserstein' → mapped to 'mean_diff' for backward
    compatibility with old scripts. Do NOT use the word Wasserstein in
    the paper when reporting mean_diff results.

    Shape contract (enforced):
      y0_pred, y1_pred : (B, 1) or (B,)
      y_true, w        : (B,)
      representation   : (B, D) or None
    """
    def __init__(
        self,
        ipm_weight: float = 0.1,
        ipm_type: str = "mean_diff",
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 20,
        sliced_projections: int = 50,
    ):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.ipm_weight = ipm_weight
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.sliced_projections = sliced_projections

        # Legacy alias: old code that still passes 'wasserstein' gets mean_diff
        if ipm_type in ("wasserstein", "wass", "mean", "linear_mmd"):
            ipm_type = "mean_diff"
        self.ipm_type = ipm_type

    @staticmethod
    def _flat(t: torch.Tensor) -> torch.Tensor:
        """Force tensor to shape (B,) without ever becoming a 0-dim scalar."""
        return t.reshape(-1)

    def forward(self, y0_pred, y1_pred, y_true, w, representation=None):
        y0 = self._flat(y0_pred)
        y1 = self._flat(y1_pred)
        y = self._flat(y_true)
        w_flat = self._flat(w).float()

        if not (y0.shape == y1.shape == y.shape == w_flat.shape):
            raise RuntimeError(
                f"CausalLoss shape mismatch after flatten: "
                f"y0={y0.shape}, y1={y1.shape}, y={y.shape}, w={w_flat.shape}"
            )

        y_pred = w_flat * y1 + (1.0 - w_flat) * y0
        outcome_loss = self.mse_loss(y_pred, y)

        ipm_loss = y_pred.new_tensor(0.0)
        if representation is not None and self.ipm_weight > 0:
            ipm_loss = self._compute_ipm(representation, w_flat)

        total_loss = outcome_loss + self.ipm_weight * ipm_loss
        return total_loss, outcome_loss, ipm_loss

    def _compute_ipm(self, representation: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w = w.reshape(-1)
        rep_t = representation[w == 1]
        rep_c = representation[w == 0]

        if rep_t.size(0) == 0 or rep_c.size(0) == 0:
            return representation.new_tensor(0.0)

        if self.ipm_type == "mean_diff":
            return torch.norm(rep_t.mean(0) - rep_c.mean(0), p=2)
        if self.ipm_type == "mmd":
            return self._compute_mmd(rep_t, rep_c)
        if self.ipm_type == "sinkhorn":
            return self._sinkhorn_wasserstein(rep_t, rep_c)
        if self.ipm_type == "sliced":
            return self._sliced_wasserstein(rep_t, rep_c)
        return representation.new_tensor(0.0)

    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        xx = torch.exp(-torch.cdist(x, x) ** 2 / (2 * sigma ** 2)).mean()
        yy = torch.exp(-torch.cdist(y, y) ** 2 / (2 * sigma ** 2)).mean()
        xy = torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma ** 2)).mean()
        return xx + yy - 2 * xy

    def _sinkhorn_wasserstein(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Entropic approx of W₂² via log-domain Sinkhorn (numerically stable).
        Cost = squared Euclidean, median-normalized so eps is scale-free.
        """
        n, m = x.size(0), y.size(0)
        C = torch.cdist(x, y, p=2).pow(2)  # [n, m]

        med = C.detach().median().clamp(min=1e-6)
        C_scaled = C / med
        eps = max(float(self.sinkhorn_eps), 1e-3)

        log_K = -C_scaled / eps
        log_a = x.new_full((n,), -torch.log(torch.tensor(float(n), device=x.device, dtype=x.dtype)))
        log_b = y.new_full((m,), -torch.log(torch.tensor(float(m), device=y.device, dtype=y.dtype)))
        log_u = x.new_zeros(n)
        log_v = y.new_zeros(m)

        for _ in range(self.sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)

        log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        P = torch.exp(log_P)
        return (P * C).sum()

    def _sliced_wasserstein(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Sliced Wasserstein-2 (Monte-Carlo). Differentiable, O(n log n) per proj.
        More stable than full Sinkhorn on small batches.
        """
        d = x.size(1)
        n_proj = self.sliced_projections
        # Random directions on the sphere
        theta = torch.randn(n_proj, d, device=x.device, dtype=x.dtype)
        theta = theta / (theta.norm(dim=1, keepdim=True) + 1e-8)

        # Project
        x_proj = x @ theta.t()  # [n, n_proj]
        y_proj = y @ theta.t()  # [m, n_proj]

        # Sort along sample dimension for each projection
        x_sorted, _ = torch.sort(x_proj, dim=0)
        y_sorted, _ = torch.sort(y_proj, dim=0)

        # Handle unequal sizes by interpolating the longer one (simple subsample)
        n, m = x_sorted.size(0), y_sorted.size(0)
        if n != m:
            # Quantile matching: interpolate both to a common grid of size min(n,m)
            k = min(n, m)
            idx_x = torch.linspace(0, n - 1, k, device=x.device).long()
            idx_y = torch.linspace(0, m - 1, k, device=y.device).long()
            x_sorted = x_sorted[idx_x]
            y_sorted = y_sorted[idx_y]

        # Mean squared difference of sorted projections ≈ SW₂²
        return (x_sorted - y_sorted).pow(2).mean()


# =========================================================
# Training Loop
# =========================================================

class NeuralTrainer:
    """
    Trainer for neural causal models.
    """
    def __init__(self, model, device='cpu', lr=0.001, weight_decay=1e-4,
                 ipm_weight=0.1, grad_clip=1.0):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.ipm_weight = ipm_weight
        self.grad_clip = grad_clip
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.causal_loss = CausalLoss(ipm_weight=ipm_weight)
    
    def train_epoch(self, dataloader):
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        outcome_loss_sum = 0.0
        ipm_loss_sum = 0.0
        
        for batch in dataloader:
            x_cont = batch[0].to(self.device)
            w = batch[1].to(self.device)
            y = batch[2].to(self.device)
            
            self.optimizer.zero_grad()
            
            # One shared stochastic forward pass keeps heads and IPM on the same representation.
            y0_pred, y1_pred, representation = self.model(
                x_cont, w=None, return_representation=True
            )
            
            # Compute loss
            loss, outcome_loss, ipm_loss = self.causal_loss(
                y0_pred, y1_pred, y, w, representation
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            outcome_loss_sum += outcome_loss.item()
            ipm_loss_sum += ipm_loss.item()
        
        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'outcome_loss': outcome_loss_sum / n_batches,
            'ipm_loss': ipm_loss_sum / n_batches
        }
    
    def validate(self, dataloader):
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        all_y0_pred = []
        all_y1_pred = []
        all_y_true = []
        all_w = []
        
        with torch.no_grad():
            for batch in dataloader:
                x_cont = batch[0].to(self.device)
                w = batch[1].to(self.device)
                y = batch[2].to(self.device)
                
                # Get both heads (don't pass w)
                y0_pred, y1_pred = self.model(x_cont, w=None)
                
                loss, outcome_loss, _ = self.causal_loss(y0_pred, y1_pred, y, w)
                
                total_loss += loss.item()
                all_y0_pred.append(y0_pred.cpu())
                all_y1_pred.append(y1_pred.cpu())
                all_y_true.append(y.cpu())
                all_w.append(w.cpu())
        
        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'y0_pred': torch.cat(all_y0_pred),
            'y1_pred': torch.cat(all_y1_pred),
            'y_true': torch.cat(all_y_true),
            'w': torch.cat(all_w)
        }
    
    def train(self, train_loader, val_loader, epochs=100, patience=20, 
              log_to_wandb=False, run_name=None):
        """
        Full training loop with early stopping.
        """
        if log_to_wandb:
            if not WANDB_AVAILABLE:
                raise ImportError('wandb is not installed but log_to_wandb=True was requested')
            wandb.init(project="cate-geometry-shift", name=run_name)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': []}
        
        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Logging
            if log_to_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'outcome_loss': train_metrics['outcome_loss'],
                    'ipm_loss': train_metrics['ipm_loss'],
                    'epoch': epoch
                })
            
            pbar.set_postfix({
                'train_loss': f"{train_metrics['loss']:.4f}",
                'val_loss': f"{val_metrics['loss']:.4f}"
            })
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        if log_to_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        return history


# =========================================================
# Utility Functions
# =========================================================

def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python, NumPy and PyTorch RNGs for auditable repetitions.

    Default deterministic=False: seeds RNGs + cudnn.deterministic, without
    torch.use_deterministic_algorithms (avoids CuBLAS warnings/crashes on
    CUDA>=10.2 during attention training).

    For strict tests, call set_global_seed(seed, deterministic=True) and set:
      CUBLAS_WORKSPACE_CONFIG=:4096:8
    """
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def create_dataloaders(X, w, y, batch_size=64, test_size=0.2, random_state=42):
    """
    Create PyTorch DataLoaders from numpy arrays.

    Guarantees:
      - X  : FloatTensor (N, F)
      - w  : FloatTensor (N,)   (kept float so CausalLoss indexing is uniform)
      - y  : FloatTensor (N,)
    drop_last=False so the last incomplete batch is kept; CausalLoss is
    now shape-safe for B=1.
    """
    from sklearn.model_selection import train_test_split

    X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
    w_tensor = torch.as_tensor(np.asarray(w, dtype=np.float32).ravel())
    y_tensor = torch.as_tensor(np.asarray(y, dtype=np.float32).ravel())

    X_train, X_val, w_train, w_val, y_train, y_val = train_test_split(
        X_tensor, w_tensor, y_tensor,
        test_size=test_size,
        random_state=random_state,
        stratify=w_tensor.numpy().astype(int) if len(np.unique(w_tensor.numpy())) > 1 else None,
    )

    train_dataset = TensorDataset(X_train, w_train, y_train)
    val_dataset = TensorDataset(X_val, w_val, y_val)

    generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def evaluate_model(model, X_test, w_test, y_test, po0_test, po1_test, device='cpu'):
    """
    Evaluate model on test set using canonical evaluation metrics.

    Returns dict compatible with Phase 0:
      PEHE, ATE_Error, Correlation, Policy_Value, Regret, CATE_ECE,
      plus tau_pred / tau_true and backward-compatible aliases.
    """
    from evaluation import evaluate_cate

    model.eval()
    model = model.to(device)

    X_tensor = torch.FloatTensor(np.asarray(X_test, dtype=np.float32)).to(device)

    with torch.no_grad():
        tau_pred_tensor = model.predict_effect(X_tensor)
        tau_pred = tau_pred_tensor.detach().cpu().numpy().ravel()

    po0 = np.asarray(po0_test, dtype=np.float64).ravel()
    po1 = np.asarray(po1_test, dtype=np.float64).ravel()
    tau_true = po1 - po0

    metrics = evaluate_cate(po0, po1, tau_pred)
    metrics["tau_pred"] = tau_pred
    metrics["tau_true"] = tau_true
    metrics["pehe"] = metrics["PEHE"]
    metrics["ate_error"] = metrics["ATE_Error"]
    metrics["ate_pred"] = float(tau_pred.mean())
    metrics["ate_true"] = float(tau_true.mean())
    return metrics


if __name__ == "__main__":
    # Test the training pipeline
    print("Testing Neural Training Pipeline...")
    
    from canonical_data_loader import load_ihdp_canonical
    from models_neural import TARNet_FTTransformer
    
    # Load data (canonical)
    data = load_ihdp_canonical(replication=1, scale_features=True)
    X_train = data['train']['X']
    w_train = data['train']['w']
    y_train = data['train']['y']
    
    # Create model
    model = TARNet_FTTransformer(
        num_continuous=25,
        d_token=32,
        num_layers=3,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1
    )
    
    # =========================================================
    # SANITY CHECK: Test loss function on one batch BEFORE training
    # =========================================================
    print("\n[Sanity Check] Testing CausalLoss on one batch...")
    dummy_x = torch.FloatTensor(X_train[:8])
    dummy_w = torch.LongTensor(w_train[:8].astype(int))
    dummy_y = torch.FloatTensor(y_train[:8])
    
    test_model = TARNet_FTTransformer(num_continuous=25, d_token=32, num_layers=2, nhead=4)
    y0_p, y1_p = test_model(dummy_x, w=None)
    with torch.no_grad():
        tok = test_model.tokenizer(dummy_x)
        rep = test_model.encoder(tok).mean(dim=1)
    
    test_loss_fn = CausalLoss(ipm_weight=0.1)
    try:
        total, out_l, ipm_l = test_loss_fn(y0_p, y1_p, dummy_y, dummy_w, rep)
        print(f"  ✅ Loss works! total={total.item():.4f}, outcome={out_l.item():.4f}, ipm={ipm_l.item():.4f}")
    except Exception as e:
        print(f"  ❌ Loss failed: {e}")
        raise
    print("  Shapes check:")
    print(f"    y0_pred: {y0_p.shape}, y1_pred: {y1_p.shape}")
    print(f"    w (flat expected): {dummy_w.shape}, representation: {rep.shape}")
    print("  ✅ Shape alignment verified.")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, w_train, y_train)
    
    # Create trainer
    trainer = NeuralTrainer(model, lr=0.001, ipm_weight=0.1)
    
    # Train (just 2 epochs for testing)
    history = trainer.train(train_loader, val_loader, epochs=2, patience=20)
    
    print(f"✅ Training test passed! Final val_loss: {history['val_loss'][-1]:.4f}")
