import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))
"""
Training utilities for RIPO v2 variants.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from train_neural import set_global_seed


def hybrid_v2_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    attention_weights: torch.Tensor,
    ipm_loss: torch.Tensor = None,
    distill_loss: torch.Tensor = None,
    sparsity_weight: float = 1e-3,  # کاهش ضریب از 1e-3 به 1e-4
    ipm_weight: float = 0.1,
    distill_weight: float = 0.1,
    use_bounded_entropy: bool = True,
    target_entropy: float = 2.5
) -> torch.Tensor:
    
    # Shape-safe MSE (both sides forced to 1-D)
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    main_loss = F.mse_loss(y_pred, y_true, reduction="mean")
    total_loss = main_loss
    
    # ۲. افزودن زیان IPM
    if ipm_loss is not None:
        total_loss += ipm_weight * ipm_loss
        
    # ۳. افزودن زیان تقطیر (Distillation)
    if distill_loss is not None:
        total_loss += distill_weight * distill_loss
        
    # ۴. تنظیم‌کننده آنتروپی توجه
    if sparsity_weight > 0.0 and attention_weights is not None:
        eps = 1e-8
        probs = torch.clamp(attention_weights, min=eps, max=1.0)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1).mean()
        
        if use_bounded_entropy:
            # تنها زمانی جریمه اعمال می‌شود که آنتروپی از حد آستانه پایین‌تر بیاید
            sparsity_penalty = F.relu(target_entropy - entropy)
        else:
            sparsity_penalty = -entropy
            
        total_loss += sparsity_weight * sparsity_penalty
        
    return total_loss

class HybridV2Trainer:
    def __init__(self, model, variant, lr=3e-4, weight_decay=1e-4, alpha=0.1, distill_weight=0.5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.variant = variant
        self.alpha = alpha
        self.distill_weight = distill_weight
        # تغییر به:
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    @staticmethod
    def loader(X, tree, w, y, batch_size, shuffle, seed):
        ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(tree, dtype=torch.float32), torch.as_tensor(w, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)
    
    def _epoch(self, loader, training, tree_po0=None, tree_po1=None):
        self.model.train(training)
        total_loss = 0.0
        context = torch.enable_grad() if training else torch.no_grad()
        
        with context:
            for X, tree, w, y in loader:
                X, tree, w, y = [v.to(self.device) for v in (X, tree, w, y)]
                
                attn_weights = None
                distill_loss = None
                
                # دریافت دینامیک خروجی‌ها بر اساس معماری
                if self.variant == 'distill' and training:
                    y0, y1, z, extra_out = self.model(
                        X, tree_po0=tree[:, 0], tree_po1=tree[:, 1], training=True
                    )
                    distill_loss = extra_out
                elif self.variant == 'attention':
                    # استخراج درست وزن‌های توجه از خروجی مدل
                    y0, y1, z, extra_info = self.model(X, tree)
                    if extra_info is not None:
                        attn_weights, _, _ = extra_info
                else:
                    y0, y1, z, extra_out = self.model(X, tree)
                
# Shape-safe factual prediction (B=1 safe: never bare .squeeze())
                y0f = y0.reshape(-1)
                y1f = y1.reshape(-1)
                wf  = w.reshape(-1).float()
                yf  = y.reshape(-1)
                y_pred = wf * y1f + (1.0 - wf) * y0f

                loss = hybrid_v2_loss(
                    y_true=yf,
                    y_pred=y_pred,
                    attention_weights=attn_weights,
                    distill_loss=distill_loss,
                    distill_weight=self.distill_weight,
                )
                if loss.ndim != 0:
                    raise RuntimeError(f"Expected scalar loss, got shape {tuple(loss.shape)}")
                
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                total_loss += loss.item() * len(X)
        
        return total_loss / len(loader.dataset)
    
    def fit(self, train_loader, val_loader, epochs=150, patience=20, 
            tree_po0_val=None, tree_po1_val=None):
        """Train with early stopping."""
        best, best_state, stale = float("inf"), None, 0
        
        for epoch in range(epochs):
            train_loss = self._epoch(train_loader, True)
            val_loss = self._epoch(val_loader, False)
            
            if val_loss < best - 1e-6:
                best, stale = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return best
    
    def predict(self, X, tree):
        """Predict outcomes and treatment effect."""
        self.model.eval()
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            tree = torch.as_tensor(tree, dtype=torch.float32, device=self.device)
            
            if self.variant == 'distill':
                # For distill, tree signals are not needed at inference
                y0, y1, _, _ = self.model(X, training=False)
            else:
                y0, y1, _, _ = self.model(X, tree)
        
        return y0.cpu().numpy().ravel(), y1.cpu().numpy().ravel()
    def evaluate(self, test_loader, po0_test, po1_test):
        """Evaluate the model on a test loader and return PEHE and ATE error."""
        all_y0, all_y1 = [], []
        for X, tree, w, y in test_loader:
            y0_b, y1_b = self.predict(X, tree)
            all_y0.append(y0_b)
            all_y1.append(y1_b)
        y0_pred = np.concatenate(all_y0)
        y1_pred = np.concatenate(all_y1)
        return pehe_ate(po0_test, po1_test, y0_pred, y1_pred)

def pehe_ate(po0, po1, y0, y1):
    """Calculate PEHE and ATE Error."""
    true_tau, pred_tau = np.asarray(po1) - np.asarray(po0), y1 - y0
    return {
        "pehe": float(np.sqrt(np.mean((true_tau - pred_tau) ** 2))),
        "ate_error": float(abs(true_tau.mean() - pred_tau.mean())),
        "tau_true": true_tau,
        "tau_pred": pred_tau,
    }