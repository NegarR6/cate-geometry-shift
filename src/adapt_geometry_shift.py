"""
Geometry-Shift Adaptation — STRONG VERSION

Two modes of capacity:
  router_only   : freeze experts, fine-tune router (previous; small gains)
  full_adapt    : fine-tune router + both experts with lower LR on k samples
                  using factual MSE + (optional) oracle routing + direct τ loss

The full_adapt mode is what can break the Tree-only ceiling under shift,
because the NeuralExpert can re-specialise to the target geometry from
the k samples.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from typing import Any, Dict, Optional, Tuple

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from models_geometry_adaptive import GeometryAdaptiveCATE
except ImportError:
    from src.models_geometry_adaptive import GeometryAdaptiveCATE

try:
    from train_neural import CausalLoss
except ImportError:
    try:
        from src.train_neural import CausalLoss
    except ImportError:
        from train_geometry_adaptive import CausalLoss  # type: ignore


def _pehe(tau_true: np.ndarray, tau_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau_true - tau_pred) ** 2)))


@torch.no_grad()
def _expert_preds(
    model: GeometryAdaptiveCATE,
    X: np.ndarray,
    e: np.ndarray,
    tree_sig: Optional[np.ndarray],
    device: str,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    tau_n, tau_t = [], []
    n = len(X)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xt = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
        et = torch.as_tensor(e[start:end], dtype=torch.float32, device=device)
        tst = (
            torch.as_tensor(tree_sig[start:end], dtype=torch.float32, device=device)
            if tree_sig is not None
            else None
        )
        out = model(xt, et, tree_signals=tst)
        tau_n.append(out["tau_n"].cpu().numpy())
        tau_t.append(out["tau_t"].cpu().numpy())
    return np.concatenate(tau_n), np.concatenate(tau_t)


def build_fewshot_oracle(
    model: GeometryAdaptiveCATE,
    X_k: np.ndarray,
    e_k: np.ndarray,
    tree_k: Optional[np.ndarray],
    tau_k: np.ndarray,
    device: str,
) -> np.ndarray:
    tau_n, tau_t = _expert_preds(model, X_k, e_k, tree_k, device)
    err_n = np.abs(tau_n - tau_k)
    err_t = np.abs(tau_t - tau_k)
    return (err_t < err_n).astype(np.int64)


@torch.no_grad()
def compute_dr_pseudo_outcomes(
    model: GeometryAdaptiveCATE,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    e: np.ndarray,
    tree_sig: Optional[np.ndarray],
    device: str,
    batch_size: int = 256,
    e_clip: float = 0.05,
) -> np.ndarray:
    """
    Doubly-robust CATE pseudo-outcome using the *current* model as nuisance:

      φ = (μ1−μ0) + T(Y−μ1)/ê − (1−T)(Y−μ0)/(1−ê)

    No ground-truth τ is used. Suitable as a τ-free supervision target on k samples.
    """
    model.eval()
    n = len(X)
    phi = np.zeros(n, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    e = np.clip(np.asarray(e, dtype=np.float64).ravel(), e_clip, 1.0 - e_clip)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xt = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
        et = torch.as_tensor(e[start:end], dtype=torch.float32, device=device)
        tst = (
            torch.as_tensor(tree_sig[start:end], dtype=torch.float32, device=device)
            if tree_sig is not None
            else None
        )
        out = model(xt, et, tree_signals=tst)
        y0 = out["y0"].squeeze(-1).cpu().numpy().astype(np.float64)
        y1 = out["y1"].squeeze(-1).cpu().numpy().astype(np.float64)
        t = T[start:end]
        y = Y[start:end]
        ee = e[start:end]
        phi[start:end] = (y1 - y0) + t * (y - y1) / ee - (1.0 - t) * (y - y0) / (1.0 - ee)
    return phi.astype(np.float32)


def adapt_fewshot(
    model: GeometryAdaptiveCATE,
    X_k: np.ndarray,
    T_k: np.ndarray,
    Y_k: np.ndarray,
    e_k: np.ndarray,
    tree_k: Optional[np.ndarray],
    *,
    tau_k: Optional[np.ndarray] = None,
    mode: str = "full_adapt",
    device: str = "cpu",
    steps: int = 150,
    lr_router: float = 1e-3,
    lr_expert: float = 3e-4,
    batch_size: int = 32,
    entropy_weight: float = 0.01,
    route_weight: float = 0.5,
    factual_weight: float = 1.0,
    tau_weight: float = 1.0,
    ipm_weight: float = 0.05,
    refresh_oracle_every: int = 25,
    verbose: bool = False,
) -> GeometryAdaptiveCATE:
    """
    Strong few-shot adaptation.

    mode
      router_only     : freeze experts; only router + optional route CE
      full_adapt      : unfreeze experts; factual + true τ loss + route CE
      supervised_factual : unfreeze; factual only (real-data friendly)
      unsupervised_tta   : no labels; entropy + agreement
      dr_adapt        : unfreeze; factual + DR pseudo-outcome loss (τ-free)
    """
    model = copy.deepcopy(model).to(device)

    freeze = mode == "router_only"
    if freeze:
        for p in model.neural_expert.parameters():
            p.requires_grad = False
        for p in model.tree_expert.parameters():
            p.requires_grad = False
        params = [{"params": model.router.parameters(), "lr": lr_router}]
    else:
        for p in model.parameters():
            p.requires_grad = True
        params = [
            {"params": model.router.parameters(), "lr": lr_router},
            {
                "params": list(model.neural_expert.parameters())
                + list(model.tree_expert.parameters()),
                "lr": lr_expert,
            },
        ]

    opt = optim.AdamW(params, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    causal = CausalLoss(ipm_weight=ipm_weight, ipm_type="mean_diff")

    n = len(X_k)
    # DR mode: build pseudo τ from observables only (no ground-truth τ in loss)
    if mode == "dr_adapt":
        tau_k = compute_dr_pseudo_outcomes(
            model, X_k, T_k, Y_k, e_k, tree_k, device
        )

    use_tau = tau_k is not None and mode in ("full_adapt", "router_only", "dr_adapt")
    use_oracle = use_tau

    oracle_np = None
    if use_oracle:
        oracle_np = build_fewshot_oracle(model, X_k, e_k, tree_k, tau_k, device)

    def _make_loader(oracle_arr: Optional[np.ndarray], tau_arr: Optional[np.ndarray]) -> DataLoader:
        ts = (
            torch.as_tensor(tree_k, dtype=torch.float32)
            if tree_k is not None
            else torch.zeros(n, 0)
        )
        oc = (
            torch.as_tensor(oracle_arr, dtype=torch.long)
            if oracle_arr is not None
            else torch.full((n,), -1, dtype=torch.long)
        )
        tau_t = (
            torch.as_tensor(tau_arr, dtype=torch.float32)
            if tau_arr is not None
            else torch.zeros(n)
        )
        ds = TensorDataset(
            torch.as_tensor(X_k, dtype=torch.float32),
            torch.as_tensor(T_k, dtype=torch.float32),
            torch.as_tensor(Y_k, dtype=torch.float32),
            torch.as_tensor(e_k, dtype=torch.float32),
            ts,
            oc,
            tau_t,
        )
        return DataLoader(ds, batch_size=min(batch_size, max(n, 1)), shuffle=True)

    loader = _make_loader(oracle_np, tau_k)

    for step in range(1, steps + 1):
        # Periodically refresh oracle / DR targets as experts improve
        if (
            use_oracle
            and refresh_oracle_every > 0
            and step > 1
            and (step - 1) % refresh_oracle_every == 0
        ):
            if mode == "dr_adapt":
                tau_k = compute_dr_pseudo_outcomes(
                    model, X_k, T_k, Y_k, e_k, tree_k, device
                )
            oracle_np = build_fewshot_oracle(model, X_k, e_k, tree_k, tau_k, device)
            loader = _make_loader(oracle_np, tau_k)

        model.train()
        if freeze:
            model.neural_expert.eval()
            model.tree_expert.eval()

        total_loss = 0.0
        n_seen = 0
        for batch in loader:
            x, w, y, e, tree_sig, oc_b, tau_b = [t.to(device) for t in batch]
            tree_sig = (
                tree_sig if tree_sig.numel() > 0 and tree_sig.size(-1) > 0 else None
            )

            opt.zero_grad(set_to_none=True)
            out = model(x, e, tree_signals=tree_sig)
            loss = x.new_tensor(0.0)

            if mode == "router_only":
                mask = oc_b >= 0
                if mask.any():
                    log_p = F.log_softmax(out["logits"][mask], dim=-1)
                    loss = loss + route_weight * F.nll_loss(log_p, oc_b[mask])

            elif mode in ("full_adapt", "dr_adapt"):
                # 1) factual + light IPM
                c_loss, _, _ = causal(out["y0"], out["y1"], y, w, out["z"])
                loss = loss + factual_weight * c_loss
                # 2) τ or DR-pseudo supervision
                if use_tau:
                    loss = loss + tau_weight * F.mse_loss(out["tau"], tau_b)
                    loss = loss + 0.5 * tau_weight * (
                        F.mse_loss(out["tau_n"], tau_b)
                        + F.mse_loss(out["tau_t"], tau_b)
                    )
                # 3) routing CE toward better expert under the same target
                mask = oc_b >= 0
                if mask.any():
                    log_p = F.log_softmax(out["logits"][mask], dim=-1)
                    loss = loss + route_weight * F.nll_loss(log_p, oc_b[mask])

            elif mode == "supervised_factual":
                c_loss, _, _ = causal(out["y0"], out["y1"], y, w, out["z"])
                loss = loss + c_loss

            elif mode == "unsupervised_tta":
                wts = out["weights"]
                ent = -(wts.clamp(1e-8) * wts.clamp(1e-8).log()).sum(-1).mean()
                tau_tgt = 0.5 * (out["tau_n"].detach() + out["tau_t"].detach())
                agree = F.mse_loss(out["tau"], tau_tgt)
                loss = loss + agree + entropy_weight * ent
            else:
                raise ValueError(f"Unknown mode {mode}")

            if mode != "unsupervised_tta" and entropy_weight > 0:
                wts = out["weights"]
                ent = -(wts.clamp(1e-8) * wts.clamp(1e-8).log()).sum(-1).mean()
                loss = loss - 0.5 * entropy_weight * ent

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            total_loss += float(loss.detach()) * x.size(0)
            n_seen += x.size(0)

        sched.step()
        if verbose and (step % 25 == 0 or step == 1):
            print(
                f"  [adapt {mode}] step={step:03d} loss={total_loss / max(n_seen, 1):.4f}"
            )

    model.eval()
    return model


# backward-compatible alias
def adapt_router_fewshot(*args, **kwargs):
    if "freeze_experts" in kwargs:
        fe = kwargs.pop("freeze_experts")
        if fe and "mode" not in kwargs:
            kwargs["mode"] = "router_only"
        elif not fe and kwargs.get("mode", "full_adapt") == "supervised_oracle":
            kwargs["mode"] = "full_adapt"
    if kwargs.get("mode") == "supervised_oracle":
        kwargs["mode"] = "full_adapt"
    return adapt_fewshot(*args, **kwargs)


def evaluate_shift_with_adaptation(
    model: GeometryAdaptiveCATE,
    X_te: np.ndarray,
    T_te: np.ndarray,
    Y_te: np.ndarray,
    e_te: np.ndarray,
    tree_te: Optional[np.ndarray],
    tau_te: np.ndarray,
    *,
    k: int = 100,
    adapt_mode: str = "full_adapt",
    adapt_steps: int = 150,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Split target into k adaptation + held-out test.
    Compare PEHE before vs after strong adaptation.
    """
    # map legacy name
    if adapt_mode == "supervised_oracle":
        adapt_mode = "full_adapt"

    rng = np.random.default_rng(seed)
    n = len(X_te)
    idx = rng.permutation(n)
    k = min(k, max(n // 2, 1))
    adapt_idx = idx[:k]
    test_idx = idx[k:]

    model.eval()
    with torch.no_grad():
        tau_n0, tau_t0 = _expert_preds(
            model,
            X_te[test_idx],
            e_te[test_idx],
            None if tree_te is None else tree_te[test_idx],
            device,
        )
        xt = torch.as_tensor(X_te[test_idx], dtype=torch.float32, device=device)
        et = torch.as_tensor(e_te[test_idx], dtype=torch.float32, device=device)
        tst = (
            torch.as_tensor(tree_te[test_idx], dtype=torch.float32, device=device)
            if tree_te is not None
            else None
        )
        out0 = model(xt, et, tree_signals=tst)
        tau_a0 = out0["tau"].cpu().numpy()
        w0 = out0["weights"].cpu().numpy()

    pehe_before = _pehe(tau_te[test_idx], tau_a0)

    model_adapted = adapt_fewshot(
        model,
        X_te[adapt_idx],
        T_te[adapt_idx],
        Y_te[adapt_idx],
        e_te[adapt_idx],
        None if tree_te is None else tree_te[adapt_idx],
        tau_k=tau_te[adapt_idx],
        mode=adapt_mode,
        device=device,
        steps=adapt_steps,
        verbose=verbose,
    )

    with torch.no_grad():
        out1 = model_adapted(xt, et, tree_signals=tst)
        tau_a1 = out1["tau"].cpu().numpy()
        w1 = out1["weights"].cpu().numpy()
        tau_n1 = out1["tau_n"].cpu().numpy()
        tau_t1 = out1["tau_t"].cpu().numpy()

    pehe_after = _pehe(tau_te[test_idx], tau_a1)
    return {
        "k": k,
        "adapt_mode": adapt_mode,
        "pehe_before": pehe_before,
        "pehe_after": pehe_after,
        "pehe_neural_before": _pehe(tau_te[test_idx], tau_n0),
        "pehe_tree_before": _pehe(tau_te[test_idx], tau_t0),
        "pehe_neural_after": _pehe(tau_te[test_idx], tau_n1),
        "pehe_tree_after": _pehe(tau_te[test_idx], tau_t1),
        "w_neural_before": float(w0[:, 0].mean()),
        "w_neural_after": float(w1[:, 0].mean()),
        "delta_pehe": pehe_before - pehe_after,
    }
