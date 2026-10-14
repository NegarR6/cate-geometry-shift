"""
Canonical evaluation metrics for causal effect estimation.

All metrics that require ground-truth potential outcomes (PEHE, ATE Error,
Correlation, Policy Value, Regret, CATE ECE) return None when po0/po1 contain
NaNs (e.g. Jobs dataset). Callers must handle None gracefully.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from typing import Any, Dict, Optional, Tuple

import numpy as np


def _to_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def _has_ground_truth(po0: Optional[np.ndarray], po1: Optional[np.ndarray]) -> bool:
    if po0 is None or po1 is None:
        return False
    po0 = _to_1d(po0)
    po1 = _to_1d(po1)
    if po0.size == 0 or po1.size == 0:
        return False
    return not (np.any(np.isnan(po0)) or np.any(np.isnan(po1)))


def calc_pehe(
    po0_true: Optional[np.ndarray],
    po1_true: Optional[np.ndarray],
    tau_pred: np.ndarray,
) -> Optional[float]:
    """
    Root PEHE = sqrt( mean( (tau_true - tau_pred)^2 ) )
    Preferred interface: pass predicted CATE directly.
    """
    if not _has_ground_truth(po0_true, po1_true):
        return None
    tau_true = _to_1d(po1_true) - _to_1d(po0_true)
    tau_pred = _to_1d(tau_pred)
    if tau_true.shape != tau_pred.shape:
        raise ValueError(f"Shape mismatch: tau_true {tau_true.shape} vs tau_pred {tau_pred.shape}")
    return float(np.sqrt(np.mean((tau_true - tau_pred) ** 2)))


def calc_ate_error(
    po0_true: Optional[np.ndarray],
    po1_true: Optional[np.ndarray],
    tau_pred: np.ndarray,
) -> Optional[float]:
    """Absolute error of the estimated ATE."""
    if not _has_ground_truth(po0_true, po1_true):
        return None
    tau_true = _to_1d(po1_true) - _to_1d(po0_true)
    tau_pred = _to_1d(tau_pred)
    return float(np.abs(tau_true.mean() - tau_pred.mean()))


def calc_correlation(
    po0_true: Optional[np.ndarray],
    po1_true: Optional[np.ndarray],
    tau_pred: np.ndarray,
) -> Optional[float]:
    """Pearson correlation between true and predicted CATE."""
    if not _has_ground_truth(po0_true, po1_true):
        return None
    tau_true = _to_1d(po1_true) - _to_1d(po0_true)
    tau_pred = _to_1d(tau_pred)
    if tau_true.std() < 1e-12 or tau_pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(tau_true, tau_pred)[0, 1])


def calc_policy_value_and_regret(
    po0_true: Optional[np.ndarray],
    po1_true: Optional[np.ndarray],
    tau_pred: np.ndarray,
    threshold: float = 0.0,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Policy Value and Regret under the decision rule: treat if tau_pred > threshold.
    """
    if not _has_ground_truth(po0_true, po1_true):
        return None, None
    tau_true = _to_1d(po1_true) - _to_1d(po0_true)
    tau_pred = _to_1d(tau_pred)
    po0 = _to_1d(po0_true)

    pi_pred = (tau_pred > threshold).astype(np.float64)
    pi_opt = (tau_true > threshold).astype(np.float64)

    policy_value = float(np.mean(po0 + pi_pred * tau_true))
    opt_value = float(np.mean(po0 + pi_opt * tau_true))
    regret = opt_value - policy_value
    return policy_value, float(regret)


def calc_cate_ece(
    po0_true: Optional[np.ndarray],
    po1_true: Optional[np.ndarray],
    tau_pred: np.ndarray,
    n_bins: int = 10,
) -> Optional[float]:
    """
    Expected Calibration Error for CATE (bin by predicted CATE).
    """
    if not _has_ground_truth(po0_true, po1_true):
        return None
    tau_true = _to_1d(po1_true) - _to_1d(po0_true)
    tau_pred = _to_1d(tau_pred)

    order = np.argsort(tau_pred)
    pred_sorted = tau_pred[order]
    true_sorted = tau_true[order]

    bins = np.array_split(np.arange(len(tau_pred)), n_bins)
    ece = 0.0
    n = len(tau_pred)
    for b in bins:
        if len(b) == 0:
            continue
        ece += (len(b) / n) * np.abs(pred_sorted[b].mean() - true_sorted[b].mean())
    return float(ece)


def calc_factual_rmse(
    y_true: Optional[np.ndarray],
    w: Optional[np.ndarray],
    mu0_pred: Optional[np.ndarray],
    mu1_pred: Optional[np.ndarray],
) -> Optional[float]:
    """RMSE of factual outcome: y vs μ_w(x). Primary metric for Jobs/Twins."""
    if w is None or np.any(np.isnan(_to_1d(w))):
        return None
    y = _to_1d(y_true)
    w = _to_1d(w)
    mu0 = _to_1d(mu0_pred)
    mu1 = _to_1d(mu1_pred)
    if y.size == 0 or np.any(np.isnan(y)):
        return None
    y_hat = np.where(w > 0.5, mu1, mu0)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def calc_att_error(
    y_true: Optional[np.ndarray],
    w: Optional[np.ndarray],
    tau_pred: np.ndarray,
) -> Optional[float]:
    """
    |ATT_obs - mean(τ̂ | treated)| where ATT_obs = mean(y|T=1)-mean(y|T=0).
    Diagnostic for observational data without potential outcomes.
    """
    if w is None or np.any(np.isnan(_to_1d(w))):
        return None
    y = _to_1d(y_true)
    w = _to_1d(w).astype(int)
    tau = _to_1d(tau_pred)
    treated = w == 1
    control = w == 0
    if treated.sum() < 2 or control.sum() < 2:
        return None
    att_obs = float(y[treated].mean() - y[control].mean())
    att_pred = float(tau[treated].mean())
    return float(np.abs(att_obs - att_pred))


def evaluate_cate(
    po0_true: Optional[np.ndarray],
    po1_true: Optional[np.ndarray],
    tau_pred: np.ndarray,
    threshold: float = 0.0,
    n_bins: int = 10,
    y_true: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    mu0_pred: Optional[np.ndarray] = None,
    mu1_pred: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """
    Single entry point for CATE + factual metrics.

    PO-based (None when potential outcomes unavailable):
        PEHE, ATE_Error, Correlation, Policy_Value, Regret, CATE_ECE
    Factual (Jobs / Twins):
        Factual_RMSE, ATT_Error
    """
    pv, regret = calc_policy_value_and_regret(po0_true, po1_true, tau_pred, threshold)
    return {
        "PEHE": calc_pehe(po0_true, po1_true, tau_pred),
        "ATE_Error": calc_ate_error(po0_true, po1_true, tau_pred),
        "Correlation": calc_correlation(po0_true, po1_true, tau_pred),
        "Policy_Value": pv,
        "Regret": regret,
        "CATE_ECE": calc_cate_ece(po0_true, po1_true, tau_pred, n_bins),
        "Factual_RMSE": calc_factual_rmse(y_true, w, mu0_pred, mu1_pred),
        "ATT_Error": calc_att_error(y_true, w, tau_pred),
    }


# Backward-compatible aliases (old code may still call these)
def calc_pehe_legacy(po0_true, po1_true, po0_pred, po1_pred):
    tau_pred = _to_1d(po1_pred) - _to_1d(po0_pred)
    return calc_pehe(po0_true, po1_true, tau_pred)


def evaluate_predictions(po0_true, po1_true, po0_pred, po1_pred):
    tau_pred = _to_1d(po1_pred) - _to_1d(po0_pred)
    return evaluate_cate(po0_true, po1_true, tau_pred)


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    po0 = np.random.randn(n)
    po1 = po0 + 1.5 + 0.3 * np.random.randn(n)
    tau_true = po1 - po0
    tau_pred = tau_true + 0.2 * np.random.randn(n)

    metrics = evaluate_cate(po0, po1, tau_pred)
    print("evaluation.py self-test:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if v is not None else f"  {k}: None")

    # NaN case (Jobs-style)
    metrics_nan = evaluate_cate(np.full(n, np.nan), np.full(n, np.nan), tau_pred)
    assert all(v is None for v in metrics_nan.values()), "NaN handling failed"
    print("  NaN handling: OK")
    print("All checks passed.")
