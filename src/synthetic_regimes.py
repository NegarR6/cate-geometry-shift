"""
Synthetic controlled regimes for CATE ranking experiments (SPEC v1 LOCKED).

Question:
  Holding n, p, noise, propensity, and training protocol fixed, does changing
  only the structure of tau(x) (and signal strength) reverse estimator rankings?

Regimes:
  A — smooth / distributed
  B — sparse threshold / tree-like
  C — smooth + interaction

Do not change formulas without amending synthetic-spec.md.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from dataclasses import asdict, dataclass
from typing import Dict, Literal, Tuple

import numpy as np

RegimeName = Literal["A", "B", "C", "A2", "C2"]

# ---------------------------------------------------------------------------
# Locked constants (synthetic-spec)
# ---------------------------------------------------------------------------
N_DEFAULT = 3000
P_DEFAULT = 12
NOISE_STD = 0.5
TRAIN_FRAC = 0.7
NORM_SAMPLE_SIZE = 20_000
SIGNAL_LEVELS = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class SyntheticConfig:
    n: int = N_DEFAULT
    p: int = P_DEFAULT
    noise_std: float = NOISE_STD
    train_frac: float = TRAIN_FRAC
    seed: int = 0
    regime: RegimeName = "A"
    signal: float = 1.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def sample_X(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    assert p >= 5, "Spec requires p >= 5 for tau formulas"
    return rng.normal(0.0, 1.0, size=(n, p)).astype(np.float64)


def propensity(X: np.ndarray) -> np.ndarray:
    """e(x) = σ(0.4 x1 - 0.3 x2 + 0.2 x3) — locked moderate confounding."""
    z = 0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
    return _sigmoid(z)


def mu0(X: np.ndarray) -> np.ndarray:
    """Baseline outcome — identical across regimes."""
    return (
        0.5 * X[:, 0]
        + 0.3 * X[:, 1]
        - 0.2 * X[:, 3]
        + 0.1 * X[:, 0] * X[:, 1]
    )


def tau_tilde_A(X: np.ndarray) -> np.ndarray:
    """Smooth / distributed."""
    return (
        np.tanh(0.8 * X[:, 0] + 0.5 * X[:, 1])
        + 0.4 * np.sin(X[:, 2])
        + 0.3 * X[:, 4]
    )


def tau_tilde_B(X: np.ndarray) -> np.ndarray:
    """Sparse threshold / tree-like."""
    return (
        ((X[:, 0] > 0) & (X[:, 1] > -0.5)).astype(np.float64)
        + 0.5 * (X[:, 2] > 0.5).astype(np.float64)
        - 0.3 * (X[:, 0] <= 0).astype(np.float64)
    )


def tau_tilde_C(X: np.ndarray) -> np.ndarray:
    """Smooth + interaction."""
    return np.tanh(X[:, 0]) * X[:, 1] + 0.4 * (X[:, 2] ** 2) - 0.2 * X[:, 3]


def tau_tilde_A2(X: np.ndarray) -> np.ndarray:
    """Control: same smooth geometry class as A, different coefficients / variables.
    Used for A->A' geometry-within-class shift experiments (P0 control).
    """
    return (
        np.tanh(0.5 * X[:, 1] - 0.8 * X[:, 0])
        + 0.4 * np.cos(X[:, 2])
        - 0.3 * X[:, 4]
    )


def tau_tilde_C2(X: np.ndarray) -> np.ndarray:
    """Control: same interaction geometry class as C, different coefficients."""
    return np.tanh(X[:, 1]) * X[:, 0] + 0.3 * (X[:, 3] ** 2) - 0.25 * X[:, 2]


TAU_FN = {
    "A": tau_tilde_A,
    "B": tau_tilde_B,
    "C": tau_tilde_C,
    "A2": tau_tilde_A2,
    "C2": tau_tilde_C2,
}


def normalization_constant(regime: RegimeName, rng: np.random.Generator, p: int = P_DEFAULT) -> float:
    """
    Seed-independent in intent: use a fixed RNG stream for norm constants.
    Std(tau_tilde) on NORM_SAMPLE_SIZE draws -> scale so std ≈ 1.
    """
    X = sample_X(NORM_SAMPLE_SIZE, p, rng)
    t = TAU_FN[regime](X)
    s = float(np.std(t))
    if s < 1e-8:
        return 1.0
    return 1.0 / s


def compute_all_norm_constants(p: int = P_DEFAULT, base_seed: int = 12345) -> Dict[str, float]:
    """Fixed constants shared across all experiment seeds."""
    rng = np.random.default_rng(base_seed)
    return {r: normalization_constant(r, rng, p=p) for r in ("A", "B", "C", "A2", "C2")}  # type: ignore


def tau(X: np.ndarray, regime: RegimeName, signal: float, norm: float) -> np.ndarray:
    return signal * norm * TAU_FN[regime](X)


def generate_synthetic(
    cfg: SyntheticConfig,
    norm_constants: Dict[str, float] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Returns data dict compatible with typical CATE runners:
      X, T, Y, mu0, mu1, tau, e, train_idx, test_idx
    and a meta dict.
    """
    if cfg.regime not in TAU_FN:
        raise ValueError(f"Unknown regime {cfg.regime}")
    if norm_constants is None:
        norm_constants = compute_all_norm_constants(p=cfg.p)

    rng = np.random.default_rng(cfg.seed)
    X = sample_X(cfg.n, cfg.p, rng)
    e = propensity(X)
    T = rng.binomial(1, e).astype(np.float64)
    m0 = mu0(X)
    norm = float(norm_constants[cfg.regime])
    ttrue = tau(X, cfg.regime, cfg.signal, norm)
    m1 = m0 + ttrue
    Y = m0 + T * ttrue + rng.normal(0.0, cfg.noise_std, size=cfg.n)

    idx = rng.permutation(cfg.n)
    n_tr = int(cfg.train_frac * cfg.n)
    train_idx = np.sort(idx[:n_tr])
    test_idx = np.sort(idx[n_tr:])

    data = {
        "X": X,
        "T": T,
        "Y": Y,
        "mu0": m0,
        "mu1": m1,
        "tau": ttrue,
        "e": e,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    meta = {
        "config": asdict(cfg),
        "norm_constant": norm,
        "norm_constants": norm_constants,
        "n_train": int(n_tr),
        "n_test": int(cfg.n - n_tr),
        "spec": "synthetic-spec",
    }
    return data, meta


def split_train_test(data: Dict[str, np.ndarray]):
    tr, te = data["train_idx"], data["test_idx"]
    out_tr, out_te = {}, {}
    for k, v in data.items():
        if k in ("train_idx", "test_idx"):
            continue
        out_tr[k] = v[tr]
        out_te[k] = v[te]
    return out_tr, out_te


# ---------------------------------------------------------------------------
# Overlap diagnostics (for IPM × α reporting)
# ---------------------------------------------------------------------------

def overlap_metrics(e: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    """Classic propensity overlap diagnostics.

    ESS_treated / ESS_control use self-normalized IPW weights on the sample.
    """
    e = np.asarray(e, dtype=np.float64).ravel()
    e = np.clip(e, eps, 1.0 - eps)
    # IPW weights for treated/control potential outcomes
    w1 = 1.0 / e
    w0 = 1.0 / (1.0 - e)
    ess1 = float((w1.sum() ** 2) / (np.square(w1).sum() + eps))
    ess0 = float((w0.sum() ** 2) / (np.square(w0).sum() + eps))
    return {
        "n": float(len(e)),
        "min_e": float(e.min()),
        "max_e": float(e.max()),
        "mean_abs_e_minus_half": float(np.mean(np.abs(e - 0.5))),
        "frac_e_outside_05_95": float(np.mean((e < 0.05) | (e > 0.95))),
        "ess_treated": ess1,
        "ess_control": ess0,
        "ess_ratio_min": float(min(ess1, ess0) / max(len(e), 1)),
    }


def propensity_scaled(X: np.ndarray, alpha: float) -> np.ndarray:
    """e_alpha(x) = σ(alpha * (0.4 x1 - 0.3 x2 + 0.2 x3)) — same form as IPM SPEC."""
    z = 0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
    return _sigmoid(alpha * z)


# ---------------------------------------------------------------------------
# Harder stress-test DGP (NOT SPEC v1 — appendix / stress only)
# ---------------------------------------------------------------------------
# Higher dimension + noise; same tau geometry A/B/C so ranking-reversal
# question remains identifiable. Never mix these numbers into locked SPEC tables.

@dataclass(frozen=True)
class SyntheticConfigHard(SyntheticConfig):
    """Stress config: p=50, noise_std=1.0, n=5000 by default."""
    n: int = 5000
    p: int = 50
    noise_std: float = 1.0


def generate_synthetic_hard(
    cfg: SyntheticConfigHard,
    norm_constants: Dict[str, float] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Same tau geometry as SPEC v1, higher p/noise. Meta.spec marks hard."""
    data, meta = generate_synthetic(cfg, norm_constants=norm_constants)
    meta["spec"] = "SYNTHETIC_HARD_v1"
    meta["note"] = (
        "Stress-test only: higher p and noise. Do not replace SPEC v1 tables."
    )
    # attach overlap under default propensity
    meta["overlap"] = overlap_metrics(data["e"])
    return data, meta


# ---------------------------------------------------------------------------
# Geometry continuum (continuum phase) — additive blend of normalized τ
# ---------------------------------------------------------------------------

def tau_blend(
    X: np.ndarray,
    lam: float,
    regime_a: RegimeName = "A",
    regime_b: RegimeName = "B",
    signal: float = 1.0,
    norm_constants: Dict[str, float] | None = None,
) -> np.ndarray:
    """τ_λ = (1-λ) τ_a + λ τ_b with each side scaled by its locked norm constant."""
    if norm_constants is None:
        norm_constants = compute_all_norm_constants(p=X.shape[1])
    ta = signal * float(norm_constants[regime_a]) * TAU_FN[regime_a](X)
    tb = signal * float(norm_constants[regime_b]) * TAU_FN[regime_b](X)
    lam = float(np.clip(lam, 0.0, 1.0))
    return (1.0 - lam) * ta + lam * tb


def generate_continuum(
    n: int,
    p: int,
    lam: float,
    seed: int,
    noise_std: float = NOISE_STD,
    train_frac: float = TRAIN_FRAC,
    signal: float = 1.0,
    regime_a: RegimeName = "A",
    regime_b: RegimeName = "B",
    norm_constants: Dict[str, float] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Same observational laws as generate_synthetic; only τ is a continuum blend."""
    if norm_constants is None:
        norm_constants = compute_all_norm_constants(p=p)
    rng = np.random.default_rng(seed)
    X = sample_X(n, p, rng)
    e = propensity(X)
    T = rng.binomial(1, e).astype(np.float64)
    m0 = mu0(X)
    ttrue = tau_blend(X, lam, regime_a, regime_b, signal, norm_constants)
    m1 = m0 + ttrue
    Y = m0 + T * ttrue + rng.normal(0.0, noise_std, size=n)
    idx = rng.permutation(n)
    n_tr = int(train_frac * n)
    train_idx = np.sort(idx[:n_tr])
    test_idx = np.sort(idx[n_tr:])
    data = {
        "X": X, "T": T, "Y": Y, "mu0": m0, "mu1": m1,
        "tau": ttrue, "e": e, "train_idx": train_idx, "test_idx": test_idx,
    }
    meta = {
        "lam": lam, "regime_a": regime_a, "regime_b": regime_b,
        "signal": signal, "noise_std": noise_std, "n": n, "p": p, "seed": seed,
        "norm_constants": norm_constants, "spec": "GEOMETRY_CONTINUUM_v1",
    }
    return data, meta
