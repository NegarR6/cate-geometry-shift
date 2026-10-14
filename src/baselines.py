"""
Tree-based causal baselines for Phase 0.

Models:
  - X-Learner (XGBoost base learners) via EconML
  - Causal Forest DML via EconML
  - (Optional) DoubleML if doubleml is installed

All models expose a uniform interface:
  fit(X, w, y) -> self
  predict_cate(X) -> np.ndarray of shape (n,)
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


from typing import Any, Dict, Optional

import numpy as np

try:
    from econml.metalearners import XLearner
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    import xgboost as xgb
    _HAS_ECONML = True
except ImportError:
    _HAS_ECONML = False

try:
    import doubleml as dml
    from sklearn.base import clone
    _HAS_DOUBLEML = True
except ImportError:
    _HAS_DOUBLEML = False


class BaseCATEModel:
    """Minimal interface for CATE estimators used in this project."""

    def fit(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> "BaseCATEModel":
        raise NotImplementedError

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class XLearnerXGBoost(BaseCATEModel):
    """X-Learner with XGBoost regressors and RandomForest propensity."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 3,
        propensity_n_estimators: int = 100,
        random_state: int = 42,
    ):
        if not _HAS_ECONML:
            raise ImportError("econml and xgboost are required for XLearnerXGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.propensity_n_estimators = propensity_n_estimators
        self.random_state = random_state
        self._model: Optional[XLearner] = None

    def fit(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> "XLearnerXGBoost":
        X = np.asarray(X, dtype=np.float64)
        w = np.asarray(w).ravel().astype(int)
        y = np.asarray(y).ravel()

        self._model = XLearner(
            models=xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                random_state=self.random_state,
                n_jobs=4,
                verbosity=0,
            ),
            propensity_model=RandomForestClassifier(
                n_estimators=self.propensity_n_estimators,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=4,
            ),
        )
        self._model.fit(y, w, X=X)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(self._model.effect(np.asarray(X, dtype=np.float64))).ravel()


class CausalForestBaseline(BaseCATEModel):
    """Causal Forest DML (EconML)."""

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 10,
        model_y_n_estimators: int = 200,
        model_t_n_estimators: int = 100,
        random_state: int = 42,
    ):
        if not _HAS_ECONML:
            raise ImportError("econml is required for CausalForestBaseline")
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.model_y_n_estimators = model_y_n_estimators
        self.model_t_n_estimators = model_t_n_estimators
        self.random_state = random_state
        self._model: Optional[CausalForestDML] = None

    def fit(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> "CausalForestBaseline":
        X = np.asarray(X, dtype=np.float64)
        w = np.asarray(w).ravel().astype(int)
        y = np.asarray(y).ravel()

        self._model = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=self.model_y_n_estimators,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=4,
            ),
            model_t=RandomForestClassifier(
                n_estimators=self.model_t_n_estimators,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=4,
            ),
            discrete_treatment=True,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=4,
        )
        self._model.fit(y, w, X=X)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(self._model.effect(np.asarray(X, dtype=np.float64))).ravel()


class DoubleMLIRMBaseline(BaseCATEModel):
    """
    DoubleML Interactive Regression Model (IRM).

    IMPORTANT (ICLR honesty):
      Standard DoubleML IRM targets ATE. Individual CATE is not reliably
      available in all doubleml versions. When cate() is missing this class
      returns a constant ATE prediction → Correlation will be 0.
      Do NOT list this model as a CATE baseline in the main paper table.
      Use X-Learner / CausalForest for CATE comparisons; keep DoubleML
      only as an ATE reference or in an appendix.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        n_folds: int = 3,
        random_state: int = 42,
    ):
        if not _HAS_DOUBLEML:
            raise ImportError(
                "doubleml is required for DoubleMLIRMBaseline. "
                "Install with: pip install doubleml"
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_folds = n_folds
        self.random_state = random_state
        self._model = None
        self._X_train = None

    def fit(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> "DoubleMLIRMBaseline":
        import pandas as pd
        from doubleml import DoubleMLData, DoubleMLIRM

        X = np.asarray(X, dtype=np.float64)
        w = np.asarray(w).ravel().astype(int)
        y = np.asarray(y).ravel()
        self._X_train = X

        # DoubleML expects a DataFrame
        cols = [f"x{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df["y"] = y
        df["d"] = w

        dml_data = DoubleMLData(df, y_col="y", d_cols="d", x_cols=cols)

        ml_g = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=4,
            verbosity=0,
        )
        ml_m = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=4,
            verbosity=0,
            eval_metric="logloss",
        )

        self._model = DoubleMLIRM(
            dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            n_folds=self.n_folds,
            score="ATE",  # we still extract individual effects via cate
        )
        self._model.fit()
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        DoubleMLIRM primarily targets ATE. For individual CATE we use the
        orthogonalized residual method when available, otherwise fall back to
        constant ATE (honest degradation).
        """
        if self._model is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)

        # Prefer cate() if the installed doubleml version supports it
        if hasattr(self._model, "cate"):
            try:
                return np.asarray(self._model.cate(X)).ravel()
            except Exception:
                pass

        # Fallback: constant ATE prediction (still a valid, if weak, baseline)
        ate = float(self._model.coef[0]) if hasattr(self._model, "coef") else float(self._model.summary.coef["coef"].iloc[0])
        return np.full(X.shape[0], ate, dtype=np.float64)


def get_baseline(name: str, random_state: int = 42, **kwargs) -> BaseCATEModel:
    """Factory for baselines used in Phase 0."""
    name = name.lower().strip()
    if name in ("xlearner", "x-learner", "x_learner"):
        return XLearnerXGBoost(random_state=random_state, **kwargs)
    if name in ("causalforest", "causal_forest", "cf", "causalforest_dml"):
        return CausalForestBaseline(random_state=random_state, **kwargs)
    if name in ("doubleml", "doubleml_irm", "dml", "irm"):
        return DoubleMLIRMBaseline(random_state=random_state, **kwargs)
    raise ValueError(
        f"Unknown baseline '{name}'. Supported: xlearner, causalforest, doubleml"
    )


def list_available_baselines() -> Dict[str, bool]:
    return {
        "xlearner": _HAS_ECONML,
        "causalforest": _HAS_ECONML,
        "doubleml": _HAS_DOUBLEML,  # ATE-oriented; see class docstring
    }


# Models that produce genuine individual CATE estimates (safe for PEHE tables)
CATE_BASELINES = ("xlearner", "causalforest")
# Models that should only appear in ATE / appendix sections
ATE_ONLY_BASELINES = ("doubleml",)
