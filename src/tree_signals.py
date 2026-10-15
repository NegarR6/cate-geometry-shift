"""Leakage-safe tree signals for hybrid / geometry-adaptive models.

P0 FIXES:
- xgboost is REQUIRED (no silent sklearn GBR fallback that changes results).
- Propensity uses LogisticRegression (better calibrated) instead of
  RandomForestClassifier with class_weight='balanced' that pulls toward 0.5.
- Explicit documentation that transform() uses full-data models fitted after
  OOF scoring; callers must freeze source trees when evaluating on target.
"""
from __future__ import annotations

import sys as _sys
import pathlib as _pl

_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "xgboost is required for tree_signals (results are not reproducible "
        "with a silent sklearn fallback). Install via: pip install xgboost>=1.7.0"
    ) from exc


class CrossFittedTreeSignals:
    """Build out-of-fold outcome/propensity features for neural training.

    Training rows are never scored by a nuisance model fitted on those rows.
    After cross-fitting, full-data models are fitted solely for validation/test
    transformation. When transferring under pure geometry shift, callers MUST
    freeze the source-fitted extractor (call transform on target X only; never
    re-fit on target outcomes).
    """

    def __init__(self, n_splits=5, seed=42, n_estimators=200, max_depth=4):
        self.n_splits = n_splits
        self.seed = seed
        self.outcome_template = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
            verbosity=0,
        )
        # Calibrated propensity (logistic) instead of RF + class_weight=balanced
        self.propensity_template = LogisticRegression(
            max_iter=500,
            random_state=seed,
            solver="lbfgs",
        )

    @staticmethod
    def _pack(mu0, mu1, propensity):
        propensity = np.clip(propensity, 0.025, 0.975)
        return np.column_stack([mu0, mu1, mu1 - mu0, propensity]).astype(np.float32)

    def fit_transform(self, X, w, y):
        X = np.asarray(X)
        w = np.asarray(w).astype(int).ravel()
        y = np.asarray(y).ravel()
        mu0 = np.zeros(len(y), dtype=np.float32)
        mu1 = np.zeros(len(y), dtype=np.float32)
        propensity = np.zeros(len(y), dtype=np.float32)

        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        for train_idx, held_idx in splitter.split(X, w):
            model0 = clone(self.outcome_template)
            model1 = clone(self.outcome_template)
            prop = clone(self.propensity_template)
            tr0 = train_idx[w[train_idx] == 0]
            tr1 = train_idx[w[train_idx] == 1]
            # Leakage-safe guard: NEVER expand tr0/tr1 with held-out rows.
            if len(tr0) < 2 or len(tr1) < 2:
                if len(np.unique(w[train_idx])) > 1 and len(train_idx) >= 2:
                    prop.fit(X[train_idx], w[train_idx])
                    propensity[held_idx] = prop.predict_proba(X[held_idx])[:, 1]
                continue
            model0.fit(X[tr0], y[tr0])
            model1.fit(X[tr1], y[tr1])
            prop.fit(X[train_idx], w[train_idx])
            mu0[held_idx] = model0.predict(X[held_idx])
            mu1[held_idx] = model1.predict(X[held_idx])
            propensity[held_idx] = prop.predict_proba(X[held_idx])[:, 1]

        ctrl = np.where(w == 0)[0]
        treated = np.where(w == 1)[0]
        if len(ctrl) < 2 or len(treated) < 2:
            raise RuntimeError(
                f"CrossFittedTreeSignals: need >=2 samples per arm, "
                f"got control={len(ctrl)}, treated={len(treated)}"
            )
        self.model0_ = clone(self.outcome_template).fit(X[ctrl], y[ctrl])
        self.model1_ = clone(self.outcome_template).fit(X[treated], y[treated])
        self.propensity_ = clone(self.propensity_template).fit(X, w)
        importance = (
            getattr(self.model0_, "feature_importances_", np.ones(X.shape[1]))
            + getattr(self.model1_, "feature_importances_", np.ones(X.shape[1]))
        ) / 2
        self.feature_importance_ = importance / (importance.sum() + 1e-12)
        return self._pack(mu0, mu1, propensity)

    def transform(self, X):
        """Transform using full-data models fitted on the SOURCE training set.

        Under pure geometry shift this must be called on target X WITHOUT
        re-fitting, so that no target outcomes leak into the tree pathway.
        """
        if not hasattr(self, "model0_"):
            raise RuntimeError("fit_transform must be called before transform")
        X = np.asarray(X)
        return self._pack(
            self.model0_.predict(X),
            self.model1_.predict(X),
            self.propensity_.predict_proba(X)[:, 1],
        )
