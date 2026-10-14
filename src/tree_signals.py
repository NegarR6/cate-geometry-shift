import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))
"""Leakage-safe tree signals for the RIPO hybrid model."""

import numpy as np
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    xgb = None
    _HAS_XGB = False
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


class CrossFittedTreeSignals:
    """Build out-of-fold outcome/propensity features for neural training.

    Training rows are never scored by a nuisance model fitted on those rows.
    After cross-fitting, full-data models are fitted solely for validation/test
    transformation.
    """

    def __init__(self, n_splits=5, seed=42, n_estimators=200, max_depth=4):
        self.n_splits = n_splits
        self.seed = seed
        if _HAS_XGB:
            self.outcome_template = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=4,
            )
        else:
            self.outcome_template = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=seed,
            )
        self.propensity_template = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=4,
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
            # Old code used np.where(w==0) which included held_idx → outcome leakage.
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
            self.model0_.feature_importances_ + self.model1_.feature_importances_
        ) / 2
        self.feature_importance_ = importance / (importance.sum() + 1e-12)
        return self._pack(mu0, mu1, propensity)

    def transform(self, X):
        if not hasattr(self, "model0_"):
            raise RuntimeError("fit_transform must be called before transform")
        X = np.asarray(X)
        return self._pack(
            self.model0_.predict(X),
            self.model1_.predict(X),
            self.propensity_.predict_proba(X)[:, 1],
        )
