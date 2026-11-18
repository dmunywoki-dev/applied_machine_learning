"""Optional non-negative positive-unlabelled logistic learner.

This learner is not invoked unless a researcher supplies a defensible class prior
and a traceable set of verified positive examples. Its output is a ranking score,
not a calibrated probability.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy import sparse


class NnPULogistic:
    """Linear non-negative PU risk estimator using logistic losses."""

    def __init__(self, positive_prior: float, l2: float = 1e-3, max_iter: int = 500):
        if not 0.0 < positive_prior < 1.0:
            raise ValueError("positive_prior must be strictly between 0 and 1")
        self.positive_prior = float(positive_prior)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _dot(X, w):
        return np.asarray(X @ w).ravel()

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        pos = np.flatnonzero(y == 1)
        unl = np.flatnonzero(y == 0)
        if len(pos) < 2 or len(unl) < 5:
            raise ValueError("nnPU requires at least 2 verified positives and 5 unlabelled observations")
        Xp, Xu = X[pos], X[unl]
        n_features = X.shape[1]

        def objective(params):
            w, bias = params[:-1], params[-1]
            zp = self._dot(Xp, w) + bias
            zu = self._dot(Xu, w) + bias
            # Positive risk: pi * E_P[log(1 + exp(-f(x)))]
            lp_pos = np.logaddexp(0.0, -zp)
            rp_pos = self.positive_prior * lp_pos.mean()
            # Corrected negative risk: E_U[l(f,0)] - pi*E_P[l(f,0)]
            lp_neg = np.logaddexp(0.0, zp)
            ru_neg = np.logaddexp(0.0, zu).mean()
            corrected_negative = ru_neg - self.positive_prior * lp_neg.mean()
            use_negative = corrected_negative > 0.0
            risk = rp_pos + max(0.0, corrected_negative) + 0.5 * self.l2 * np.dot(w, w)

            grad_zp = self.positive_prior * (-expit(-zp)) / len(zp)
            if use_negative:
                grad_zp += -self.positive_prior * expit(zp) / len(zp)
                grad_zu = expit(zu) / len(zu)
            else:
                grad_zu = np.zeros_like(zu)
            grad_w = np.asarray(Xp.T @ grad_zp).ravel() + np.asarray(Xu.T @ grad_zu).ravel() + self.l2 * w
            grad_b = grad_zp.sum() + grad_zu.sum()
            return risk, np.r_[grad_w, grad_b]

        start = np.zeros(n_features + 1, dtype=float)
        result = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = result.x[:-1]
        self.intercept_ = float(result.x[-1])
        self.optimisation_ = {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit)}
        return self

    def decision_function(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")
        return self._dot(X, self.coef_) + self.intercept_
