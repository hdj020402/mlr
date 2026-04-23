"""Multiple Linear Regression with memory-efficient batched fitting.

Two fitting strategies are provided:

OLS (method="ols")
    Exact solution via Normal Equations accumulated batch-by-batch.
    Memory cost is O(p²) where p = number of features, regardless of
    how many samples are processed.  Each batch updates two running
    accumulators (XtX and Xty) and the final coefficients are solved
    once at the end.  This gives bitwise-identical results to loading
    all data at once.

Ridge (method="ridge")
    Exact solution via the same batched Normal Equations as OLS, with an
    L2 penalty term added to the diagonal of XtX before solving:
        theta = (XtX + alpha * I)^{-1} Xty
    Memory cost is also O(p²) and independent of sample size.

Iterative methods (method="lasso" | "elasticnet" | "huber")
    These have no closed-form solution and rely on sklearn iterative
    optimisers (coordinate descent for Lasso/ElasticNet, IRLS for Huber).
    The full dataset must fit in RAM.  Use them only when N*p is
    manageable; for very large datasets prefer OLS or Ridge.

Public interface
----------------
    model = MLR(method="ols")
    model.fit(dataset)           # dataset is any object with .iter_batches()
                                 # and .feature_names
    pred  = model.predict(X)     # numpy array
    coef  = model.coefficients   # dict {name: value, ..., "intercept": value}
    model.save("coef.json")
    model.load("coef.json")

Example:
    >>> from mlr.dataset import CSVDataset
    >>> from mlr.regression import MLR
    >>>
    >>> ds = CSVDataset("data.csv", feature_columns=["a", "b"], target_column="y")
    >>> model = MLR(method="ols")
    >>> metrics = model.fit(ds)
    >>> print(metrics)          # {"MAE": ..., "R2": ...}
    >>> model.save("coef.json")
"""

import json
import logging
from typing import Any

import numpy as np
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
)
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

_SKLEARN_METHODS: dict[str, Any] = {
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "huber": HuberRegressor,
}


class MLR:
    """Multiple Linear Regression with pluggable methods.

    Args:
        method:       Regression method.  One of:
                        "ols"        – Ordinary Least Squares via Normal
                                       Equations (exact, memory-bounded).
                        "ridge"      – L2 regularisation via Normal
                                       Equations (exact, memory-bounded).
                        "lasso"      – L1 regularisation (sparse coefficients).
                        "elasticnet" – L1 + L2 regularisation.
                        "huber"      – Robust regression, less sensitive to
                                       outliers.
        alpha:        Regularisation strength for ridge / lasso / elasticnet
                      (ignored for ols and huber).  Default 1.0.
        l1_ratio:     Mix ratio for elasticnet: 0 = pure Ridge, 1 = pure Lasso.
                      Default 0.5.
        huber_epsilon: Robustness parameter for Huber (values above this
                       threshold are treated as outliers).  Default 1.35.
        fit_intercept: Whether to fit an intercept term.  Default True.
    """

    def __init__(
        self,
        method: str = "ols",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        huber_epsilon: float = 1.35,
        fit_intercept: bool = True,
    ) -> None:
        method = method.lower()
        if method not in ("ols", "ridge", "lasso", "elasticnet", "huber"):
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from: ols, ridge, lasso, elasticnet, huber."
            )
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.huber_epsilon = huber_epsilon
        self.fit_intercept = fit_intercept

        self.feature_names: list[str] | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._sklearn_model = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, dataset) -> dict[str, float]:
        """Fit the model on a dataset.

        Args:
            dataset: Any object that provides:
                       .feature_names   – list[str]
                       .iter_batches()  – yields (X_batch, y_batch) numpy arrays

        Returns:
            Dict with training-set metrics: {"MAE": float, "R2": float}.
        """
        self.feature_names = list(dataset.feature_names)

        if self.method == "ols":
            return self._fit_ols(dataset)
        elif self.method == "ridge":
            return self._fit_ridge(dataset)
        else:
            return self._fit_sklearn(dataset)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for feature matrix X.

        Args:
            X: Shape (n_samples, n_features).

        Returns:
            Shape (n_samples, 1).

        Raises:
            RuntimeError: If the model has not been fitted or loaded.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted or loaded.")
        if self._sklearn_model is not None:
            return self._sklearn_model.predict(X).reshape(-1, 1)
        return (X @ self.coef_ + self.intercept_).reshape(-1, 1)

    @property
    def coefficients(self) -> dict[str, float]:
        """Named coefficient dict: {feature_name: coef, ..., "intercept": value}.

        Raises:
            RuntimeError: If the model has not been fitted or loaded.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted or loaded.")
        names = self.feature_names or [f"x{i}" for i in range(len(self.coef_))]
        d = {name: float(v) for name, v in zip(names, self.coef_)}
        d["intercept"] = float(self.intercept_)
        return d

    def save(self, path: str, extra_info: dict | None = None) -> None:
        """Save coefficients and metadata to a JSON file.

        Args:
            path:       Output path.
            extra_info: Optional dict merged into the saved JSON
                        (e.g. error metrics).
        """
        payload: dict[str, Any] = {
            "method": self.method,
            "fit_intercept": self.fit_intercept,
            "coefficients": self.coefficients,
        }
        if extra_info:
            payload["info"] = extra_info
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load coefficients from a JSON file produced by :meth:`save`.

        Args:
            path: Input path.
        """
        with open(path) as f:
            payload: dict = json.load(f)

        coef_dict: dict[str, float] = payload["coefficients"]
        self.method = payload.get("method", "ols")
        self.fit_intercept = payload.get("fit_intercept", True)
        self.intercept_ = coef_dict.pop("intercept", 0.0)
        self.feature_names = list(coef_dict.keys())
        self.coef_ = np.array(list(coef_dict.values()))
        self._sklearn_model = None
        logger.info(f"Model loaded from {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_ols(self, dataset) -> dict[str, float]:
        """Exact OLS via batched Normal Equations accumulation.

        Accumulates  XtX += X.T @ X  and  Xty += X.T @ y  for each
        batch.  If fit_intercept is True, X is augmented with a column
        of ones before accumulation so the intercept is part of the same
        solve.  This is identical to the closed-form solution on the
        full dataset but requires only O(p²) memory.
        """
        p = len(dataset.feature_names)
        aug_p = p + 1 if self.fit_intercept else p

        XtX = np.zeros((aug_p, aug_p), dtype=np.float64)
        Xty = np.zeros((aug_p, 1), dtype=np.float64)
        n_total = 0

        for X_batch, y_batch in dataset.iter_batches():
            if self.fit_intercept:
                ones = np.ones((len(X_batch), 1), dtype=np.float64)
                X_aug = np.hstack([X_batch, ones])
            else:
                X_aug = X_batch

            XtX += X_aug.T @ X_aug
            Xty += X_aug.T @ y_batch
            n_total += len(X_batch)
            logger.debug(f"OLS accumulate: processed {n_total} rows so far")

        logger.info(f"OLS solve: {n_total} total rows, {aug_p} parameters")
        theta = np.linalg.lstsq(XtX, Xty, rcond=None)[0].flatten()

        if self.fit_intercept:
            self.coef_ = theta[:p]
            self.intercept_ = float(theta[p])
        else:
            self.coef_ = theta
            self.intercept_ = 0.0

        return self._eval_metrics(dataset)

    def _fit_ridge(self, dataset) -> dict[str, float]:
        """Exact Ridge via batched Normal Equations with L2 penalty.

        Identical accumulation loop as OLS.  After all batches are
        processed, alpha is added to the diagonal of XtX (excluding
        the intercept row/column when fit_intercept is True) before
        solving:
            theta = (XtX + alpha * I_features)^{-1} Xty

        Memory cost is O(p²), same as OLS.
        """
        p = len(dataset.feature_names)
        aug_p = p + 1 if self.fit_intercept else p

        XtX = np.zeros((aug_p, aug_p), dtype=np.float64)
        Xty = np.zeros((aug_p, 1), dtype=np.float64)
        n_total = 0

        for X_batch, y_batch in dataset.iter_batches():
            if self.fit_intercept:
                ones = np.ones((len(X_batch), 1), dtype=np.float64)
                X_aug = np.hstack([X_batch, ones])
            else:
                X_aug = X_batch

            XtX += X_aug.T @ X_aug
            Xty += X_aug.T @ y_batch
            n_total += len(X_batch)
            logger.debug(f"Ridge accumulate: processed {n_total} rows so far")

        # Add L2 penalty to feature dimensions only (not the intercept column)
        XtX[:p, :p] += self.alpha * np.eye(p, dtype=np.float64)

        logger.info(f"Ridge solve: {n_total} total rows, alpha={self.alpha}")
        theta = np.linalg.solve(XtX, Xty).flatten()

        if self.fit_intercept:
            self.coef_ = theta[:p]
            self.intercept_ = float(theta[p])
        else:
            self.coef_ = theta
            self.intercept_ = 0.0

        return self._eval_metrics(dataset)

    def _fit_sklearn(self, dataset) -> dict[str, float]:
        """Fit a sklearn model after loading all data into memory."""
        X_parts, y_parts = [], []
        for X_batch, y_batch in dataset.iter_batches():
            X_parts.append(X_batch)
            y_parts.append(y_batch)

        X_all = np.vstack(X_parts)
        y_all = np.vstack(y_parts).ravel()
        del X_parts, y_parts

        cls = _SKLEARN_METHODS[self.method]
        kwargs: dict[str, Any] = {"fit_intercept": self.fit_intercept}
        if self.method in ("lasso", "elasticnet"):
            kwargs["alpha"] = self.alpha
        if self.method == "elasticnet":
            kwargs["l1_ratio"] = self.l1_ratio
        if self.method == "huber":
            kwargs["epsilon"] = self.huber_epsilon

        sk = cls(**kwargs)
        sk.fit(X_all, y_all)
        self._sklearn_model = sk

        coef = sk.coef_
        self.coef_ = coef.flatten()
        self.intercept_ = float(
            sk.intercept_ if np.isscalar(sk.intercept_) else sk.intercept_[0]
        )

        y_pred = sk.predict(X_all)
        mae = mean_absolute_error(y_all, y_pred)
        r2 = r2_score(y_all, y_pred)
        del X_all, y_all

        return {"MAE": float(mae), "R2": float(r2)}

    def _eval_metrics(self, dataset) -> dict[str, float]:
        """Compute MAE and R² over the full dataset in batches."""
        sum_ae = 0.0
        y_mean_acc = 0.0
        n = 0

        # Pass 1: predict and accumulate running sums
        y_preds, y_actuals = [], []
        for X_batch, y_batch in dataset.iter_batches():
            y_pred_batch = self.predict(X_batch)
            y_preds.append(y_pred_batch)
            y_actuals.append(y_batch)
            n += len(y_batch)
            sum_ae += np.abs(y_pred_batch - y_batch).sum()
            y_mean_acc += y_batch.sum()

        y_mean = y_mean_acc / n

        # Pass 2: compute SS from already-batched lists (no extra I/O)
        ss_res = 0.0
        ss_tot = 0.0
        for yp, ya in zip(y_preds, y_actuals):
            ss_res += ((ya - yp) ** 2).sum()
            ss_tot += ((ya - y_mean) ** 2).sum()

        mae = float(sum_ae / n)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        logger.info(f"Train metrics: MAE={mae:.6f}, R2={r2:.6f}")
        return {"MAE": mae, "R2": r2}
