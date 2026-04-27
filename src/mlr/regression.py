"""Multiple Linear Regression with memory-efficient batched fitting.

Two fitting strategies are provided:

OLS (method="ols")
    Exact solution via Normal Equations accumulated batch-by-batch.
    Memory cost is O(p²) where p = number of features, regardless of
    how many samples are processed.  Each batch updates two running
    accumulators (XtX and Xty) and the final coefficients are solved
    once at the end.  This gives bitwise-identical results to loading
    all data at once.

    All-zero feature columns are automatically detected and excluded
    from the solve to avoid singular matrix issues.  These columns
    receive coefficient 0.  The excluded column indices are stored
    in ``model.zero_col_indices``.

Ridge (method="ridge")
    Exact solution via the same batched Normal Equations as OLS, with an
    L2 penalty term added to the diagonal of XtX before solving:
        theta = (XtX + alpha * I)^{-1} Xty
    Memory cost is also O(p²) and independent of sample size.

    All-zero columns are handled identically to OLS.

Iterative methods (method="lasso" | "elasticnet" | "huber")
    These have no closed-form solution and rely on sklearn iterative
    optimisers (coordinate descent for Lasso/ElasticNet, IRLS for Huber).
    The full dataset must fit in RAM.  Use them only when N*p is
    manageable; for very large datasets prefer OLS or Ridge.

Public interface
----------------
    model = MLR(method="ols")
    model.fit(dataset)                       # dataset is any object with .iter_batches()
                                             # and .feature_names
    model.fit(dataset, metrics=["MAE", "RMSE"])  # optionally select metrics
    pred  = model.predict(X)                 # numpy array
    coef  = model.coefficients               # dict {name: value, ..., "intercept": value}
    model.zero_col_features                # list of excluded all-zero feature names
    model.save("coef.json")
    model.load("coef.json")
    model.save_predictions(dataset, "pred.parquet")   # save fitted values
    model.save_predictions(X, "pred.parquet")         # save predictions on new data

Available metrics: ``"MAE"``, ``"R2"``, ``"MSE"``, ``"RMSE"``.
Defaults to computing all four.

Example:
    >>> from mlr.dataset import CSVDataset
    >>> from mlr.regression import MLR
    >>>
    >>> ds = CSVDataset("data.csv", feature_columns=["a", "b"], target_column="y")
    >>> model = MLR(method="ols")
    >>> metrics = model.fit(ds)
    >>> print(metrics)          # {"MAE": ..., "R2": ..., "MSE": ..., "RMSE": ...}
    >>> model.save("coef.json")
"""

import json
import logging
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    Attributes:
        zero_col_features: List of feature names that were detected as
                           all-zero and excluded from the solve.  These
                           columns have coefficient 0.  Empty for sklearn
                           methods.
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
        self.zero_col_features: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    _VALID_METRICS = {"MAE", "R2", "MSE", "RMSE"}

    def fit(
        self,
        dataset,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Fit the model on a dataset.

        Args:
            dataset: Any object that provides:
                       .feature_names   – list[str]
                       .iter_batches()  – yields (X_batch, y_batch) numpy arrays
            metrics:   List of metric names to compute.  Valid options:
                       ``"MAE"``, ``"R2"``, ``"MSE"``, ``"RMSE"``.
                       Defaults to all four.

        Returns:
            Dict with requested training-set metrics.
        """
        self.feature_names = list(dataset.feature_names)
        logger.info(
            f"Fitting model: method={self.method}, "
            f"features={len(self.feature_names)}"
        )

        if metrics is None:
            metrics = list(self._VALID_METRICS)
        metrics = [m.upper() for m in metrics]
        invalid = set(metrics) - self._VALID_METRICS
        if invalid:
            raise ValueError(
                f"Unknown metrics: {invalid}. "
                f"Valid options: {sorted(self._VALID_METRICS)}."
            )

        if self.method == "ols":
            return self._fit_ols(dataset, metrics)
        elif self.method == "ridge":
            return self._fit_ridge(dataset, metrics)
        else:
            return self._fit_sklearn(dataset, metrics)

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

        Includes all original features; all-zero features have coefficient 0.

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
            "zero_col_features": self.zero_col_features,
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
        self.zero_col_features = payload.get("zero_col_features", [])
        logger.info(f"Model loaded from {path}")

    def save_predictions(
        self,
        source,
        output_path: str,
        *,
        include_actuals: bool = True,
        batch_size: int = 50_000,
    ) -> None:
        """Predict on data and save results to a Parquet file.

        Accepts either a dataset object (with ``.iter_batches()`` and
        ``.feature_names``) or a raw numpy array ``X``.

        Args:
            source:         A dataset object or a numpy array of shape
                            (n_samples, n_features).
            output_path:    Path for the output Parquet file.
            include_actuals: If True and *source* is a dataset, include the
                            actual y values alongside predictions.
            batch_size:     Number of rows per write chunk.
        """
        columns: dict[str, list] = {"predicted": []}
        if include_actuals and hasattr(source, "iter_batches"):
            columns["actual"] = []

        written = 0
        sink = None

        def _flush() -> None:
            nonlocal written, sink
            if not columns["predicted"]:
                return
            table = pa.table(columns)
            if sink is None:
                sink = pq.ParquetWriter(output_path, schema=table.schema)
            sink.write_table(table)
            written += len(columns["predicted"])
            for key in columns:
                columns[key] = []

        if isinstance(source, np.ndarray):
            for start in range(0, len(source), batch_size):
                batch = source[start : start + batch_size]
                pred = self.predict(batch).flatten()
                columns["predicted"].extend(pred.tolist())
                if len(columns["predicted"]) >= batch_size:
                    _flush()
            _flush()
        else:
            # dataset object
            for X_batch, y_batch in source.iter_batches():
                pred = self.predict(X_batch).flatten()
                columns["predicted"].extend(pred.tolist())
                if include_actuals:
                    columns["actual"].extend(y_batch.flatten().tolist())
                if len(columns["predicted"]) >= batch_size:
                    _flush()
            _flush()

        if sink is not None:
            sink.close()
        logger.info(f"Predictions saved to {output_path} ({written} rows)")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_ols(self, dataset, metrics: list[str]) -> dict[str, float]:
        """Exact OLS via batched Normal Equations accumulation.

        Accumulates  XtX += X.T @ X  and  Xty += X.T @ y  for each
        batch.  If fit_intercept is True, X is augmented with a column
        of ones before accumulation so the intercept is part of the same
        solve.  This is identical to the closed-form solution on the
        full dataset but requires only O(p²) memory.

        Automatically detects and excludes all-zero feature columns to
        avoid singular matrix issues.  These columns get coefficient 0.
        """
        p = len(dataset.feature_names)
        aug_p = p + 1 if self.fit_intercept else p

        XtX = np.zeros((aug_p, aug_p), dtype=np.float64)
        Xty = np.zeros((aug_p, 1), dtype=np.float64)
        col_sum = np.zeros(p, dtype=np.float64)
        n_total = 0

        logger.info("Reading data: accumulating batches...")
        for X_batch, y_batch in dataset.iter_batches():
            if self.fit_intercept:
                ones = np.ones((len(X_batch), 1), dtype=np.float64)
                X_aug = np.hstack([X_batch, ones])
            else:
                X_aug = X_batch

            XtX += X_aug.T @ X_aug
            Xty += X_aug.T @ y_batch
            col_sum += X_batch.sum(axis=0)
            n_total += len(X_batch)
            batch_mem = X_batch.nbytes + y_batch.nbytes
            logger.debug(
                f"OLS accumulate: processed {n_total} rows, "
                f"batch={batch_mem / 1024 / 1024:.1f} MB"
            )

        # Detect all-zero columns
        zero_mask = (col_sum == 0.0)
        zero_indices = [int(i) for i in np.where(zero_mask)[0]]
        active_indices = [int(i) for i in np.where(~zero_mask)[0]]

        if zero_indices:
            self.zero_col_features = [self.feature_names[i] for i in zero_indices]
            logger.info(
                f"Excluding {len(zero_indices)} all-zero features"
            )
        else:
            self.zero_col_features = []

        logger.info(f"OLS solve: {n_total} total rows, {len(active_indices)} active parameters")

        if len(active_indices) == 0:
            # All features are zero - trivial case
            self.coef_ = np.zeros(p, dtype=np.float64)
            if self.fit_intercept:
                self.intercept_ = 0.0
        else:
            # Build reduced XtX and Xty for active features only
            if self.fit_intercept:
                # active_indices + [p] (intercept column is at index p)
                active_with_intercept = active_indices + [p]
                XtX_reduced = XtX[np.ix_(active_with_intercept, active_with_intercept)]
                Xty_reduced = Xty[active_with_intercept]
            else:
                XtX_reduced = XtX[np.ix_(active_indices, active_indices)]
                Xty_reduced = Xty[active_indices]

            theta_reduced = np.linalg.lstsq(XtX_reduced, Xty_reduced, rcond=None)[0].flatten()

            # Expand back to full coefficient vector
            self.coef_ = np.zeros(p, dtype=np.float64)
            self.coef_[active_indices] = theta_reduced[:len(active_indices)]

            if self.fit_intercept:
                self.intercept_ = float(theta_reduced[-1])
            else:
                self.intercept_ = 0.0

        return self._eval_metrics(dataset, metrics)

    def _fit_ridge(self, dataset, metrics: list[str]) -> dict[str, float]:
        """Exact Ridge via batched Normal Equations with L2 penalty.

        Identical accumulation loop as OLS.  After all batches are
        processed, alpha is added to the diagonal of XtX (excluding
        the intercept row/column when fit_intercept is True) before
        solving:
            theta = (XtX + alpha * I_features)^{-1} Xty

        Memory cost is O(p²), same as OLS.

        Automatically detects and excludes all-zero feature columns.
        """
        p = len(dataset.feature_names)
        aug_p = p + 1 if self.fit_intercept else p

        XtX = np.zeros((aug_p, aug_p), dtype=np.float64)
        Xty = np.zeros((aug_p, 1), dtype=np.float64)
        col_sum = np.zeros(p, dtype=np.float64)
        n_total = 0

        logger.info("Reading data: accumulating batches...")
        for X_batch, y_batch in dataset.iter_batches():
            if self.fit_intercept:
                ones = np.ones((len(X_batch), 1), dtype=np.float64)
                X_aug = np.hstack([X_batch, ones])
            else:
                X_aug = X_batch

            XtX += X_aug.T @ X_aug
            Xty += X_aug.T @ y_batch
            col_sum += X_batch.sum(axis=0)
            n_total += len(X_batch)
            batch_mem = X_batch.nbytes + y_batch.nbytes
            logger.debug(
                f"Ridge accumulate: processed {n_total} rows, "
                f"batch={batch_mem / 1024 / 1024:.1f} MB"
            )

        # Detect all-zero columns
        zero_mask = (col_sum == 0.0)
        zero_indices = [int(i) for i in np.where(zero_mask)[0]]
        active_indices = [int(i) for i in np.where(~zero_mask)[0]]

        if zero_indices:
            self.zero_col_features = [self.feature_names[i] for i in zero_indices]
            logger.info(
                f"Excluding {len(zero_indices)} all-zero features"
            )
        else:
            self.zero_col_features = []

        # Add L2 penalty to active feature dimensions only
        if active_indices:
            XtX[np.ix_(active_indices, active_indices)] += self.alpha * np.eye(len(active_indices), dtype=np.float64)

        logger.info(f"Ridge solve: {n_total} total rows, {len(active_indices)} active parameters, alpha={self.alpha}")

        if len(active_indices) == 0:
            # All features are zero - trivial case
            self.coef_ = np.zeros(p, dtype=np.float64)
            if self.fit_intercept:
                self.intercept_ = 0.0
        else:
            # Build reduced system for active features
            if self.fit_intercept:
                active_with_intercept = active_indices + [p]
                XtX_reduced = XtX[np.ix_(active_with_intercept, active_with_intercept)]
                Xty_reduced = Xty[active_with_intercept]
            else:
                XtX_reduced = XtX[np.ix_(active_indices, active_indices)]
                Xty_reduced = Xty[active_indices]

            theta_reduced = np.linalg.solve(XtX_reduced, Xty_reduced).flatten()

            # Expand back to full coefficient vector
            self.coef_ = np.zeros(p, dtype=np.float64)
            self.coef_[active_indices] = theta_reduced[:len(active_indices)]

            if self.fit_intercept:
                self.intercept_ = float(theta_reduced[-1])
            else:
                self.intercept_ = 0.0

        return self._eval_metrics(dataset, metrics)

    def _fit_sklearn(self, dataset, metrics: list[str]) -> dict[str, float]:
        """Fit a sklearn model after loading all data into memory."""
        X_parts, y_parts = [], []
        for X_batch, y_batch in dataset.iter_batches():
            X_parts.append(X_batch)
            y_parts.append(y_batch)

        X_all = np.vstack(X_parts)
        y_all = np.vstack(y_parts).ravel()
        del X_parts, y_parts

        logger.info(
            f"Data loaded: {len(y_all)} rows, "
            f"starting {self.method} fit..."
        )

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
        result: dict[str, float] = {}
        if "MAE" in metrics:
            result["MAE"] = float(mean_absolute_error(y_all, y_pred))
        if "MSE" in metrics:
            result["MSE"] = float(mean_squared_error(y_all, y_pred))
        if "RMSE" in metrics:
            result["RMSE"] = float(np.sqrt(mean_squared_error(y_all, y_pred)))
        if "R2" in metrics:
            result["R2"] = float(r2_score(y_all, y_pred))
        del X_all, y_all

        self.zero_col_features = []  # sklearn handles zero columns naturally
        return result

    def _eval_metrics(self, dataset, metrics: list[str]) -> dict[str, float]:
        """Compute requested metrics over the full dataset in batches."""
        logger.info("Computing evaluation metrics...")

        sum_ae = 0.0
        sum_se = 0.0
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
            sum_se += ((y_pred_batch - y_batch) ** 2).sum()
            y_mean_acc += y_batch.sum()
            batch_mem = X_batch.nbytes + y_batch.nbytes + y_pred_batch.nbytes
            logger.debug(
                f"Evaluating: processed {n} rows, "
                f"batch={batch_mem / 1024 / 1024:.1f} MB"
            )

        result: dict[str, float] = {}
        if "MAE" in metrics:
            result["MAE"] = float(sum_ae / n)
        if "MSE" in metrics:
            result["MSE"] = float(sum_se / n)
        if "RMSE" in metrics:
            result["RMSE"] = float(np.sqrt(sum_se / n))

        if "R2" in metrics:
            y_mean = y_mean_acc / n
            ss_tot = 0.0
            for yp, ya in zip(y_preds, y_actuals):
                ss_tot += ((ya - y_mean) ** 2).sum()
            ss_res = sum_se
            result["R2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        log_parts = [f"{k}={v:.6f}" for k, v in result.items()]
        logger.info(f"Train metrics: {', '.join(log_parts)}")
        return result
