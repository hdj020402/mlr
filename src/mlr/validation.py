"""Model validation utilities: k-fold cross-validation.

Provides a ``CrossValidator`` class that clones an MLR model template
for each fold, fits on the training portion, and exposes per-fold results
(metrics, coefficients, predictions, actuals) through dedicated methods.
Predictions and actuals are lazily computed on first access.
"""

import logging
from collections.abc import Callable

import numpy as np

from .dataset import get_num_rows, make_subset
from .regression import MLR

logger = logging.getLogger(__name__)


class CrossValidator:
    """K-fold cross-validation with lazy prediction / actual access.

    Metrics and coefficients are computed eagerly during :meth:`fit`.
    Predictions and actuals are computed lazily on first call and cached
    for subsequent access.

    Usage::

        cv = CrossValidator(MLR(method="ols"), cv=5, random_state=42)
        cv.fit(dataset)

        # Eager results ― available immediately after fit
        print(cv.metrics_)       # {"fold_0": {"train": {...}, "val": {...}}, ...}
        print(cv.coefficients_)  # {"fold_0": {"a": 1.5, "intercept": 0.3}, ...}

        # Lazy ― first call iterates the data, subsequent calls use cache
        val_preds = cv.predictions("val")   # {"fold_0": ndarray, ...}
        val_y     = cv.actuals("val")       # {"fold_0": ndarray, ...}
    """

    def __init__(
        self,
        model: MLR,
        cv: int = 5,
        metrics: list[str] | None = None,
        random_state: int | None = None,
    ) -> None:
        """Configure the cross-validator.

        Args:
            model:        An :class:`MLR` instance used as a template.  Its
                          hyperparameters (method, alpha, …) are copied to a
                          fresh model for each fold.
            cv:           Number of folds.  Must be >= 2.
            metrics:      Metric names to compute.  Defaults to all four
                          (MAE, R2, MSE, RMSE).
            random_state: Seed for reproducible fold assignments.
        """
        self._method = model.method
        self._alpha = model.alpha
        self._l1_ratio = model.l1_ratio
        self._huber_epsilon = model.huber_epsilon
        self._fit_intercept = model.fit_intercept

        self.cv = cv
        self._metrics_list = metrics
        self._random_state = random_state

        # Populated by fit()
        self._folds: list[dict] | None = None

        # Lazy caches
        self._pred_cache: dict[tuple[int, str], np.ndarray] = {}
        self._actual_cache: dict[tuple[int, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, dataset) -> "CrossValidator":
        """Run k-fold cross-validation.

        For each fold the model template is cloned, fitted on the
        training portion, and evaluated on both train and validation
        subsets.

        Args:
            dataset: Any dataset object (ParquetDataset, CSVDataset,
                     or MemoryDataset).

        Returns:
            ``self`` so that ``fit`` can be chained.

        Raises:
            ValueError: If ``cv`` < 2.
        """
        if self.cv < 2:
            raise ValueError(f"cv must be at least 2, got {self.cv}")

        n = get_num_rows(dataset)
        rng = np.random.RandomState(self._random_state)
        indices = rng.permutation(n)

        fold_sizes = np.full(self.cv, n // self.cv, dtype=int)
        fold_sizes[: n % self.cv] += 1

        self._folds = []
        start = 0
        for fold in range(self.cv):
            end = start + fold_sizes[fold]
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            start = end

            train_ds = make_subset(dataset, train_idx.tolist())
            val_ds = make_subset(dataset, val_idx.tolist())

            fold_model = MLR(
                method=self._method,
                alpha=self._alpha,
                l1_ratio=self._l1_ratio,
                huber_epsilon=self._huber_epsilon,
                fit_intercept=self._fit_intercept,
            )
            fold_model.fit(train_ds)
            train_eval = fold_model.evaluate(train_ds, self._metrics_list)
            val_eval = fold_model.evaluate(val_ds, self._metrics_list)

            self._folds.append({
                "model": fold_model,
                "train_ds": train_ds,
                "val_ds": val_ds,
                "metrics": {
                    "train": train_eval,
                    "val": val_eval,
                },
            })

            logger.info(
                f"Fold {fold}/{self.cv}: "
                + "  ".join(f"train_{k}={v:.6f}" for k, v in train_eval.items())
                + "  "
                + "  ".join(f"val_{k}={v:.6f}" for k, v in val_eval.items())
            )

        return self

    @property
    def metrics_(self) -> dict[str, dict]:
        """Per-fold metrics computed during :meth:`fit`.

        Returns:
            .. code-block:: python

                {
                    "fold_0": {
                        "train": {"MAE": 0.08, "R2": 0.99, ...},
                        "val":   {"MAE": 0.09, "R2": 0.98, ...},
                    },
                    "fold_1": {...},
                    ...
                }
        """
        self._check_fitted()
        return {f"fold_{i}": f["metrics"] for i, f in enumerate(self._folds)}

    @property
    def coefficients_(self) -> dict[str, dict[str, float]]:
        """Per-fold coefficient dictionaries.

        Returns:
            .. code-block:: python

                {
                    "fold_0": {"a": 1.23, "b": 2.0, "intercept": 0.3},
                    "fold_1": {...},
                    ...
                }
        """
        self._check_fitted()
        return {f"fold_{i}": f["model"].coefficients for i, f in enumerate(self._folds)}

    @property
    def fold_models_(self) -> list[MLR]:
        """Fitted :class:`MLR` instances, one per fold."""
        self._check_fitted()
        return [f["model"] for f in self._folds]

    def predictions(self, split: str = "val") -> dict[str, np.ndarray]:
        """Per-fold predictions, lazily computed on first access.

        Args:
            split: ``"train"`` or ``"val"``.

        Returns:
            Dict mapping fold keys to column vectors of shape
            ``(n_fold_samples, 1)``.

        Raises:
            ValueError: If *split* is not ``"train"`` or ``"val"``.
        """
        self._check_fitted()
        self._check_split(split)
        return self._gather(
            split, self._pred_cache,
            lambda model, X, _y: model.predict(X),
        )

    def actuals(self, split: str = "val") -> dict[str, np.ndarray]:
        """Per-fold actual target values, lazily computed on first access.

        Args:
            split: ``"train"`` or ``"val"``.

        Returns:
            Dict mapping fold keys to column vectors of shape
            ``(n_fold_samples, 1)``.

        Raises:
            ValueError: If *split* is not ``"train"`` or ``"val"``.
        """
        self._check_fitted()
        self._check_split(split)
        return self._gather(
            split, self._actual_cache,
            lambda _model, _X, y: y,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._folds is None:
            raise RuntimeError("CrossValidator has not been fitted. Call fit() first.")

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    def _gather(
        self,
        split: str,
        cache: dict[tuple[int, str], np.ndarray],
        collect: Callable[[MLR, np.ndarray, np.ndarray], np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Generic batch collection with per-fold caching.

        Args:
            split:   ``"train"`` or ``"val"``.
            cache:   The cache dict to use (``_pred_cache`` or ``_actual_cache``).
            collect: Callable ``(model, X_batch, y_batch) -> ndarray``.

        Returns:
            Dict ``{fold_key: stacked_ndarray}``.
        """
        result: dict[str, np.ndarray] = {}
        for i, f in enumerate(self._folds):
            cache_key = (i, split)
            if cache_key not in cache:
                ds = f["train_ds"] if split == "train" else f["val_ds"]
                batches = []
                for X_batch, y_batch in ds.iter_batches():
                    batches.append(collect(f["model"], X_batch, y_batch))
                cache[cache_key] = np.vstack(batches)
            result[f"fold_{i}"] = cache[cache_key]
        return result
