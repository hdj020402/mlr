"""Model validation utilities: k-fold cross-validation.

Provides a standalone ``cross_validate`` function that clones an MLR model
template for each fold, fits on the training portion, and returns per-fold
results including coefficients, training metrics, and validation metrics.
"""

import logging

import numpy as np

from .dataset import get_num_rows, make_subset
from .regression import MLR

logger = logging.getLogger(__name__)


def cross_validate(
    model: MLR,
    dataset,
    cv: int = 5,
    metrics: list[str] | None = None,
    random_state: int | None = None,
) -> dict:
    """K-fold cross-validation.

    Args:
        model:        An :class:`MLR` instance used as a template.  Its
                      hyperparameters (method, alpha, …) are copied to a
                      fresh model for each fold.
        dataset:      Any dataset object (ParquetDataset, CSVDataset,
                      or MemoryDataset).
        cv:           Number of folds.  Must be >= 2.
        metrics:      Metric names to compute.  Defaults to all four.
        random_state: Seed for reproducible fold assignments.

    Returns:
        .. code-block:: python

            {
                "fold_0": {
                    "method": "ols",
                    "coefficients": {"a": 1.23, "b": 2.0, "intercept": 0.3},
                    "n_samples": 160,
                    "metrics": {
                        "train": {"MAE": 0.08, "R2": 0.99},
                        "val":   {"MAE": 0.09, "R2": 0.98},
                    },
                },
                "fold_1": { ... },
                ...
            }

    Raises:
        ValueError: If *cv* < 2.
    """
    if cv < 2:
        raise ValueError(f"cv must be at least 2, got {cv}")

    n = get_num_rows(dataset)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)

    fold_sizes = np.full(cv, n // cv, dtype=int)
    fold_sizes[: n % cv] += 1

    result: dict = {}
    start = 0
    for fold in range(cv):
        end = start + fold_sizes[fold]
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        start = end

        train_ds = make_subset(dataset, train_idx.tolist())
        val_ds = make_subset(dataset, val_idx.tolist())

        fold_model = MLR(
            method=model.method,
            alpha=model.alpha,
            l1_ratio=model.l1_ratio,
            huber_epsilon=model.huber_epsilon,
            fit_intercept=model.fit_intercept,
        )
        fold_model.fit(train_ds)
        train_eval = fold_model.evaluate(train_ds, metrics)
        val_eval = fold_model.evaluate(val_ds, metrics)

        key = f"fold_{fold}"
        result[key] = {
            "method": fold_model.method,
            "coefficients": fold_model.coefficients,
            "n_samples": fold_model.n_samples_,
            "metrics": {
                "train": train_eval,
                "val": val_eval,
            },
        }

        logger.info(
            f"Fold {fold}/{cv}: "
            + "  ".join(f"train_{k}={v:.6f}" for k, v in train_eval.items())
            + "  "
            + "  ".join(f"val_{k}={v:.6f}" for k, v in val_eval.items())
        )

    return result
