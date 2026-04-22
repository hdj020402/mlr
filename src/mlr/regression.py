"""Multiple Linear Regression (MLR) module.

Provides a reusable, domain-agnostic MLR engine that supports:
- Fitting via ordinary least squares (sklearn LinearRegression)
- Prediction with fitted or loaded coefficients
- Coefficient save/load in JSON format
- Feature name tracking for interpretability

This module has no domain-specific logic. Feature names and data
are provided by the caller, making it reusable across any application.

Example:
    >>> from mlr.mlr import MLR
    >>> model = MLR(feature_names=['k', 'C', 'N', 'O'])
    >>> error = model.fit(X_train, y_train)
    >>> pred = model.predict(X_test)
    >>> model.save('coefficients.json', extra_info=error)
"""

import numpy as np
import json
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class MLR:
    """Multiple Linear Regression engine.

    Domain-agnostic: accepts numpy arrays and feature names,
    with no assumptions about data origin or semantics.
    """

    def __init__(self, feature_names: list[str] | None = None):
        """
        Args:
            feature_names: Names of the features (for interpretability
                and save/load). Must match the column order of X.
                If None, features are unnamed.
        """
        self.feature_names = feature_names
        self.model: LinearRegression | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Fit the linear regression model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples, 1) or (n_samples,).

        Returns:
            Dict with 'MAE' and 'R2' of the fit on training data.
        """
        logger.info(f"Fitting MLR: X shape {X.shape}, y shape {y.shape}")

        model = LinearRegression()
        model.fit(X, y)

        self.model = model
        self.coef_ = model.coef_.flatten()
        self.intercept_ = model.intercept_.flatten()

        prediction = model.predict(X)
        mae = mean_absolute_error(y, prediction)
        r2 = r2_score(y, prediction)

        logger.info(f"Fit complete: MAE={mae:.6f}, R2={r2:.6f}")

        return {'MAE': mae, 'R2': r2}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted or loaded model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples, 1).

        Raises:
            ValueError: If model is not trained or loaded.
        """
        if self.model is not None:
            prediction = self.model.predict(X)
        elif self.coef_ is not None and self.intercept_ is not None:
            prediction = X @ self.coef_ + self.intercept_
        else:
            raise ValueError("Model not trained or loaded.")

        return prediction.reshape(-1, 1)

    def get_coefficients(self) -> dict[str, float]:
        """Get model coefficients as a named dictionary.

        Returns:
            Dict mapping feature names to coefficient values,
            plus 'c' for the intercept.

        Raises:
            ValueError: If model is not trained or loaded.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model not trained or loaded.")

        coef_dict = {}
        if self.feature_names:
            for name, value in zip(self.feature_names, self.coef_):
                coef_dict[name] = float(value)
        else:
            for i, value in enumerate(self.coef_):
                coef_dict[f'x{i}'] = float(value)
        coef_dict['c'] = float(self.intercept_[0])

        return coef_dict

    def save(self, path: str, extra_info: dict | None = None) -> None:
        """Save model coefficients to a JSON file.

        Args:
            path: Output file path.
            extra_info: Additional info to include (e.g., error metrics).
        """
        result = {'coefficient': self.get_coefficients()}
        if extra_info:
            result['error'] = extra_info

        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model coefficients from a JSON file.

        The JSON file must contain a 'coefficient' key with feature
        names as keys and 'c' for the intercept.

        Args:
            path: Input file path.
        """
        with open(path) as f:
            params = json.load(f)

        coefficient_dict = params['coefficient']
        k_list = [v for k, v in coefficient_dict.items() if k != 'c']
        c = coefficient_dict['c']

        self.coef_ = np.array(k_list)
        self.intercept_ = np.array([c])
        self.feature_names = [k for k in coefficient_dict.keys() if k != 'c']

        logger.info(f"Model loaded from {path}")
