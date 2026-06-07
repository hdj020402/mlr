# Usage

Complete end-to-end examples covering the most common workflows.
All examples use in-memory data for brevity; replace `MemoryDataset` with
`ParquetDataset` or `CSVDataset` for large files.

## Basic fit, evaluate, save

```python
import numpy as np
from mlr import MLR, MemoryDataset

# -- 1. Prepare data --------------------------------------------------
X = np.random.RandomState(42).randn(1000, 5)
coef = np.array([1.5, -2.0, 0.5, 3.0, -1.0]).reshape(-1, 1)
y = X @ coef + 0.3 + np.random.RandomState(43).randn(1000, 1) * 0.1
ds = MemoryDataset(X, y, feature_names=["a", "b", "c", "d", "e"])

# -- 2. Fit -----------------------------------------------------------
model = MLR(method="ols")
model.fit(ds)

# -- 3. Inspect -------------------------------------------------------
print(model.coefficients)
# {"a": 1.50, "b": -1.99, "c": 0.50, "d": 3.01, "e": -1.00, "intercept": 0.30}
print(model.evaluate(ds, metrics=["MAE", "R2"]))
# {"MAE": 0.078, "R2": 0.999}

# -- 4. Save coefficients & metrics -----------------------------------
model.save("coef.json", extra_info=model.evaluate(ds))

# -- 5. Save predictions ----------------------------------------------
model.save_predictions(ds, "predictions.parquet")
```

## Train / test split

```python
from mlr import split_dataset

# -- 1. Split ---------------------------------------------------------
train_ds, test_ds = split_dataset(ds, test_size=0.2, random_state=42)

# -- 2. Fit on training set -------------------------------------------
model = MLR(method="ols")
model.fit(train_ds)

# -- 3. Evaluate on both sets -----------------------------------------
print("train:", model.evaluate(train_ds, metrics=["MAE", "R2"]))
print("test: ", model.evaluate(test_ds, metrics=["MAE", "R2"]))
# train: {"MAE": 0.078, "R2": 0.999}
# test:  {"MAE": 0.077, "R2": 0.999}

# -- 4. Save ----------------------------------------------------------
model.save("coef.json", extra_info=model.evaluate(test_ds))
model.save_predictions(test_ds, "test_predictions.parquet")
```

## K-fold cross-validation

```python
from mlr import CrossValidator

# -- 1. Run CV --------------------------------------------------------
cv = CrossValidator(MLR(method="ols"), cv=5, random_state=42)
cv.fit(ds)

# -- 2. Inspect per-fold results --------------------------------------
for name, metrics in cv.metrics_.items():
    r2_train = metrics["train"]["R2"]
    r2_val = metrics["val"]["R2"]
    coef = cv.coefficients_[name]
    print(f"{name}: train R2={r2_train:.4f}  val R2={r2_val:.4f}  "
          f"a={coef['a']:.2f}  intercept={coef['intercept']:.2f}")
# fold_0: train R2=0.9994  val R2=0.9994  a=1.49  intercept=0.31
# fold_1: train R2=0.9995  val R2=0.9994  a=1.49  intercept=0.31
# ...

# -- 3. Get predictions (lazy — computed on first access) --------------
val_preds = cv.predictions("val")   # {"fold_0": ndarray, ...}
val_y     = cv.actuals("val")       # {"fold_0": ndarray, ...}
# Residual analysis, custom metrics, etc.
for name in cv.metrics_:
    residuals = val_preds[name] - val_y[name]

# -- 4. Fit final model on all data -----------------------------------
model = MLR(method="ols")
model.fit(ds)
model.save("coef.json", extra_info=model.evaluate(ds))
```

## Lasso with CV-based alpha selection

```python
# -- 1. Try several alpha values --------------------------------------
best_alpha, best_r2 = None, -float("inf")
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
    cv = CrossValidator(
        MLR(method="lasso", alpha=alpha), cv=5, random_state=42,
    ).fit(ds)
    mean_val_r2 = np.mean([m["val"]["R2"] for m in cv.metrics_.values()])
    print(f"alpha={alpha:.3f}  mean val R2={mean_val_r2:.4f}")
    if mean_val_r2 > best_r2:
        best_alpha, best_r2 = alpha, mean_val_r2

# -- 2. Fit final model with best alpha -------------------------------
model = MLR(method="lasso", alpha=best_alpha)
model.fit(ds)
print(model.coefficients)
# Zero coefficients indicate features that Lasso dropped.
```
