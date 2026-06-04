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
from mlr import cross_validate

# -- 1. Run CV --------------------------------------------------------
scores = cross_validate(MLR(method="ols"), ds, cv=5, random_state=42)

# -- 2. Inspect per-fold results --------------------------------------
for name, fold in scores.items():
    r2_train = fold["metrics"]["train"]["R2"]
    r2_val = fold["metrics"]["val"]["R2"]
    print(f"{name}: train R2={r2_train:.4f}  val R2={r2_val:.4f}  "
          f"a={fold['coefficients']['a']:.2f}  intercept={fold['coefficients']['intercept']:.2f}")
# fold_0: train R2=0.9994  val R2=0.9994  a=1.49  intercept=0.31
# fold_1: train R2=0.9995  val R2=0.9994  a=1.49  intercept=0.31
# ...

# -- 3. Fit final model on all data -----------------------------------
model = MLR(method="ols")
model.fit(ds)
model.save("coef.json", extra_info=model.evaluate(ds))
```

## Lasso with CV-based alpha selection

```python
# -- 1. Try several alpha values --------------------------------------
best_alpha, best_r2 = None, -float("inf")
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
    cv = cross_validate(
        MLR(method="lasso", alpha=alpha), ds, cv=5, random_state=42,
    )
    mean_val_r2 = np.mean([f["metrics"]["val"]["R2"] for f in cv.values()])
    print(f"alpha={alpha:.3f}  mean val R2={mean_val_r2:.4f}")
    if mean_val_r2 > best_r2:
        best_alpha, best_r2 = alpha, mean_val_r2

# -- 2. Fit final model with best alpha -------------------------------
model = MLR(method="lasso", alpha=best_alpha)
model.fit(ds)
print(model.coefficients)
# Zero coefficients indicate features that Lasso dropped.
```
