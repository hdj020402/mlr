# mlr

A memory-efficient, domain-agnostic Multiple Linear Regression toolkit.

- **Exact OLS and Ridge on arbitrarily large data** — Normal Equations accumulated batch-by-batch; memory cost is O(p²) in the number of features, independent of sample size.
- **Pluggable methods** — OLS, Ridge, Lasso, ElasticNet, Huber.
- **Unified data layer** — Parquet (row-group chunked), CSV (line-batched), or in-memory arrays/DataFrames through the same interface.
- **Row-by-row Parquet writer** — build a Parquet file directly from memory without an intermediate CSV.

## Installation

```bash
pip install -e .
```

## Project structure

```
src/mlr/
├── converter.py   # ParquetWriter, csv_to_parquet
├── dataset.py     # ParquetDataset, CSVDataset, MemoryDataset, split_dataset
├── regression.py  # MLR
└── validation.py  # cross_validate
```

---

## Data preparation

### Option A — Write Parquet row by row (recommended)

Use this when features are extracted one record at a time from individual
source files.  `extract_features` is intentionally left to the caller —
feature extraction is application-specific.

```python
from mlr import ParquetWriter

with ParquetWriter(
    "output.parquet",
    columns=["a", "b", "c", "y"],   # defines schema and column order
    row_group_size=100_000,          # rows buffered before each flush
    compression="snappy",
) as writer:
    for source_file in source_files:
        row = extract_features(source_file)  # returns dict
        writer.append(row)
        # e.g. {"a": 1.2, "b": 3.4, "c": 5.6, "y": 0.9}
```

### Option B — Convert an existing CSV

```python
from mlr import csv_to_parquet

csv_to_parquet(
    "raw_data.csv",
    "data.parquet",
    batch_size=100_000,   # rows per row group
    compression="snappy",
)
```

---

## Creating a dataset

All three dataset types share the same `.feature_names` / `.iter_batches()` interface and can be passed directly to `MLR.fit`.

### From Parquet

```python
from mlr import ParquetDataset

ds = ParquetDataset(
    "data.parquet",
    feature_columns=["a", "b", "c"],   # columns to use as X
    target_column="y",                  # column to use as y
)
```

With an external array merged as a feature (e.g. predictions from another model):

```python
ds = ParquetDataset(
    "features.parquet",
    feature_columns=["a", "b"],
    external_columns={"pred": pred_array},   # 1-D array, len == file row count
    target_column="y",
)
```

With the target coming from an external array:

```python
ds = ParquetDataset(
    "features.parquet",
    feature_columns=["a", "b", "c"],
    external_columns={"label": label_array},
    target_column="label",   # resolved from external_columns first
)
```

With a row subset:

```python
ds = ParquetDataset(
    "data.parquet",
    feature_columns=["a", "b", "c"],
    target_column="y",
    index_filter=[0, 1, 2, 100, 200],   # physical row positions (0-based)
)
```

### From CSV

```python
from mlr import CSVDataset

ds = CSVDataset(
    "data.csv",
    feature_columns=["a", "b", "c"],
    target_column="y",
    batch_size=50_000,
)
```

### From memory

```python
from mlr import MemoryDataset

ds = MemoryDataset(X, y, feature_names=["a", "b", "c"])
# feature_names inferred automatically if X is a pd.DataFrame
```

---

## Fitting

```python
from mlr import MLR

model = MLR(method="ols")
model.fit(ds)
print(model.evaluate(ds))      # {"MAE": ..., "R2": ..., "MSE": ..., "RMSE": ...}
print(model.coefficients)      # {"a": ..., "b": ..., "c": ..., "intercept": ...}
```

`feature_names` are taken from the dataset — no need to pass them separately.

### Standalone evaluation

Use `evaluate()` to compute metrics on any dataset without re-fitting:

```python
model.evaluate(some_other_ds, metrics=["MAE", "R2"])
# -> {"MAE": 0.08, "R2": 0.999}
```

Raises `RuntimeError` if called before the model is fitted or loaded.

### Selecting metrics

```python
model.evaluate(ds, metrics=["MAE", "RMSE"])
```

Available metrics: `"MAE"`, `"R2"`, `"MSE"`, `"RMSE"` (default: all four).

### All-zero feature handling

OLS and Ridge automatically detect and exclude all-zero feature columns to avoid singular matrix issues. These columns receive coefficient 0. The excluded feature names are available via `model.zero_col_features`:

```python
print(model.zero_col_features)  # e.g. ["feat_c", "feat_f"] — names of all-zero features
```

### Saving predictions

```python
model.save_predictions(ds, "predictions.parquet")            # fitted values on training data
model.save_predictions(X_new, "new_predictions.parquet")     # predictions on new data
```

---

## Prediction

```python
import numpy as np

X_new = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
pred = model.predict(X_new)   # shape (n, 1)
```

---

## Dataset splitting

Use `split_dataset` to create independent train/validation/test subsets from any dataset type without loading all data into memory.

```python
from mlr import split_dataset

# Train / test split (80/20)
train_ds, test_ds = split_dataset(ds, test_size=0.2, random_state=42)

# Train / validation / test split (70/10/20)
train_ds, val_ds, test_ds = split_dataset(
    ds, test_size=0.2, val_size=0.1, random_state=42,
)
```

For `MemoryDataset` the arrays are sliced directly. For `ParquetDataset` and `CSVDataset` the existing `index_filter` mechanism is used so no data is copied. The returned subsets are independent dataset objects that can be passed to `MLR.fit()` or `MLR.evaluate()`.

```python
# Full workflow: split → fit → evaluate
train_ds, test_ds = split_dataset(ds, test_size=0.2, random_state=42)
model.fit(train_ds)
train_metrics = model.evaluate(train_ds)
test_metrics  = model.evaluate(test_ds)
```

---

## Cross-validation

K-fold cross-validation is provided by `cross_validate`. It clones the model template for each fold, fits on the training portion, and returns per-fold results including coefficients, training metrics, and validation metrics.

```python
from mlr import cross_validate, MLR

model = MLR(method="ols")
scores = cross_validate(model, ds, cv=5, metrics=["MAE", "R2"], random_state=42)

for fold_name, fold_info in scores.items():
    print(fold_name, fold_info["coefficients"], fold_info["metrics"]["val"]["R2"])
# fold_0 {'a': 1.23, 'b': -2.01, 'intercept': 0.30} 0.9991
# fold_1 {'a': 1.25, 'b': -1.98, 'intercept': 0.31} 0.9988
# ...
```

Works with any method, including Lasso and ElasticNet where cross-validation is essential for tuning the regularisation strength.

---

## Save and load

```python
model.save("coef.json", extra_info=model.evaluate(ds))

model2 = MLR()
model2.load("coef.json")
pred2 = model2.predict(X_new)   # identical results
```

Saved JSON format:

```json
{
  "method": "ols",
  "fit_intercept": true,
  "n_samples": 200,
  "coefficients": {"a": 1.23, "b": -0.45, "c": 0.67, "intercept": 0.70},
  "zero_col_features": [],
  "metrics": {"MAE": 0.008, "R2": 0.9999}
}
```

---

## Regression methods

| `method=`      | Description | Memory for fit |
|---------------|-------------|----------------|
| `"ols"`        | Ordinary Least Squares via batched Normal Equations. Exact solution. Auto-excludes all-zero columns. | O(p²) — bounded |
| `"ridge"`      | L2 regularisation via batched Normal Equations. Exact solution. Auto-excludes all-zero columns. | O(p²) — bounded |
| `"lasso"`      | L1 regularisation. Drives irrelevant coefficients to exactly zero. | Full data in RAM |
| `"elasticnet"` | L1 + L2. Combines feature selection and coefficient stability. | Full data in RAM |
| `"huber"`      | Robust regression. Reduces the influence of outliers. | Full data in RAM |

For datasets that do not fit in RAM, use `method="ols"` or `method="ridge"`.

### Key parameters

```python
MLR(
    method="ols",
    alpha=1.0,           # regularisation strength (ridge / lasso / elasticnet)
    l1_ratio=0.5,        # 0 = pure Ridge, 1 = pure Lasso (elasticnet only)
    huber_epsilon=1.35,  # outlier threshold (huber only)
    fit_intercept=True,
)
```

---

## Memory behaviour

| Scenario | Memory usage |
|----------|-------------|
| OLS, N rows, p features | O(p²) — accumulators XᵀX and Xᵀy only |
| Ridge, N rows, p features | O(p²) — same accumulators, alpha added to diagonal before solve |
| Lasso / ElasticNet / Huber | O(N·p) — full data must fit in RAM (iterative solvers) |
| Parquet read per batch | One row group at a time (size set at write time via `row_group_size` / `batch_size`) |
| CSV read per batch | `batch_size` rows at a time |
| `ParquetWriter` buffer | At most `row_group_size` rows at any time |
