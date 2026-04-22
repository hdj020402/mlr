# mlr

A reusable, domain-agnostic Multiple Linear Regression toolkit with memory-efficient chunked parquet data loading.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from mlr import ChunkedParquetLoader, MLR

# 1. Load data from parquet (chunked, memory-efficient)
loader = ChunkedParquetLoader(
    "features.parquet",
    external_columns={"pred": pred_array, "target": target_array},
)

# Discover available columns
print(loader.columns)         # parquet 内部列名
print(loader.external_column_names)  # 外部合并列名

# 2. Load as numpy arrays
X, y = loader.load_array(
    feature_columns=["feature_a", "feature_b"],
    index=[0, 1, 2, 3],          # only load specific rows
    merge_columns=["pred"],       # prepend external columns to X
    target_column="target",       # designate one external column as y
)

# 3. Fit model
model = MLR(feature_names=["pred", "feature_a", "feature_b"])
error = model.fit(X, y)  # returns {'MAE': ..., 'R2': ...}

# 4. Predict
prediction = model.predict(X_test)

# 5. Save / Load coefficients
model.save("coef.json", extra_info=error)
model2 = MLR()
model2.load("coef.json")
prediction2 = model2.predict(X_test)  # identical results
```

## API

### `ChunkedParquetLoader`

Memory-efficient parquet loader that reads data in row groups, with support for index-based filtering and external column merging.

| Method / Property | Description |
|---|---|
| `columns` | Column names in the parquet file |
| `external_column_names` | Names of externally merged columns |
| `iter_chunks(feature_columns, index, merge_columns, target_column)` | Generator yielding filtered `pd.DataFrame` chunks |
| `load_array(feature_columns, index, merge_columns, target_column)` | Returns `(X, y)` as numpy arrays |

**Parameters:**

- `parquet_path` — Path to the parquet file.
- `external_columns` — `dict[str, array_like]`, 1-D arrays aligned row-by-row with the parquet file.
- `feature_columns` — Column names to read from parquet. `None` reads all.
- `index` — Row indices to include. `None` includes all rows.
- `merge_columns` — External column names to prepend to X (in order given).
- `target_column` — One external column name used as y (not included in X).

X column order: `[merge_columns..., feature_columns...]`

### `MLR`

Domain-agnostic multiple linear regression engine.

| Method | Description |
|---|---|
| `fit(X, y)` | Fit OLS regression. Returns `{'MAE', 'R2'}`. |
| `predict(X)` | Predict. Returns array of shape `(n, 1)`. |
| `get_coefficients()` | Returns `dict` mapping feature names to coefficients, plus `'c'` for intercept. |
| `save(path, extra_info)` | Save coefficients to JSON. Optionally include extra metadata. |
| `load(path)` | Load coefficients from JSON (produced by `save`). |

## Project Structure

```
mlr/
├── pyproject.toml
├── README.md
└── src/
    └── mlr/
        ├── __init__.py
        ├── data.py          # ChunkedParquetLoader
        └── regression.py    # MLR
```
