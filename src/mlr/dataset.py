"""Unified dataset abstraction for MLR data loading.

Supports multiple data sources with a consistent iterator interface:
- Parquet files (row-group chunked, recommended for large data)
- CSV files (line-batched chunked)
- In-memory numpy arrays or pandas DataFrames

All sources expose the same API:
    dataset.feature_names   -> list[str]
    dataset.iter_batches()  -> Iterator[tuple[np.ndarray, np.ndarray]]

Example – Parquet with external columns:
    >>> ds = ParquetDataset(
    ...     "features.parquet",
    ...     feature_columns=["a", "b"],
    ...     external_columns={"pred": pred_array},
    ...     target_column="pred",   # from external_columns
    ... )

Example – CSV:
    >>> ds = CSVDataset(
    ...     "data.csv",
    ...     feature_columns=["a", "b", "c"],
    ...     target_column="y",
    ...     batch_size=50_000,
    ... )

Example – in-memory:
    >>> ds = MemoryDataset(X, y, feature_names=["a", "b", "c"])
"""

import gc
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetDataset:
    """Chunked dataset backed by a Parquet file.

    Reads one row group at a time; memory usage is bounded by the
    largest single row group regardless of total file size.

    Feature columns and external columns are kept as separate parameters
    for clarity.  ``target_column`` is always a plain string; if the
    target lives in an external array, pass it via ``external_columns``
    and refer to it by name in ``target_column``.

    Args:
        parquet_path:     Path to the Parquet file.
        feature_columns:  Ordered list of column names to use as X.
                          Must all be present in the Parquet file.
        target_column:    Column name for y.  Looked up first in
                          ``external_columns``, then in the Parquet file.
        external_columns: Optional dict of 1-D arrays to merge into each
                          batch.  Keys are column names; values must have
                          length equal to the total row count of the file.
                          Columns here take precedence over file columns
                          of the same name.
        index_filter:     Optional list of physical row positions (0-based)
                          to include.  Rows not in this list are skipped.
    """

    def __init__(
        self,
        parquet_path: str,
        feature_columns: list[str],
        target_column: str,
        external_columns: dict[str, np.ndarray | list] | None = None,
        index_filter: list[int] | None = None,
    ) -> None:
        self._pf = pq.ParquetFile(parquet_path)
        self.parquet_path = parquet_path
        self.feature_names: list[str] = list(feature_columns)
        self._target_column = target_column

        # Discover total row count via a minimal read (one column only)
        first_col = self._pf.schema.names[0]
        _tmp = pd.read_parquet(parquet_path, columns=[first_col])
        self._num_rows: int = len(_tmp)
        del _tmp
        gc.collect()

        # Validate and store external columns as aligned Series
        self._ext: dict[str, pd.Series] = {}
        if external_columns:
            for name, arr in external_columns.items():
                values = np.asarray(arr).flatten()
                if len(values) != self._num_rows:
                    raise ValueError(
                        f"External column '{name}' has {len(values)} rows, "
                        f"expected {self._num_rows}."
                    )
                self._ext[name] = pd.Series(
                    values, index=pd.RangeIndex(self._num_rows)
                )

        self._filter_set: set[int] | None = (
            set(index_filter) if index_filter is not None else None
        )

        logger.info(
            f"ParquetDataset: {parquet_path}, {self._num_rows} rows, "
            f"{self._pf.num_row_groups} row groups, "
            f"features={self.feature_names}"
        )

    @property
    def columns(self) -> list[str]:
        """All column names available in the Parquet file."""
        return self._pf.schema.names

    def iter_batches(self) -> "Generator[tuple[np.ndarray, np.ndarray]]":
        """Yield (X, y) numpy arrays one row group at a time."""
        # Determine which columns to read from the file
        file_feature_cols = [
            c for c in self.feature_names if c not in self._ext
        ]
        file_target_col = (
            self._target_column if self._target_column not in self._ext else None
        )

        cols_to_read = list(file_feature_cols)
        if file_target_col and file_target_col not in cols_to_read:
            cols_to_read.append(file_target_col)
        if not cols_to_read:
            cols_to_read = [self._pf.schema.names[0]]

        cursor = 0
        for rg_idx in range(self._pf.num_row_groups):
            chunk = self._pf.read_row_group(rg_idx, columns=cols_to_read).to_pandas()
            n = len(chunk)
            chunk.index = pd.RangeIndex(start=cursor, stop=cursor + n)
            cursor += n

            if self._filter_set is not None:
                chunk = chunk[chunk.index.isin(self._filter_set)].copy()

            if chunk.empty:
                del chunk
                gc.collect()
                continue

            X, y = self._build_arrays(chunk)
            del chunk
            gc.collect()
            yield X, y

    def _build_arrays(
        self, chunk: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble X and y arrays from a chunk, merging external columns."""
        idx = chunk.index
        feature_parts: list[np.ndarray] = []
        for name in self.feature_names:
            if name in self._ext:
                feature_parts.append(
                    self._ext[name].iloc[idx].values.reshape(-1, 1)
                )
            else:
                feature_parts.append(chunk[[name]].values)
        X = np.hstack(feature_parts).astype(np.float64)

        if self._target_column in self._ext:
            y = self._ext[self._target_column].iloc[idx].values.reshape(-1, 1)
        else:
            y = chunk[[self._target_column]].values
        y = y.astype(np.float64)

        return X, y


class CSVDataset:
    """Chunked dataset backed by a CSV file.

    Reads the file in fixed-size line batches using ``pd.read_csv``
    with ``chunksize``.  All feature and target columns must be present
    in the CSV file; external array merging is not supported for CSV
    sources because rows are streamed sequentially without a stable index.

    Args:
        csv_path:         Path to the CSV file.
        feature_columns:  List of column names to use as X.
        target_column:    Column name for y.
        batch_size:       Number of rows per batch (default 100_000).
        index_filter:     Optional list of 0-based row positions to include.
        **read_csv_kwargs: Extra keyword arguments forwarded to
                          ``pd.read_csv`` (e.g. ``sep``, ``dtype``).
    """

    def __init__(
        self,
        csv_path: str,
        feature_columns: list[str],
        target_column: str,
        batch_size: int = 100_000,
        index_filter: list[int] | None = None,
        **read_csv_kwargs,
    ) -> None:
        self.csv_path = csv_path
        self.feature_names: list[str] = list(feature_columns)
        self._target_column = target_column
        self._batch_size = batch_size
        self._filter_set: set[int] | None = (
            set(index_filter) if index_filter is not None else None
        )
        self._read_csv_kwargs = read_csv_kwargs

        logger.info(
            f"CSVDataset: {csv_path}, batch_size={batch_size}, "
            f"features={self.feature_names}"
        )

    def iter_batches(self) -> "Generator[tuple[np.ndarray, np.ndarray]]":
        """Yield (X, y) numpy arrays one batch at a time."""
        usecols = self.feature_names + [self._target_column]
        reader = pd.read_csv(
            self.csv_path,
            usecols=usecols,
            chunksize=self._batch_size,
            **self._read_csv_kwargs,
        )
        row_cursor = 0
        for chunk in reader:
            chunk_len = len(chunk)
            if self._filter_set is not None:
                mask = [
                    p in self._filter_set
                    for p in range(row_cursor, row_cursor + chunk_len)
                ]
                chunk = chunk[mask]
            row_cursor += chunk_len

            if chunk.empty:
                del chunk
                gc.collect()
                continue

            X = chunk[self.feature_names].values.astype(np.float64)
            y = chunk[[self._target_column]].values.astype(np.float64)
            del chunk
            gc.collect()
            yield X, y


class MemoryDataset:
    """Dataset wrapping in-memory numpy arrays or a pandas DataFrame.

    The entire dataset is yielded as a single batch.  Suitable for small
    data or when data is already fully loaded.

    Args:
        X:             Feature matrix (n_samples, n_features) or DataFrame.
        y:             Target vector (n_samples,) or (n_samples, 1).
        feature_names: Column names for X.  Inferred automatically if X
                       is a DataFrame and this argument is omitted.
    """

    def __init__(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
    ) -> None:
        if isinstance(X, pd.DataFrame):
            self.feature_names: list[str] = feature_names or list(X.columns)
            self._X = X.values.astype(np.float64)
        else:
            self._X = np.asarray(X, dtype=np.float64)
            self.feature_names = feature_names or [
                f"x{i}" for i in range(self._X.shape[1])
            ]
        self._y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        logger.info(
            f"MemoryDataset: X={self._X.shape}, y={self._y.shape}, "
            f"features={self.feature_names}"
        )

    def iter_batches(self) -> "Generator[tuple[np.ndarray, np.ndarray]]":
        """Yield the entire dataset as a single batch."""
        yield self._X, self._y
