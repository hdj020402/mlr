"""Data loading module with chunked parquet reading support.

Provides memory-efficient loading of large parquet files with:
- Row-group based chunked reading
- Index-based row filtering
- External column merging (arbitrary named columns)
- Column name discovery from parquet schema
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


class ChunkedParquetLoader:
    """Memory-efficient chunked loader for parquet files.

    Reads parquet files in row groups, filters by index,
    and merges arbitrary external columns as needed.
    Feature column names are discovered from the parquet schema,
    not hardcoded.

    Example:
        >>> # Merge external 'pred' and 'target' arrays
        >>> loader = ChunkedParquetLoader(
        ...     'data.parquet',
        ...     external_columns={'pred': pred_array, 'target': target_array},
        ... )
        >>> print(loader.columns)  # discover available columns
        ['feature_a', 'feature_b', 'feature_c', ...]
        >>> X, y = loader.load_array(
        ...     feature_columns=['feature_a', 'feature_b'],
        ...     index=[0, 1, 2, 3],
        ...     merge_columns=['pred'],
        ...     target_column='target',
        ... )
    """

    def __init__(
        self,
        parquet_path: str,
        external_columns: Optional[dict[str, Union[np.ndarray, list]]] = None,
    ):
        """
        Args:
            parquet_path: Path to the parquet file containing feature data.
            external_columns: Dict mapping column names to 1-D arrays.
                Each array must have length equal to the number of rows in
                the parquet file. These columns are merged into each chunk
                on demand during iteration.
                Accepts numpy arrays, lists, or any sequence convertible
                via pd.Series.
        """
        self.parquet_path = parquet_path
        self._parquet_file = pq.ParquetFile(parquet_path)

        # Discover full index by reading only the first column
        first_col = self._parquet_file.schema.names[0]
        temp_df = pd.read_parquet(parquet_path, columns=[first_col])
        self.full_index = temp_df.index
        self.num_rows = len(self.full_index)
        del temp_df
        gc.collect()

        logger.info(
            f"Parquet: {parquet_path}, total rows: {self.num_rows}"
        )

        # Align external columns with parquet index
        self._external_series: dict[str, pd.Series] = {}
        if external_columns:
            for name, arr in external_columns.items():
                values = np.asarray(arr).flatten()
                if len(values) != self.num_rows:
                    raise ValueError(
                        f"External column '{name}' has {len(values)} rows, "
                        f"expected {self.num_rows} (parquet row count)."
                    )
                self._external_series[name] = pd.Series(
                    values, index=self.full_index
                )

    @property
    def columns(self) -> list[str]:
        """Available column names in the parquet file."""
        return self._parquet_file.schema.names

    @property
    def external_column_names(self) -> list[str]:
        """Names of externally merged columns."""
        return list(self._external_series.keys())

    def iter_chunks(
        self,
        feature_columns: Optional[list[str]] = None,
        index: Optional[list[int]] = None,
        merge_columns: Optional[list[str]] = None,
        target_column: Optional[str] = None,
    ):
        """Iterate over filtered data chunks.

        Reads the parquet file row-group by row-group, filters rows by
        index, and optionally merges external columns.

        Args:
            feature_columns: Columns to read from parquet.
                If None, reads all columns.
            index: Row indices to include. If None, includes all rows.
            merge_columns: External column names to merge into each chunk.
                Must be keys of the external_columns dict passed at init.
            target_column: An external column name to designate as target.
                Also merged into the chunk. Must be a key of external_columns.

        Yields:
            pd.DataFrame: Filtered chunk with requested columns merged.
        """
        if index is not None:
            filter_set = set(index) & set(self.full_index)
        else:
            filter_set = set(self.full_index)

        # Determine columns to read from parquet
        columns_to_read = feature_columns
        if not columns_to_read:
            # Need at least one column to construct DataFrame with correct row count
            columns_to_read = [self._parquet_file.schema.names[0]]

        # Collect all external columns to merge
        merge_names = set(merge_columns or [])
        if target_column:
            merge_names.add(target_column)

        current_index = 0
        for i in range(self._parquet_file.num_row_groups):
            chunk_df = self._parquet_file.read_row_group(
                i, columns=columns_to_read
            ).to_pandas()
            num_rows = len(chunk_df)
            chunk_df.index = pd.RangeIndex(
                start=current_index, stop=current_index + num_rows
            )
            current_index += num_rows

            filtered_chunk = chunk_df[chunk_df.index.isin(filter_set)].copy()

            if not filtered_chunk.empty:
                # Merge requested external columns
                for name in merge_names:
                    if name in self._external_series:
                        filtered_chunk[name] = self._external_series[name].loc[
                            filtered_chunk.index
                        ].values
                yield filtered_chunk

            del chunk_df, filtered_chunk
            gc.collect()

    def load_array(
        self,
        feature_columns: Optional[list[str]] = None,
        index: Optional[list[int]] = None,
        merge_columns: Optional[list[str]] = None,
        target_column: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Load all data into numpy arrays.

        Reads the parquet file in chunks (memory-efficient I/O)
        and returns the concatenated result as numpy arrays.

        Args:
            feature_columns: Columns to read from parquet as features.
            index: Row indices to include. If None, includes all rows.
            merge_columns: External column names to prepend as feature
                columns (in the order given).
            target_column: External column name to use as target (y).
                Not included in X.

        Returns:
            Tuple of (X, y) where X is the feature matrix and
            y is the target vector (or None if target_column is None).
            X column order: [merge_columns..., feature_columns...]
        """
        all_features = []
        all_targets = []

        for chunk in self.iter_chunks(
            feature_columns=feature_columns,
            index=index,
            merge_columns=merge_columns,
            target_column=target_column,
        ):
            # Build feature columns: [merge_columns..., feature_columns...]
            feature_parts = []
            if merge_columns:
                for name in merge_columns:
                    if name in chunk.columns:
                        feature_parts.append(chunk[[name]].values)
            if feature_columns:
                feature_parts.append(chunk[feature_columns].values)

            if feature_parts:
                all_features.append(np.hstack(feature_parts))

            if target_column and target_column in chunk.columns:
                all_targets.append(chunk[[target_column]].values)

            del chunk
            gc.collect()

        X = np.vstack(all_features) if all_features else np.array([])
        y = np.vstack(all_targets) if all_targets else None

        del all_features, all_targets
        gc.collect()

        logger.info(
            f"Loaded array: X shape {X.shape}, "
            f"y shape {y.shape if y is not None else None}"
        )

        return X, y
