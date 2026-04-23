"""Data preparation utilities: CSV-to-Parquet converter and row-by-row Parquet writer.

Two tools are provided:

csv_to_parquet
    Converts a CSV file to Parquet in a single streaming pass.  Memory
    usage is bounded by ``batch_size`` rows at any time.  Each batch
    becomes one row group, so ParquetDataset will later read it with
    the same chunk granularity.

ParquetWriter
    Accepts rows one at a time (or in small dicts) and buffers them
    until ``row_group_size`` rows have accumulated, then flushes one
    row group to disk.  Designed for workflows where features are
    extracted file-by-file and written directly to Parquet without
    an intermediate CSV.

    The column schema is fixed at construction time via the ``columns``
    argument, which also determines the column order in the output file.

    Usage::

        with ParquetWriter("output.parquet", columns=["a", "b", "c"]) as writer:
            for source_file in source_files:
                row = extract_features(source_file)  # your own function
                writer.append(row)                   # {"a": 1.2, "b": 3.4, "c": 5.6}

    ``extract_features`` is intentionally not provided here — feature
    extraction is application-specific and left to the caller.
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def csv_to_parquet(
    csv_path: str,
    parquet_path: str,
    batch_size: int = 100_000,
    compression: str = "snappy",
    columns: list[str] | None = None,
    **read_csv_kwargs,
) -> None:
    """Convert a CSV file to Parquet, streaming in fixed-size batches.

    Memory usage is bounded by ``batch_size`` rows at any point.
    Each batch becomes one row group in the output Parquet file,
    so subsequent ``ParquetDataset`` reads use the same chunk size.

    Args:
        csv_path:        Input CSV file path.
        parquet_path:    Output Parquet file path (created or overwritten).
        batch_size:      Rows per batch / row group (default 100_000).
        compression:     Parquet compression codec: "snappy", "gzip",
                         "brotli", or "none" (default "snappy").
        columns:         Subset of columns to include.  None keeps all.
        **read_csv_kwargs: Extra keyword arguments forwarded to
                           ``pd.read_csv`` (e.g. ``sep``, ``dtype``).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError:        If the CSV is empty or has no readable columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    num_batches = 0

    try:
        reader = pd.read_csv(
            csv_path,
            usecols=columns,
            chunksize=batch_size,
            **read_csv_kwargs,
        )
        for chunk in reader:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    parquet_path, table.schema, compression=compression
                )
            writer.write_table(table)
            total_rows += len(chunk)
            num_batches += 1
            logger.info(
                f"Converted batch {num_batches}: {len(chunk)} rows "
                f"(total so far: {total_rows})"
            )

        if writer is None:
            raise ValueError(f"CSV file is empty or unreadable: {csv_path}")

    finally:
        if writer is not None:
            writer.close()

    logger.info(
        f"Done: {csv_path} -> {parquet_path} "
        f"({total_rows} rows, {num_batches} row groups, "
        f"compression={compression})"
    )


class ParquetWriter:
    """Row-by-row Parquet writer with automatic row-group flushing.

    Buffers appended rows in memory and writes one row group to disk
    every ``row_group_size`` rows.  Any remaining buffered rows are
    flushed on ``close()`` or when exiting the context manager.

    Args:
        parquet_path:   Output Parquet file path (created or overwritten).
        columns:        Ordered list of column names that define the schema.
                        Every dict passed to ``append`` must contain exactly
                        these keys (extra keys are ignored, missing keys raise
                        a KeyError).
        row_group_size: Number of rows per row group (default 100_000).
        compression:    Parquet compression codec: "snappy", "gzip",
                        "brotli", or "none" (default "snappy").

    Raises:
        ValueError: On ``close()`` if no rows were appended (empty file).
    """

    def __init__(
        self,
        parquet_path: str,
        columns: list[str],
        row_group_size: int = 100_000,
        compression: str = "snappy",
    ) -> None:
        self._path = Path(parquet_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._columns = list(columns)
        self._row_group_size = row_group_size
        self._compression = compression

        self._buffer: list[dict] = []
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self._total_rows = 0
        self._num_groups = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ParquetWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def append(self, row: dict) -> None:
        """Append one row to the buffer.

        Args:
            row: A dict mapping column names to scalar values.  Must
                 contain all keys declared in ``columns``.

        Flushes a row group to disk automatically when the buffer
        reaches ``row_group_size`` rows.
        """
        self._buffer.append({col: row[col] for col in self._columns})
        if len(self._buffer) >= self._row_group_size:
            self._flush()

    def close(self) -> None:
        """Flush remaining rows and close the underlying Parquet file.

        Must be called exactly once after all rows have been appended,
        unless the instance is used as a context manager (which calls
        this automatically).

        Raises:
            ValueError: If no rows were ever appended.
        """
        if self._buffer:
            self._flush()
        if self._writer is None:
            raise ValueError("ParquetWriter closed with no rows written.")
        self._writer.close()
        logger.info(
            f"ParquetWriter closed: {self._path} "
            f"({self._total_rows} rows, {self._num_groups} row groups)"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Write the current buffer as one row group and clear it."""
        df = pd.DataFrame(self._buffer, columns=self._columns)
        table = pa.Table.from_pandas(df, preserve_index=False)

        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self._path, self._schema, compression=self._compression
            )

        self._writer.write_table(table)
        self._total_rows += len(self._buffer)
        self._num_groups += 1
        logger.debug(
            f"Flushed row group {self._num_groups}: {len(self._buffer)} rows "
            f"(total so far: {self._total_rows})"
        )
        self._buffer.clear()
