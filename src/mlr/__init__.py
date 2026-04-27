import io
import logging
import os
import sys

from .dataset import ParquetDataset, CSVDataset, MemoryDataset
from .regression import MLR
from .converter import csv_to_parquet, ParquetWriter

logger = logging.getLogger(__name__)

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False


class _FlushHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit, bypassing Python buffering."""

    def emit(self, record):
        try:
            msg = self.format(record) + self.terminator
            stream = self.stream
            stream.write(msg)
            stream.flush()
            # Force OS-level flush for nohup/redirected scenarios
            try:
                fd = stream.fileno()
                os.fsync(fd)
            except (OSError, io.UnsupportedOperation, AttributeError):
                pass
        except Exception:
            self.handleError(record)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the mlr package.

    Can be called multiple times to change the level after import.

    Args:
        level: Logging level.  One of "DEBUG", "INFO", "WARNING",
               "ERROR", "CRITICAL".  Default "INFO".
    """
    global _configured

    handler = _FlushHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        _DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT,
    ))
    handler.setLevel(level.upper())

    # Configure all mlr sub-loggers
    for sub in ("mlr", "mlr.dataset", "mlr.regression", "mlr.converter"):
        sub_logger = logging.getLogger(sub)
        sub_logger.handlers = []
        sub_logger.addHandler(handler)
        sub_logger.setLevel(level.upper())
        sub_logger.propagate = False

    _configured = True


# Auto-configure logging at import time with INFO level
setup_logging()
