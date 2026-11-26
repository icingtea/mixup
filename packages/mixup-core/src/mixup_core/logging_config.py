"""Shared logging configuration for all mixup entry points.

One consistent, timestamped format so node logs, orchestrator logs, and the REPL are all
parseable the same way. Pure stdlib — no torch.
"""

import logging

# %(asctime)s is "YYYY-MM-DD HH:MM:SS,mmm" (millisecond precision) by default.
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] | %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with the shared timestamped format."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
