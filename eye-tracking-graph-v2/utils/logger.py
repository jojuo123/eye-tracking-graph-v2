"""Console/file logging setup."""

import logging
import os
import sys


def get_logger(name: str = "pytorch_template", log_file: str = None, level=logging.INFO):
    """Return a configured logger. Safe to call multiple times with the same
    name (handlers are only attached once)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
