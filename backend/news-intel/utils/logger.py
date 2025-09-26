# news-intel/utils/logger.py
"""
Unified logger for news-intel project.

Features
--------
- Console handler with color (default).
- Optional JSON logs (structured logging).
- Optional rotating file handler.
- UTC timestamps.
- Configurable via environment variables:
    LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
    LOG_JSON=1       # enable JSON format
    LOG_FILE=path    # log to file (rotating, 10 MB, 5 backups)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict


# ------------------ formatters ------------------

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            base.update(record.extra) # type: ignore
        return json.dumps(base, ensure_ascii=False)


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",   # white
        "INFO": "\033[36m",    # cyan
        "WARNING": "\033[33m", # yellow
        "ERROR": "\033[31m",   # red
        "CRITICAL": "\033[41m" # red bg
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(record.created))
        msg = record.getMessage()
        return f"{color}{ts} [{record.levelname}] {record.name}: {msg}{self.RESET}"


# ------------------ setup ------------------

def get_logger(name: str = "news-intel") -> logging.Logger:
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    use_json = os.getenv("LOG_JSON", "0") in {"1", "true", "True"}
    logfile = os.getenv("LOG_FILE")

    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(level)
    logger.propagate = False

    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = ColorFormatter()

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(level)
    logger.addHandler(ch)

    # optional file handler
    if logfile:
        fh = RotatingFileHandler(logfile, maxBytes=10 * 1024 * 1024, backupCount=5)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)

    return logger


# ------------------ self-test ------------------

if __name__ == "__main__":
    log = get_logger("demo")
    log.debug("debug message")
    log.info("info message with %s", "arg")
    log.warning("warn message")
    try:
        1 / 0 # type: ignore
    except ZeroDivisionError:
        log.exception("oops")