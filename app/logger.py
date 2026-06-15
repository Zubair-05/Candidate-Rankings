"""
Centralised logging for the candidate ranking system.

Provides:
- Structured JSON log lines in production (LOG_FORMAT=json)
- Human-readable coloured output in development (LOG_FORMAT=text, the default)
- A context manager and decorator for timing any code block with automatic
  start/end/latency log lines
- get_logger(name): drop-in replacement for logging.getLogger that always
  returns a logger bound to the configured level and format

Usage
-----
    from app.logger import get_logger, log_latency

    logger = get_logger(__name__)

    # Time a block:
    with log_latency(logger, "layer1_retrieval", candidates=2000):
        result = retrieval.retrieve(query)

    # Time a single call inline (returns the value):
    result = log_latency(logger, "bm25_index_build").run(build_bm25, corpus)

Environment variables
---------------------
    LOG_LEVEL   DEBUG | INFO | WARNING | ERROR   (default: INFO)
    LOG_FORMAT  text | json                       (default: text)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypeVar

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

_LEVEL: int  = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
_FORMAT: str = os.getenv("LOG_FORMAT", "text").lower()

# ANSI colour codes — used only in text mode
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class _TextFormatter(logging.Formatter):
    """
    Coloured single-line format for development:
        2026-06-15 12:34:56.789  INFO  app.services.retrieval — message  [key=val ...]
    """
    def format(self, record: logging.LogRecord) -> str:
        ts    = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname.ljust(8)
        colour = _COLOURS.get(record.levelname, "")
        name  = record.name

        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        # Extra structured fields attached via logger.info(..., extra={...})
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__ and not k.startswith("_")
            and k not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            )
        }
        extra_str = ""
        if extras:
            extra_str = "  " + _DIM + "  ".join(f"{k}={v}" for k, v in extras.items()) + _RESET

        return (
            f"{_DIM}{ts}{_RESET}  "
            f"{colour}{_BOLD}{level}{_RESET}  "
            f"{_DIM}{name}{_RESET} — "
            f"{msg}"
            f"{extra_str}"
        )


class _JsonFormatter(logging.Formatter):
    """
    Newline-delimited JSON for production log aggregators (Loki, Cloud Logging, Datadog).
    Every line is a valid JSON object — machine-parseable, no colour codes.
    """
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Attach any extra= fields
        for k, v in record.__dict__.items():
            if k not in logging.LogRecord.__dict__ and not k.startswith("_") and k not in payload and k not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                payload[k] = v

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Root handler — configured once
# ---------------------------------------------------------------------------

def _configure_root() -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. by uvicorn)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter() if _FORMAT == "json" else _TextFormatter())
    root.addHandler(handler)
    root.setLevel(_LEVEL)

    # Silence chatty third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_root()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Drop-in replacement for logging.getLogger(name).
    Returns a logger that inherits the root handler configured above.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_LEVEL)
    return logger


# ---------------------------------------------------------------------------
# Latency context manager
# ---------------------------------------------------------------------------

@contextmanager
def log_latency(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    **extra_fields: Any,
) -> Generator[dict, None, None]:
    """
    Context manager that logs start, end, and elapsed time for any block.

    Usage:
        with log_latency(logger, "layer1_retrieval", query_len=len(query)) as ctx:
            result = retrieval.retrieve(query)
            ctx["output_count"] = len(result.candidates)   # add fields mid-block

    Log output (text mode):
        12:34:56.100  INFO  app.services — ▶ layer1_retrieval started
        12:34:56.850  INFO  app.services — ✓ layer1_retrieval done  latency_ms=750  query_len=50  output_count=2000
    """
    ctx: dict[str, Any] = {}
    logger.log(level, "▶ %s started", operation, extra=extra_fields or None)
    t0 = time.perf_counter()

    try:
        yield ctx
    except Exception:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.error(
            "✗ %s failed after %.1fms",
            operation, elapsed_ms,
            extra={"latency_ms": elapsed_ms, **extra_fields, **ctx},
            exc_info=True,
        )
        raise
    else:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.log(
            level,
            "✓ %s done",
            operation,
            extra={"latency_ms": elapsed_ms, **extra_fields, **ctx},
        )


# ---------------------------------------------------------------------------
# Latency decorator
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def timed(operation: Optional[str] = None, level: int = logging.INFO) -> Callable[[F], F]:
    """
    Decorator that wraps a function with log_latency.

    Usage:
        @timed("bm25_build")
        def build_bm25_index(corpus): ...

        @timed()   # uses function name
        def score_candidate(candidate, context): ...
    """
    def decorator(fn: F) -> F:
        op_name = operation or fn.__qualname__

        if _is_async(fn):
            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _logger = get_logger(fn.__module__)
                with log_latency(_logger, op_name, level=level):
                    return await fn(*args, **kwargs)
            return async_wrapper  # type: ignore[return-value]
        else:
            @wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                _logger = get_logger(fn.__module__)
                with log_latency(_logger, op_name, level=level):
                    return fn(*args, **kwargs)
            return sync_wrapper  # type: ignore[return-value]

    return decorator


def _is_async(fn: Callable) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# Pipeline summary helper — logs a clean summary table per ranking request
# ---------------------------------------------------------------------------

def log_pipeline_summary(
    logger: logging.Logger,
    request_id: str,
    jd_preview: str,
    layer1_ms: float,
    layer1_count: int,
    layer2_ms: float,
    layer2_count: int,
    layer3_ms: Optional[float] = None,
    layer3_count: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Emits a single structured log line summarising the full pipeline run.
    Useful for dashboards and latency tracking.
    """
    total_ms = layer1_ms + layer2_ms + (layer3_ms or 0.0)

    fields: dict[str, Any] = {
        "request_id":    request_id,
        "jd_preview":    jd_preview[:80].replace("\n", " "),
        "layer1_ms":     round(layer1_ms, 1),
        "layer1_out":    layer1_count,
        "layer2_ms":     round(layer2_ms, 1),
        "layer2_out":    layer2_count,
        "total_ms":      round(total_ms, 1),
    }
    if layer3_ms is not None:
        fields["layer3_ms"]  = round(layer3_ms, 1)
        fields["layer3_out"] = layer3_count
        fields["llm_provider"] = provider
        fields["llm_model"]    = model

    logger.info("pipeline_summary", extra=fields)
