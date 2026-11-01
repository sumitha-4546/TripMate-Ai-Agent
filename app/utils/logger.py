import logging
import sys
from pathlib import Path
from typing import Any, Optional


def setup_logger(name: str = "tripmate", log_level: str = "INFO") -> logging.Logger:
    """Configure structured logger writing to console and daily log file."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        from app.config import settings

        log_file = settings.logs_dir / "tripmate.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass

    return logger


logger = setup_logger()


class TraceLogger:
    """Helper for formatted agent and tool execution traces."""

    @staticmethod
    def agent_event(message: str, request_id: Optional[str] = None) -> None:
        prefix = f"[{request_id}] " if request_id else ""
        logger.info(f"{prefix}[AGENT] {message}")

    @staticmethod
    def tool_call(
        tool_name: str,
        arguments: Any,
        request_id: Optional[str] = None,
    ) -> None:
        prefix = f"[{request_id}] " if request_id else ""
        logger.info(f"{prefix}[TOOL] {tool_name}")
        logger.info(f"{prefix}[ARGS] {arguments}")

    @staticmethod
    def tool_status(
        tool_name: str,
        status: str,
        duration_ms: Optional[float] = None,
        summary: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        prefix = f"[{request_id}] " if request_id else ""
        dur_str = f" in {duration_ms:.1f}ms" if duration_ms is not None else ""
        summary_str = f" - {summary}" if summary else ""
        logger.info(f"{prefix}[STATUS] {status.upper()}{dur_str}{summary_str}")

    @staticmethod
    def error(message: str, request_id: Optional[str] = None) -> None:
        prefix = f"[{request_id}] " if request_id else ""
        logger.error(f"{prefix}[ERROR] {message}")

