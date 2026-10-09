"""Apply the user's logging settings: level, redaction, console noise and the rotating log file."""
from __future__ import annotations

import logging
import logging.handlers
import os
from typing import Any, Mapping, Optional

from app.core import log_queue
from app.core.constants import LOG_FILE_PATH
from app.core.redaction import LOG_REDACTION_STATE


def apply_logging_config(config: Mapping[str, Any],
                         file_handler: Optional[logging.Handler]) -> Optional[logging.Handler]:
    """Reconfigure logging from ``config``; returns the new file handler (or None) to keep."""
    logging.getLogger().setLevel(logging.DEBUG if config["debug_logging"] else logging.INFO)
    # Shared redaction switch, honored by every log site including worker threads.
    LOG_REDACTION_STATE["enabled"] = bool(config.get("redact_transcription_in_log", True))

    # In windowed/pythonw mode stderr is the crash-log file; at DEBUG the console handler would
    # copy every line into it. Keep it at WARNING there — the rotating log file has full detail.
    if os.environ.get("WHISPERTYPER_WINDOWED") == "1":
        for handler in log_queue.sinks():
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.WARNING)

    if file_handler is not None:
        try:
            log_queue.remove_sink(file_handler)
        except Exception:
            pass
    if not config["file_logging"]:
        return None
    try:
        # Daily rotation at midnight; the previous day becomes "WhisperTyper.log.YYYY-MM-DD".
        retention_days = max(0, int(config.get("log_retention_days", 3) or 0))
        handler = logging.handlers.TimedRotatingFileHandler(
            LOG_FILE_PATH, when="midnight", interval=1, backupCount=retention_days,
            encoding="utf-8", delay=True,
        )
        handler.suffix = "%Y-%m-%d"
        handler.setLevel(logging.DEBUG)  # The file always keeps full detail.
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log_queue.add_sink(handler)
        logging.info(f"File logging enabled (daily rotation, keeping {retention_days} days): {LOG_FILE_PATH}")
        return handler
    except Exception as e:
        logging.error(f"Failed to enable file logging: {e}")
        return None
