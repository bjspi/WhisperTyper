"""Log redaction — keep sensitive transcript/prompt text out of log files.

Single responsibility: given a string, return a redacted version per shared state.
Module-level state so worker threads (no config access) redact consistently.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Shared, module-level state so that classes without direct access to the config
# (e.g. TranscriptionWorker running in its own thread) can redact consistently.
LOG_REDACTION_STATE: Dict[str, Any] = {"enabled": True, "keep": 10}


def redact_for_log(text: Any) -> Optional[str]:
    """
    Redacts potentially sensitive text (voice transcripts / prompts) for logging.

    When redaction is enabled, only the first ``keep`` characters are preserved and the
    remainder is replaced with a placeholder, so log files never contain the full transcript.
    """
    if text is None:
        return text
    text = str(text)
    if not LOG_REDACTION_STATE.get("enabled", True):
        return text
    keep = int(LOG_REDACTION_STATE.get("keep", 10))
    if len(text) <= keep:
        return text
    return f"{text[:keep]}…[redacted {len(text) - keep} chars]"
