"""Tests for the log redaction helper in app/core/redaction.py."""
from __future__ import annotations

import pytest

from app.core.redaction import LOG_REDACTION_STATE, redact_for_log


@pytest.fixture(autouse=True)
def reset_redaction_state():
    original = dict(LOG_REDACTION_STATE)
    yield
    LOG_REDACTION_STATE.clear()
    LOG_REDACTION_STATE.update(original)


class TestRedactForLog:
    def test_none_passes_through(self):
        assert redact_for_log(None) is None

    def test_short_text_is_kept(self):
        LOG_REDACTION_STATE.update({"enabled": True, "keep": 10})
        assert redact_for_log("hi there") == "hi there"

    def test_long_text_is_clipped(self):
        LOG_REDACTION_STATE.update({"enabled": True, "keep": 10})
        secret = "this is a very private transcript"
        result = redact_for_log(secret)
        assert result.startswith(secret[:10])
        assert "redacted" in result
        assert secret[15:] not in result

    def test_disabled_redaction_returns_full_text(self):
        LOG_REDACTION_STATE.update({"enabled": False})
        secret = "this is a very private transcript"
        assert redact_for_log(secret) == secret

    def test_non_string_is_coerced(self):
        LOG_REDACTION_STATE.update({"enabled": True, "keep": 10})
        assert redact_for_log(12345) == "12345"
