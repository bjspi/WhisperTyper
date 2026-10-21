"""Settings-change log line, the native crash log and the overlay opacity setting."""
from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from app import bootstrap
from app.core.config_diff import describe_changes


def test_changed_settings_are_listed_without_credentials_or_user_texts():
    before = {"play_sounds": True, "gain_db": 0.0, "prompt": "secret prompt", "proxy_url": "",
              "api_endpoint": "http://user:pw@localhost:8000/v1/audio/transcriptions", "same": 1,
              "api_key_profiles": [{"key": "sk-1"}]}
    after = {**before, "play_sounds": False, "gain_db": 6.0, "prompt": "other", "proxy_url": "http://u:p@proxy:8080",
             "api_endpoint": "http://127.0.0.1:9000/v1/audio/transcriptions", "api_key_profiles": [{"key": "sk-2"}]}
    line = describe_changes(before, after)
    assert "play_sounds=True->False" in line and "gain_db=0.0->6.0" in line
    assert "prompt=changed" in line and "proxy_url=changed" in line and "api_key_profiles=changed" in line
    assert "api_endpoint='host:localhost'->'host:127.0.0.1'" in line
    for secret in ("secret", "sk-", "pw", "u:p", "same"):
        assert secret not in line
    assert describe_changes(before, dict(before)) == "none"


def test_crash_log_reports_a_native_crash_of_the_previous_session(tmp_path, monkeypatch, caplog):
    enable = Mock()
    monkeypatch.setattr(bootstrap.faulthandler, "enable", enable)
    monkeypatch.setattr(bootstrap, "_crash_log_stream", None)
    path = tmp_path / bootstrap.CRASH_LOG_NAME
    path.write_text(f"{bootstrap._CRASH_SESSION_MARKER}1 started=x ===\n"
                    f"{bootstrap._CRASH_SESSION_MARKER}2 started=y ===\n"
                    "Fatal Python error: Segmentation fault\nThread 0x1 (most recent call first):\n", encoding="utf-8")
    with caplog.at_level(logging.INFO):
        assert bootstrap.enable_native_crash_log(str(tmp_path)) == str(path)
    assert "previous session crashed natively" in caplog.text and "Segmentation fault" in caplog.text
    assert enable.call_args.kwargs["all_threads"] is True

    # A clean session leaves nothing to report at the next start.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        bootstrap.enable_native_crash_log(str(tmp_path))
    assert "crashed" not in caplog.text
    bootstrap._crash_log_stream.close()
    monkeypatch.setattr(bootstrap, "_crash_log_stream", None)


@pytest.mark.parametrize("value,expected", [(100, 1.0), (75, 0.75), (5, 0.3), (250, 1.0), ("bad", 1.0)])
def test_overlay_opacity_is_clamped(value, expected):
    floating_buttons = pytest.importorskip("app.ui.floating_buttons")
    assert floating_buttons.overlay_opacity({"recording_overlay_opacity": value}) == expected
