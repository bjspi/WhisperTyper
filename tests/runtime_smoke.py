"""Runtime smoke test: launches the REAL WhisperTyper app and drives it end to end.

Run manually with:  python tests/runtime_smoke.py

Unlike the unit tests (which cover the Qt-free layers), this exercises the composed
application — startup, tray, hotkey registration, the transcription and LivePrompt
worker pipelines, their error paths, and clean shutdown — against a local fake
OpenAI-compatible API.

It is fully isolated and safe to run while a production instance is running:
  * own USERPROFILE/HOME  -> own config + logs (never touches ~/.WhisperTyper)
  * own TMP/TEMP          -> own single-instance lock + recordings directory
  * offscreen Qt platform -> no visible windows or tray icon
  * seeded hotkeys        -> combos that don't collide with the defaults
  * microphone/speaker streams are never opened; the system clipboard is snapshotted and restored

Requires the full runtime environment (PyQt6, pyaudio, ...), so it is NOT part of CI.
It deliberately has no ``test_`` prefix — pytest must never collect it — and the whole
run lives behind ``__main__`` so importing this module has no side effects.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from unittest.mock import Mock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeAPI(BaseHTTPRequestHandler):
    """Answers transcription/chat requests; /fail paths and FAILME payloads get HTTP 500."""

    last_chat_body = b""
    authorization_headers: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """Serve one fake transcription/chat response (500 for /fail or FAILME payloads)."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        FakeAPI.authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path.endswith("/fail") or b"FAILME" in body:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "simulated failure"}')
            return
        if "audio/transcriptions" in self.path:
            payload = {"text": "TRANSCRIBED_FAKE_RESULT"}
        else:
            FakeAPI.last_chat_body = body
            payload = {"choices": [{"message": {"content": "REPHRASED_FAKE_RESULT"}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        """Silence per-request logging."""


def _isolate_environment() -> str:
    """Point HOME/TMP at fresh temp dirs (must run before any ``app.*`` import)."""
    iso = tempfile.mkdtemp(prefix="wt_smoke_")
    iso_home = os.path.join(iso, "home")
    iso_tmp = os.path.join(iso, "tmp")
    os.makedirs(iso_home, exist_ok=True)
    os.makedirs(iso_tmp, exist_ok=True)
    os.environ["USERPROFILE"] = iso_home
    os.environ["HOME"] = iso_home
    os.environ["TMP"] = iso_tmp
    os.environ["TEMP"] = iso_tmp
    tempfile.tempdir = iso_tmp  # mkdtemp above cached the real temp dir before the environment switch.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return iso


def _seed_config(iso_home: str, port: int) -> None:
    """Write a config with non-colliding hotkeys, no mic access, and the fake API endpoints."""
    app_data = os.path.join(iso_home, ".WhisperTyper")
    os.makedirs(app_data, exist_ok=True)
    with open(os.path.join(app_data, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "api_key": "sk-test-dummy",
            "api_endpoint": f"http://127.0.0.1:{port}/v1/audio/transcriptions",
            "rephrasing_api_url": f"http://127.0.0.1:{port}/v1/chat/completions",
            "rephrasing_api_key": "sk-test-dummy",
            "rephrasing_model": "fake-model",
            "hotkey": "<ctrl>+<shift>+<f12>",
            "post_rephrase_hotkey": "<ctrl>+<shift>+<f11>",
            "windows_keep_mic_hot": False,      # never touch the microphone in this harness
            "quit_without_confirmation": True,  # allow programmatic quit_app()
            "liveprompt_enabled": False,
            "generic_rephrase_enabled": False,
        }, f)


def main() -> int:
    """Run the full smoke sequence; return a process exit code (0 = all checks passed)."""
    iso = _isolate_environment()
    iso_home = os.path.join(iso, "home")
    iso_tmp = os.path.join(iso, "tmp")
    sys.path.insert(0, PROJECT_ROOT)

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _seed_config(iso_home, port)

    # Snapshot the system clipboard so it can be restored afterwards.
    import copykitten
    try:
        clipboard_before: str | None = copykitten.paste()
    except Exception:
        clipboard_before = None

    # App imports happen only now, after the environment is isolated.
    from PyQt6.QtCore import QPoint, Qt, QTimer
    from PyQt6.QtGui import QColor, QCursor, QFont, QFontDatabase
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit, QPushButton, QStyle, QStyleOptionComboBox

    from app.bootstrap import configure_base_logging
    configure_base_logging()

    from app.application import WhisperTyperApp
    from app.core.api_keys import selected_api_key
    from app.core.constants import CONFIG_SCHEMA_VERSION
    from app.core.netutil import generate_test_wav_bytes
    from app.core.timing import OperationTiming, add_log_handler, flush_timing_logs
    from app.ui.floating_buttons import RecordingPromptOverlay
    from app.ui.tooltip import MouseFollowerTooltip

    qapp = QApplication([sys.argv[0]])
    if sys.platform.startswith("win"):
        # The offscreen platform otherwise renders placeholder glyphs after HOME isolation.
        QFontDatabase.addApplicationFont(os.path.join(os.environ["WINDIR"], "Fonts", "segoeui.ttf"))
        qapp.setFont(QFont("Segoe UI", 9))
    qapp.setQuitOnLastWindowClosed(False)

    results: list[str] = []
    failed = False
    latency_summaries: list[str] = []

    class LatencyCollector(logging.Handler):
        """Collect completed operations after the background logger formats them."""

        def emit(self, record: logging.LogRecord) -> None:
            """Keep summaries only, without changing the application's existing sinks."""
            message = record.getMessage()
            if message.startswith("latency_summary"):
                latency_summaries.append(message)

    add_log_handler(LatencyCollector())

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failed
        if not cond:
            failed = True
        results.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    def clipboard() -> str:
        try:
            return copykitten.paste()
        except Exception:
            return "<unreadable>"

    with patch("app.audio.sound.SoundPlayer.preload"):
        wt = WhisperTyperApp()
    wt.play_sound = lambda _filename: None
    assert os.path.commonpath([wt.recordings.directory, iso_tmp]) == iso_tmp

    ok_wav = os.path.join(iso_tmp, "ok_input.wav")
    fail_wav = os.path.join(iso_tmp, "failing_input.wav")
    with open(ok_wav, "wb") as f:
        f.write(generate_test_wav_bytes())
    with open(fail_wav, "wb") as f:
        f.write(generate_test_wav_bytes() + b"FAILME")  # marker makes the fake server 500

    def poll(predicate: Callable[[], bool], timeout_s: float,
             on_ok: Callable[[], None], on_timeout: Callable[[], None]) -> None:
        """Poll ``predicate`` on the event loop every 150 ms until true or timeout."""
        deadline = time.monotonic() + timeout_s

        def _tick() -> None:
            if predicate():
                on_ok()
            elif time.monotonic() > deadline:
                on_timeout()
            else:
                QTimer.singleShot(150, _tick)
        QTimer.singleShot(150, _tick)

    def step1_startup() -> None:
        """Assert config creation/migration, tray, and hotkey registration."""
        check("startup: config created + migrated",
              wt.config["hotkey"] == "<ctrl>+<shift>+<f12>" and wt.config["config_schema_version"] == CONFIG_SCHEMA_VERSION)
        check("startup: tray icon visible", wt.tray_icon.isVisible())
        menu_actions = [a for a in wt.tray_menu.actions() if not a.isSeparator()]
        check("startup: tray menu populated", len(menu_actions) >= 9, f"{len(menu_actions)} actions")
        transformation_actions = [a for a in menu_actions if a.data() == "t"]
        check("startup: transformation templates tray action", len(transformation_actions) == 1)
        if transformation_actions:
            transformation_actions[0].trigger()
            check(
                "startup: transformation tray action opens its tab",
                wt.tabs.currentWidget() is wt.post_rephrasing_tab,
            )
        check(
            "startup: transformation warning hidden with complete API settings",
            wt.transformations_unavailable_label.isHidden(),
        )
        saved_rephrasing_id = wt.rephrasing_key_profile_selector.currentData()
        wt.rephrasing_key_profile_selector.setCurrentIndex(0)
        check(
            "startup: transformation warning shown without API key",
            not wt.transformations_unavailable_label.isHidden(),
        )
        wt.rephrasing_key_profile_selector.setCurrentIndex(wt.rephrasing_key_profile_selector.findData(saved_rephrasing_id))
        probe_api_key_management()
        probe_automatic_key_selection()
        probe_provider_models()
        check(
            "startup: transcription status includes language",
            wt._transcription_progress_message("de") == "Transkribiere [German]...",
        )
        check(
            "startup: system-positioned recording prompt palette is enabled by default",
            wt.recording_prompt_overlay_system_position_checkbox.isChecked()
            and wt.config["recording_prompt_overlay_system_position"] is True,
        )
        wt._show_recording_prompt_overlay([{"caption": "Polish", "text": "CUSTOM_OVERLAY_PROMPT"}])
        overlay = RecordingPromptOverlay._instance
        mouse_status = MouseFollowerTooltip._instance
        has_system_status_area = sys.platform.startswith("win") or sys.platform == "darwin"
        check(
            "startup: system-positioned prompt palette keeps recording status by the mouse",
            (
                bool(
                    mouse_status
                    and mouse_status.label.text() == wt.translator.tr("recording_running_message")
                )
                if has_system_status_area
                else mouse_status is None
            ),
        )
        check(
            "startup: recording prompt overlay is non-activating",
            bool(overlay and overlay.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus),
        )
        if overlay and sys.platform.startswith("win"):
            screen_geometry = QApplication.primaryScreen().availableGeometry()
            right_gap = screen_geometry.left() + screen_geometry.width() - (overlay.x() + overlay.width())
            bottom_gap = screen_geometry.top() + screen_geometry.height() - (overlay.y() + overlay.height())
            check(
                "startup: recording prompt overlay is positioned by the Windows tray",
                0 <= right_gap <= 24 and 0 <= bottom_gap <= 24,
                f"right_gap={right_gap}, bottom_gap={bottom_gap}",
            )
        overlay_buttons = overlay.findChildren(QPushButton) if overlay else []
        check(
            "startup: recording prompt overlay uses compact labels",
            len(overlay_buttons) >= 2
            and overlay_buttons[0].text() == "Standard"
            and overlay_buttons[1].text() == "Pol"
            and overlay_buttons[1].toolTip() == "Polish",
        )
        if len(overlay_buttons) >= 2:
            overlay_buttons[1].click()
        check(
            "startup: recording prompt overlay updates selection",
            wt.current_recording_prompt == "CUSTOM_OVERLAY_PROMPT",
        )
        wt._abandon_recording_prompt_selection()
        check(
            "startup: closing the prompt palette also closes its mouse status",
            MouseFollowerTooltip._instance is None,
        )
        wt._show_recording_prompt_overlay([{"caption": "Polish", "text": "CUSTOM_OVERLAY_PROMPT"}])
        MouseFollowerTooltip.show_tooltip("Replacement status", 60_000, spinner=True)
        replacement_status = MouseFollowerTooltip._instance
        wt._abandon_recording_prompt_selection()
        check(
            "startup: closing prompt palette preserves a replacement tooltip",
            replacement_status is not None and MouseFollowerTooltip._instance is replacement_status,
        )
        MouseFollowerTooltip.hide_tooltip()
        wt.config["recording_prompt_overlay_system_position"] = False
        QCursor.setPos(QPoint(120, 140))
        wt._show_recording_prompt_overlay([{"caption": "Polish", "text": "CUSTOM_OVERLAY_PROMPT"}])
        mouse_overlay = RecordingPromptOverlay._instance
        check(
            "startup: recording prompt overlay can use the original mouse-relative position",
            bool(mouse_overlay and abs(mouse_overlay.x() - 135) <= 2 and abs(mouse_overlay.y() - 155) <= 2),
            f"overlay=({mouse_overlay.x()},{mouse_overlay.y()})" if mouse_overlay else "missing overlay",
        )
        check(
            "startup: mouse-positioned prompt palette does not duplicate recording status",
            MouseFollowerTooltip._instance is None,
        )
        wt._abandon_recording_prompt_selection()
        wt.config["recording_prompt_overlay_system_position"] = True
        with (
            patch("app.ui.floating_buttons.is_WINDOWS", False),
            patch("app.ui.floating_buttons.is_MACOS", True),
        ):
            mac_overlay = RecordingPromptOverlay(
                prompts=[{"caption": "Polish", "text": "CUSTOM_OVERLAY_PROMPT"}],
                status_text="Recording",
                standard_text="Standard",
                on_selection_changed=lambda _prompt: None,
                use_system_position=True,
                # Deliberately left of center: macOS must still choose top-right.
                system_anchor=QPoint(300, 10),
            )
        mac_screen = QApplication.primaryScreen().availableGeometry()
        mac_right_gap = mac_screen.left() + mac_screen.width() - (mac_overlay.x() + mac_overlay.width())
        mac_top_gap = mac_overlay.y() - mac_screen.top()
        check(
            "startup: macOS system position remains top-right for a left-side menu-bar icon",
            0 <= mac_right_gap <= 24 and 0 <= mac_top_gap <= 24,
            f"right_gap={mac_right_gap}, top_gap={mac_top_gap}",
        )
        mac_overlay.close()
        probe_dropdown_arrows()
        probe_window_resize()
        wt.hide()  # Keep QApplication.quit() from being intercepted by the tray-style closeEvent.
        check("startup: hotkey bindings parsed", len(wt.hotkey_bindings) == 2,
              "; ".join(b["display"] for b in wt.hotkey_bindings))
        step2_recording_stop()

    def probe_dropdown_arrows() -> None:
        """Verify painted arrows on every combo, including table cells, in both themes."""
        original_theme = wt.config["color_theme"]
        for mode in ("light", "dark"):
            wt.config["color_theme"] = mode
            wt.apply_theme()
            qapp.processEvents()
            missing = []
            accent = QColor(wt._theme_palette["accent"])
            for selector in wt.findChildren(QComboBox):
                for index in range(wt.tabs.count()):
                    page = wt.tabs.widget(index)
                    if page.isAncestorOf(selector):
                        wt.tabs.setCurrentWidget(page)
                        qapp.processEvents()
                        break
                option = QStyleOptionComboBox()
                selector.initStyleOption(option)
                rect = selector.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                                       QStyle.SubControl.SC_ComboBoxArrow, selector)
                rendered = selector.grab().toImage()
                ratio = rendered.devicePixelRatio()
                colored = sum(rendered.pixelColor(int(x * ratio), int(y * ratio)) == accent
                              for x in range(rect.left() + 5, rect.right() - 4)
                              for y in range(rect.top() + 4, rect.bottom() - 3))
                if colored < 8:
                    missing.append(selector.objectName() or "API key provider")
                    rendered.save(os.path.join(iso_tmp, f"dropdown-{mode}-{len(missing)}.png"))
            check(f"theme: all dropdowns paint visible arrows in {mode} mode", not missing, ", ".join(missing))
        wt.tabs.setCurrentWidget(wt._api_keys_tab)
        qapp.processEvents()
        provider = wt._api_keys_tab.table.cellWidget(0, 1).findChild(QComboBox)
        QTest.mouseClick(provider, Qt.MouseButton.LeftButton, pos=QPoint(provider.width() - 14, provider.height() // 2))
        qapp.processEvents()
        check("theme: API key provider arrow opens the dropdown", provider.view().isVisible())
        provider.hidePopup()
        wt.config["color_theme"] = original_theme
        wt.apply_theme()
        wt.tabs.setCurrentWidget(wt.transcription_tab)

    def probe_window_resize() -> None:
        """Shrink the real settings window and check prompt allocation and control reachability."""
        original_size = wt.size()
        wt.tabs.setCurrentWidget(wt.transcription_tab)
        wt.resize(760, 1280)
        qapp.processEvents()
        large_prompt_height = wt.prompt_input.height()
        controls_size = (wt.transcription_api_group.size(), wt.recording_group.size())
        prompt = wt.prompt_input.toPlainText()
        wt.resize(760, 1100)
        qapp.processEvents()
        check("resize: prompt absorbs the height change", abs(large_prompt_height - wt.prompt_input.height() - 180) <= 2,
              f"prompt={large_prompt_height}->{wt.prompt_input.height()}")
        check("resize: API and recording controls retain their sizes",
              controls_size == (wt.transcription_api_group.size(), wt.recording_group.size()))
        wt.resize(760, 600)
        qapp.processEvents()
        check("resize: window reaches 600px height", wt.height() == 600, f"height={wt.height()}")
        check("resize: prompt content is preserved", wt.prompt_input.toPlainText() == prompt)
        check("resize: short transcription page scrolls", wt.transcription_scroll_area.verticalScrollBar().maximum() > 0)
        wt.tabs.setCurrentWidget(wt.rephrasing_tab)
        qapp.processEvents()
        from app.core.constants import REPHRASING_MODEL_OPTIONS
        for width in (680, 760):
            wt.resize(width, 600)
            qapp.processEvents()
            editor = wt.rephrasing_model_input.lineEdit()
            needed = max(editor.fontMetrics().horizontalAdvance(model)
                         for models in REPHRASING_MODEL_OPTIONS.values() for model in models)
            check(f"resize: rephrasing model names fit at {width}px window width",
                  editor.contentsRect().width() - 8 >= needed,
                  f"available={editor.contentsRect().width() - 8}, needed={needed}, window={wt.width()}, page={wt.rephrasing_scroll_widget.width()}")
            right = wt.rephrasing_model_input.mapTo(wt.rephrasing_scroll_area.viewport(),
                                                    QPoint(wt.rephrasing_model_input.width(), 0)).x()
            check(f"resize: rephrasing dropdown stays inside the visible page at {width}px",
                  right <= wt.rephrasing_scroll_area.viewport().width())
        wt.rephrasing_scroll_area.ensureWidgetVisible(wt.rephrasing_provider_selector)
        qapp.processEvents()
        check("resize: rephrasing provider dropdown remain usable at 600px", wt.rephrasing_provider_selector.isVisible()
              and wt.rephrasing_provider_selector.height() >= wt.rephrasing_provider_selector.minimumSizeHint().height()
              and wt.rephrasing_provider_selector.mapTo(wt.rephrasing_tab, QPoint(0, wt.rephrasing_provider_selector.height())).y()
              <= wt.rephrasing_tab.height())
        wt.tabs.setCurrentWidget(wt.general_tab)
        qapp.processEvents()
        wt.general_scroll_area.ensureWidgetVisible(wt.play_g_button)
        qapp.processEvents()
        check("resize: short General page scrolls", wt.general_scroll_area.verticalScrollBar().maximum() > 0)
        bottom = wt.play_g_button.mapTo(wt.general_tab, QPoint(0, wt.play_g_button.height())).y()
        check("resize: last General control is reachable", bottom <= wt.general_tab.height(),
              f"last-control-bottom={bottom}, page-height={wt.general_tab.height()}")
        general_controls = [wt.general_layout.itemAt(i).widget() for i in range(wt.general_layout.count())]
        compressed = [widget.objectName() for widget in general_controls if widget is not None
                      and widget.isVisible() and widget.height() < widget.minimumSizeHint().height()]
        check("resize: General controls retain usable heights", not compressed, ", ".join(compressed))
        wt.tabs.setCurrentWidget(wt._api_keys_tab)
        qapp.processEvents()
        secret = wt._api_keys_tab.table.cellWidget(0, 2).findChild(QLineEdit)
        check("resize: API key editor remains usable at 600px", secret.height() >= secret.minimumSizeHint().height()
              and wt._api_keys_tab.groq_rotation.mapTo(wt, QPoint(0, wt._api_keys_tab.groq_rotation.height())).y() <= wt.height(),
              f"input={secret.height()}, minimum={secret.minimumSizeHint().height()}")
        table = wt._api_keys_tab.table
        check("resize: key editor stays within its table row",
              table.cellWidget(0, 2).geometry().bottom() <= table.visualRect(table.model().index(0, 2)).bottom())
        check("resize: save button stays in the window", wt.save_button.mapTo(wt, QPoint(0, wt.save_button.height())).y() <= wt.height())
        wt.resize(original_size)
        wt.tabs.setCurrentWidget(wt.transcription_tab)
        qapp.processEvents()

    def probe_api_key_management() -> None:
        """Edit masked profiles, verify independent selections, save, and test live form keys."""
        tab = wt._api_keys_tab
        table = tab.table
        original_profiles = tab.profiles()
        original_selection = (wt.config["transcription_key_profile_id"], wt.config["rephrasing_key_profile_id"])
        check("keys: table uses theme separators instead of the native grid", not table.showGrid())
        check("keys: central tab is between Transformations and General", wt.tabs.indexOf(tab) == 3 and wt.tabs.indexOf(wt.general_tab) == 4)
        check("keys: legacy identical keys migrated into one profile", len(original_profiles) == 1
              and original_selection[0] == original_selection[1] and "api_key" not in wt.config and "rephrasing_api_key" not in wt.config)
        ids = []
        for key in ("gsk-smoke-a", "gsk-smoke-b"):
            tab.add_button.click()
            row = table.rowCount() - 1
            provider, secret = table.cellWidget(row, 1).findChild(QComboBox), table.cellWidget(row, 2).findChild(QLineEdit)
            assert isinstance(provider, QComboBox) and isinstance(secret, QLineEdit)
            provider.setCurrentIndex(provider.findData("groq"))
            secret.setText(key)
            ids.append(tab.profiles()[-1]["id"])
            check("keys: secret is masked", secret.echoMode() == QLineEdit.EchoMode.Password)
        wt.api_endpoint_input.setText("https://api.groq.com/openai/v1/audio/transcriptions")
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("groq"))
        check("providers: Groq selection sets the chat-completions URL",
              wt.rephrasing_api_url_input.text() == "https://api.groq.com/openai/v1/chat/completions")
        check("providers: manually entered URL updates the provider dropdown", wt.transcription_provider_selector.currentData() == "groq")
        check("providers: both panels select the first matching Groq key automatically",
              wt.transcription_key_profile_selector.currentData() == ids[0] and wt.rephrasing_key_profile_selector.currentData() == ids[0])
        wt.transcription_key_profile_selector.setCurrentIndex(wt.transcription_key_profile_selector.findData(ids[0]))
        wt.rephrasing_key_profile_selector.setCurrentIndex(wt.rephrasing_key_profile_selector.findData(ids[1]))
        check("keys: task selections are independent", wt._ui_api_key("transcription") == "gsk-smoke-a"
              and wt._ui_api_key("rephrasing") == "gsk-smoke-b")
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("groq"))
        check("providers: clicking the current provider retains its key selection", wt.rephrasing_key_profile_selector.currentData() == ids[1])
        tab.add_button.click()
        openai_row = table.rowCount() - 1
        provider = table.cellWidget(openai_row, 1).findChild(QComboBox)
        provider.setCurrentIndex(provider.findData("openai"))
        table.cellWidget(openai_row, 2).findChild(QLineEdit).setText("sk-smoke-openai")
        openai_id = tab.profiles()[-1]["id"]
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("openai"))
        check("providers: OpenAI selection sets the chat-completions URL",
              wt.rephrasing_api_url_input.text() == "https://api.openai.com/v1/chat/completions")
        check("providers: changing provider selects the matching OpenAI profile and filters keys",
              wt.rephrasing_key_profile_selector.currentData() == openai_id
              and wt.rephrasing_key_profile_selector.findData(ids[1]) == -1
              and wt.rephrasing_key_profile_selector.findData(openai_id) > 0)
        wt.rephrasing_key_profile_selector.setCurrentIndex(wt.rephrasing_key_profile_selector.findData(openai_id))
        check("providers: OpenAI profile resolves from the current form", wt._ui_api_key("rephrasing") == "sk-smoke-openai")
        wt.rephrasing_api_url_input.setText(f"http://127.0.0.1:{port}/v1/chat/completions")
        check("providers: custom URLs remain editable with Custom profiles", wt.rephrasing_key_profile_selector.findData(original_profiles[0]["id"]) > 0
              and wt.rephrasing_key_profile_selector.findData(openai_id) == -1)
        check("providers: changes await saving and preserve the model and transcription selection",
              wt.config["rephrasing_api_url"] == f"http://127.0.0.1:{port}/v1/chat/completions"
              and wt.rephrasing_model_input.currentText() == "fake-model"
              and wt.transcription_key_profile_selector.currentData() == ids[0])
        table.selectRow(openai_row)
        tab.remove_button.click()
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("groq"))
        wt.rephrasing_key_profile_selector.setCurrentIndex(wt.rephrasing_key_profile_selector.findData(ids[1]))
        check("keys: provider filters exclude the custom profile", wt.transcription_key_profile_selector.findData(original_profiles[0]["id"]) == -1)
        table.item(1, 0).setText("Renamed Groq key")
        check("keys: renaming retains the selected ID", wt.transcription_key_profile_selector.currentData() == ids[0])
        secret = table.cellWidget(1, 2).findChild(QLineEdit)
        secret.setText("gsk-smoke-a-edited")
        check("keys: preview shows only the first 10 and last 4 characters",
              table.cellWidget(1, 2).findChild(QLabel).text() == "gsk-smoke-••••••ited"
              and wt.transcription_key_profile_selector.currentText().endswith("gsk-smoke-••••••ited")
              and secret.echoMode() == QLineEdit.EchoMode.Password
              and tab.profiles()[1]["key"] == "gsk-smoke-a-edited")
        check("keys: unsaved edit does not affect runtime credentials", selected_api_key(wt.config, "transcription") == "sk-test-dummy")
        with (patch("app.ui.connection_tester.net.run_transcription_connection_test", return_value=("ok", "")) as connection,
              patch("app.ui.connection_tester.QMessageBox.information")):
            wt._connection_tester.test_transcription()
        check("keys: connection test uses the unsaved selected key", connection.call_args.args[1] == "gsk-smoke-a-edited")
        tab.groq_rotation.setChecked(True)
        with patch.object(wt, "_collect_validation_warnings", return_value=[]):
            wt.save_settings()
        with open(os.path.join(iso_home, ".WhisperTyper", "config.json"), encoding="utf-8") as saved:
            persisted = json.load(saved)
        check("keys: save persists profiles, independent choices and rotation", persisted["transcription_key_profile_id"] == ids[0]
              and persisted["rephrasing_key_profile_id"] == ids[1] and persisted["groq_key_rotation"]
              and "api_key" not in persisted and "rephrasing_api_key" not in persisted)
        check("providers: save persists the provider-selected URL", persisted["rephrasing_api_url"] == "https://api.groq.com/openai/v1/chat/completions")
        table.selectRow(1)
        tab.remove_button.click()
        check("keys: deleting a selected key never selects its neighbour", wt.transcription_key_profile_selector.currentData() == ""
              and wt.rephrasing_key_profile_selector.currentData() == ids[1])
        table.selectRow(1)
        tab.remove_button.click()
        wt.api_endpoint_input.setText(f"http://127.0.0.1:{port}/v1/audio/transcriptions")
        wt.rephrasing_api_url_input.setText(f"http://127.0.0.1:{port}/v1/chat/completions")
        wt.transcription_key_profile_selector.setCurrentIndex(wt.transcription_key_profile_selector.findData(original_selection[0]))
        wt.rephrasing_key_profile_selector.setCurrentIndex(wt.rephrasing_key_profile_selector.findData(original_selection[1]))
        tab.groq_rotation.setChecked(False)
        with patch.object(wt, "_collect_validation_warnings", return_value=[]):
            wt.save_settings()

    def probe_automatic_key_selection() -> None:
        """Select saved or first valid profiles only on provider changes, independently per task."""
        tab = wt._api_keys_tab
        table = tab.table
        original_count = table.rowCount()
        saved_transcription = wt.config["transcription_key_profile_id"]
        ids = []
        for key in ("", "invalid\nkey", "gsk-first-valid", "gsk-saved-valid"):
            tab.add_button.click()
            row = table.rowCount() - 1
            provider = table.cellWidget(row, 1).findChild(QComboBox)
            provider.setCurrentIndex(provider.findData("groq"))
            table.cellWidget(row, 2).findChild(QLineEdit).setText(key)
            ids.append(tab.profiles()[-1]["id"])
        wt.config["transcription_key_profile_id"] = ids[-1]
        wt.transcription_provider_selector.setCurrentIndex(wt.transcription_provider_selector.findData("groq"))
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("groq"))
        check("keys: provider switch prefers the saved valid profile", wt.transcription_key_profile_selector.currentData() == ids[-1])
        check("models: plain Groq model names do not trigger provider-label warnings",
              not any("selected model" in warning for warning in wt._collect_validation_warnings(wt.model_dropdown.currentText())))
        check("keys: provider switch skips empty and invalid keys", wt.rephrasing_key_profile_selector.currentData() == ids[2])
        wt.rephrasing_key_profile_selector.setCurrentIndex(wt.rephrasing_key_profile_selector.findData(ids[-1]))
        wt.rephrasing_api_url_input.setText("https://api.groq.com/openai/v1/chat/completions?test=1")
        check("keys: same-provider URL edits preserve manual selection", wt.rephrasing_key_profile_selector.currentData() == ids[-1])
        wt.transcription_provider_selector.setCurrentIndex(wt.transcription_provider_selector.findData("openai"))
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("openai"))
        check("keys: provider without a key leaves both selectors empty", wt.transcription_key_profile_selector.currentData() == ""
              and wt.rephrasing_key_profile_selector.currentData() == "")
        wt.transcription_provider_selector.setCurrentIndex(wt.transcription_provider_selector.findData("groq"))
        check("keys: switching back displays the saved Groq profile", wt.transcription_key_profile_selector.currentData() == ids[-1])
        check("keys: transcription provider change does not alter rephrasing selection", wt.rephrasing_key_profile_selector.currentData() == "")
        for row in range(table.rowCount() - 1, original_count - 1, -1):
            table.selectRow(row)
            tab.remove_button.click()
        check("keys: removing selected profiles still leaves the selection empty", wt.transcription_key_profile_selector.currentData() == "")
        wt.config["transcription_key_profile_id"] = saved_transcription
        wt.api_endpoint_input.setText(wt.config["api_endpoint"])
        wt.rephrasing_api_url_input.setText(wt.config["rephrasing_api_url"])
        check("keys: custom provider switch selects a valid Custom profile", wt._ui_api_key("transcription") == "sk-test-dummy"
              and wt._ui_api_key("rephrasing") == "sk-test-dummy")
        check("providers: manual custom URL updates both provider dropdowns", wt.transcription_provider_selector.currentData() == "custom"
              and wt.rephrasing_provider_selector.currentData() == "custom")
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("openai"))
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("custom"))
        check("providers: choosing Custom prepares an editable empty URL", wt.rephrasing_api_url_input.text() == "")
        wt.rephrasing_api_url_input.setText(wt.config["rephrasing_api_url"])

    def probe_provider_models() -> None:
        """Check provider catalogs, custom/persisted models and real worker request parameters."""
        from app.core.constants import REPHRASING_MODEL_OPTIONS, TRANSCRIPTION_MODEL_OPTIONS
        from app.services.transcription_worker import TranscriptionWorker

        wt.transcription_provider_selector.setCurrentIndex(wt.transcription_provider_selector.findData("groq"))
        check("models: Groq transcription has only Groq suggestions", [wt.model_dropdown.itemText(i) for i in range(wt.model_dropdown.count())]
              == TRANSCRIPTION_MODEL_OPTIONS["groq"])
        check("models: incompatible transcription builtin switches to Groq turbo", wt.model_dropdown.currentText() == "whisper-large-v3-turbo")
        wt._refresh_model_selectors(preserve_saved=True)
        saved_transcription_model = wt.model_dropdown.currentText()
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("openai"))
        check("models: rephrasing provider change preserves the transcription model",
              wt.model_dropdown.currentText() == saved_transcription_model)
        wt.transcription_provider_selector.setCurrentIndex(wt.transcription_provider_selector.findData("openai"))
        check("models: OpenAI transcription has only OpenAI suggestions", [wt.model_dropdown.itemText(i) for i in range(wt.model_dropdown.count())]
              == TRANSCRIPTION_MODEL_OPTIONS["openai"])
        check("models: transcription catalog includes current GPT models without provider suffixes",
              {"gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-mini-transcribe-2025-12-15", "gpt-4o-transcribe-diarize"}
              <= set(TRANSCRIPTION_MODEL_OPTIONS["openai"])
              and all(" (" not in name for names in TRANSCRIPTION_MODEL_OPTIONS.values() for name in names))
        check("models: OpenAI provider switch starts with recommended GPT transcription", wt.model_dropdown.currentText() == "gpt-transcribe"
              and not wt.transcription_temp_slider.isEnabled())
        wt.model_dropdown.setCurrentText("gpt-4o-transcribe-diarize")
        check("models: diarization disables unsupported prompt and temperature", not wt.prompt_input.isEnabled()
              and not wt.transcription_temp_slider.isEnabled())
        wt.model_dropdown.setCurrentText("gpt-4o-transcribe")
        check("models: ordinary transcription restores prompt and temperature", wt.prompt_input.isEnabled() and wt.transcription_temp_slider.isEnabled())
        check("models: GPT transcription has no Whisper prompt limit", "230" not in wt.prompt_token_label.text())
        wt.tabs.setCurrentWidget(wt.transcription_tab)
        qapp.processEvents()
        selector = wt.model_dropdown
        QTest.mouseClick(selector, Qt.MouseButton.LeftButton, pos=QPoint(selector.width() - 11, selector.height() // 2))
        qapp.processEvents()
        check("models: clicking the transcription dropdown arrow opens its suggestions", selector.isEditable()
              and selector.view().isVisible() and selector.view().model().rowCount() == len(TRANSCRIPTION_MODEL_OPTIONS["openai"]))
        selector.hidePopup()
        selector.lineEdit().selectAll()
        QTest.keyClicks(selector.lineEdit(), "my-custom-transcriber")
        check("models: typing a transcription model edits the dropdown directly", selector.currentText() == "my-custom-transcriber")
        with patch("app.ui.connection_tester.net.run_transcription_connection_test", return_value=("ok", "")) as connection:
            wt._connection_tester._run_transcription_connection_test("https://api.openai.com/v1/audio/transcriptions", "test-key", None)
        check("models: transcription connection test uses the typed model", connection.call_args.args[2] == "my-custom-transcriber")
        wt.transcription_provider_selector.setCurrentIndex(wt.transcription_provider_selector.findData("groq"))
        check("models: custom transcription name survives a provider change", wt.model_dropdown.currentText() == "my-custom-transcriber")
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("openai"))
        wt.rephrasing_model_input.setCurrentText("gpt-6-luna")
        check("models: OpenAI chat suggestions and fixed temperature", [wt.rephrasing_model_input.itemText(i) for i in range(wt.rephrasing_model_input.count())]
              == REPHRASING_MODEL_OPTIONS["openai"] and not wt.rephrasing_temp_slider.isEnabled())
        wt.tabs.setCurrentWidget(wt.rephrasing_tab)
        qapp.processEvents()
        selector = wt.rephrasing_model_input
        QTest.mouseClick(selector, Qt.MouseButton.LeftButton, pos=QPoint(selector.width() - 11, selector.height() // 2))
        qapp.processEvents()
        check("models: clicking the rephrasing dropdown arrow opens the OpenAI list", isinstance(selector, QComboBox)
              and selector.view().isVisible() and selector.view().model().rowCount() == len(REPHRASING_MODEL_OPTIONS["openai"]))
        selector.hidePopup()
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("groq"))
        check("models: Groq chat suggestions replace the incompatible OpenAI model", wt.rephrasing_model_input.currentText() == "openai/gpt-oss-120b"
              and [wt.rephrasing_model_input.itemText(i) for i in range(wt.rephrasing_model_input.count())] == REPHRASING_MODEL_OPTIONS["groq"]
              and wt.rephrasing_temp_slider.isEnabled())
        QTest.mouseClick(selector, Qt.MouseButton.LeftButton, pos=QPoint(selector.width() - 11, selector.height() // 2))
        qapp.processEvents()
        check("models: clicking the rephrasing dropdown arrow opens the Groq list", selector.view().isVisible()
              and selector.view().model().rowCount() == len(REPHRASING_MODEL_OPTIONS["groq"])
              and selector.view().model().index(0, 0).data() == "openai/gpt-oss-120b")
        selector.hidePopup()
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("openai"))
        check("models: switching to OpenAI chat selects GPT-6 Luna", wt.rephrasing_model_input.currentText() == "gpt-6-luna")
        wt.rephrasing_model_input.setCurrentText("my-custom-chat-model")
        wt.rephrasing_provider_selector.setCurrentIndex(wt.rephrasing_provider_selector.findData("openai"))
        check("models: editable chat dropdown retains custom names", wt.rephrasing_model_input.currentText() == "my-custom-chat-model")
        wt.config["rephrasing_model"] = "my-custom-chat-model"
        wt._refresh_model_selectors(preserve_saved=True)
        check("models: saved unknown chat model is preserved", wt.rephrasing_model_input.currentText() == "my-custom-chat-model")
        wt.config["rephrasing_model"] = "fake-model"
        wt.api_endpoint_input.setText(wt.config["api_endpoint"])
        wt.rephrasing_api_url_input.setText(wt.config["rephrasing_api_url"])
        wt._refresh_model_selectors(preserve_saved=True)
        wt.transcription_key_profile_selector.setCurrentIndex(wt.transcription_key_profile_selector.findData(wt.config["transcription_key_profile_id"]))
        wt.rephrasing_key_profile_selector.setCurrentIndex(wt.rephrasing_key_profile_selector.findData(wt.config["rephrasing_key_profile_id"]))
        original_model = wt.config["model"]
        wt.model_dropdown.setCurrentText("my-custom-transcriber")
        wt.rephrasing_model_input.setCurrentText("my-custom-chat-model")
        with patch.object(wt, "_collect_validation_warnings", return_value=[]):
            wt.save_settings()
        with open(os.path.join(iso_home, ".WhisperTyper", "config.json"), encoding="utf-8") as saved:
            persisted = json.load(saved)
        check("models: editable chat name is saved as a string", persisted["rephrasing_model"] == "my-custom-chat-model")
        check("models: typed transcription name is saved as a string", persisted["model"] == "my-custom-transcriber")
        reloaded, _changed = wt._get_config_store().load()
        check("models: typed transcription name survives configuration reload", reloaded["model"] == "my-custom-transcriber")
        wt.model_dropdown.setCurrentText(original_model)
        wt.rephrasing_model_input.setCurrentText("fake-model")
        with patch.object(wt, "_collect_validation_warnings", return_value=[]):
            wt.save_settings()

        for model, language in (("gpt-transcribe", "de"), ("gpt-transcribe", ""),
                                ("gpt-4o-transcribe", "de"), ("gpt-4o-mini-transcribe", "de"),
                                ("gpt-4o-mini-transcribe-2025-12-15", "de"),
                                ("gpt-4o-transcribe-diarize", "de"), ("whisper-large-v3-turbo", "de")):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"text": "MODEL_PARAMETER_PROBE"}
            results = []
            worker = TranscriptionWorker("test-key", "https://api.openai.com/v1/audio/transcriptions", ok_wav,
                                         "Context", model, language, 0.0)
            worker.finished.connect(results.append)
            with patch("app.services.transcription_worker.requests.post", return_value=response) as post:
                worker.run()
            worker.timing.finish("model_probe")
            data = post.call_args.kwargs["data"]
            if model.startswith("gpt-4o-transcribe-diarize"):
                parameters_ok = "prompt" not in data and "temperature" not in data and data["chunking_strategy"] == "auto" and data["response_format"] == "json"
            elif model.startswith("gpt-transcribe"):
                parameters_ok = data["prompt"] == "Context" and "temperature" not in data and "language" not in data and data.get("languages[]") == (language or None)
            else:
                parameters_ok = data["prompt"] == "Context" and data["temperature"] == 0.0 and data["language"] == language
            check(f"models: {model} request parameters ({language or 'auto'})", results == ["MODEL_PARAMETER_PROBE"]
                  and data["model"] == model.split(" (")[0] and parameters_ok)

    def step2_recording_stop() -> None:
        """Stop synthesized PCM via the real hotkey signal, without opening a microphone."""
        with wave.open(ok_wav, "rb") as audio:
            wt.recorded_frames = [audio.readframes(audio.getnframes())]
            wt.current_input_samplerate = audio.getframerate()
        wt.config["min_recording_seconds"] = 0
        wt.config["generic_rephrase_enabled"] = True
        wt.is_recording = True
        detected_ns = time.perf_counter_ns() - 5_000_000
        # Deliver to clipboard so the real stop path cannot send paste keys to another application.
        start_worker = wt.start_transcription_worker
        with patch.object(wt, "start_transcription_worker", side_effect=lambda path, **kw: start_worker(
            path, output_mode="clipboard", **kw,
        )):
            wt.hotkey_action_signal.emit("stop_transcription", detected_ns)
        def recording_done() -> None:
            check("stop hotkey: transcription and rephrasing pipeline completed", not wt.is_recording)
            wt.config["generic_rephrase_enabled"] = False
            step2_transcribe()

        poll(lambda: wt.last_transcription == "REPHRASED_FAKE_RESULT", 10,
             recording_done,
             lambda: (check("stop hotkey: transcription finished", False, "timeout"), step2_transcribe()))

    def step2_transcribe() -> None:
        """Drive one file transcription through the worker pipeline into the clipboard."""
        wt.last_transcription = ""
        wt.start_transcription_worker(ok_wav, output_mode="clipboard")
        poll(lambda: wt.last_transcription == "TRANSCRIBED_FAKE_RESULT", 10,
             lambda: (check("transcribe: delivered to clipboard",
                            clipboard() == "TRANSCRIBED_FAKE_RESULT", repr(clipboard())),
                      step3_liveprompt()),
             lambda: (check("transcribe: worker finished", False, "timeout"), step3_liveprompt()))

    def step3_liveprompt() -> None:
        """Drive a LivePrompt-triggered transcription through the rephrasing worker."""
        wt.config["liveprompt_enabled"] = True
        wt.config["liveprompt_trigger_words"] = "prompt,"
        wt.config["liveprompt_strip_trigger"] = True
        wt.on_transcription_finished("Prompt, please write hello world", "clipboard")
        poll(lambda: wt.last_transcription == "REPHRASED_FAKE_RESULT", 10,
             lambda: (check("liveprompt: rephrased text delivered",
                            clipboard() == "REPHRASED_FAKE_RESULT", repr(clipboard())),
                      step4_recording_prompt()),
             lambda: (check("liveprompt: rephrase finished", False, "timeout"), step4_recording_prompt()))

    def step4_recording_prompt() -> None:
        """An explicit recording-palette prompt must override automatic LivePrompting."""
        wt.last_transcription = ""
        wt.config["liveprompt_enabled"] = True
        wt.config["liveprompt_system_prompt"] = "LIVEPROMPT_SHOULD_NOT_WIN"
        wt.on_transcription_finished(
            "prompt, keep this as ordinary transcript text",
            "clipboard",
            "CUSTOM_RECORDING_PROMPT",
        )

        def recording_prompt_done() -> None:
            body = FakeAPI.last_chat_body
            check(
                "recording prompt: explicit choice overrides LivePrompt",
                b"CUSTOM_RECORDING_PROMPT" in body and b"LIVEPROMPT_SHOULD_NOT_WIN" not in body,
                body.decode(errors="replace"),
            )
            probe1_rephrase_failure()

        poll(lambda: wt.last_transcription == "REPHRASED_FAKE_RESULT", 10,
             recording_prompt_done,
             lambda: (check("recording prompt: rephrase finished", False, "timeout"),
                      probe1_rephrase_failure()))

    def probe1_rephrase_failure() -> None:
        """PROBE: a failing rephrase endpoint must fall back to the raw transcription."""
        wt.config["rephrasing_api_url"] = f"http://127.0.0.1:{port}/v1/chat/fail"
        wt.config["liveprompt_strip_trigger"] = False
        wt.on_transcription_finished("prompt, translate this text", "clipboard")
        expected = "prompt, translate this text"
        poll(lambda: wt.last_transcription == expected, 10,
             lambda: (check("PROBE rephrase-500 falls back to raw text",
                            clipboard() == expected, repr(clipboard())),
                      probe2_batch()),
             lambda: (check("PROBE rephrase-500 falls back to raw text", False, "timeout"), probe2_batch()))

    def probe2_batch() -> None:
        """PROBE: a batch with one failing file must skip it and join the rest (no modal)."""
        wt.config["liveprompt_enabled"] = False
        wt._start_batch_transcription([ok_wav, fail_wav, ok_wav])
        poll(lambda: not getattr(wt, "_batch_active", True), 20,
             lambda: (check("PROBE batch skips failing file, joins rest",
                            clipboard() == "TRANSCRIBED_FAKE_RESULT\n\nTRANSCRIBED_FAKE_RESULT",
                            repr(clipboard())),
                      probe3_rotation()),
             lambda: (check("PROBE batch completes despite failure", False, "timeout"), probe3_rotation()))

    def probe3_rotation() -> None:
        """Overlapping workers must retain their different credential snapshots."""
        original_profiles = wt.config["api_key_profiles"]
        original_selection = wt.config["transcription_key_profile_id"]
        wt.config["api_key_profiles"] = [
            {"id": "rotation-a", "name": "First", "provider": "groq", "key": "gsk-smoke-rotation-a"},
            {"id": "rotation-b", "name": "Second", "provider": "groq", "key": "gsk-smoke-rotation-b"},
        ]
        wt.config["transcription_key_profile_id"] = "rotation-a"
        wt.config["groq_key_rotation"] = True
        before = len(FakeAPI.authorization_headers)
        # Only credential classification is patched; both HTTP requests go to the local fake API.
        with patch("app.core.api_keys.provider_for_url", return_value="groq"):
            wt.start_transcription_worker(ok_wav, output_mode="clipboard")
            wt.start_transcription_worker(ok_wav, output_mode="clipboard")
        wt.config["api_key_profiles"][0]["key"] = "gsk-smoke-edited-during-request"

        def rotation_done() -> None:
            check("rotation: overlapping workers use distinct immutable key snapshots", sorted(FakeAPI.authorization_headers[before:])
                  == ["Bearer gsk-smoke-rotation-a", "Bearer gsk-smoke-rotation-b"])
            wt.config["api_key_profiles"] = original_profiles
            wt.config["transcription_key_profile_id"] = original_selection
            wt.config["groq_key_rotation"] = False
            finish()

        poll(lambda: not wt.active_threads, 10, rotation_done,
             lambda: (check("rotation: workers finish", False, "timeout"), finish()))

    def finish() -> None:
        """Shut the app down through its real quit path."""
        operation = OperationTiming("output_failure_probe")
        with patch("app.mixins.transcription_mixin.copykitten.copy", side_effect=RuntimeError("synthetic clipboard failure")):
            try:
                wt._finalize_transcription_output("probe", output_mode="clipboard", timing=operation)
            except RuntimeError:
                check("timings: output failure preserves exception behavior", True)
            else:
                check("timings: output failure preserves exception behavior", False)
        wt.quit_app()

    QTimer.singleShot(400, step1_startup)
    QTimer.singleShot(60_000, qapp.quit)  # watchdog: never hang

    rc = qapp.exec()

    check("timings: background writer drained", flush_timing_logs(3))
    recordings = [line for line in latency_summaries if "source=recording " in line]
    check("timings: recording has stop-to-API and stop-to-output measurements",
          len(recordings) == 1 and all(field in recordings[0] for field in (
              "stop_to_api_ms=", "stop_to_output_ms=", "event_queue_ms=", "file_write_ms=",
              "rephrase_request_ms=",
          )))
    if recordings:
        queue_ms = float(recordings[0].split("event_queue_ms=")[1].split()[0])
        check("timings: hotkey carries the original 64-bit timestamp", queue_ms >= 5)
    check("timings: file jobs do not invent stop measurements", all(
        "stop_to_" not in line for line in latency_summaries if "source=file " in line
    ))
    check("timings: failed request and rephrase fallback have summaries", all(
        any(f"outcome={outcome} " in line for line in latency_summaries)
        for outcome in ("transcription_failed", "rephrase_failed_fallback", "batch_buffered", "output_failed")
    ))
    ids = [line.split("op=")[1].split()[0] for line in latency_summaries]
    check("timings: operations keep separate IDs and finish once", len(ids) >= 8 and len(ids) == len(set(ids)))

    try:
        if clipboard_before is not None:
            copykitten.copy(clipboard_before)
        else:
            copykitten.clear()
    except Exception:
        pass

    print("\n".join(results))
    print(f"[INFO] Qt event loop exited with rc={rc}")
    return 1 if (failed or rc != 0) else 0


if __name__ == "__main__":
    sys.exit(main())
