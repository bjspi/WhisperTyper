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
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeAPI(BaseHTTPRequestHandler):
    """Answers transcription/chat requests; /fail paths and FAILME payloads get HTTP 500."""

    last_chat_body = b""
    last_transcription_body = b""
    authorization_headers: list[str] = []
    protocol_version = "HTTP/1.1"

    def do_HEAD(self) -> None:  # noqa: N802 - http.server API
        """Warm the local transport without including HEAD requests in key-rotation checks."""
        self.send_response(401)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """Serve one fake transcription/chat response (500 for /fail or FAILME payloads)."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        FakeAPI.authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path.endswith("/fail") or b"FAILME" in body:
            self.send_response(500)
            self.send_header("Content-Length", str(len(b'{"error": "simulated failure"}')))
            self.end_headers()
            self.wfile.write(b'{"error": "simulated failure"}')
            return
        if "audio/transcriptions" in self.path:
            FakeAPI.last_transcription_body = body
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
            # Older top-level LivePrompt key: the migration turns it into the instruction entry.
            "liveprompt_enabled": False,
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
    from app.core import log_queue
    from app.core.api_keys import selected_api_key
    from app.core.constants import CONFIG_SCHEMA_VERSION
    from app.core.replacements import Replacements
    from app.core.timing import OperationTiming
    from app.services.netutil import generate_test_wav_bytes
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
    replacement_logs: list[str] = []
    transport_logs: list[str] = []

    class LatencyCollector(logging.Handler):
        """Collect completed operations after the background logger formats them."""

        def emit(self, record: logging.LogRecord) -> None:
            """Keep summaries only, without changing the application's existing sinks."""
            message = record.getMessage()
            if message.startswith("latency_summary"):
                latency_summaries.append(message)
            elif message.startswith("replacements_check"):
                replacement_logs.append(message)
            elif message.startswith("http_transport"):
                transport_logs.append(message)

    log_queue.add_sink(LatencyCollector())

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
    sw = wt.settings
    # Settings tests select official providers; warming must stay on the fake local API.
    original_schedule = wt.warmup.http.schedule
    wt.warmup.http.schedule = lambda _endpoints, proxy, px: original_schedule(
        (f"http://127.0.0.1:{port}/",), proxy, px)
    wt.sound_player.play = lambda _filename: None
    assert os.path.commonpath([wt.ctx.recordings.directory, iso_tmp]) == iso_tmp

    ok_wav = os.path.join(iso_tmp, "ok_input.wav")
    fail_wav = os.path.join(iso_tmp, "failing_input.wav")
    with open(ok_wav, "wb") as f:
        f.write(generate_test_wav_bytes())
    with open(fail_wav, "wb") as f:
        f.write(generate_test_wav_bytes() + b"FAILME")  # marker makes the fake server 500

    def set_instruction(**fields: Any) -> None:
        """Change the instruction (LivePrompt) entry of the live prompt list."""
        entries = wt.ctx.config["post_rephrasing_entries"]
        index = next(i for i, entry in enumerate(entries) if entry.get("kind") == "instruction")
        entries[index] = {**entries[index], **fields}

    check("startup: LivePrompt settings migrated into the inactive instruction entry",
          [entry.get("kind") for entry in wt.ctx.config["post_rephrasing_entries"]][:1] == ["instruction"]
          and wt.ctx.config["post_rephrasing_entries"][0]["enabled"] is False
          and "liveprompt_enabled" not in wt.ctx.config)

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
              wt.ctx.config["hotkey"] == "<ctrl>+<shift>+<f12>" and wt.ctx.config["config_schema_version"] == CONFIG_SCHEMA_VERSION)
        check("startup: tray icon visible", wt.tray.tray_icon.isVisible())
        menu_actions = [a for a in wt.tray.tray_menu.actions() if not a.isSeparator()]
        check("startup: tray menu populated", len(menu_actions) >= 9, f"{len(menu_actions)} actions")
        transformation_actions = [a for a in menu_actions if a.data() == "t"]
        check("startup: transformation templates tray action", len(transformation_actions) == 1)
        if transformation_actions:
            transformation_actions[0].trigger()
            check(
                "startup: transformation tray action opens its tab",
                sw.tabs.currentWidget() is sw.post_rephrasing_tab,
            )
        check(
            "startup: transformation warning hidden with complete API settings",
            sw.transformations_unavailable_label.isHidden(),
        )
        saved_rephrasing_url = sw.rephrasing_api_url_input.text()
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("openai"))
        check(
            "startup: transformation warning and key status shown without a matching API key",
            not sw.transformations_unavailable_label.isHidden()
            and sw.rephrasing_key_status_label.property("key_status") == "missing"
            and not sw.rephrasing_key_add_button.isHidden(),
        )
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("custom"))
        check("providers: switching back to Custom restores the custom URL",
              sw.rephrasing_api_url_input.text() == saved_rephrasing_url)
        probe_api_key_management()
        probe_automatic_key_selection()
        probe_provider_models()
        probe_replacements()
        probe_windows_text_input()
        field = QLineEdit()
        commit_timing = OperationTiming("text_commit_probe")
        commit_timing.mark("stop")

        def paste_into_test_field(char: str) -> bool:
            """Deliver paste to this offscreen Qt widget without emitting OS hotkeys."""
            assert char == "v"
            # Offscreen Qt owns a virtual clipboard; bridge the native text for this test.
            qapp.clipboard().setText(copykitten.paste())
            modifier = Qt.KeyboardModifier.MetaModifier if sys.platform == "darwin" else Qt.KeyboardModifier.ControlModifier
            QTest.keyClick(field, Qt.Key.Key_V, modifier)
            return True

        with patch.object(wt.text_output, "_simulate_key_combination", side_effect=paste_into_test_field):
            check("text commit: real Qt field receives text", wt.text_output.insert("COMMIT_PROBE", timing=commit_timing)
                  and field.text() == "COMMIT_PROBE")
        check("text commit: dispatch ends before settling wait", commit_timing._events["text_commit_end"]
              < commit_timing._events["paste_settle_end"])
        commit_timing.finish("ok")
        wt.text_output._restore_timer.stop()
        wt.text_output.restore_clipboard_now()
        for post_rephrase in (False, True):
            operation = OperationTiming("text_commit_pipeline_probe")
            with patch.object(wt.text_output, "_simulate_key_combination", side_effect=paste_into_test_field):
                if post_rephrase:
                    wt.post_rephrase.on_rephrasing_finished("COMMIT_PROBE", timing=operation)
                else:
                    wt.pipeline.finalize_output("COMMIT_PROBE", timing=operation)
            check(f"text commit: {'rephrasing' if post_rephrase else 'transcription'} passes timing to insertion",
                  "text_commit_end" in operation._events and operation._finished)
            wt.text_output._restore_timer.stop()
            wt.text_output.restore_clipboard_now()
        check(
            "startup: transcription status includes language",
            wt.pipeline._transcription_progress_message("de") == "Transkribiere [German]...",
        )
        check(
            "startup: system-positioned recording prompt palette is enabled by default",
            sw.recording_prompt_overlay_system_position_checkbox.isChecked()
            and wt.ctx.config["recording_prompt_overlay_system_position"] is True,
        )
        wt.recording.show_prompt_palette([{"caption": "Polish", "text": "CUSTOM_OVERLAY_PROMPT"}])
        overlay = RecordingPromptOverlay._instance
        mouse_status = MouseFollowerTooltip._instance
        has_system_status_area = sys.platform.startswith("win") or sys.platform == "darwin"
        check(
            "startup: system-positioned prompt palette keeps recording status by the mouse",
            (
                bool(
                    mouse_status
                    and mouse_status.label.text() == wt.ctx.translator.tr("recording_running_message")
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
        if len(overlay_buttons) >= 2:
            with patch.dict(wt.ctx.config, {"rephrasing_api_url": "https://api.openai.com/v1/chat/completions"}), \
                    patch.object(wt.warmup.http, "schedule") as warmup:
                wt.warmup.http._warm_until = 0
                overlay_buttons[1].click()
                check("HTTP: selecting an overlay prompt immediately prioritizes OpenAI rephrasing warmup",
                      warmup.call_args.args == (("https://api.openai.com/v1/chat/completions", wt.ctx.config["api_endpoint"]),
                                               wt.ctx.config["proxy_url"], wt.ctx.config["use_local_px_proxy"])
                      and wt.warmup.http.active)
                overlay_buttons[0].click()
                check("HTTP: None selection skips the automatic prompt without another warmup",
                      wt.recording.current_prompt is None and not wt.recording.use_auto_prompt
                      and warmup.call_count == 1)
                overlay_buttons[1].click()
        check(
            "startup: recording prompt overlay updates selection",
            wt.recording.current_prompt == "CUSTOM_OVERLAY_PROMPT",
        )
        wt.recording.abandon_prompt_selection()
        check(
            "startup: closing the prompt palette also closes its mouse status",
            MouseFollowerTooltip._instance is None,
        )
        wt.recording.show_prompt_palette([{"caption": "Polish", "text": "CUSTOM_OVERLAY_PROMPT"}])
        MouseFollowerTooltip.show_tooltip("Replacement status", 60_000, spinner=True)
        replacement_status = MouseFollowerTooltip._instance
        wt.recording.abandon_prompt_selection()
        check(
            "startup: closing prompt palette preserves a replacement tooltip",
            replacement_status is not None and MouseFollowerTooltip._instance is replacement_status,
        )
        MouseFollowerTooltip.hide_tooltip()
        wt.ctx.config["recording_prompt_overlay_system_position"] = False
        QCursor.setPos(QPoint(120, 140))
        wt.recording.show_prompt_palette([{"caption": "Polish", "text": "CUSTOM_OVERLAY_PROMPT"}])
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
        wt.recording.abandon_prompt_selection()
        wt.ctx.config["recording_prompt_overlay_system_position"] = True
        with (
            patch("app.ui.floating_buttons.is_WINDOWS", False),
            patch("app.ui.floating_buttons.is_MACOS", True),
        ):
            mac_overlay = RecordingPromptOverlay(
                prompts=[{"caption": "Polish", "text": "CUSTOM_OVERLAY_PROMPT"}],
                status_text="Recording",
                none_text="None",
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
        sw.hide()  # Keep QApplication.quit() from being intercepted by the tray-style closeEvent.
        check("startup: hotkey bindings parsed", len(wt.hotkeys.bindings) == 2,
              "; ".join(b["display"] for b in wt.hotkeys.bindings))
        step2_recording_stop()

    def probe_windows_text_input() -> None:
        """Check settings persistence and direct dispatch with all OS input intercepted."""
        windows = sys.platform.startswith("win")
        check("insertion options: group belongs to scrollable General settings", sw.general_layout.indexOf(sw.text_insertion_group) >= 0)
        check("insertion options: alternative library is inside the group", sw.alt_clipboard_lib_checkbox.parentWidget() is sw.text_insertion_group)
        check("insertion options: Windows features are only shown on Windows", all(
            checkbox.isHidden() != windows for checkbox in (
                sw.windows_sendinput_text_checkbox, sw.windows_sendinput_fallback_checkbox)
        ))
        check("fast paste: available on Windows and macOS", sw.fast_paste_checkbox.isHidden() != (windows or sys.platform == "darwin"))
        check("SendInput: default is off and fallback control is disabled",
              not sw.windows_sendinput_text_checkbox.isChecked() and not sw.windows_sendinput_fallback_checkbox.isEnabled())
        if not windows:
            if sys.platform == "darwin":
                sw.fast_paste_checkbox.setChecked(True)
                with patch.object(sw, "_collect_validation_warnings", return_value=[]):
                    sw.save_settings()
                check("fast paste: macOS choice persists", wt.ctx.config["fast_paste"])
                sw.fast_paste_checkbox.setChecked(False)
                with patch.object(sw, "_collect_validation_warnings", return_value=[]):
                    sw.save_settings()
            return
        check("fast paste: default is off", not sw.fast_paste_checkbox.isChecked())
        sw.windows_sendinput_text_checkbox.setChecked(True)
        check("SendInput: selecting direct mode enables fallback control", sw.windows_sendinput_fallback_checkbox.isEnabled())
        sw.windows_sendinput_fallback_checkbox.setChecked(False)
        sw.fast_paste_checkbox.setChecked(True)
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()
        with open(os.path.join(iso_home, ".WhisperTyper", "config.json"), encoding="utf-8") as saved:
            persisted = json.load(saved)
        check("SendInput: both choices persist on save", persisted["windows_sendinput_text"]
              and not persisted["windows_sendinput_fallback"])
        check("fast paste: separate setting persists", persisted["fast_paste"])
        operation = OperationTiming("sendinput_probe")
        operation.mark("stop")
        with (patch("app.controllers.text_output.send_unicode_text", return_value=(8, 8)) as send,
              patch("app.controllers.text_output.copykitten.copy", side_effect=AssertionError("clipboard access")),
              patch.object(wt.text_output, "_simulate_key_combination", side_effect=AssertionError("keyboard access"))):
            check("SendInput: composed application dispatches directly", wt.text_output.insert("ä🦄文", operation))
            check("SendInput: Unicode is passed intact", send.call_args.args == ("ä🦄文",))
        operation.finish("ok")
        check("SendInput: direct timings omit clipboard and paste", "text_commit_end" in operation._events
              and "sendinput_dispatch_end" in operation._events and "paste_dispatch_start" not in operation._events
              and "clipboard_snapshot_start" not in operation._events)
        sw.windows_sendinput_text_checkbox.setChecked(False)
        wt.ctx.config["windows_sendinput_text"] = False
        field = QLineEdit()

        def paste_into_probe(_char: str) -> bool:
            """Paste only into the owned offscreen widget without OS input."""
            field.paste()
            return True

        with (patch("app.controllers.text_output.copykitten.copy", side_effect=qapp.clipboard().setText),
              patch.object(wt.text_output, "_simulate_key_combination", side_effect=paste_into_probe)):
            check("fast paste: real Qt field receives text", wt.text_output.insert("FAST_PASTE_PROBE")
                  and field.text() == "FAST_PASTE_PROBE")
        check("fast paste: restore timer keeps 600 ms after dispatch", wt.text_output._restore_timer.interval() == 600)
        sw.fast_paste_checkbox.setChecked(False)
        sw.windows_sendinput_fallback_checkbox.setChecked(True)
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()

    def probe_dropdown_arrows() -> None:
        """Verify painted arrows on every combo, including table cells, in both themes."""
        original_theme = wt.ctx.config["color_theme"]
        original_format = sw.recording_format_selector.currentIndex()
        aac_index = sw.recording_format_selector.findData("aac")
        if aac_index >= 0:
            sw.recording_format_selector.setCurrentIndex(aac_index)
        for mode in ("light", "dark"):
            wt.ctx.config["color_theme"] = mode
            sw.apply_theme()
            qapp.processEvents()
            missing = []
            accent = QColor(sw._theme_palette["accent"])
            for selector in sw.findChildren(QComboBox):
                if selector.isHidden():
                    continue
                for index in range(sw.tabs.count()):
                    page = sw.tabs.widget(index)
                    if page.isAncestorOf(selector):
                        sw.tabs.setCurrentWidget(page)
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
        sw.tabs.setCurrentWidget(sw._api_keys_tab)
        qapp.processEvents()
        provider = sw._api_keys_tab.table.cellWidget(0, 1).findChild(QComboBox)
        QTest.mouseClick(provider, Qt.MouseButton.LeftButton, pos=QPoint(provider.width() - 14, provider.height() // 2))
        qapp.processEvents()
        check("theme: API key provider arrow opens the dropdown", provider.view().isVisible())
        provider.hidePopup()
        wt.ctx.config["color_theme"] = original_theme
        sw.recording_format_selector.setCurrentIndex(original_format)
        sw.apply_theme()
        sw.tabs.setCurrentWidget(sw.transcription_tab)

    def probe_window_resize() -> None:
        """Shrink the real settings window and check prompt allocation and control reachability."""
        original_size = sw.size()
        sw.tabs.setCurrentWidget(sw.transcription_tab)
        sw.resize(760, 1280)
        qapp.processEvents()
        large_prompt_height = sw.prompt_input.height()
        controls_size = (sw.transcription_api_group.size(), sw.recording_group.size())
        prompt = sw.prompt_input.toPlainText()
        sw.resize(760, 1100)
        qapp.processEvents()
        check("resize: prompt absorbs the height change", abs(large_prompt_height - sw.prompt_input.height() - 180) <= 2,
              f"prompt={large_prompt_height}->{sw.prompt_input.height()}")
        check("resize: API and recording controls retain their sizes",
              controls_size == (sw.transcription_api_group.size(), sw.recording_group.size()))
        sw.resize(760, 600)
        qapp.processEvents()
        check("resize: window reaches 600px height", sw.height() == 600, f"height={sw.height()}")
        check("resize: prompt content is preserved", sw.prompt_input.toPlainText() == prompt)
        check("resize: short transcription page scrolls", sw.transcription_scroll_area.verticalScrollBar().maximum() > 0)
        sw.tabs.setCurrentWidget(sw.rephrasing_tab)
        qapp.processEvents()
        from app.core.models import REPHRASING_MODEL_OPTIONS
        for width in (680, 760):
            sw.resize(width, 600)
            qapp.processEvents()
            editor = sw.rephrasing_model_input.lineEdit()
            needed = max(editor.fontMetrics().horizontalAdvance(model)
                         for models in REPHRASING_MODEL_OPTIONS.values() for model in models)
            check(f"resize: rephrasing model names fit at {width}px window width",
                  editor.contentsRect().width() - 8 >= needed,
                  f"available={editor.contentsRect().width() - 8}, needed={needed}, window={sw.width()}, page={sw.rephrasing_scroll_widget.width()}")
            right = sw.rephrasing_model_input.mapTo(sw.rephrasing_scroll_area.viewport(),
                                                    QPoint(sw.rephrasing_model_input.width(), 0)).x()
            check(f"resize: rephrasing dropdown stays inside the visible page at {width}px",
                  right <= sw.rephrasing_scroll_area.viewport().width())
        sw.rephrasing_scroll_area.ensureWidgetVisible(sw.rephrasing_provider_selector)
        qapp.processEvents()
        check("resize: rephrasing provider dropdown remain usable at 600px", sw.rephrasing_provider_selector.isVisible()
              and sw.rephrasing_provider_selector.height() >= sw.rephrasing_provider_selector.minimumSizeHint().height()
              and sw.rephrasing_provider_selector.mapTo(sw.rephrasing_tab, QPoint(0, sw.rephrasing_provider_selector.height())).y()
              <= sw.rephrasing_tab.height())
        sw.tabs.setCurrentWidget(sw.general_tab)
        qapp.processEvents()
        last_control = sw.quit_without_confirmation_checkbox
        sw.general_scroll_area.ensureWidgetVisible(last_control)
        qapp.processEvents()
        check("resize: short General page scrolls", sw.general_scroll_area.verticalScrollBar().maximum() > 0)
        bottom = last_control.mapTo(sw.general_tab, QPoint(0, last_control.height())).y()
        check("resize: last General control is reachable", bottom <= sw.general_tab.height(),
              f"last-control-bottom={bottom}, page-height={sw.general_tab.height()}")
        general_controls = [sw.general_layout.itemAt(i).widget() for i in range(sw.general_layout.count())]
        compressed = [widget.objectName() for widget in general_controls if widget is not None
                      and widget.isVisible() and widget.height() < widget.minimumSizeHint().height()]
        check("resize: General controls retain usable heights", not compressed, ", ".join(compressed))
        sw.tabs.setCurrentWidget(sw._api_keys_tab)
        qapp.processEvents()
        secret = sw._api_keys_tab.table.cellWidget(0, 2).findChild(QLineEdit)
        check("resize: API key editor remains usable at 600px", secret.height() >= secret.minimumSizeHint().height()
              and sw._api_keys_tab.groq_rotation.mapTo(wt, QPoint(0, sw._api_keys_tab.groq_rotation.height())).y() <= sw.height(),
              f"input={secret.height()}, minimum={secret.minimumSizeHint().height()}")
        table = sw._api_keys_tab.table
        check("resize: key editor stays within its table row",
              table.cellWidget(0, 2).geometry().bottom() <= table.visualRect(table.model().index(0, 2)).bottom())
        check("resize: save button stays in the window", sw.save_button.mapTo(wt, QPoint(0, sw.save_button.height())).y() <= sw.height())
        sw.resize(original_size)
        sw.tabs.setCurrentWidget(sw.transcription_tab)
        qapp.processEvents()

    def probe_api_key_management() -> None:
        """Edit masked profiles, verify independent selections, save, and test live form keys."""
        tab = sw._api_keys_tab
        table = tab.table
        original_profiles = tab.profiles()
        original_selection = (wt.ctx.config["transcription_key_profile_id"], wt.ctx.config["rephrasing_key_profile_id"])
        check("keys: table uses theme separators instead of the native grid", not table.showGrid())
        check("keys: central tab is between Prompts and General", sw.tabs.indexOf(tab) == 3 and sw.tabs.indexOf(sw.general_tab) == 5)
        check("keys: legacy identical keys migrated into one profile", len(original_profiles) == 1
              and original_selection[0] == original_selection[1] and "api_key" not in wt.ctx.config and "rephrasing_api_key" not in wt.ctx.config)
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
        sw.api_endpoint_input.setText("https://api.groq.com/openai/v1/audio/transcriptions")
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("groq"))
        check("providers: Groq selection sets the chat-completions URL",
              sw.rephrasing_api_url_input.text() == "https://api.groq.com/openai/v1/chat/completions")
        check("providers: manually entered URL updates the provider dropdown", sw.transcription_provider_selector.currentData() == "groq")
        check("providers: both panels use the first matching Groq key automatically",
              sw._ui_api_key("transcription") == "gsk-smoke-a" and sw._ui_api_key("rephrasing") == "gsk-smoke-a"
              and sw.transcription_key_profile_selector.currentData() == "" and sw.transcription_key_profile_selector.isHidden())
        check("providers: official providers hide URL and key selector", sw.api_endpoint_input.isHidden()
              and sw.rephrasing_api_url_input.isHidden() and not sw.rephrasing_key_choose_button.isHidden())
        sw.rephrasing_key_choose_button.click()
        check("keys: 'choose another key' reveals the selector", not sw.rephrasing_key_profile_selector.isHidden())
        sw.rephrasing_key_profile_selector.setCurrentIndex(sw.rephrasing_key_profile_selector.findData(ids[1]))
        check("keys: task selections are independent", sw._ui_api_key("transcription") == "gsk-smoke-a"
              and sw._ui_api_key("rephrasing") == "gsk-smoke-b")
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("groq"))
        check("providers: clicking the current provider retains its key selection", sw.rephrasing_key_profile_selector.currentData() == ids[1])
        tab.add_button.click()
        openai_row = table.rowCount() - 1
        provider = table.cellWidget(openai_row, 1).findChild(QComboBox)
        provider.setCurrentIndex(provider.findData("openai"))
        table.cellWidget(openai_row, 2).findChild(QLineEdit).setText("sk-smoke-openai")
        openai_id = tab.profiles()[-1]["id"]
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("openai"))
        check("providers: OpenAI selection sets the chat-completions URL",
              sw.rephrasing_api_url_input.text() == "https://api.openai.com/v1/chat/completions")
        check("providers: changing provider returns to the automatic OpenAI key and filters keys",
              sw.rephrasing_key_profile_selector.currentData() == ""
              and sw.rephrasing_key_profile_selector.isHidden()
              and sw.rephrasing_key_profile_selector.findData(ids[1]) == -1
              and sw.rephrasing_key_profile_selector.findData(openai_id) > 0
              and sw._ui_api_key("rephrasing") == "sk-smoke-openai")
        sw.rephrasing_api_url_input.setText(f"http://127.0.0.1:{port}/v1/chat/completions")
        check("providers: custom URLs show URL and Custom profiles", sw.rephrasing_key_profile_selector.findData(original_profiles[0]["id"]) > 0
              and sw.rephrasing_key_profile_selector.findData(openai_id) == -1
              and not sw.rephrasing_api_url_input.isHidden() and not sw.rephrasing_key_profile_selector.isHidden())
        check("providers: changes await saving and preserve the model and transcription selection",
              wt.ctx.config["rephrasing_api_url"] == f"http://127.0.0.1:{port}/v1/chat/completions"
              and sw.rephrasing_model_input.currentText() == "fake-model"
              and sw._ui_api_key("transcription") == "gsk-smoke-a")
        table.selectRow(openai_row)
        tab.remove_button.click()
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("groq"))
        sw.rephrasing_key_choose_button.click()
        sw.rephrasing_key_profile_selector.setCurrentIndex(sw.rephrasing_key_profile_selector.findData(ids[1]))
        check("keys: provider filters exclude the custom profile", sw.transcription_key_profile_selector.findData(original_profiles[0]["id"]) == -1)
        table.item(1, 0).setText("Renamed Groq key")
        check("keys: the status line follows a renamed automatic key", "Renamed Groq key" in sw.transcription_key_status_label.text())
        secret = table.cellWidget(1, 2).findChild(QLineEdit)
        secret.setText("gsk-smoke-a-edited")
        selector = sw.transcription_key_profile_selector
        check("keys: preview shows only the first 10 and last 4 characters",
              table.cellWidget(1, 2).findChild(QLabel).text() == "gsk-smoke-••••••ited"
              and selector.itemText(selector.findData(ids[0])).endswith("gsk-smoke-••••••ited")
              and "gsk-smoke-••••••ited" in sw.transcription_key_status_label.text()
              and secret.echoMode() == QLineEdit.EchoMode.Password
              and tab.profiles()[1]["key"] == "gsk-smoke-a-edited")
        check("keys: unsaved edit does not affect runtime credentials", selected_api_key(wt.ctx.config, "transcription") == "sk-test-dummy")
        with (patch("app.ui.connection_tester.net.run_transcription_connection_test", return_value=("ok", "")) as connection,
              patch("app.ui.connection_tester.QMessageBox.information")):
            sw._connection_tester.test_transcription()
        check("keys: connection test uses the unsaved automatic key", connection.call_args.args[1] == "gsk-smoke-a-edited")
        tab.groq_rotation.setChecked(True)
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()
        with open(os.path.join(iso_home, ".WhisperTyper", "config.json"), encoding="utf-8") as saved:
            persisted = json.load(saved)
        check("keys: save persists automatic and explicit choices and rotation", persisted["transcription_key_profile_id"] == ""
              and persisted["rephrasing_key_profile_id"] == ids[1] and persisted["groq_key_rotation"]
              and "api_key" not in persisted and "rephrasing_api_key" not in persisted)
        check("providers: save persists the provider-selected URL and the custom URL",
              persisted["rephrasing_api_url"] == "https://api.groq.com/openai/v1/chat/completions"
              and persisted["rephrasing_custom_url"] == f"http://127.0.0.1:{port}/v1/chat/completions")
        table.selectRow(1)
        tab.remove_button.click()
        check("keys: deleting the automatic key moves on to the next usable one", sw._ui_api_key("transcription") == "gsk-smoke-b"
              and sw.rephrasing_key_profile_selector.currentData() == ids[1])
        table.selectRow(1)
        tab.remove_button.click()
        sw.api_endpoint_input.setText(f"http://127.0.0.1:{port}/v1/audio/transcriptions")
        sw.rephrasing_api_url_input.setText(f"http://127.0.0.1:{port}/v1/chat/completions")
        sw.transcription_key_profile_selector.setCurrentIndex(sw.transcription_key_profile_selector.findData(original_selection[0]))
        sw.rephrasing_key_profile_selector.setCurrentIndex(sw.rephrasing_key_profile_selector.findData(original_selection[1]))
        tab.groq_rotation.setChecked(False)
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()

    def probe_automatic_key_selection() -> None:
        """Select saved or first valid profiles only on provider changes, independently per task."""
        tab = sw._api_keys_tab
        table = tab.table
        original_count = table.rowCount()
        saved_transcription = wt.ctx.config["transcription_key_profile_id"]
        ids = []
        for key in ("", "invalid\nkey", "gsk-first-valid", "gsk-saved-valid"):
            tab.add_button.click()
            row = table.rowCount() - 1
            provider = table.cellWidget(row, 1).findChild(QComboBox)
            provider.setCurrentIndex(provider.findData("groq"))
            table.cellWidget(row, 2).findChild(QLineEdit).setText(key)
            ids.append(tab.profiles()[-1]["id"])
        wt.ctx.config["transcription_key_profile_id"] = ids[-1]
        sw.transcription_provider_selector.setCurrentIndex(sw.transcription_provider_selector.findData("groq"))
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("groq"))
        check("keys: provider switch uses the first valid key, skipping empty and invalid ones",
              sw._ui_api_key("transcription") == "gsk-first-valid" and sw._ui_api_key("rephrasing") == "gsk-first-valid"
              and sw.transcription_key_profile_selector.currentData() == "")
        check("models: plain Groq model names do not trigger provider-label warnings",
              not any("selected model" in warning for warning in sw._collect_validation_warnings(sw.model_dropdown.currentText())))
        sw.rephrasing_key_choose_button.click()
        sw.rephrasing_key_profile_selector.setCurrentIndex(sw.rephrasing_key_profile_selector.findData(ids[-1]))
        sw.rephrasing_api_url_input.setText("https://api.groq.com/openai/v1/chat/completions?test=1")
        check("keys: same-provider URL edits preserve manual selection", sw.rephrasing_key_profile_selector.currentData() == ids[-1])
        sw.transcription_provider_selector.setCurrentIndex(sw.transcription_provider_selector.findData("openai"))
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("openai"))
        check("keys: provider without a key warns in both sections", sw._ui_api_key("transcription") == ""
              and sw.transcription_key_status_label.property("key_status") == "missing"
              and sw.rephrasing_key_status_label.property("key_status") == "missing"
              and sw.transcription_api_group.property("incomplete") is True)
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("groq"))
        check("keys: switching back returns to the automatic Groq key", sw.rephrasing_key_profile_selector.currentData() == ""
              and sw._ui_api_key("rephrasing") == "gsk-first-valid")
        check("keys: rephrasing provider change does not alter transcription", sw._ui_api_key("transcription") == "")
        for row in range(table.rowCount() - 1, original_count - 1, -1):
            table.selectRow(row)
            tab.remove_button.click()
        check("keys: removing all Groq profiles leaves no key", sw._ui_api_key("rephrasing") == "")
        wt.ctx.config["transcription_key_profile_id"] = saved_transcription
        sw.api_endpoint_input.setText(wt.ctx.config["api_endpoint"])
        sw.rephrasing_api_url_input.setText(wt.ctx.config["rephrasing_api_url"])
        check("keys: custom provider switch selects a valid Custom profile", sw._ui_api_key("transcription") == "sk-test-dummy"
              and sw._ui_api_key("rephrasing") == "sk-test-dummy")
        check("providers: manual custom URL updates both provider dropdowns", sw.transcription_provider_selector.currentData() == "custom"
              and sw.rephrasing_provider_selector.currentData() == "custom")
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("openai"))
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("custom"))
        check("providers: choosing Custom restores the user's custom URL",
              sw.rephrasing_api_url_input.text() == wt.ctx.config["rephrasing_api_url"])
        sw.rephrasing_api_url_input.setText(wt.ctx.config["rephrasing_api_url"])

    def probe_provider_models() -> None:
        """Check provider catalogs, custom/persisted models and real worker request parameters."""
        from app.core.models import REPHRASING_MODEL_OPTIONS, TRANSCRIPTION_MODEL_OPTIONS
        from app.services.transcription import TranscriptionRequest
        from app.services.transcription_worker import TranscriptionWorker

        sw.transcription_provider_selector.setCurrentIndex(sw.transcription_provider_selector.findData("groq"))
        check("models: Groq transcription has only Groq suggestions", [sw.model_dropdown.itemText(i) for i in range(sw.model_dropdown.count())]
              == TRANSCRIPTION_MODEL_OPTIONS["groq"])
        check("models: incompatible transcription builtin switches to Groq turbo", sw.model_dropdown.currentText() == "whisper-large-v3-turbo")
        sw._refresh_model_selectors(preserve_saved=True)
        saved_transcription_model = sw.model_dropdown.currentText()
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("openai"))
        check("models: rephrasing provider change preserves the transcription model",
              sw.model_dropdown.currentText() == saved_transcription_model)
        sw.transcription_provider_selector.setCurrentIndex(sw.transcription_provider_selector.findData("openai"))
        check("models: OpenAI transcription has only OpenAI suggestions", [sw.model_dropdown.itemText(i) for i in range(sw.model_dropdown.count())]
              == TRANSCRIPTION_MODEL_OPTIONS["openai"])
        check("models: OpenAI provider switch starts with recommended GPT transcription", sw.model_dropdown.currentText() == "gpt-transcribe"
              and not sw.transcription_temp_slider.isEnabled())
        sw.model_dropdown.setCurrentText("gpt-4o-transcribe-diarize")
        check("models: diarization disables unsupported prompt and temperature", not sw.prompt_input.isEnabled()
              and not sw.transcription_temp_slider.isEnabled())
        sw.model_dropdown.setCurrentText("gpt-4o-transcribe")
        check("models: ordinary transcription restores prompt and temperature", sw.prompt_input.isEnabled() and sw.transcription_temp_slider.isEnabled())
        check("models: GPT transcription has no Whisper prompt limit", "230" not in sw.prompt_token_label.text())
        sw.tabs.setCurrentWidget(sw.transcription_tab)
        qapp.processEvents()
        selector = sw.model_dropdown
        QTest.mouseClick(selector, Qt.MouseButton.LeftButton, pos=QPoint(selector.width() - 11, selector.height() // 2))
        qapp.processEvents()
        check("models: clicking the transcription dropdown arrow opens its suggestions", selector.isEditable()
              and selector.view().isVisible() and selector.view().model().rowCount() == len(TRANSCRIPTION_MODEL_OPTIONS["openai"]))
        selector.hidePopup()
        selector.lineEdit().selectAll()
        QTest.keyClicks(selector.lineEdit(), "my-custom-transcriber")
        check("models: typing a transcription model edits the dropdown directly", selector.currentText() == "my-custom-transcriber")
        with (patch("app.ui.connection_tester.net.run_transcription_connection_test", return_value=("ok", "")) as connection,
              patch("app.ui.connection_tester.QMessageBox")):
            sw._connection_tester.test_transcription()
        check("models: transcription connection test uses the typed model",
              connection.called and connection.call_args.args[2] == "my-custom-transcriber")
        sw.transcription_provider_selector.setCurrentIndex(sw.transcription_provider_selector.findData("groq"))
        check("models: custom transcription name survives a provider change", sw.model_dropdown.currentText() == "my-custom-transcriber")
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("openai"))
        sw.rephrasing_model_input.setCurrentText("gpt-6-luna")
        check("models: OpenAI chat suggestions and fixed temperature", [sw.rephrasing_model_input.itemText(i) for i in range(sw.rephrasing_model_input.count())]
              == REPHRASING_MODEL_OPTIONS["openai"] and not sw.rephrasing_temp_slider.isEnabled())
        sw.tabs.setCurrentWidget(sw.rephrasing_tab)
        qapp.processEvents()
        selector = sw.rephrasing_model_input
        QTest.mouseClick(selector, Qt.MouseButton.LeftButton, pos=QPoint(selector.width() - 11, selector.height() // 2))
        qapp.processEvents()
        check("models: clicking the rephrasing dropdown arrow opens the OpenAI list", isinstance(selector, QComboBox)
              and selector.view().isVisible() and selector.view().model().rowCount() == len(REPHRASING_MODEL_OPTIONS["openai"]))
        selector.hidePopup()
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("groq"))
        check("models: Groq chat suggestions replace the incompatible OpenAI model", sw.rephrasing_model_input.currentText() == "openai/gpt-oss-120b"
              and [sw.rephrasing_model_input.itemText(i) for i in range(sw.rephrasing_model_input.count())] == REPHRASING_MODEL_OPTIONS["groq"]
              and sw.rephrasing_temp_slider.isEnabled())
        QTest.mouseClick(selector, Qt.MouseButton.LeftButton, pos=QPoint(selector.width() - 11, selector.height() // 2))
        qapp.processEvents()
        check("models: clicking the rephrasing dropdown arrow opens the Groq list", selector.view().isVisible()
              and selector.view().model().rowCount() == len(REPHRASING_MODEL_OPTIONS["groq"])
              and selector.view().model().index(0, 0).data() == "openai/gpt-oss-120b")
        selector.hidePopup()
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("openai"))
        check("models: switching to OpenAI chat selects GPT-6 Luna", sw.rephrasing_model_input.currentText() == "gpt-6-luna")
        sw.rephrasing_model_input.setCurrentText("my-custom-chat-model")
        sw.rephrasing_provider_selector.setCurrentIndex(sw.rephrasing_provider_selector.findData("openai"))
        check("models: editable chat dropdown retains custom names", sw.rephrasing_model_input.currentText() == "my-custom-chat-model")
        wt.ctx.config["rephrasing_model"] = "my-custom-chat-model"
        sw._refresh_model_selectors(preserve_saved=True)
        check("models: saved unknown chat model is preserved", sw.rephrasing_model_input.currentText() == "my-custom-chat-model")
        wt.ctx.config["rephrasing_model"] = "fake-model"
        sw.api_endpoint_input.setText(wt.ctx.config["api_endpoint"])
        sw.rephrasing_api_url_input.setText(wt.ctx.config["rephrasing_api_url"])
        sw._refresh_model_selectors(preserve_saved=True)
        sw.transcription_key_profile_selector.setCurrentIndex(sw.transcription_key_profile_selector.findData(wt.ctx.config["transcription_key_profile_id"]))
        sw.rephrasing_key_profile_selector.setCurrentIndex(sw.rephrasing_key_profile_selector.findData(wt.ctx.config["rephrasing_key_profile_id"]))
        original_model = wt.ctx.config["model"]
        sw.model_dropdown.setCurrentText("my-custom-transcriber")
        sw.rephrasing_model_input.setCurrentText("my-custom-chat-model")
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()
        with open(os.path.join(iso_home, ".WhisperTyper", "config.json"), encoding="utf-8") as saved:
            persisted = json.load(saved)
        check("models: editable chat name is saved as a string", persisted["rephrasing_model"] == "my-custom-chat-model")
        check("models: typed transcription name is saved as a string", persisted["model"] == "my-custom-transcriber")
        reloaded, _changed = wt.ctx._store.load()
        check("models: typed transcription name survives configuration reload", reloaded["model"] == "my-custom-transcriber")
        sw.model_dropdown.setCurrentText(original_model)
        sw.rephrasing_model_input.setCurrentText("fake-model")
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()

        for model, language in (("gpt-transcribe", "de"), ("gpt-transcribe", ""),
                                ("gpt-4o-transcribe", "de"), ("gpt-4o-mini-transcribe", "de"),
                                ("gpt-4o-mini-transcribe-2025-12-15", "de"),
                                ("gpt-4o-transcribe-diarize", "de"), ("whisper-large-v3-turbo", "de")):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"text": "MODEL_PARAMETER_PROBE"}
            results = []
            worker = TranscriptionWorker(TranscriptionRequest(
                "test-key", "https://api.openai.com/v1/audio/transcriptions", ok_wav, "Context", model, language, 0.0))
            worker.finished.connect(results.append)
            with patch("app.services.transcription.request", return_value=response) as post:
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

    def probe_replacements() -> None:
        """Validate edits, persistence and corrections before every rephrasing route."""
        tab = sw._replacements_tab
        highlighted = "🤖 Croc ; Groq ; 1\nchat gpt ; ChatGPT"
        tab.editor.setPlainText(highlighted)
        original_theme = wt.ctx.config["color_theme"]
        theme_colors = []
        for mode in ("light", "dark"):
            wt.ctx.config["color_theme"] = mode
            sw.apply_theme()
            formats = tab.editor.document().firstBlock().layout().formats()
            colors = [span.format.foreground().color().name() for span in formats]
            theme_colors.append(colors)
            check(f"replacements: three distinct field colors in {mode} mode", len(colors) == 3 and len(set(colors)) == 3)
            check(f"replacements: field coloring respects emoji offsets in {mode} mode",
                  [span.start for span in formats] == [0, 9, 16])
            check(f"replacements: highlighting preserves editable plain text in {mode} mode", tab.editor.toPlainText() == highlighted)
        check("replacements: colors adapt to theme changes", theme_colors[0] != theme_colors[1])
        wt.ctx.config["color_theme"] = original_theme
        sw.apply_theme()
        raw = "Croc, Krog, Krok ; Groq ; 1\nanweisung ; prompt ; 1"
        tab.editor.setPlainText(raw)
        check("replacements: unsaved rules do not affect runtime", wt.pipeline.replacement_rules.term_count == 0)
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()
        loaded, _changed = wt.ctx._store.load()
        check("replacements: rules and enabled state survive save/reload", loaded["replacements_rules"] == raw
              and loaded["replacements_enabled"] and wt.pipeline.replacement_rules.term_count == 4)
        with patch.object(wt.pipeline, "finalize_output") as output:
            wt.pipeline.on_transcription_finished("Krog und Croc", "clipboard")
            check("replacements: plain transcription is corrected", output.call_args.args[0] == "Groq und Groq")
        set_instruction(enabled=True, trigger_words="prompt,")
        with patch.object(wt.pipeline, "_start_post_transcription_rephrase") as rephrase:
            wt.pipeline.on_transcription_finished("Anweisung, erkläre Krog", "clipboard")
            check("replacements: correction precedes LivePrompt trigger detection", rephrase.call_args.args[0].user_prompt == "prompt, erkläre Groq")
            wt.pipeline.on_transcription_finished("Croc", "clipboard", "CUSTOM")
            check("replacements: explicit transformation receives corrected text", rephrase.call_args.args[0].user_prompt == "Groq"
                  and rephrase.call_args.args[1] == "Groq")
        set_instruction(enabled=False)
        with patch.object(wt.pipeline, "_start_post_transcription_rephrase") as rephrase:
            wt.pipeline.on_transcription_finished("Krog", "clipboard", None, None, "AUTO_PROMPT")
            check("replacements: the automatic prompt receives corrected text",
                  rephrase.call_args.args[0].user_prompt == "Groq"
                  and rephrase.call_args.args[0].system_prompt == "AUTO_PROMPT")
        tab.enabled.setChecked(False)
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()
        with patch.object(wt.pipeline, "finalize_output") as output:
            wt.pipeline.on_transcription_finished("Krog", "clipboard")
            check("replacements: disabled corrections leave text untouched", output.call_args.args[0] == "Krog")
        tab.editor.setPlainText("valid ; rule\nbroken")
        with patch("app.ui.settings.window.QMessageBox.warning") as warning:
            sw.save_settings()
        check("replacements: invalid edits show their line and preserve saved rules", warning.call_count == 1
              and "2" in warning.call_args.args[2] and wt.ctx.config["replacements_rules"] == raw
              and wt.pipeline.replacement_rules.term_count == 4 and sw.tabs.currentWidget() is tab)
        tab.editor.clear()
        tab.enabled.setChecked(True)
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()
        sw.tabs.setCurrentWidget(sw.transcription_tab)

    def step2_recording_stop() -> None:
        """Stop synthesized PCM via the real hotkey signal, without opening a microphone."""
        with wave.open(ok_wav, "rb") as audio:
            frames = [audio.readframes(audio.getnframes())]
            wt.recording.microphone.samplerate = audio.getframerate()
        captured = SimpleNamespace(stop_recording=lambda _timing: lambda: frames)
        wt.ctx.config["min_recording_seconds"] = 0
        saved_prompts = wt.ctx.config["post_rephrasing_entries"]
        # An automatic prompt routes the stopped recording through rephrasing.
        wt.ctx.config["post_rephrasing_entries"] = [{"caption": "Auto", "text": "AUTO_PROMPT", "auto_apply": True}]
        wt.recording.is_recording = True
        detected_ns = time.perf_counter_ns() - 5_000_000
        # Deliver to clipboard so the real stop path cannot send paste keys to another application.
        start_worker = wt.pipeline.start
        with patch.object(wt.pipeline, "start", side_effect=lambda path, **kw: start_worker(
            path, output_mode="clipboard", **kw,
        )), patch.object(wt.recording, "active_capture", return_value=captured):
            wt.hotkeys.action_triggered.emit("stop_transcription", detected_ns)
        def recording_done() -> None:
            check("stop hotkey: transcription and rephrasing pipeline completed", not wt.recording.is_recording)
            check("recording format: default WAV is uploaded as PCM", b"RIFF" in FakeAPI.last_transcription_body)
            wt.ctx.config["post_rephrasing_entries"] = saved_prompts
            step2_native_aac()

        poll(lambda: wt.pipeline.last_transcription == "REPHRASED_FAKE_RESULT", 10,
             recording_done,
             lambda: (check("stop hotkey: transcription finished", False, "timeout"), step2_transcribe()))

    def step2_native_aac() -> None:
        """Encode the retained recording with the real PyAV AAC encoder and upload to the fake API."""
        check("recording format: deferred AAC probe finished after startup", sw._aac_bitrates_probed)
        check("recording format: WAV is the default and hides bitrate", sw.recording_format_selector.currentData() == "wav"
              and not sw.recording_bitrate_selector.isEnabled())
        index = sw.recording_format_selector.findData("aac")
        if index < 0:
            step2_transcribe()
            return
        sw.recording_format_selector.setCurrentIndex(index)
        check("recording format: AAC enables supported bitrate choices", sw.recording_bitrate_selector.isEnabled()
              and sw.recording_bitrate_selector.currentData() == 64)
        with patch.object(sw, "_collect_validation_warnings", return_value=[]):
            sw.save_settings()
        check("recording format: selected codec and bitrate are saved", wt.ctx.config["recording_format"] == "aac"
              and wt.ctx.config["recording_aac_bitrate_kbps"] == 64)
        path = wt.ctx.recordings.latest()
        assert path is not None
        with open(path, "rb") as recording:
            original = recording.read()
        wt.pipeline.last_transcription = ""
        with patch("app.controllers.transcription.resolve_ffmpeg", side_effect=AssertionError("No FFmpeg for recordings")):
            wt.pipeline.start(path, output_mode="clipboard")

        def done() -> None:
            body = FakeAPI.last_transcription_body
            check("recording format: real AAC/M4A reaches HTTP upload", b".m4a" in body
                  and b"audio/mp4" in body and b"ftyp" in body and b"mdat" in body)
            with open(path, "rb") as recording:
                check("recording format: original WAV remains playable", recording.read() == original)
            check("recording format: temporary encoded upload is removed", not any(
                name.startswith("whispertyper_upload_") for name in os.listdir(iso_tmp)))
            sw.recording_format_selector.setCurrentIndex(0)
            with patch.object(sw, "_collect_validation_warnings", return_value=[]):
                sw.save_settings()
            step2_transcribe()

        poll(lambda: wt.pipeline.last_transcription == "TRANSCRIBED_FAKE_RESULT", 10, done,
             lambda: (check("recording format: native upload completed", False, "timeout"), step2_transcribe()))

    def step2_transcribe() -> None:
        """Drive one file transcription through the worker pipeline into the clipboard."""
        wt.pipeline.last_transcription = ""
        wt.pipeline.replacement_rules = Replacements("TRANSCRIBED_FAKE_RESULT ; CORRECTED_FAKE_RESULT ; 1")
        wt.pipeline.start(ok_wav, output_mode="clipboard")
        poll(lambda: wt.pipeline.last_transcription == "CORRECTED_FAKE_RESULT", 10,
             lambda: (check("transcribe: delivered to clipboard",
                            clipboard() == "CORRECTED_FAKE_RESULT", repr(clipboard())),
                      step3_liveprompt()),
             lambda: (check("transcribe: worker finished", False, "timeout"), step3_liveprompt()))

    def step3_liveprompt() -> None:
        """Drive a LivePrompt-triggered transcription through the rephrasing worker."""
        wt.pipeline.replacement_rules = Replacements("")
        set_instruction(enabled=True, trigger_words="prompt,", strip_trigger=True)
        wt.pipeline.on_transcription_finished("Prompt, please write hello world", "clipboard")
        poll(lambda: wt.pipeline.last_transcription == "REPHRASED_FAKE_RESULT", 10,
             lambda: (check("liveprompt: rephrased text delivered",
                            clipboard() == "REPHRASED_FAKE_RESULT", repr(clipboard())),
                      step4_recording_prompt()),
             lambda: (check("liveprompt: rephrase finished", False, "timeout"), step4_recording_prompt()))

    def step4_recording_prompt() -> None:
        """An explicit recording-palette prompt must override automatic LivePrompting."""
        wt.pipeline.last_transcription = ""
        set_instruction(enabled=True, text="LIVEPROMPT_SHOULD_NOT_WIN")
        wt.pipeline.on_transcription_finished(
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

        poll(lambda: wt.pipeline.last_transcription == "REPHRASED_FAKE_RESULT", 10,
             recording_prompt_done,
             lambda: (check("recording prompt: rephrase finished", False, "timeout"),
                      probe1_rephrase_failure()))

    def probe1_rephrase_failure() -> None:
        """PROBE: a failing rephrase endpoint must fall back to the raw transcription."""
        wt.ctx.config["rephrasing_api_url"] = f"http://127.0.0.1:{port}/v1/chat/fail"
        set_instruction(strip_trigger=False)
        wt.pipeline.on_transcription_finished("prompt, translate this text", "clipboard")
        expected = "prompt, translate this text"
        poll(lambda: wt.pipeline.last_transcription == expected, 10,
             lambda: (check("PROBE rephrase-500 falls back to raw text",
                            clipboard() == expected, repr(clipboard())),
                      probe2_batch()),
             lambda: (check("PROBE rephrase-500 falls back to raw text", False, "timeout"), probe2_batch()))

    def probe2_batch() -> None:
        """PROBE: a batch with one failing file must skip it and join the rest (no modal)."""
        set_instruction(enabled=False)
        wt.pipeline.start_batch([ok_wav, fail_wav, ok_wav])
        poll(lambda: not getattr(wt.pipeline, "_batch_active", True), 20,
             lambda: (check("PROBE batch skips failing file, joins rest",
                            clipboard() == "TRANSCRIBED_FAKE_RESULT\n\nTRANSCRIBED_FAKE_RESULT",
                            repr(clipboard())),
                      probe3_rotation()),
             lambda: (check("PROBE batch completes despite failure", False, "timeout"), probe3_rotation()))

    def probe3_rotation() -> None:
        """Overlapping workers must retain their different credential snapshots."""
        original_profiles = wt.ctx.config["api_key_profiles"]
        original_selection = wt.ctx.config["transcription_key_profile_id"]
        wt.ctx.config["api_key_profiles"] = [
            {"id": "rotation-a", "name": "First", "provider": "groq", "key": "gsk-smoke-rotation-a"},
            {"id": "rotation-b", "name": "Second", "provider": "groq", "key": "gsk-smoke-rotation-b"},
        ]
        wt.ctx.config["transcription_key_profile_id"] = "rotation-a"
        wt.ctx.config["groq_key_rotation"] = True
        before = len(FakeAPI.authorization_headers)
        # Only credential classification is patched; both HTTP requests go to the local fake API.
        with patch("app.core.api_keys.provider_for_url", return_value="groq"):
            wt.pipeline.start(ok_wav, output_mode="clipboard")
            wt.pipeline.start(ok_wav, output_mode="clipboard")
        wt.ctx.config["api_key_profiles"][0]["key"] = "gsk-smoke-edited-during-request"

        def rotation_done() -> None:
            check("rotation: overlapping workers use distinct immutable key snapshots", sorted(FakeAPI.authorization_headers[before:])
                  == ["Bearer gsk-smoke-rotation-a", "Bearer gsk-smoke-rotation-b"])
            wt.ctx.config["api_key_profiles"] = original_profiles
            wt.ctx.config["transcription_key_profile_id"] = original_selection
            wt.ctx.config["groq_key_rotation"] = False
            finish()

        poll(lambda: not wt.workers.running, 10, rotation_done,
             lambda: (check("rotation: workers finish", False, "timeout"), finish()))

    def finish() -> None:
        """Shut the app down through its real quit path."""
        operation = OperationTiming("output_failure_probe")
        with patch("app.controllers.transcription.copykitten.copy", side_effect=RuntimeError("synthetic clipboard failure")):
            try:
                wt.pipeline.finalize_output("probe", output_mode="clipboard", timing=operation)
            except RuntimeError:
                check("timings: output failure preserves exception behavior", True)
            else:
                check("timings: output failure preserves exception behavior", False)
        for post_rephrase in (False, True):
            operation = OperationTiming("text_commit_failed_probe")
            with patch.object(wt.text_output, "_simulate_key_combination", return_value=False):
                if post_rephrase:
                    wt.post_rephrase.on_rephrasing_finished("probe", timing=operation)
                else:
                    wt.pipeline.finalize_output("probe", timing=operation)
            check(f"text commit: {'rephrasing' if post_rephrase else 'transcription'} finishes failed dispatch",
                  "text_commit_failed" in operation._events and "text_commit_end" not in operation._events and operation._finished)
            wt.text_output._restore_timer.stop()
            wt.text_output.restore_clipboard_now()
        operation = OperationTiming("text_commit_exception_probe")
        with patch.object(wt.text_output, "insert", side_effect=RuntimeError("synthetic insertion exception")):
            try:
                wt.post_rephrase.on_rephrasing_finished("probe", timing=operation)
            except RuntimeError:
                check("text commit: standalone insertion exception finishes timing", operation._finished)
            else:
                check("text commit: standalone insertion exception finishes timing", False)
        wt.quit_app()

    QTimer.singleShot(400, step1_startup)
    QTimer.singleShot(60_000, qapp.quit)  # watchdog: never hang

    rc = qapp.exec()

    log_queue.flush(3)
    check("replacements: asynchronous logs include matches and disabled state without transcript text",
          any("matches=2" in line and "matched_rules=[1]" in line for line in replacement_logs)
          and any("enabled=False" in line and "matches=0" in line for line in replacement_logs)
          and all("CORRECTED_FAKE_RESULT" not in line and "Krog" not in line for line in replacement_logs))
    check("timings: correction duration is included", any("replacements_ms=" in line for line in latency_summaries))
    if sw.recording_format_selector.findData("aac") >= 0:
        check("recording format: encoding duration appears in the summary", any("audio_encode_ms=" in line for line in latency_summaries))
    check("text commit: summary includes stop-to-commit, clipboard and both waits", any(
        "source=text_commit_probe " in line and all(field in line for field in (
            "stop_to_text_commit_ms=", "text_commit_ms=", "clipboard_write_ms=",
            "paste_prepare_wait_ms=", "paste_dispatch_ms=", "paste_settle_wait_ms=",
        )) for line in latency_summaries
    ))
    recordings = [line for line in latency_summaries if "source=recording " in line]
    check("timings: recording has stop-to-API and stop-to-output measurements",
          len(recordings) == 1 and all(field in recordings[0] for field in (
              "stop_to_api_ms=", "stop_to_output_ms=", "event_queue_ms=", "file_write_ms=",
              "rephrase_request_ms=",
              "stop_to_request_sent_ms=", "stop_to_first_byte_ms=",
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
    check("HTTP: transcription and rephrasing timings include first byte and post-upload wait", all(
        any(f"stage={stage} " in line and "ttfb_ms=" in line and "after_upload_wait_ms=" in line
            and "upload_ms=" in line for line in transport_logs) for stage in ("transcription", "rephrase")
    ))
    check("HTTP: API calls reuse the warm connection", any("reused=True" in line for line in transport_logs))

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
