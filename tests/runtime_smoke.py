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
  * the microphone is never opened; the system clipboard is snapshotted and restored

Requires the full runtime environment (PyQt6, pyaudio, ...), so it is NOT part of CI.
It deliberately has no ``test_`` prefix — pytest must never collect it — and the whole
run lives behind ``__main__`` so importing this module has no side effects.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeAPI(BaseHTTPRequestHandler):
    """Answers transcription/chat requests; /fail paths and FAILME payloads get HTTP 500."""

    last_chat_body = b""

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """Serve one fake transcription/chat response (500 for /fail or FAILME payloads)."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
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
    from PyQt6.QtGui import QCursor
    from PyQt6.QtWidgets import QApplication, QPushButton

    from app.bootstrap import configure_base_logging
    configure_base_logging()

    from app.application import WhisperTyperApp
    from app.core.netutil import generate_test_wav_bytes
    from app.ui.floating_buttons import RecordingPromptOverlay
    from app.ui.tooltip import MouseFollowerTooltip

    qapp = QApplication([sys.argv[0]])
    qapp.setQuitOnLastWindowClosed(False)

    results: list[str] = []
    failed = False

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

    wt = WhisperTyperApp()

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
              wt.config["hotkey"] == "<ctrl>+<shift>+<f12>" and wt.config["config_schema_version"] == 2)
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
        wt.rephrasing_api_key_input.clear()
        check(
            "startup: transformation warning shown without API key",
            not wt.transformations_unavailable_label.isHidden(),
        )
        wt.rephrasing_api_key_input.setText("sk-test-dummy")
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
        wt.hide()  # Keep QApplication.quit() from being intercepted by the tray-style closeEvent.
        check("startup: hotkey bindings parsed", len(wt.hotkey_bindings) == 2,
              "; ".join(b["display"] for b in wt.hotkey_bindings))
        step2_transcribe()

    def step2_transcribe() -> None:
        """Drive one file transcription through the worker pipeline into the clipboard."""
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
                      finish()),
             lambda: (check("PROBE batch completes despite failure", False, "timeout"), finish()))

    def finish() -> None:
        """Shut the app down through its real quit path."""
        wt.quit_app()

    QTimer.singleShot(400, step1_startup)
    QTimer.singleShot(60_000, qapp.quit)  # watchdog: never hang

    rc = qapp.exec()

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
