"""Live connection tests for the settings window (transcription / internet / rephrasing).

Single responsibility: drive the three "Test connection" buttons — read the current form
values, run the network probe (via ``services.net`` / ``services.rephrasing``) off the GUI
thread and report the outcome in message dialogs. The networking itself lives in ``services``.
"""
from __future__ import annotations

import logging
import threading
from functools import partial
from typing import Any, Callable, Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication, QPushButton

from app.core.api_keys import LOCAL_PROVIDER, provider_for_url
from app.core.models import LOCAL_TRANSCRIPTION_MODEL
from app.core.redaction import redact_for_log
from app.core.textutil import clean_model_name
from app.local_engine.supervisor import READY
from app.services import net
from app.services.http_transport import request
from app.services.netutil import PX_PROXY_PORT, is_px_running
from app.services.rephrasing import rephrase_text
from app.ui import message_dialog

#: Tiny, well-known connectivity endpoint (answers HTTP 204).
_CONNECTIVITY_URL = "https://www.google.com/generate_204"

#: A probe runs on its own thread and returns the report to show on the GUI thread.
_Probe = Callable[[], Callable[[], None]]


class ConnectionTester(QObject):
    """Runs the settings window's three connection-test flows against the current form.

    One test runs at a time: its button shows "testing…" until the probe thread has
    reported back through ``_reported`` (queued onto the GUI thread).
    """

    _reported = pyqtSignal(object)

    def __init__(self, window: Any) -> None:
        """Keep the settings window for its form widgets and translator."""
        super().__init__(window)
        self._w = window
        self._thread: Optional[threading.Thread] = None
        self._reported.connect(self._on_reported)

    @property
    def busy(self) -> bool:
        """Whether a test is still running."""
        return self._thread is not None and self._thread.is_alive()

    def _start(self, button: QPushButton, probe: _Probe) -> None:
        """Show "testing…" on ``button`` and run ``probe`` on a thread; a click meanwhile is ignored."""
        if self.busy:
            return
        caption = button.text()
        button.setEnabled(False)
        button.setText(self._w.translator.tr("api_test_testing_button"))

        def run() -> None:
            try:
                report = probe()
            except Exception as error:
                report = partial(self._crash, error)
            try:
                self._reported.emit((button, caption, report))
            except RuntimeError:
                pass  # the window was deleted while the probe ran

        self._thread = threading.Thread(target=run, name="ConnectionTest", daemon=True)
        self._thread.start()

    def _on_reported(self, result: Any) -> None:
        """GUI thread: restore the button and show the probe's report."""
        button, caption, report = result
        button.setEnabled(True)
        button.setText(caption)
        report()

    def wait(self, timeout: float = 10.0) -> None:
        """Block until the running test has reported (for tests; the app never waits)."""
        if self._thread is not None:
            self._thread.join(timeout)
        QApplication.processEvents()

    def _form_proxies(self) -> Optional[Dict[str, str]]:
        """Proxies from the current (unsaved) proxy fields."""
        w = self._w
        return net.proxies_for({"proxy_url": w.proxy_url_input.text(),
                                "use_local_px_proxy": w.use_px_proxy_checkbox.isChecked()})

    def _fail(self, title_key: str, text_key: str, **kwargs: Any) -> None:
        """Warn about missing settings or a failed test."""
        tr = self._w.translator.tr
        message_dialog.inform(self._w, tr(title_key), tr(text_key, **kwargs), tr("ok_button"), message_dialog.WARNING)

    def _crash(self, error: Exception) -> None:
        """Report an unexpected exception raised by a test."""
        tr = self._w.translator.tr
        message_dialog.inform(self._w, tr("api_test_fail_title"), tr("api_test_exception_text", error=str(error)),
                              tr("ok_button"), message_dialog.ERROR)

    def test_transcription(self) -> None:
        """Send a tiny test audio to the endpoint and report the classified outcome."""
        w = self._w
        api_url = w.api_endpoint_input.text().strip()
        api_key = w._ui_api_key("transcription")
        model = clean_model_name(w.model_dropdown.currentText())
        if provider_for_url(api_url) == LOCAL_PROVIDER:
            # The running engine is tested; it uses the saved settings of its page.
            if w.local_engine.status.state != READY:
                self._fail("api_test_fail_title", "local_test_not_ready")
                return
            api_url, api_key, model = w.local_engine.endpoint(), "", LOCAL_TRANSCRIPTION_MODEL
        elif not api_url or not w._ui_key_ready("transcription"):
            self._fail("api_test_fail_title", "recording_no_api_keys")
            return
        proxies = self._form_proxies()

        def probe() -> Callable[[], None]:
            result_key, detail = net.run_transcription_connection_test(api_url, api_key, model, proxies)
            return partial(self._report_transcription, result_key, detail)

        self._start(w.test_transcription_api_button, probe)

    def _report_transcription(self, result_key: str, detail: str) -> None:
        """Show the classified outcome of the transcription test."""
        w = self._w
        logging.info(f"Transcription connection test result: {result_key} ({redact_for_log(detail)})")
        message = w.translator.tr(f"conn_test_{result_key}", detail=detail)
        # Only a genuine 200 is a success; anything else must not read as "everything is fine".
        if result_key == "ok":
            message_dialog.inform(w, w.translator.tr("conn_test_title"), message, w.translator.tr("ok_button"),
                                  message_dialog.SUCCESS)
        else:
            message_dialog.inform(w, w.translator.tr("conn_test_title"), message, w.translator.tr("ok_button"),
                                  message_dialog.WARNING)

    def test_internet(self) -> None:
        """Test plain internet reachability, honoring the proxy / px settings."""
        w = self._w
        proxies = self._form_proxies()
        manual_proxy = bool(w.proxy_url_input.text().strip())
        used_px = (proxies or {}).get("https", "").endswith(f":{PX_PROXY_PORT}") and not manual_proxy

        def probe() -> Callable[[], None]:
            try:
                response = request("GET", _CONNECTIVITY_URL, stage="internet_test", proxies=proxies, timeout=10)
                if response.status_code in (200, 204):
                    return partial(self._report_internet_ok, used_px, response.status_code)
                reason_key, detail = "unknown", f"HTTP {response.status_code}: {(response.text or '')[:200]}"
            except Exception as e:
                reason_key, detail = net.classify_transport_error(e, _CONNECTIVITY_URL, proxies), str(e)
            # A running px the user has not opted into is a likely fix.
            px_hint = is_px_running() and not used_px and not manual_proxy
            return partial(self._report_internet_failed, reason_key, detail, px_hint)

        self._start(w.test_internet_button, probe)

    def _report_internet_ok(self, used_px: bool, status_code: int) -> None:
        """Show the reachability success, naming px when it carried the request."""
        w = self._w
        key = "internet_test_ok_px" if used_px else "internet_test_ok"
        message_dialog.inform(w, w.translator.tr("internet_test_title"),
                              w.translator.tr(key, detail=f"HTTP {status_code}"), w.translator.tr("ok_button"),
                              message_dialog.SUCCESS)

    def _report_internet_failed(self, reason_key: str, detail: str, px_hint: bool) -> None:
        """Show the reachability failure with its classified reason."""
        w = self._w
        logging.info(f"Internet test failed: {reason_key} ({redact_for_log(detail)})")
        message = w.translator.tr("internet_test_fail", detail=f"[{reason_key}] {detail}")
        if px_hint:
            message += "\n\n" + w.translator.tr("internet_test_px_hint")
        message_dialog.inform(w, w.translator.tr("internet_test_title"), message, w.translator.tr("ok_button"),
                              message_dialog.WARNING)

    def test_rephrasing(self) -> None:
        """Send a minimal chat request with the current rephrasing settings."""
        w = self._w
        api_url = w.rephrasing_api_url_input.text().strip()
        api_key = w._ui_api_key("rephrasing")
        model = w.rephrasing_model_input.currentText().strip()
        if not api_url or not model or not w._ui_key_ready("rephrasing"):
            self._fail("api_test_fail_title", "rephrase_api_settings_missing")
            return
        proxies = self._form_proxies()

        def probe() -> Callable[[], None]:
            response_text = rephrase_text(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with only the word 'Success'.",
                api_url=api_url, api_key=api_key, model=model, temperature=0.0,
                proxies=proxies,
            )
            return partial(self._report_rephrasing, response_text)

        self._start(w.test_rephrasing_api_button, probe)

    def _report_rephrasing(self, response_text: str) -> None:
        """Show whether the model answered with the expected word."""
        w = self._w
        if "success" in response_text.lower():
            message_dialog.inform(w, w.translator.tr("api_test_success_title"),
                                  w.translator.tr("api_test_success_text", response=response_text),
                                  w.translator.tr("ok_button"), message_dialog.SUCCESS)
        else:
            self._fail("api_test_fail_title", "api_test_unexpected_response_text", response=response_text)
