"""Live connection tests for the settings window (transcription / internet / rephrasing).

Single responsibility: drive the three "Test connection" buttons — read the current form
values, run the network probe (via ``services.net`` / ``services.rephrasing``) and report
the outcome through QMessageBox. The networking itself lives in ``services``.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton

from app.core.redaction import redact_for_log
from app.core.textutil import clean_model_name
from app.services import net
from app.services.http_transport import request
from app.services.netutil import PX_PROXY_PORT, is_px_running
from app.services.rephrasing import rephrase_text

#: Tiny, well-known connectivity endpoint (answers HTTP 204).
_CONNECTIVITY_URL = "https://www.google.com/generate_204"


class ConnectionTester:
    """Runs the settings window's three connection-test flows against the current form."""

    def __init__(self, window: Any) -> None:
        """Keep the settings window for its form widgets and translator."""
        self._w = window

    def _form_proxies(self) -> Optional[Dict[str, str]]:
        """Proxies from the current (unsaved) proxy fields."""
        w = self._w
        return net.proxies_for({"proxy_url": w.proxy_url_input.text(),
                                "use_local_px_proxy": w.use_px_proxy_checkbox.isChecked()})

    @contextmanager
    def _busy(self, button: QPushButton) -> Iterator[None]:
        """Show "testing…" on ``button`` while a test runs; restore its caption afterwards."""
        caption = button.text()
        button.setEnabled(False)
        button.setText(self._w.translator.tr("api_test_testing_button"))
        QApplication.processEvents()
        try:
            yield
        finally:
            button.setEnabled(True)
            button.setText(caption)

    def _fail(self, title_key: str, text_key: str, **kwargs: Any) -> None:
        """Warn about missing settings or a failed test."""
        QMessageBox.warning(self._w, self._w.translator.tr(title_key), self._w.translator.tr(text_key, **kwargs))

    def _crash(self, error: Exception) -> None:
        """Report an unexpected exception raised by a test."""
        tr = self._w.translator.tr
        QMessageBox.critical(self._w, tr("api_test_fail_title"), tr("api_test_exception_text", error=str(error)))

    def test_transcription(self) -> None:
        """Send a tiny test audio to the endpoint and report the classified outcome."""
        w = self._w
        api_url = w.api_endpoint_input.text().strip()
        api_key = w._ui_api_key("transcription")
        if not api_url or not api_key:
            self._fail("api_test_fail_title", "recording_no_api_keys")
            return
        model = clean_model_name(w.model_dropdown.currentText())
        with self._busy(w.test_transcription_api_button):
            try:
                result_key, detail = net.run_transcription_connection_test(api_url, api_key, model, self._form_proxies())
            except Exception as e:
                self._crash(e)
                return
        logging.info(f"Transcription connection test result: {result_key} ({redact_for_log(detail)})")
        message = w.translator.tr(f"conn_test_{result_key}", detail=detail)
        # Only a genuine 200 is a success; anything else must not read as "everything is fine".
        if result_key == "ok":
            QMessageBox.information(w, w.translator.tr("conn_test_title"), message)
        else:
            QMessageBox.warning(w, w.translator.tr("conn_test_title"), message)

    def test_internet(self) -> None:
        """Test plain internet reachability, honoring the proxy / px settings."""
        w = self._w
        proxies = self._form_proxies()
        manual_proxy = bool(w.proxy_url_input.text().strip())
        used_px = (proxies or {}).get("https", "").endswith(f":{PX_PROXY_PORT}") and not manual_proxy
        with self._busy(w.test_internet_button):
            try:
                response = request("GET", _CONNECTIVITY_URL, stage="internet_test", proxies=proxies, timeout=10)
                if response.status_code in (200, 204):
                    key = "internet_test_ok_px" if used_px else "internet_test_ok"
                    QMessageBox.information(w, w.translator.tr("internet_test_title"),
                                            w.translator.tr(key, detail=f"HTTP {response.status_code}"))
                    return
                reason_key, detail = "unknown", f"HTTP {response.status_code}: {(response.text or '')[:200]}"
            except Exception as e:
                reason_key, detail = net.classify_transport_error(e, _CONNECTIVITY_URL, proxies), str(e)
        logging.info(f"Internet test failed: {reason_key} ({redact_for_log(detail)})")
        message = w.translator.tr("internet_test_fail", detail=f"[{reason_key}] {detail}")
        # A running px the user has not opted into is a likely fix.
        if is_px_running() and not used_px and not manual_proxy:
            message += "\n\n" + w.translator.tr("internet_test_px_hint")
        QMessageBox.warning(w, w.translator.tr("internet_test_title"), message)

    def test_rephrasing(self) -> None:
        """Send a minimal chat request with the current rephrasing settings."""
        w = self._w
        api_url = w.rephrasing_api_url_input.text().strip()
        api_key = w._ui_api_key("rephrasing")
        model = w.rephrasing_model_input.currentText().strip()
        if not all([api_url, api_key, model]):
            self._fail("api_test_fail_title", "rephrase_api_settings_missing")
            return
        with self._busy(w.test_rephrasing_api_button):
            try:
                response_text = rephrase_text(
                    system_prompt="You are a test assistant.",
                    user_prompt="Reply with only the word 'Success'.",
                    api_url=api_url, api_key=api_key, model=model, temperature=0.0,
                    proxies=self._form_proxies(),
                )
            except Exception as e:
                self._crash(e)
                return
        if "success" in response_text.lower():
            QMessageBox.information(w, w.translator.tr("api_test_success_title"),
                                    w.translator.tr("api_test_success_text", response=response_text))
        else:
            self._fail("api_test_fail_title", "api_test_unexpected_response_text", response=response_text)
