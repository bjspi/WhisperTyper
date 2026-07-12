"""Live connection tests for the settings window (transcription / internet / rephrasing).

Single responsibility: drive the three "Test connection" buttons — read the current form
values, run the network probe (via ``services.net`` / ``services.rephrasing``) and report
the outcome through QMessageBox. The pure networking lives in ``services.net``; this class only
owns the UI orchestration, keeping it out of the already large settings mixin.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.core.netutil import PX_PROXY_PORT, is_px_running
from app.core.redaction import redact_for_log
from app.core.textutil import clean_model_name
from app.services import net
from app.services.rephrasing import rephrase_text


class ConnectionTester:
    """Runs the settings window's three connection-test flows against the current form."""

    def __init__(self, window: Any) -> None:
        """
        Args:
            window: The settings window (WhisperTyperApp). Used for its form widgets,
                translator, proxy resolution and rephrasing helper.
        """
        self._w = window

    def _run_transcription_connection_test(self, api_url: str, api_key: str,
                                           proxies: Optional[Dict[str, str]]) -> tuple[str, str]:
        """Send a tiny test audio to the endpoint and classify (delegates to services.net)."""
        w = self._w
        model = clean_model_name(w.model_dropdown.currentText())
        if w.model_dropdown.currentText() == "Custom":
            model = w.model_input.text().strip() or "whisper-1"
        return net.run_transcription_connection_test(api_url, api_key, model, proxies)

    def test_transcription(self) -> None:
        """Tests the transcription API connection (incl. proxy) and reports a differentiated result."""
        w = self._w
        api_url = w.api_endpoint_input.text().strip()
        api_key = w.api_key_input.text().strip()

        if not api_url or not api_key:
            QMessageBox.warning(
                w,
                w.translator.tr("api_test_fail_title"),
                w.translator.tr("recording_no_api_keys")
            )
            return

        proxies = w._ui_proxies()

        w.test_transcription_api_button.setEnabled(False)
        w.test_transcription_api_button.setText(w.translator.tr("api_test_testing_button"))
        QApplication.processEvents()

        try:
            result_key, detail = self._run_transcription_connection_test(api_url, api_key, proxies)
            logging.info(f"Transcription connection test result: {result_key} ({redact_for_log(detail)})")

            message = w.translator.tr(f"conn_test_{result_key}", detail=detail)
            # Only a genuine 200 counts as success; any other outcome (wrong URL, wrong key,
            # rejected request, proxy/connectivity issue) is shown as a warning so it is not
            # mistaken for "everything is fine".
            if result_key == "ok":
                QMessageBox.information(w, w.translator.tr("conn_test_title"), message)
            else:
                QMessageBox.warning(w, w.translator.tr("conn_test_title"), message)
        except Exception as e:
            QMessageBox.critical(
                w,
                w.translator.tr("api_test_fail_title"),
                w.translator.tr("api_test_exception_text", error=str(e))
            )
        finally:
            w.test_transcription_api_button.setEnabled(True)
            w.test_transcription_api_button.setText(w.translator.tr("test_connection_button"))

    def test_internet(self) -> None:
        """Tests plain internet reachability (google.com) honoring proxy / px settings."""
        w = self._w
        proxies = w._ui_proxies()
        px_running = is_px_running()
        used_px = bool(proxies) and proxies.get("https", "").endswith(f":{PX_PROXY_PORT}") \
            and not w.proxy_url_input.text().strip()

        w.test_internet_button.setEnabled(False)
        w.test_internet_button.setText(w.translator.tr("api_test_testing_button"))
        QApplication.processEvents()

        try:
            # generate_204 is a tiny, well-known connectivity endpoint (returns HTTP 204).
            try:
                response = requests.get(
                    "https://www.google.com/generate_204", proxies=proxies, timeout=10
                )
                reachable = response.status_code in (200, 204)
                detail = f"HTTP {response.status_code}"
                if reachable:
                    key = "internet_test_ok_px" if used_px else "internet_test_ok"
                    QMessageBox.information(
                        w, w.translator.tr("internet_test_title"),
                        w.translator.tr(key, detail=detail)
                    )
                    return
                reason_key, detail = "unknown", f"HTTP {response.status_code}: {(response.text or '')[:200]}"
            except requests.exceptions.ProxyError as e:
                reason_key, detail = "proxy", str(e)
            except requests.exceptions.SSLError as e:
                reason_key, detail = "ssl", str(e)
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                reason_key, detail = net.diagnose_connectivity("https://www.google.com", proxies), str(e)
            except Exception as e:
                reason_key, detail = "unknown", str(e)

            logging.info(f"Internet test failed: {reason_key} ({redact_for_log(detail)})")
            message = w.translator.tr("internet_test_fail", detail=f"[{reason_key}] {detail}")
            # If px is running but the user hasn't opted to use it, hint at that.
            if px_running and not used_px and not w.proxy_url_input.text().strip():
                message += "\n\n" + w.translator.tr("internet_test_px_hint")
            QMessageBox.warning(w, w.translator.tr("internet_test_title"), message)
        finally:
            w.test_internet_button.setEnabled(True)
            w.test_internet_button.setText(w.translator.tr("test_internet_button"))

    def test_rephrasing(self) -> None:
        """Tests the rephrasing API settings by sending a simple request."""
        w = self._w
        api_url = w.rephrasing_api_url_input.text().strip()
        api_key = w.rephrasing_api_key_input.text().strip() or w.api_key_input.text().strip()
        model = w.rephrasing_model_input.text().strip()

        if not all([api_url, api_key, model]):
            QMessageBox.warning(
                w,
                w.translator.tr("api_test_fail_title"),
                w.translator.tr("rephrase_api_settings_missing")
            )
            return

        # Honor the current (unsaved) proxy settings, like the other two test buttons.
        proxies = w._ui_proxies()

        # Disable button to prevent multiple clicks
        w.test_rephrasing_api_button.setEnabled(False)
        w.test_rephrasing_api_button.setText(w.translator.tr("api_test_testing_button"))
        QApplication.processEvents()  # Update UI

        try:
            response_text = rephrase_text(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with only the word 'Success'.",
                api_url=api_url,
                api_key=api_key,
                model=model,
                temperature=0.0,
                proxies=proxies,
            )
            if "success" in response_text.lower():
                QMessageBox.information(
                    w,
                    w.translator.tr("api_test_success_title"),
                    w.translator.tr("api_test_success_text", response=response_text)
                )
            else:
                QMessageBox.warning(
                    w,
                    w.translator.tr("api_test_fail_title"),
                    w.translator.tr("api_test_unexpected_response_text", response=response_text)
                )
        except Exception as e:
            QMessageBox.critical(
                w,
                w.translator.tr("api_test_fail_title"),
                w.translator.tr("api_test_exception_text", error=str(e))
            )
        finally:
            # Re-enable button
            w.test_rephrasing_api_button.setEnabled(True)
            w.test_rephrasing_api_button.setText(w.translator.tr("test_api_button"))
