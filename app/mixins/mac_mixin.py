"""MacMixin — macOS-native tray/status icon rendering and permission prompts."""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import pyaudio
from PyQt6.QtCore import QRectF, Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.core.env import is_MACOS
from app.core.frameworks import (
    AXIsProcessTrustedWithOptions,
    CGPreflightListenEventAccess,
    CGRequestListenEventAccess,
    kAXTrustedCheckOptionPrompt,
)
from app.core.paths import resource_path


class MacMixin:
    """macOS-native tray/status icon rendering and permission prompts."""

    def _get_macos_status_icon(self) -> QIcon:
        """Return the macOS menu bar status icon, preferring a template image."""
        if not is_MACOS:
            return QIcon()

        icon_candidates = [
            resource_path("resources", "whispertyperStatusTemplate.png"),
            resource_path("resources", "whispertyper-status.png"),
        ]

        for icon_path in icon_candidates:
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
                if not icon.isNull():
                    return icon
        return QIcon()

    def _tint_pixmap(self, pixmap: QPixmap, color: QColor) -> QPixmap:
        """Tint a pixmap while preserving its original alpha mask."""
        if pixmap.isNull():
            return QPixmap()

        tinted = QPixmap(pixmap.size())
        tinted.fill(Qt.GlobalColor.transparent)

        painter = QPainter(tinted)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), color)
        painter.end()
        return tinted

    def _crop_pixmap_to_visible_bounds(self, pixmap: QPixmap) -> QPixmap:
        """Trim transparent padding around a pixmap so status icons render at full size."""
        if pixmap.isNull():
            return QPixmap()

        image = pixmap.toImage()
        min_x = image.width()
        min_y = image.height()
        max_x = -1
        max_y = -1

        for y in range(image.height()):
            for x in range(image.width()):
                if image.pixelColor(x, y).alpha() > 0:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

        if max_x < min_x or max_y < min_y:
            return pixmap

        content_width = max_x - min_x + 1
        content_height = max_y - min_y + 1
        padding = max(1, int(round(max(content_width, content_height) * 0.05)))

        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.width() - 1, max_x + padding)
        max_y = min(image.height() - 1, max_y + padding)

        return QPixmap.fromImage(image.copy(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))

    def _build_macos_status_symbol_pixmap(self, size: int, color: QColor) -> QPixmap:
        """Draw a simplified tray-only WhisperTyper mark that stays legible in the macOS menu bar."""
        if size <= 0:
            return QPixmap()

        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)

        design_size = 24.0
        scale = size / design_size
        bar_height = 3.2 * scale
        radius = bar_height / 2.0
        left_bars = [
            (4.2, 1.8, 5.8),
            (2.8, 6.3, 7.8),
            (1.5, 10.8, 9.8),
            (3.2, 15.4, 8.2),
            (5.0, 19.8, 6.4),
        ]

        for x, y, width in left_bars:
            painter.drawRoundedRect(QRectF(x * scale, y * scale, width * scale, bar_height), radius, radius)
            mirrored_x = design_size - x - width
            painter.drawRoundedRect(QRectF(mirrored_x * scale, y * scale, width * scale, bar_height), radius, radius)

        painter.end()
        return pixmap

    def _get_macos_status_pixmap(self, size: int, color: Optional[QColor] = None) -> QPixmap:
        """Load a scaled menu bar status pixmap for macOS and optionally recolor it."""
        if is_MACOS:
            tray_color = color if color is not None else QColor(0, 0, 0)
            symbol_pixmap = self._build_macos_status_symbol_pixmap(size, tray_color)
            if not symbol_pixmap.isNull():
                return symbol_pixmap

        icon = self._get_macos_status_icon()
        if icon.isNull():
            return QPixmap()

        source_size = max(size * 8, 128)
        pixmap = icon.pixmap(source_size, source_size)
        pixmap = self._crop_pixmap_to_visible_bounds(pixmap)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        if pixmap.isNull() or color is None:
            return pixmap
        return self._tint_pixmap(pixmap, color)

    def _get_macos_tray_device_pixel_ratio(self) -> float:
        """Return the device pixel ratio used for crisp macOS menu bar icons."""
        screen = QApplication.primaryScreen()
        if screen is not None:
            try:
                return max(1.0, float(screen.devicePixelRatio()))
            except Exception:
                pass
        return 2.0 if is_MACOS else 1.0

    def _get_macos_tray_icon(self, size: int = 18) -> QIcon:
        """Return the macOS tray icon as a white monochrome symbol."""
        scale = self._get_macos_tray_device_pixel_ratio()
        pixel_size = max(18, int(round(size * scale)))
        pixmap = self._get_macos_status_pixmap(pixel_size, QColor(255, 255, 255))
        if pixmap.isNull():
            return QIcon()
        pixmap.setDevicePixelRatio(scale)
        return QIcon(pixmap)

    def _get_brand_pixmap(self, size: int) -> QPixmap:
        """Load a scaled brand pixmap for macOS header and tray usage."""
        icon = self._get_app_icon()
        if icon.isNull():
            return QPixmap()
        return icon.pixmap(size, size)

    def _build_macos_recording_tray_icon(self, level: float) -> QIcon:
        """Render a branded macOS tray icon that keeps the logo visible while recording."""
        scale = self._get_macos_tray_device_pixel_ratio()
        base_size = max(20, int(round(20 * scale)))
        base_pixmap = self._get_macos_status_pixmap(base_size, QColor(255, 255, 255))
        if base_pixmap.isNull():
            base_pixmap = self._get_brand_pixmap(base_size)
        if base_pixmap.isNull():
            return QIcon()

        canvas_size = max(24, int(round(24 * scale)))
        pixmap = QPixmap(canvas_size, canvas_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        pixmap.setDevicePixelRatio(scale)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        icon_offset = int(round(2 * scale))
        painter.drawPixmap(icon_offset, icon_offset, base_pixmap)

        indicator_size = 5 + max(0, min(3, int(round(level * 3))))
        indicator_color = QColor(16, 199, 222) if level < 0.82 else QColor(18, 124, 243)
        indicator_size_px = max(4, int(round(indicator_size * scale)))
        indicator_offset = int(round(2 * scale))
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1))
        painter.setBrush(indicator_color)
        painter.drawEllipse(
            canvas_size - indicator_size_px - indicator_offset,
            canvas_size - indicator_size_px - indicator_offset,
            indicator_size_px,
            indicator_size_px,
        )
        painter.end()
        return QIcon(pixmap)

    def _check_and_warn_macos_permissions(self, permission_type: str) -> None:
        """
        Checks if on macOS and if a permission warning is needed.
        If so, emits a signal to show the warning dialog in the main thread.
        This function is non-blocking and safe to call from any thread.
        Args:
            permission_type (str): 'input_monitoring', 'accessibility' or 'microphone'.
        """
        if not is_MACOS:
            return

        config_key = f"macos_{permission_type}_info_shown"
        if not self.config.get(config_key, False):
            # Mark as shown immediately to prevent repeated dialogs
            self.config[config_key] = True
            self.save_config()

            # Points to the dedicated macOS installation guide (permissions walkthrough).
            url = "https://github.com/bjspi/WhisperTyper/blob/main/docs/INSTALL_MACOS.md"

            if permission_type == 'input_monitoring':
                title = self.translator.tr("macos_input_monitoring_title")
                text = self.translator.tr("macos_input_monitoring_text")
            elif permission_type == 'accessibility':
                title = self.translator.tr("macos_accessibility_title")
                text = self.translator.tr("macos_accessibility_text")
            elif permission_type == 'microphone':
                title = self.translator.tr("macos_microphone_title")
                text = self.translator.tr("macos_microphone_text")
            else:
                return

            self.show_permission_dialog_signal.emit(title, text, url)

    def _should_request_macos_startup_permissions(self) -> bool:
        """Return whether the proactive macOS permission flow should run on launch."""
        return is_MACOS and bool(getattr(sys, "frozen", False))

    def _request_macos_input_monitoring_permission(self) -> bool:
        """Ask macOS for Input Monitoring access used for global hotkeys."""
        if not self._should_request_macos_startup_permissions():
            return True
        if not CGPreflightListenEventAccess or not CGRequestListenEventAccess:
            return True

        try:
            if CGPreflightListenEventAccess():
                return True
            granted = bool(CGRequestListenEventAccess())
            if not granted:
                logging.info("macOS Input Monitoring permission is not granted yet.")
                self._check_and_warn_macos_permissions('input_monitoring')
            return granted
        except Exception as e:
            logging.warning(f"Failed to request macOS Input Monitoring permission: {e}")
            self._check_and_warn_macos_permissions('input_monitoring')
            return False

    def _request_macos_accessibility_permission(self) -> bool:
        """Ask macOS for Accessibility access used for hotkeys and clipboard insertion."""
        if not self._should_request_macos_startup_permissions():
            return True
        if not AXIsProcessTrustedWithOptions or not kAXTrustedCheckOptionPrompt:
            return True

        try:
            trusted = bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}))
            if not trusted:
                logging.info("macOS Accessibility permission is not granted yet.")
                self._check_and_warn_macos_permissions('accessibility')
            return trusted
        except Exception as e:
            logging.warning(f"Failed to request macOS Accessibility permission: {e}")
            self._check_and_warn_macos_permissions('accessibility')
            return False

    def _request_macos_microphone_permission(self) -> bool:
        """Trigger the macOS microphone prompt once on app startup."""
        if not self._should_request_macos_startup_permissions():
            return True

        if self._can_use_macos_native_recorder():
            try:
                self._start_macos_native_recording()
                self._stop_macos_native_recording(discard=True)
                logging.info("macOS microphone permission preflight succeeded.")
                return True
            except Exception as e:
                logging.info(f"macOS native microphone permission preflight did not complete: {e}")
                self._check_and_warn_macos_permissions('microphone')
                return False

        temp_audio = None
        temp_stream = None
        try:
            temp_audio = pyaudio.PyAudio()
            stream_candidates = self._get_preferred_input_stream_candidates(temp_audio)
            last_error: Optional[Exception] = None
            for candidate in stream_candidates:
                open_kwargs = {
                    "format": pyaudio.paInt16,
                    "channels": candidate["channels"],
                    "rate": candidate["samplerate"],
                    "input": True,
                    "frames_per_buffer": self.chunk_size,
                }
                if candidate["index"] is not None:
                    open_kwargs["input_device_index"] = candidate["index"]
                try:
                    temp_stream = temp_audio.open(**open_kwargs)
                    break
                except Exception as e:
                    last_error = e
                    logging.info(
                        "macOS extension: Microphone preflight failed on "
                        f"'{candidate['name']}' (index={candidate['index']}, rate={candidate['samplerate']} Hz): {e}"
                    )
            if not temp_stream and last_error:
                raise last_error
            logging.info("macOS microphone permission preflight succeeded.")
            return True
        except Exception as e:
            logging.info(f"macOS microphone permission preflight did not complete: {e}")
            self._check_and_warn_macos_permissions('microphone')
            return False
        finally:
            if temp_stream:
                try:
                    temp_stream.stop_stream()
                except Exception:
                    pass
                try:
                    temp_stream.close()
                except Exception:
                    pass
            if temp_audio:
                try:
                    temp_audio.terminate()
                except Exception:
                    pass

    def _request_macos_startup_permissions(self) -> None:
        """Proactively trigger the important macOS permission prompts on app launch."""
        if not self._should_request_macos_startup_permissions() or self._macos_startup_permissions_requested:
            return

        self._macos_startup_permissions_requested = True
        self._request_macos_input_monitoring_permission()
        self._request_macos_accessibility_permission()
        self._request_macos_microphone_permission()

    def _ensure_macos_hotkey_permissions(self) -> None:
        """Prompt for macOS hotkey permissions once per session, even for source-based runs."""
        if not is_MACOS or self._macos_hotkey_permissions_checked:
            return

        self._macos_hotkey_permissions_checked = True

        try:
            if CGPreflightListenEventAccess and not CGPreflightListenEventAccess():
                CGRequestListenEventAccess()
                self._check_and_warn_macos_permissions('input_monitoring')
        except Exception as e:
            logging.warning(f"Failed to request macOS Input Monitoring permission for source run: {e}")
            self._check_and_warn_macos_permissions('input_monitoring')

        try:
            if AXIsProcessTrustedWithOptions and kAXTrustedCheckOptionPrompt:
                trusted = bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}))
                if not trusted:
                    self._check_and_warn_macos_permissions('accessibility')
        except Exception as e:
            logging.warning(f"Failed to request macOS Accessibility permission for source run: {e}")
            self._check_and_warn_macos_permissions('accessibility')

    def _show_permission_dialog_slot(self, title: str, text: str, settings_url: str) -> None:
        """
        Shows the macOS permission information dialog. This runs in the main GUI thread.
        Args:
            title (str): The dialog title.
            text (str): The dialog message.
            settings_url (str): The URL to open system settings.
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.addButton(self.translator.tr("ok_button"), QMessageBox.ButtonRole.AcceptRole)
        open_instructions_button = None
        if settings_url:
            # Use the new translation key for the GitHub instructions button
            open_instructions_button = msg_box.addButton(self.translator.tr("macos_github_instructions_button"), QMessageBox.ButtonRole.ActionRole)

        msg_box.exec()

        # Check which button was clicked after the dialog is closed
        if msg_box.clickedButton() == open_instructions_button:
            QDesktopServices.openUrl(QUrl(settings_url))
