"""Floating button palette shown near the cursor."""
from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import QEasingCurve, QPoint, Qt, QTimer, QVariantAnimation
from PyQt6.QtGui import QColor, QCursor, QFont, QKeyEvent, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.env import is_MACOS, is_WINDOWS
from app.ui import theme
from app.ui.flow_layout import FlowLayout


class FloatingButtonWindow(QWidget):
    """Cross‑platform floating button palette near the cursor."""

    _instance: Optional['FloatingButtonWindow'] = None

    def __init__(self, buttons: List[Dict[str, str]], selected_text: str,
                 on_button_click_callback: Callable[..., None]) -> None:
        """Build the floating button palette near the cursor."""
        # Close previous instance
        if FloatingButtonWindow._instance:
            FloatingButtonWindow._instance.close()
        super().__init__()
        FloatingButtonWindow._instance = self

        # Platform specific flags:
        # macOS: Dialog improves stacking; avoid focus stealing issues; stays on top.
        # Windows/Linux: Tool avoids taskbar entry; Frameless + StayOnTop; show without activation.
        if is_MACOS:
            flags = (Qt.WindowType.FramelessWindowHint |
                     Qt.WindowType.Dialog |
                     Qt.WindowType.WindowStaysOnTopHint)
        else:
            flags = (Qt.WindowType.FramelessWindowHint |
                     Qt.WindowType.Tool |
                     Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # Determine if we auto-close on focus loss (avoid on macOS due to premature closes).
        self._close_on_focus_out = not is_MACOS

        pal = theme.palette(theme.is_dark_mode(QApplication.instance()))
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {pal['panel']};
                border: 1px solid {pal['border']};
                border-radius: 12px;
                color: {pal['text']};
            }}
            QPushButton {{
                background-color: {pal['panel2']};
                border: 1px solid {pal['border']};
                padding: 6px 10px;
                border-radius: 8px;
                text-align: left;
                font-size: 12px;
                color: {pal['text']};
            }}
            QPushButton:hover {{ border-color: {pal['accent']}; color: {pal['accent']}; }}
            QPushButton:pressed {{ background-color: {pal['hover']}; }}
            QPushButton#closeButton {{
                font-weight: bold;
                font-size: 14px;
                min-width: 22px; max-width: 22px;
                min-height: 22px; max-height: 22px;
                padding: 0px 0px 2px 0px;
                text-align: center;
                border-radius: 11px;
                background-color: {pal['panel2']};
                border: 1px solid {pal['border']};
                color: {pal['text']};
            }}
            QPushButton#closeButton:hover {{ border-color: {pal['accent']}; color: {pal['accent']}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(0, 0, 0, 4)
        top_bar_layout.addStretch()
        close_button = QPushButton("×")
        close_button.setObjectName("closeButton")
        close_button.setToolTip("Close (Esc)")
        close_button.clicked.connect(self.close)
        top_bar_layout.addWidget(close_button)
        layout.addLayout(top_bar_layout)

        for button_info in buttons:
            caption = button_info.get("caption", "Unnamed")
            prompt_text = button_info.get("text", "")
            btn = QPushButton(caption)
            btn.clicked.connect(partial(on_button_click_callback, prompt_text, selected_text, self))
            layout.addWidget(btn)

        self._position_near_cursor()
        self.show()

    def _position_near_cursor(self) -> None:
        """Position window near cursor and clamp inside available screen."""
        pos = QCursor.pos() + QPoint(15, 15)
        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.adjustSize()
            w, h = self.width(), self.height()
            x = min(max(pos.x(), geo.left()), geo.right() - w)
            y = min(max(pos.y(), geo.top()), geo.bottom() - h)
            self.move(x, y)
        else:
            self.move(pos)

    def keyPressEvent(self, event: Optional[QKeyEvent]) -> None:
        """Close the palette on Escape."""
        if event is not None and event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event: Any) -> None:
        """Close the palette when it loses focus (non-macOS)."""
        if self._close_on_focus_out:
            self.close()
        super().focusOutEvent(event)

    def closeEvent(self, event: Any) -> None:
        """Clear the singleton instance on close."""
        if FloatingButtonWindow._instance is self:
            FloatingButtonWindow._instance = None
        super().closeEvent(event)


class _RecordingDot(QWidget):
    """Small red dot that gently pulses while the recording runs."""

    def __init__(self, color: QColor, parent: Optional[QWidget] = None) -> None:
        """Start the pulse right away; it stops with the palette."""
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._color = QColor(color)
        self._opacity = 1.0
        self._pulse = QVariantAnimation(self)
        self._pulse.setStartValue(1.0)
        self._pulse.setKeyValueAt(0.5, 0.35)
        self._pulse.setEndValue(1.0)
        self._pulse.setDuration(1400)
        self._pulse.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._set_opacity)
        self._pulse.start()

    def _set_opacity(self, value: Any) -> None:
        """Repaint with the animated opacity."""
        self._opacity = float(value)
        self.update()

    def stop(self) -> None:
        """Stop the animation (the palette is closing)."""
        self._pulse.stop()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt API
        """Draw the dot; its opacity breathes between bright and dim."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(1, 1, 10, 10)


class RecordingPromptOverlay(QWidget):
    """Persistent, non-activating prompt selector shown for one microphone recording.

    A rounded card with a pulsing recording dot, the elapsed time and one chip per prompt; the
    chips wrap onto further lines when the captions do not fit next to each other.
    """

    _instance: Optional['RecordingPromptOverlay'] = None
    #: Room around the card for its drop shadow; the visible card sits this far inside the window.
    SHADOW_MARGIN = 14
    #: The card grows with its chips up to this width, then the chips wrap.
    MAX_CARD_WIDTH = 460
    _CARD_PADDING = 12

    def __init__(self, prompts: List[Dict[str, Any]], status_text: str, none_text: str,
                 on_selection_changed: Callable[[Optional[Dict[str, Any]]], None],
                 use_system_position: bool = True,
                 system_anchor: Optional[QPoint] = None,
                 dark: Optional[bool] = None) -> None:
        """Build the palette by the system status area or near the cursor without taking focus.

        The prompt marked ``auto_apply`` starts selected (otherwise "None"); only clicks report
        a selection, so the preselection itself leaves the automatic default in charge.
        ``dark`` follows the app's colour theme; None uses the operating system's scheme.
        """
        if RecordingPromptOverlay._instance:
            RecordingPromptOverlay._instance.close()
        super().__init__()
        RecordingPromptOverlay._instance = self
        self._on_selection_changed = on_selection_changed
        self._prompt_buttons: List[QPushButton] = []
        self._started = time.monotonic()

        window_kind = Qt.WindowType.Dialog if is_MACOS else Qt.WindowType.Tool
        flags = (
            Qt.WindowType.FramelessWindowHint
            | window_kind
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        if dark is None:
            dark = theme.is_dark_mode(QApplication.instance())
        pal = theme.palette(dark)
        recording_red = "#ff5a52" if dark else "#e5484d"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(self.SHADOW_MARGIN, self.SHADOW_MARGIN, self.SHADOW_MARGIN, self.SHADOW_MARGIN)
        self.card = QFrame(self)
        self.card.setObjectName("recordingPromptCard")
        outer.addWidget(self.card)
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 150 if dark else 60))
        self.card.setGraphicsEffect(shadow)
        self.card.setStyleSheet(f"""
            QFrame#recordingPromptCard {{
                background-color: {pal['panel']};
                border: 1px solid {pal['border']};
                border-radius: 12px;
            }}
            QLabel {{ background: transparent; border: none; color: {pal['text']}; }}
            QLabel#recordingPromptStatus {{ font-size: 13px; font-weight: 600; }}
            QLabel#recordingPromptTimer {{ font-size: 13px; color: {pal['muted']}; }}
            QPushButton {{
                background-color: {pal['panel2']};
                border: 1px solid {pal['border']};
                border-radius: 14px;
                padding: 5px 12px;
                font-size: 13px;
                color: {pal['text']};
                min-height: 18px;
            }}
            QPushButton:hover {{ border-color: {pal['accent']}; }}
            QPushButton:checked {{
                background-color: {pal['accent']};
                border-color: {pal['accent']};
                color: {pal['on_accent']};
                font-weight: 600;
            }}
        """)

        card_layout = QVBoxLayout(self.card)
        pad = self._CARD_PADDING
        card_layout.setContentsMargins(pad, pad, pad, pad)
        card_layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._dot = _RecordingDot(QColor(recording_red), self.card)
        header.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        status_label = QLabel(status_text.rstrip(" .\u2026"), self.card)
        status_label.setObjectName("recordingPromptStatus")
        header.addWidget(status_label)
        header.addStretch()
        self.timer_label = QLabel(self.card)
        self.timer_label.setObjectName("recordingPromptTimer")
        timer_font = self.timer_label.font()
        timer_font.setFeature(QFont.Tag("tnum"), 1)  # digits keep their width while counting
        self.timer_label.setFont(timer_font)
        header.addWidget(self.timer_label)
        card_layout.addLayout(header)

        self._chips = FlowLayout(spacing=6)
        card_layout.addLayout(self._chips, 1)  # the chips take the height their lines need
        none_button = self._build_prompt_button(none_text)
        none_button.clicked.connect(partial(self._select_prompt, None, none_button))
        preselected = none_button
        for prompt in prompts:
            button = self._build_prompt_button(prompt["caption"])
            button.clicked.connect(partial(self._select_prompt, prompt, button))
            if prompt.get("auto_apply"):
                preselected = button
        preselected.setChecked(True)

        self._clock = QTimer(self)
        self._clock.setInterval(250)
        self._clock.timeout.connect(self._update_elapsed)
        self._update_elapsed()
        self._clock.start()

        self._fit_card(card_layout, header)
        self._position_for_platform(use_system_position, system_anchor)
        self.show()

    def _build_prompt_button(self, caption: str) -> QPushButton:
        """Create one mouse-only, checkable chip showing the full caption."""
        button = QPushButton(caption.strip(), self.card)
        button.setCheckable(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._prompt_buttons.append(button)
        self._chips.addWidget(button)
        return button

    def _fit_card(self, card_layout: QVBoxLayout, header: QHBoxLayout) -> None:
        """Grow the card with its chips up to the maximum width; beyond that the chips wrap."""
        self.ensurePolished()  # apply the stylesheet (padding, font size) before measuring
        frame = 2 * self.card.frameWidth()  # the card's border sits outside its layout
        padding = 2 * self._CARD_PADDING + frame
        self._elide_long_captions(self.MAX_CARD_WIDTH - padding)
        chips_width = self._chips.sizeHint().width()
        width = min(self.MAX_CARD_WIDTH, max(header.sizeHint().width() + 24, chips_width) + padding)
        chips_height = self._chips.heightForWidth(width - padding)
        height = header.sizeHint().height() + card_layout.spacing() + chips_height + padding
        self.card.setFixedSize(width, height)
        self.adjustSize()

    def _elide_long_captions(self, max_chip_width: int) -> None:
        """Shorten a caption too long for one card line; the full caption becomes its tooltip."""
        for button in self._prompt_buttons:
            caption = button.text()
            chrome = button.sizeHint().width() - button.fontMetrics().horizontalAdvance(caption)
            elided = button.fontMetrics().elidedText(caption, Qt.TextElideMode.ElideRight, max_chip_width - chrome)
            if elided != caption:
                button.setText(elided)
                button.setToolTip(caption)

    def _update_elapsed(self) -> None:
        """Show the time since the palette opened as m:ss."""
        seconds = int(time.monotonic() - self._started)
        self.timer_label.setText(f"{seconds // 60}:{seconds % 60:02d}")

    def _select_prompt(self, prompt: Optional[Dict[str, Any]], selected_button: QPushButton) -> None:
        """Highlight one choice and report its palette entry (None for "None") to the app."""
        for button in self._prompt_buttons:
            button.setChecked(button is selected_button)
        self._on_selection_changed(prompt)

    def _position_for_platform(self, use_system_position: bool,
                               system_anchor: Optional[QPoint]) -> None:
        """Place the visible card by the OS status area or once near the cursor."""
        self.adjustSize()
        card_width, card_height = self.card.width(), self.card.height()
        if use_system_position and (is_WINDOWS or is_MACOS):
            screen = QApplication.screenAt(system_anchor) if system_anchor is not None else None
            screen = screen or QApplication.primaryScreen()
            if screen:
                available = screen.availableGeometry()
                full = screen.geometry()
                margin = 16
                anchor = system_anchor or (
                    QPoint(full.right(), full.bottom()) if is_WINDOWS
                    else QPoint(full.right(), full.top())
                )
                # macOS always uses the top-right status area promised by the setting. Windows
                # can place the taskbar on any edge, so follow the tray icon's nearest corner.
                if is_MACOS:
                    x = available.left() + available.width() - card_width - margin
                    y = available.top() + margin
                elif anchor.x() < full.center().x():
                    x = available.left() + margin
                else:
                    x = available.left() + available.width() - card_width - margin
                if not is_MACOS:
                    if anchor.y() < full.center().y():
                        y = available.top() + margin
                    else:
                        y = available.top() + available.height() - card_height - margin
                x = min(max(x, available.left()), max(available.left(), available.right() - card_width))
                y = min(max(y, available.top()), max(available.top(), available.bottom() - card_height))
                self._move_card(x, y)
                return

        pos = QCursor.pos() + QPoint(15, 15)
        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        if not screen:
            self._move_card(pos.x(), pos.y())
            return
        geo = screen.availableGeometry()
        x = min(max(pos.x(), geo.left()), max(geo.left(), geo.right() - card_width))
        y = min(max(pos.y(), geo.top()), max(geo.top(), geo.bottom() - card_height))
        self._move_card(x, y)

    def _move_card(self, x: int, y: int) -> None:
        """Move the window so the card itself (not its shadow margin) lands at ``x``, ``y``."""
        self.move(x - self.SHADOW_MARGIN, y - self.SHADOW_MARGIN)

    def closeEvent(self, event: Any) -> None:
        """Stop the clock and the pulse, and clear the singleton instance."""
        self._clock.stop()
        self._dot.stop()
        if RecordingPromptOverlay._instance is self:
            RecordingPromptOverlay._instance = None
        super().closeEvent(event)

    @staticmethod
    def close_current() -> None:
        """Close the active recording palette, if one exists."""
        if RecordingPromptOverlay._instance:
            RecordingPromptOverlay._instance.close()
