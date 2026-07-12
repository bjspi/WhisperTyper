"""Frameless tooltip that follows the cursor and self-closes."""
from __future__ import annotations

import html
from typing import Optional

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QCursor
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from app.ui import theme


class MouseFollowerTooltip(QWidget):
    """
    A custom frameless widget that follows the mouse cursor and closes after a timeout.
    This class is designed to be instantiated and managed via its static 'show_tooltip' method.
    """

    _instance: Optional['MouseFollowerTooltip'] = None
    _close_timer: Optional[QTimer] = None
    _move_timer: Optional[QTimer] = None
    _spinner_timer: Optional[QTimer] = None

    # Braille spinner frames — a compact "mini spinner" that renders on all platforms.
    _SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    # Completion glyph shown (in place of the spinner) when a job finishes successfully.
    _CHECK_GLYPH = "✓"
    # Safety net for persistent (spinner) tooltips so a missed hide can never leave it
    # spinning forever. Comfortably longer than the worker's own network timeouts.
    _SPINNER_SAFETY_MS = 300000

    def __init__(self, message: str, timeout_ms: int = 2000, spinner: bool = False,
                 check: bool = False) -> None:
        """
        Initializes the tooltip widget.

        Args:
            message (str): The text message to display in the tooltip.
            timeout_ms (int): The duration in milliseconds before the tooltip automatically closes.
            spinner (bool): If True, prepend an animated mini spinner and keep the tooltip
                visible (no auto-close) until it is explicitly hidden or replaced. Used for the
                "transcribing…" state that must persist until the worker actually finishes.
            check (bool): If True, prepend a static green checkmark instead of the spinner. Used
                for the "done ✓" state that replaces the spinner when a job finishes. Auto-closes
                after ``timeout_ms``. Ignored when ``spinner`` is True.
        """
        super().__init__()
        self._message = message
        self._spinner = spinner
        self._check = check and not spinner
        self._spinner_index = 0
        # Set window flags to create a frameless, top-level tooltip
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.ToolTip | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.label = QLabel(self)
        dark = theme.is_dark_mode(QApplication.instance())
        pal = theme.palette(dark)
        # A green that reads as "success" and stays legible on the panel in either theme.
        self._check_color = "#66bb6a" if dark else "#2e7d32"
        self.label.setStyleSheet(
            f"background-color: {pal['panel']}; color: {pal['text']}; "
            f"border: 1px solid {pal['accent']}; padding: 7px 11px; "
            f"border-radius: 9px; font-size: 12px;"
        )
        self._render()

        self.move_to_mouse()
        self.show()

        # Timer to close the tooltip after the specified timeout. In spinner mode the tooltip is
        # persistent, so we only arm a long safety timeout instead of the caller's value.
        if MouseFollowerTooltip._close_timer:
            MouseFollowerTooltip._close_timer.stop()
        MouseFollowerTooltip._close_timer = QTimer(self)
        MouseFollowerTooltip._close_timer.setSingleShot(True)
        MouseFollowerTooltip._close_timer.timeout.connect(self.close)
        MouseFollowerTooltip._close_timer.start(self._SPINNER_SAFETY_MS if spinner else timeout_ms)

        # Timer to update the tooltip's position to follow the mouse
        if MouseFollowerTooltip._move_timer:
            MouseFollowerTooltip._move_timer.stop()
        MouseFollowerTooltip._move_timer = QTimer(self)
        MouseFollowerTooltip._move_timer.timeout.connect(self.move_to_mouse)
        MouseFollowerTooltip._move_timer.start(16)  # ~60 FPS update rate

        # Timer to animate the spinner glyph
        if MouseFollowerTooltip._spinner_timer:
            MouseFollowerTooltip._spinner_timer.stop()
            MouseFollowerTooltip._spinner_timer = None
        if spinner:
            MouseFollowerTooltip._spinner_timer = QTimer(self)
            MouseFollowerTooltip._spinner_timer.timeout.connect(self._advance_spinner)
            MouseFollowerTooltip._spinner_timer.start(110)

        MouseFollowerTooltip._instance = self

    def _render(self) -> None:
        """Refresh the label text (spinner frame / checkmark / plain) and resize to fit."""
        if self._spinner:
            self.label.setTextFormat(Qt.TextFormat.PlainText)
            frame = self._SPINNER_FRAMES[self._spinner_index % len(self._SPINNER_FRAMES)]
            self.label.setText(f"{frame}  {self._message}")
        elif self._check:
            # Rich text so only the checkmark is tinted green; escape the message so any stray
            # angle brackets/ampersands in transcribed text don't get parsed as markup.
            self.label.setTextFormat(Qt.TextFormat.RichText)
            self.label.setText(
                f"<span style='color:{self._check_color}; font-weight:bold;'>{self._CHECK_GLYPH}</span>"
                f"&nbsp;&nbsp;{html.escape(self._message)}"
            )
        else:
            self.label.setTextFormat(Qt.TextFormat.PlainText)
            self.label.setText(self._message)
        self.label.adjustSize()
        self.resize(self.label.size())

    def _advance_spinner(self) -> None:
        """Advance to the next spinner frame."""
        self._spinner_index += 1
        self._render()

    def move_to_mouse(self) -> None:
        """Moves the tooltip to the current cursor position with a slight offset."""
        pos = QCursor.pos() + QPoint(15, 15)
        self.move(pos)

    def closeEvent(self, event: Optional[QCloseEvent]) -> None:
        """
        Overrides the close event to ensure timers are stopped and the static instance is cleared.

        Args:
            event: The close event.
        """
        # Stop timers to prevent them from running after the widget is gone
        if MouseFollowerTooltip._move_timer:
            MouseFollowerTooltip._move_timer.stop()
            MouseFollowerTooltip._move_timer = None
        if MouseFollowerTooltip._close_timer:
            MouseFollowerTooltip._close_timer.stop()
            MouseFollowerTooltip._close_timer = None
        if MouseFollowerTooltip._spinner_timer:
            MouseFollowerTooltip._spinner_timer.stop()
            MouseFollowerTooltip._spinner_timer = None

        # Clear the static instance reference if it's this instance
        if MouseFollowerTooltip._instance is self:
            MouseFollowerTooltip._instance = None

        if event is not None:
            event.accept()

    @staticmethod
    def show_tooltip(message: str, timeout_ms: int = 2000, spinner: bool = False,
                     check: bool = False) -> None:
        """
        Displays a new tooltip. If one is already visible, it is closed first.

        Args:
            message (str): The message to display.
            timeout_ms (int): How long the tooltip should be visible in milliseconds.
            spinner (bool): If True, show a persistent animated spinner (see __init__).
            check (bool): If True, prepend a static green checkmark (see __init__).
        """
        # Close the previous tooltip if it exists
        if MouseFollowerTooltip._instance:
            MouseFollowerTooltip._instance.close()

        # Create a new instance
        MouseFollowerTooltip(message, timeout_ms, spinner, check)

    @staticmethod
    def hide_tooltip() -> None:
        """Close the currently visible tooltip, if any (used to end the spinner state)."""
        if MouseFollowerTooltip._instance:
            MouseFollowerTooltip._instance.close()
