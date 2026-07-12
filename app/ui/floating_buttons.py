"""Floating button palette shown near the cursor."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QCursor, QKeyEvent
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from app.core.env import is_MACOS
from app.ui import theme


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
