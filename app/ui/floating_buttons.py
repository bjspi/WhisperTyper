"""Floating button palette shown near the cursor."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import QPoint, Qt, QTextBoundaryFinder
from PyQt6.QtGui import QCursor, QKeyEvent
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.core.env import is_MACOS, is_WINDOWS
from app.ui import theme


def graphemes(text: str) -> List[str]:
    """User-perceived characters (emoji sequences and flags stay whole), via Qt's UAX #29 rules."""
    units = text.encode("utf-16-le")  # Qt reports boundaries in UTF-16 code units
    finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Grapheme, text)
    bounds = [0]
    while (position := finder.toNextBoundary()) != -1:
        bounds.append(position)
    return [units[start * 2:end * 2].decode("utf-16-le") for start, end in zip(bounds, bounds[1:])]


def compact_caption(caption: str, length: int = 3) -> str:
    """Short palette label: a caption starting with a symbol/emoji shows it alone, else ``length`` characters."""
    characters = graphemes(caption.strip())
    if characters and not characters[0].isalnum():
        return characters[0]
    return "".join(characters[:length]).strip()


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


class RecordingPromptOverlay(QWidget):
    """Persistent, non-activating prompt selector shown for one microphone recording."""

    _instance: Optional['RecordingPromptOverlay'] = None

    def __init__(self, prompts: List[Dict[str, Any]], status_text: str, none_text: str,
                 on_selection_changed: Callable[[Optional[str]], None],
                 use_system_position: bool = True,
                 system_anchor: Optional[QPoint] = None) -> None:
        """Build a fixed palette near the cursor without taking focus from the target app.

        The prompt marked ``auto_apply`` starts selected (otherwise "None"); only clicks report
        a selection, so the preselection itself leaves the automatic default in charge.
        """
        if RecordingPromptOverlay._instance:
            RecordingPromptOverlay._instance.close()
        super().__init__()
        RecordingPromptOverlay._instance = self
        self._on_selection_changed = on_selection_changed
        self._prompt_buttons: List[QPushButton] = []

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

        pal = theme.palette(theme.is_dark_mode(QApplication.instance()))
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {pal['panel']};
                border: 1px solid {pal['border']};
                border-radius: 12px;
                color: {pal['text']};
            }}
            QLabel {{
                border: none;
                color: {pal['text']};
                font-size: 12px;
                font-weight: 600;
                padding: 2px 4px 5px 4px;
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
            QPushButton:checked {{
                background-color: {pal['hover']};
                border: 2px solid {pal['accent']};
                color: {pal['accent']};
                font-weight: 600;
            }}
            QPushButton[compactPrompt="true"] {{
                min-width: 38px;
                max-width: 38px;
                padding: 6px 2px;
                text-align: center;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        status_label = QLabel(f"●  {status_text}")
        status_label.setStyleSheet(
            f"border: none; color: {pal['warn']}; font-size: 12px; "
            "font-weight: 600; padding: 2px 4px 5px 4px;"
        )
        layout.addWidget(status_label)

        none_button = self._build_prompt_button(none_text)
        none_button.clicked.connect(partial(self._select_prompt, None, none_button))
        none_row = QHBoxLayout()
        none_row.setContentsMargins(0, 0, 0, 0)
        none_row.addWidget(none_button)
        none_row.addStretch()
        layout.addLayout(none_row)
        preselected = none_button

        prompt_row = QHBoxLayout()
        prompt_row.setContentsMargins(0, 0, 0, 0)
        prompt_row.setSpacing(4)
        for prompt in prompts:
            caption = prompt["caption"]
            button = self._build_prompt_button(compact_caption(caption), compact=True)
            button.setToolTip(caption)
            button.clicked.connect(partial(self._select_prompt, prompt["text"], button))
            prompt_row.addWidget(button)
            if prompt.get("auto_apply"):
                preselected = button
        prompt_row.addStretch()
        layout.addLayout(prompt_row)
        preselected.setChecked(True)

        self._position_for_platform(use_system_position, system_anchor)
        self.show()

    def _build_prompt_button(self, caption: str, compact: bool = False) -> QPushButton:
        """Create one mouse-only, checkable palette button."""
        button = QPushButton(caption)
        button.setCheckable(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if compact:
            button.setProperty("compactPrompt", True)
        self._prompt_buttons.append(button)
        return button

    def _select_prompt(self, prompt_text: Optional[str], selected_button: QPushButton) -> None:
        """Highlight one choice and report its immutable prompt text to the app."""
        for button in self._prompt_buttons:
            button.setChecked(button is selected_button)
        self._on_selection_changed(prompt_text)

    def _position_for_platform(self, use_system_position: bool,
                               system_anchor: Optional[QPoint]) -> None:
        """Place the palette by the OS status area or once near the cursor."""
        self.adjustSize()
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
                    x = available.left() + available.width() - self.width() - margin
                    y = available.top() + margin
                elif anchor.x() < full.center().x():
                    x = available.left() + margin
                else:
                    x = available.left() + available.width() - self.width() - margin
                if not is_MACOS:
                    if anchor.y() < full.center().y():
                        y = available.top() + margin
                    else:
                        y = available.top() + available.height() - self.height() - margin
                x = min(max(x, available.left()), max(available.left(), available.right() - self.width()))
                y = min(max(y, available.top()), max(available.top(), available.bottom() - self.height()))
                self.move(x, y)
                return

        pos = QCursor.pos() + QPoint(15, 15)
        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        if not screen:
            self.move(pos)
            return
        geo = screen.availableGeometry()
        x = min(max(pos.x(), geo.left()), max(geo.left(), geo.right() - self.width()))
        y = min(max(pos.y(), geo.top()), max(geo.top(), geo.bottom() - self.height()))
        self.move(x, y)

    def closeEvent(self, event: Any) -> None:
        """Clear the singleton instance on close."""
        if RecordingPromptOverlay._instance is self:
            RecordingPromptOverlay._instance = None
        super().closeEvent(event)

    @staticmethod
    def close_current() -> None:
        """Close the active recording palette, if one exists."""
        if RecordingPromptOverlay._instance:
            RecordingPromptOverlay._instance.close()
