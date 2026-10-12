"""Floating button palette shown near the cursor."""
from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import QByteArray, QEasingCurve, QPoint, QSize, Qt, QTimer, QVariantAnimation
from PyQt6.QtGui import QColor, QCursor, QFont, QIcon, QKeyEvent, QMouseEvent, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.env import is_MACOS, is_WINDOWS
from app.platform.macos import make_nonactivating
from app.platform.windows import hotkey_id_of, register_window_hotkey, unregister_window_hotkey
from app.ui import theme
from app.ui.flow_layout import FlowLayout


class _PromptCard(QWidget):
    """Frameless window holding a rounded, shadowed card with a wrapping row of prompt chips.

    The visible card sits ``SHADOW_MARGIN`` inside the translucent window, so the drop shadow is
    never clipped; ``_move_card`` positions the card itself.
    """

    #: Room around the card for its drop shadow; the visible card sits this far inside the window.
    SHADOW_MARGIN = 14
    #: The card grows with its chips up to this width, then the chips wrap.
    MAX_CARD_WIDTH = 460
    _CARD_PADDING = 12

    def _build_card(self, object_name: str, dark: Optional[bool]) -> Tuple[QVBoxLayout, bool]:
        """Create the card, its shadow, the shared chip styling and the chip flow.

        Returns the card's layout and whether the dark palette is used (None follows the OS).
        """
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        if dark is None:
            dark = theme.is_dark_mode(QApplication.instance())
        pal = theme.palette(dark)
        danger = "#ff5a52" if dark else "#e5484d"
        danger_tint = "rgba(255, 90, 82, 0.18)" if dark else "rgba(229, 72, 77, 0.12)"
        self._danger_color = danger
        self._prompt_buttons: List[QPushButton] = []

        outer = QVBoxLayout(self)
        margin = self.SHADOW_MARGIN
        outer.setContentsMargins(margin, margin, margin, margin)
        self.card = QFrame(self)
        self.card.setObjectName(object_name)
        outer.addWidget(self.card)
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 150 if dark else 60))
        self.card.setGraphicsEffect(shadow)
        self.card.setStyleSheet(f"""
            QFrame#{object_name} {{
                background-color: {pal['panel']};
                border: 1px solid {pal['border']};
                border-radius: 12px;
            }}
            QLabel {{ background: transparent; border: none; color: {pal['text']}; }}
            QLabel[role="title"] {{ font-size: 13px; font-weight: 600; }}
            QLabel[role="muted"] {{ font-size: 13px; color: {pal['muted']}; }}
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
            QPushButton:pressed {{ background-color: {pal['hover']}; }}
            QPushButton:checked {{
                background-color: {pal['accent']};
                border-color: {pal['accent']};
                color: {pal['on_accent']};
                font-weight: 600;
            }}
            QPushButton[role="close"] {{
                border-radius: 11px;
                padding: 0px 0px 2px 0px;
                min-width: 20px; max-width: 20px;
                min-height: 20px; max-height: 20px;
                font-size: 14px;
                color: {pal['muted']};
            }}
            QPushButton[role="close"]:hover {{ color: {pal['accent']}; }}
            QPushButton[role="icon"] {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 12px;
                padding: 0px;
                min-width: 24px; max-width: 24px;
                min-height: 24px; max-height: 24px;
            }}
            QPushButton[role="icon"]:hover {{ background-color: {pal['panel2']}; border-color: {pal['border']}; }}
            QPushButton[role="icon"]:pressed {{ background-color: {pal['hover']}; }}
            QPushButton[role="icon"][danger="true"]:hover {{
                background-color: {danger_tint}; border-color: {danger};
            }}
        """)

        card_layout = QVBoxLayout(self.card)
        pad = self._CARD_PADDING
        card_layout.setContentsMargins(pad, pad, pad, pad)
        card_layout.setSpacing(10)
        self._chips = FlowLayout(spacing=6)
        return card_layout, dark

    def _add_icon_button(self, svg: str, tooltip: str, hover_svg: Optional[str] = None,
                         danger: bool = False) -> QPushButton:
        """A round, mouse-only icon button drawn from an SVG (sharp on high-DPI screens)."""
        button = QPushButton(self.card)
        button.setProperty("role", "icon")
        button.setProperty("danger", danger)  # destructive actions turn red on hover
        button.setIcon(_svg_icon(svg, 16, self.devicePixelRatioF(), hover_svg))
        button.setIconSize(QSize(16, 16))
        button.setToolTip(tooltip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _add_chip(self, caption: str, checkable: bool) -> QPushButton:
        """Create one mouse-only chip showing the full caption."""
        button = QPushButton(caption.strip(), self.card)
        button.setCheckable(checkable)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._prompt_buttons.append(button)
        self._chips.addWidget(button)
        return button

    def _fit_card(self, card_layout: QVBoxLayout, header: QLayout) -> None:
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

    def _place_near_cursor(self) -> None:
        """Put the card just below-right of the cursor, kept inside the screen."""
        self.adjustSize()
        card_width, card_height = self.card.width(), self.card.height()
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


def _svg_pixmap(svg: str, size: int, device_pixel_ratio: float) -> QPixmap:
    """Render an SVG string at the screen's pixel density."""
    scale = max(1.0, device_pixel_ratio)
    pixmap = QPixmap(round(size * scale), round(size * scale))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return pixmap


def _svg_icon(svg: str, size: int, device_pixel_ratio: float, hover_svg: Optional[str] = None) -> QIcon:
    """An icon from an SVG string; ``hover_svg`` is shown while the pointer is over the button."""
    icon = QIcon(_svg_pixmap(svg, size, device_pixel_ratio))
    if hover_svg is not None:
        icon.addPixmap(_svg_pixmap(hover_svg, size, device_pixel_ratio), QIcon.Mode.Active)
    return icon


class FloatingButtonWindow(_PromptCard):
    """Rephrase palette near the cursor: one chip per prompt for the selected text."""

    _instance: Optional['FloatingButtonWindow'] = None

    def __init__(self, buttons: List[Dict[str, str]], selected_text: str,
                 on_button_click_callback: Callable[..., None], title: str = "",
                 close_tooltip: str = "", dark: Optional[bool] = None) -> None:
        """Build the palette near the cursor; a chip click reports its prompt and the selection.

        ``dark`` follows the app's colour theme; None uses the operating system's scheme.
        """
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
        card_layout, _dark = self._build_card("rephrasePromptCard", dark)

        # Determine if we auto-close on focus loss (avoid on macOS due to premature closes).
        self._close_on_focus_out = not is_MACOS

        header = QVBoxLayout()
        header.setSpacing(2)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_label = QLabel(title, self.card)
        title_label.setProperty("role", "title")
        title_row.addWidget(title_label)
        title_row.addStretch()
        close_button = QPushButton("\u00d7", self.card)
        close_button.setProperty("role", "close")
        close_button.setToolTip(close_tooltip)
        close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_button.clicked.connect(self.close)
        title_row.addWidget(close_button, 0, Qt.AlignmentFlag.AlignTop)
        header.addLayout(title_row)
        preview = " ".join(selected_text.split())
        self.preview_label: Optional[QLabel] = None
        if preview:
            # One muted line of the selection; the card width comes from title and chips, and the
            # preview is shortened to it once that width is known.
            self.preview_label = QLabel(" ", self.card)
            self.preview_label.setProperty("role", "muted")
            header.addWidget(self.preview_label)
        card_layout.addLayout(header)
        card_layout.addLayout(self._chips, 1)

        for button_info in buttons:
            chip = self._add_chip(button_info.get("caption", ""), checkable=False)
            chip.clicked.connect(partial(on_button_click_callback, button_info.get("text", ""), selected_text, self))

        self._fit_card(card_layout, header)
        if self.preview_label is not None:
            content_width = self.card.width() - 2 * (self._CARD_PADDING + self.card.frameWidth())
            self.preview_label.setText(self.preview_label.fontMetrics().elidedText(
                preview, Qt.TextElideMode.ElideRight, content_width))
            self.preview_label.setToolTip(preview if len(preview) <= 500 else preview[:500] + "…")
        self._place_near_cursor()
        self.show()

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

    def set_paused(self, paused: bool) -> None:
        """Hold the dot dimmed while the recording is paused; pulse again when it continues."""
        if paused:
            self._pulse.pause()
            self._set_opacity(0.35)
        else:
            self._pulse.resume()

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


class _MoveHandle(QLabel):
    """Small four-way arrow in the card's corner that marks the palette as movable.

    It does not handle the mouse itself: presses fall through to the palette, which drags.
    """

    SIZE = 13
    ICON_SIZE = 10

    def __init__(self, color: str, tooltip: str, parent: QWidget) -> None:
        """Draw the arrow cross in ``color``."""
        super().__init__(parent)
        self.setPixmap(_svg_pixmap(theme.move_svg(color), self.ICON_SIZE, self.devicePixelRatioF()))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setToolTip(tooltip)


#: First id of the overlay's digit hotkeys (application ids are 0x0000-0xBFFF).
_DIGIT_HOTKEY_BASE = 0xB000
_VK_0, _VK_NUMPAD0 = 0x30, 0x60


class RecordingPromptOverlay(_PromptCard):
    """Persistent, non-activating prompt selector shown for one microphone recording.

    A rounded card with a pulsing recording dot, the elapsed time and one chip per prompt; the
    chips wrap onto further lines when the captions do not fit next to each other.
    """

    _instance: Optional['RecordingPromptOverlay'] = None

    def __init__(self, prompts: List[Dict[str, Any]], status_text: str, none_text: str,
                 on_selection_changed: Callable[[Optional[Dict[str, Any]]], None],
                 use_system_position: bool = True,
                 system_anchor: Optional[QPoint] = None,
                 dark: Optional[bool] = None,
                 on_cancel: Optional[Callable[[], None]] = None,
                 on_restart: Optional[Callable[[], None]] = None,
                 cancel_tooltip: str = "", restart_tooltip: str = "", move_tooltip: str = "",
                 on_pause: Optional[Callable[[], None]] = None, paused_text: str = "",
                 pause_tooltip: str = "", resume_tooltip: str = "", digit_keys: bool = False) -> None:
        """Build the palette by the system status area or near the cursor without taking focus.

        The prompt marked ``auto_apply`` starts selected (otherwise "None"); only clicks report
        a selection, so the preselection itself leaves the automatic default in charge.
        ``dark`` follows the app's colour theme; None uses the operating system's scheme.
        ``on_restart`` / ``on_cancel`` add header buttons to start the take over or discard it.
        ``on_pause`` adds a pause/resume button; the app reports the state back via ``set_paused``.
        ``digit_keys`` (Windows): while the mouse is over the palette, the number keys pick the chips
        in their shown order (1 is "None", 0 the tenth chip, as on the keyboard); the keys are
        registered as hotkeys, so the focus stays where it is.
        A palette placed near the cursor can be dragged by its card; a handle in the bottom-right
        corner shows that.
        """
        if RecordingPromptOverlay._instance:
            RecordingPromptOverlay._instance.close()
        super().__init__()
        RecordingPromptOverlay._instance = self
        self._on_selection_changed = on_selection_changed
        self._started = time.monotonic()
        # Paused time is left out of the elapsed time.
        self._paused_since: Optional[float] = None
        self._paused_total = 0.0

        window_kind = Qt.WindowType.Dialog if is_MACOS else Qt.WindowType.Tool
        flags = (
            Qt.WindowType.FramelessWindowHint
            | window_kind
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setWindowFlags(flags)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        card_layout, dark = self._build_card("recordingPromptCard", dark)
        recording_red = "#ff5a52" if dark else "#e5484d"

        header = QHBoxLayout()
        header.setSpacing(8)
        self._dot = _RecordingDot(QColor(recording_red), self.card)
        header.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        self._status_text = status_text.rstrip(" .\u2026")
        self._paused_text = paused_text.rstrip(" .\u2026") or self._status_text
        self.status_label = QLabel(self._status_text, self.card)
        self.status_label.setProperty("role", "title")
        # Wide enough for either text, so the card keeps its size when the recording pauses.
        metrics = self.status_label.fontMetrics()
        self.status_label.setMinimumWidth(max(metrics.horizontalAdvance(self._status_text),
                                              metrics.horizontalAdvance(self._paused_text)))
        header.addWidget(self.status_label)
        header.addStretch()
        self.timer_label = QLabel(self.card)
        self.timer_label.setProperty("role", "muted")
        timer_font = self.timer_label.font()
        timer_font.setFeature(QFont.Tag("tnum"), 1)  # digits keep their width while counting
        self.timer_label.setFont(timer_font)
        # Timer and the take controls form one tight group at the right.
        controls = QHBoxLayout()
        controls.setSpacing(2)
        controls.addWidget(self.timer_label, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addSpacing(4)
        # Text is laid out by its baseline, so glyphs sit slightly below the line's middle;
        # a 2 px top margin moves the centred icons down by 1 px onto the same optical line.
        icons = QHBoxLayout()
        icons.setSpacing(2)
        icons.setContentsMargins(0, 2, 0, 0)
        controls.addLayout(icons)
        icon_color = theme.palette(dark)["muted"]
        self.pause_button: Optional[QPushButton] = None
        if on_pause is not None:
            self._pause_icons = (_svg_icon(theme.pause_svg(icon_color), 16, self.devicePixelRatioF()),
                                 _svg_icon(theme.play_svg(icon_color), 16, self.devicePixelRatioF()))
            self._pause_tooltips = (pause_tooltip, resume_tooltip)
            self.pause_button = self._add_icon_button(theme.pause_svg(icon_color), pause_tooltip)
            self.pause_button.clicked.connect(lambda _checked=False: on_pause())
            icons.addWidget(self.pause_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.restart_button: Optional[QPushButton] = None
        self.cancel_button: Optional[QPushButton] = None
        if on_restart is not None:
            self.restart_button = self._add_icon_button(theme.restart_svg(icon_color), restart_tooltip)
            self.restart_button.clicked.connect(lambda _checked=False: on_restart())
            icons.addWidget(self.restart_button, 0, Qt.AlignmentFlag.AlignVCenter)
        if on_cancel is not None:
            self.cancel_button = self._add_icon_button(theme.trash_svg(icon_color), cancel_tooltip,
                                                       hover_svg=theme.trash_svg(self._danger_color), danger=True)
            self.cancel_button.clicked.connect(lambda _checked=False: on_cancel())
            icons.addWidget(self.cancel_button, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addLayout(controls)
        card_layout.addLayout(header)

        card_layout.addLayout(self._chips, 1)  # the chips take the height their lines need
        none_button = self._add_chip(none_text, checkable=True)
        none_button.clicked.connect(partial(self._select_prompt, None, none_button))
        preselected = none_button
        for prompt in prompts:
            button = self._add_chip(prompt["caption"], checkable=True)
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
        self.move_handle: Optional[_MoveHandle] = None
        self._grab_offset: Optional[QPoint] = None
        self._digit_keys = digit_keys and is_WINDOWS
        self._registered_hotkeys: List[int] = []
        if not self._position_for_platform(use_system_position, system_anchor):
            self.move_handle = _MoveHandle(icon_color, move_tooltip, self.card)
            # The whole card drags; buttons and chips keep their own pointing-hand cursor.
            self.card.setCursor(Qt.CursorShape.SizeAllCursor)
            # The handle sits in the card's bottom-right padding, clear of the rounded border.
            inset = 1
            self.move_handle.move(self.card.width() - _MoveHandle.SIZE - inset,
                                  self.card.height() - _MoveHandle.SIZE - inset)
            self.move_handle.raise_()
        if is_MACOS:
            make_nonactivating(self)  # after setWindowFlags, which recreates the native window
        self.show()

    def mousePressEvent(self, event: Optional[QMouseEvent]) -> None:  # noqa: N802 - Qt API
        """Start dragging a movable palette from anywhere on the card but its buttons.

        Buttons and chips take their own clicks; everything else (status, timer, the move handle,
        the free space) reaches the palette. The drag follows the mouse inside Qt instead of
        handing it to the window manager, so the non-activating window never takes the focus
        from the application that receives the text.
        """
        if (event is not None and self.move_handle is not None and event.button() == Qt.MouseButton.LeftButton
                and self.card.geometry().contains(event.position().toPoint())):
            self._grab_offset = event.globalPosition().toPoint() - self.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Optional[QMouseEvent]) -> None:  # noqa: N802 - Qt API
        """Keep the grabbed point of the palette under the mouse."""
        if event is not None and self._grab_offset is not None:
            self.move(event.globalPosition().toPoint() - self._grab_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Optional[QMouseEvent]) -> None:  # noqa: N802 - Qt API
        """Finish the drag."""
        if event is not None and self._grab_offset is not None and event.button() == Qt.MouseButton.LeftButton:
            self._grab_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        """Take the digit keys for the chips while the mouse is over the palette."""
        super().enterEvent(event)
        if not self._digit_keys or self._registered_hotkeys:
            return
        hwnd = int(self.winId())
        for index in range(min(10, len(self._prompt_buttons))):
            digit = (index + 1) % 10
            for vk in (_VK_0 + digit, _VK_NUMPAD0 + digit):
                hotkey_id = _DIGIT_HOTKEY_BASE + vk
                if register_window_hotkey(hwnd, hotkey_id, vk):
                    self._registered_hotkeys.append(hotkey_id)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        """Give the digit keys back to the other applications."""
        super().leaveEvent(event)
        self._release_digit_keys()

    def _release_digit_keys(self) -> None:
        """Unregister every digit hotkey of this palette."""
        if not self._registered_hotkeys:
            return
        hwnd = int(self.winId())
        for hotkey_id in self._registered_hotkeys:
            unregister_window_hotkey(hwnd, hotkey_id)
        self._registered_hotkeys = []

    def nativeEvent(self, event_type: Any, message: Any) -> Any:  # noqa: N802 - Qt API
        """Pick the chip of a pressed digit key."""
        hotkey_id = hotkey_id_of(event_type, message) if self._registered_hotkeys else None
        if hotkey_id is not None and hotkey_id in self._registered_hotkeys:
            self.select_by_digit((hotkey_id - _DIGIT_HOTKEY_BASE) % 0x10)
            return True, 0
        # QWidget's own implementation only reports "not handled"; calling it from PyQt6 crashes.
        return False, 0

    def select_by_digit(self, digit: int) -> None:
        """Pick a chip like a click: 1-9 the chips in their shown order, 0 the tenth."""
        index = (digit - 1) % 10
        if index < len(self._prompt_buttons):
            self._prompt_buttons[index].click()

    def restart_timer(self) -> None:
        """Count the elapsed time from zero again (the take was started over)."""
        self._started = time.monotonic()
        self._paused_total = 0.0
        if self._paused_since is not None:
            self._paused_since = self._started
        self._update_elapsed()

    def set_paused(self, paused: bool) -> None:
        """Show the paused state: frozen timer, dimmed dot, resume button and paused status."""
        now = time.monotonic()
        if paused and self._paused_since is None:
            self._paused_since = now
        elif not paused and self._paused_since is not None:
            self._paused_total += now - self._paused_since
            self._paused_since = None
        self._dot.set_paused(paused)
        self.status_label.setText(self._paused_text if paused else self._status_text)
        if self.pause_button is not None:
            self.pause_button.setIcon(self._pause_icons[paused])
            self.pause_button.setToolTip(self._pause_tooltips[paused])
        self._update_elapsed()

    def _update_elapsed(self) -> None:
        """Show the recorded time since the palette opened as m:ss, without paused time."""
        end = self._paused_since if self._paused_since is not None else time.monotonic()
        seconds = int(end - self._started - self._paused_total)
        self.timer_label.setText(f"{seconds // 60}:{seconds % 60:02d}")

    def _select_prompt(self, prompt: Optional[Dict[str, Any]], selected_button: QPushButton) -> None:
        """Highlight one choice and report its palette entry (None for "None") to the app."""
        for button in self._prompt_buttons:
            button.setChecked(button is selected_button)
        self._on_selection_changed(prompt)

    def _position_for_platform(self, use_system_position: bool,
                               system_anchor: Optional[QPoint]) -> bool:
        """Place the visible card by the OS status area or once near the cursor; True for the status area."""
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
                return True

        self._place_near_cursor()
        return False

    def closeEvent(self, event: Any) -> None:
        """Stop the clock and the pulse, and clear the singleton instance."""
        self._clock.stop()
        self._dot.stop()
        self._release_digit_keys()
        if RecordingPromptOverlay._instance is self:
            RecordingPromptOverlay._instance = None
        super().closeEvent(event)

    @staticmethod
    def close_current() -> None:
        """Close the active recording palette, if one exists."""
        if RecordingPromptOverlay._instance:
            RecordingPromptOverlay._instance.close()
