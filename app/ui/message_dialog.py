"""Message dialogs in the look of the settings window: icon badge, heading, text, optional details, button bar.

Replaces the stock message box, whose big system icon, flat text and equal-weight buttons look out of
place next to the modern settings. The colours come from the parent's theme (``theme_palette()``), so
the dialog follows light/dark and the chosen design.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFontDatabase, QGuiApplication, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui import theme
from app.ui.line_icons import line_icon

#: Dialog kinds: (line icon, palette colour of the badge).
INFO = ("info", "accent")
QUESTION = ("help", "accent")
UPDATE = ("download", "accent")
SUCCESS = ("check", "accent")
WARNING = ("alert", "warn")
ERROR = ("alert", "danger")

#: Button styles: the filled main action, a plain one, and a destructive one.
PRIMARY = "primary"
SECONDARY = "secondary"
DANGER = "danger"
#: (caption, style) of a button; ``show_dialog`` returns the caption that was clicked.
Button = Tuple[str, str]
#: (caption, handler) of a button that acts without closing the dialog (e.g. "Open folder").
Action = Tuple[str, Callable[[], None]]
#: (label, value) rows of a facts card; values are rich text, so links open.
Fact = Tuple[str, str]

_WIDTH = 480
_WIDTH_WITH_DETAILS = 640
_DETAILS_HEIGHT = 280
_BADGE = 40
_MARGIN = 24
_HEADER_SPACING = 16
_BADGE_ICON = 22
#: Share of the available screen the dialog may use at most.
_MAX_SCREEN_SHARE = 0.85

_QSS = """
QDialog#messageDialog { background: %(panel)s; }
QWidget#messageBody { background: %(panel)s; }
QWidget#messageFooter { background: %(bg)s; border-top: 1px solid %(border)s; }
QLabel#messageTitle { color: %(heading)s; font-size: 15px; font-weight: 600; background: transparent; }
QLabel#messageText { color: %(muted)s; font-size: 13px; background: transparent; }
QLabel#messageBadge { background: %(badge_bg)s; border-radius: 10px; }
QLabel#messageBadge[plain="true"] { background: transparent; }
QFrame#messageFacts { background: %(panel2)s; border: 1px solid %(border)s; border-radius: 10px; }
QLabel#factLabel { color: %(muted)s; font-size: 12px; background: transparent; }
QLabel#factValue { color: %(text)s; font-size: 12px; background: transparent; }
QPlainTextEdit#messageDetails {
    background: %(panel2)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px;
    padding: 6px 8px; font-size: 12px;
    selection-background-color: %(accent)s; selection-color: %(on_accent)s;
}
QPushButton {
    background: %(panel)s; color: %(text)s; border: 1px solid %(field_border)s; border-radius: 7px;
    padding: 7px 18px; min-width: 64px; font-size: 13px;
}
QPushButton:hover { border-color: %(accent)s; color: %(accent)s; }
QPushButton:pressed { background: %(hover)s; }
QPushButton#primary { background: %(accent)s; color: %(on_accent)s; border: 1px solid %(accent)s; font-weight: 600; }
QPushButton#primary:hover { background: %(accent_hover)s; border-color: %(accent_hover)s; color: %(on_accent)s; }
QPushButton#primary:pressed { background: %(accent_press)s; }
QPushButton#danger { color: %(danger)s; border-color: %(danger)s; }
QPushButton#danger:hover { background: %(danger)s; color: #ffffff; }
QPushButton#detailsToggle {
    background: transparent; border: none; padding: 7px 4px; min-width: 0; color: %(accent)s;
}
QPushButton#detailsToggle:hover { color: %(accent_hover)s; text-decoration: underline; }
"""


def _colors(parent: Optional[QWidget]) -> Dict[str, str]:
    """The parent window's theme, or the modern one following the system's light/dark mode."""
    palette = getattr(parent, "theme_palette", None)
    if callable(palette):
        try:
            return dict(palette())
        except Exception:
            pass
    return theme.palette(theme.is_dark_mode(QApplication.instance()), True)


def _stylesheet(colors: Dict[str, str], badge: str) -> str:
    """The dialog's stylesheet; the classic palettes lack a few modern keys, which fall back."""
    values = dict(colors)
    values.setdefault("heading", colors["text"])
    tint = QColor(colors[badge])
    tint.setAlpha(36)
    values["badge_bg"] = f"rgba({tint.red()}, {tint.green()}, {tint.blue()}, {tint.alpha()})"
    return _QSS % values


class MessageDialog(QDialog):
    """A heading with an icon badge, a text, optional folded details and a right-aligned button bar."""

    def __init__(self, parent: Optional[QWidget], title: str, text: str, buttons: Sequence[Button],
                 kind: Tuple[str, str] = INFO, details: str = "",
                 details_captions: Optional[Tuple[str, str]] = None, window_title: str = "",
                 icon: Optional[QIcon] = None, facts: Sequence[Fact] = (),
                 actions: Sequence[Action] = ()) -> None:
        """Build the dialog; ``details_captions`` (show, hide) fold the details behind a toggle.

        ``icon`` replaces the tinted line icon (e.g. the app logo), ``facts`` adds a card of label/value
        rows below the text and ``actions`` are buttons at the left of the bar that keep the dialog open.
        """
        super().__init__(parent)
        self.setObjectName("messageDialog")
        self.setWindowTitle(window_title or title)
        self.clicked: Optional[str] = None
        colors = _colors(parent)
        icon_name, badge_color = kind
        self.setStyleSheet(_stylesheet(colors, badge_color))

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QWidget(self)
        body.setObjectName("messageBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(_MARGIN, 22, _MARGIN, 20)
        body_layout.setSpacing(16)
        header = QHBoxLayout()
        header.setSpacing(_HEADER_SPACING)
        badge = QLabel(body)
        badge.setObjectName("messageBadge")
        badge.setFixedSize(_BADGE, _BADGE)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if icon is not None:
            badge.setProperty("plain", True)
            badge.setPixmap(icon.pixmap(_BADGE, _BADGE))
        else:
            badge.setPixmap(line_icon(icon_name, colors[badge_color]).pixmap(_BADGE_ICON, _BADGE_ICON))
        header.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(6)
        heading = QLabel(title, body)
        heading.setObjectName("messageTitle")
        heading.setWordWrap(True)
        texts.addWidget(heading)
        # Labels that wrap get an explicit width in _fit(): nested layouts misjudge their height otherwise.
        self._wrapped: List[QLabel] = [heading]
        if text:
            label = QLabel(_tint_links(text, colors), body)
            label.setObjectName("messageText")
            label.setWordWrap(True)
            # Rich text (the About text) keeps its links clickable.
            label.setOpenExternalLinks(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            texts.addWidget(label)
            self._wrapped.append(label)
        texts.addStretch(1)
        header.addLayout(texts, 1)
        body_layout.addLayout(header)
        if facts:
            body_layout.addWidget(_facts_card(body, facts, colors))

        self.details: Optional[QPlainTextEdit] = None
        if details:
            self.details = QPlainTextEdit(details, body)
            self.details.setObjectName("messageDetails")
            self.details.setReadOnly(True)
            self.details.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self.details.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
            self.details.setFixedHeight(_DETAILS_HEIGHT)
            self.details.setVisible(details_captions is None)
            body_layout.addWidget(self.details)
        root.addWidget(body, 1)

        footer = QWidget(self)
        footer.setObjectName("messageFooter")
        bar = QHBoxLayout(footer)
        bar.setContentsMargins(24, 12, 24, 12)
        bar.setSpacing(8)
        self.details_toggle: Optional[QPushButton] = None
        if self.details is not None and details_captions:
            toggle = QPushButton(details_captions[0], footer)
            toggle.setObjectName("detailsToggle")
            toggle.setCheckable(True)
            toggle.setAutoDefault(False)
            toggle.setCursor(Qt.CursorShape.PointingHandCursor)
            toggle.toggled.connect(lambda shown: self._show_details(shown, details_captions))
            bar.addWidget(toggle)
            self.details_toggle = toggle
        for caption, handler in actions:
            action = QPushButton(caption, footer)
            action.setAutoDefault(False)
            action.setCursor(Qt.CursorShape.PointingHandCursor)
            action.clicked.connect(lambda _checked=False, handler=handler: handler())
            bar.addWidget(action)
        bar.addStretch(1)
        self.buttons: List[QPushButton] = []
        for caption, style in buttons:
            button = QPushButton(caption, footer)
            if style in (PRIMARY, DANGER):
                button.setObjectName(style)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, caption=caption: self._choose(caption))
            button.setAutoDefault(False)
            bar.addWidget(button)
            self.buttons.append(button)
        # Enter runs the main action; without one (a destructive question) it takes the safe way out.
        styles = [style for _caption, style in buttons]
        default_style = PRIMARY if PRIMARY in styles else SECONDARY
        default = next((b for b, style in zip(self.buttons, styles) if style == default_style),
                       self.buttons[-1] if self.buttons else None)
        if default is not None:
            default.setDefault(True)
            default.setFocus()
        root.addWidget(footer)
        self._fit(self.details is not None and self.details.isVisible())

    def _choose(self, caption: str) -> None:
        self.clicked = caption
        self.accept()

    def _show_details(self, shown: bool, captions: Tuple[str, str]) -> None:
        if self.details is None or self.details_toggle is None:
            return
        self.details.setVisible(shown)
        self.details_toggle.setText(captions[shown])
        self._fit(shown)

    def _fit(self, wide: bool) -> None:
        """A fixed width keeps the text from stretching into one long line; the height follows the content."""
        width = _WIDTH_WITH_DETAILS if wide else _WIDTH
        parent = self.parentWidget()
        screen = (parent.screen() if parent is not None else None) or QGuiApplication.primaryScreen()
        if screen is not None:
            width = min(width, int(screen.availableGeometry().width() * _MAX_SCREEN_SHARE))
        for label in self._wrapped:
            label.setFixedWidth(width - 2 * _MARGIN - _BADGE - _HEADER_SPACING)
        self.setFixedWidth(width)
        self.adjustSize()


def _tint_links(text: str, colors: Dict[str, str]) -> str:
    """Links in the accent colour; Qt's default link blue is unreadable on the dark theme."""
    return text.replace("<a ", f'<a style="color: {colors["accent"]}; text-decoration: none;" ')


def _facts_card(parent: QWidget, facts: Sequence[Fact], colors: Dict[str, str]) -> QFrame:
    """A rounded card of label/value rows; the values are selectable and their links open."""
    card = QFrame(parent)
    card.setObjectName("messageFacts")
    grid = QGridLayout(card)
    grid.setContentsMargins(16, 12, 16, 12)
    grid.setHorizontalSpacing(18)
    grid.setVerticalSpacing(8)
    for row, (label, value) in enumerate(facts):
        name = QLabel(label, card)
        name.setObjectName("factLabel")
        grid.addWidget(name, row, 0, Qt.AlignmentFlag.AlignTop)
        text = QLabel(_tint_links(value, colors), card)
        text.setObjectName("factValue")
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.RichText)
        text.setOpenExternalLinks(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        grid.addWidget(text, row, 1)
    grid.setColumnStretch(1, 1)
    return card


def show_dialog(parent: Optional[QWidget], title: str, text: str, buttons: Sequence[Button],
                kind: Tuple[str, str] = INFO, details: str = "",
                details_captions: Optional[Tuple[str, str]] = None, window_title: str = "",
                icon: Optional[QIcon] = None, facts: Sequence[Fact] = (),
                actions: Sequence[Action] = ()) -> Optional[str]:
    """Show a message dialog; the clicked button's caption, or None when it was closed."""
    dialog = MessageDialog(parent, title, text, buttons, kind, details, details_captions, window_title,
                           icon, facts, actions)
    dialog.exec()
    clicked = dialog.clicked
    dialog.deleteLater()
    return clicked


def inform(parent: Optional[QWidget], title: str, text: str, ok: str, kind: Tuple[str, str] = INFO) -> None:
    """A message with a single acknowledging button."""
    show_dialog(parent, title, text, [(ok, PRIMARY)], kind)


def ask(parent: Optional[QWidget], title: str, text: str, confirm: str, cancel: str,
        kind: Tuple[str, str] = QUESTION, window_title: str = "") -> bool:
    """A yes/no question; destructive confirmations (kind WARNING/ERROR) get a red confirm button."""
    style = DANGER if kind in (WARNING, ERROR) else PRIMARY
    return show_dialog(parent, title, text, [(cancel, SECONDARY), (confirm, style)], kind,
                       window_title=window_title) == confirm
