"""Cross-platform Qt stylesheet (light/dark, teal accent) for the settings window.

Single responsibility: produce the application QSS for a light or dark palette and
detect the OS colour scheme. No app state.
"""

from __future__ import annotations

from typing import Any, Dict

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

LIGHT: Dict[str, str] = {
    "bg": "#f5f8fa", "panel": "#ffffff", "panel2": "#eef3f6", "text": "#16212b",
    "muted": "#5a6b78", "border": "#d6dfe5", "group_border": "#9aa9b4", "field": "#ffffff", "field_border": "#cfd9e0",
    "accent": "#0e8aa8", "accent_hover": "#0b7690", "accent_press": "#095c74",
    "on_accent": "#ffffff", "warn": "#cf6a3f", "danger": "#e5484d", "hover": "#eef3f6",
}

DARK: Dict[str, str] = {
    "bg": "#0e1620", "panel": "#16212c", "panel2": "#1b2733", "text": "#dce6ee",
    "muted": "#8598a5", "border": "#273744", "group_border": "#627382", "field": "#121c26", "field_border": "#2c3d4a",
    "accent": "#35c0dd", "accent_hover": "#4fcbe4", "accent_press": "#2aa6c1",
    "on_accent": "#08222b", "warn": "#e08b4a", "danger": "#ff5a52", "hover": "#1e2b37",
}


#: The modern design: soft grey canvas, white cards, one blue accent.
MODERN_LIGHT: Dict[str, str] = {
    "bg": "#f3f5f8", "panel": "#ffffff", "panel2": "#f1f4f8", "text": "#1f2329",
    "muted": "#6b727c", "border": "#e3e7ed", "group_border": "#e3e7ed", "field": "#ffffff", "field_border": "#cfd5dd",
    "accent": "#2f7de1", "accent_hover": "#2a72cf", "accent_press": "#2565b8",
    "on_accent": "#ffffff", "warn": "#cf6a3f", "danger": "#e5484d", "hover": "#edf0f4",
    "heading": "#15191e", "sidebar": "#eaedf2", "sidebar_border": "#dfe3e9", "nav_text": "#3a4049",
    "nav_hover": "#e0e5ec", "nav_selected": "#1f5fb8", "menu_bg": "#f9f9fb", "menu_border": "#e0e3e8",
    "menu_selected": "#e8ebf0", "menu_icon": "#5f6670", "switch_off": "#c5ccd6", "selection": "#bcd5f5",
    "divider": "#eef1f4",
}

MODERN_DARK: Dict[str, str] = {
    "bg": "#1b1d21", "panel": "#25282d", "panel2": "#2c3036", "text": "#e6e8eb",
    "muted": "#9aa1ab", "border": "#33373e", "group_border": "#33373e", "field": "#1f2226", "field_border": "#40454d",
    "accent": "#4c94ef", "accent_hover": "#5ea0f2", "accent_press": "#3d84de",
    "on_accent": "#ffffff", "warn": "#e08b4a", "danger": "#ff5a52", "hover": "#2f333a",
    "heading": "#f2f4f6", "sidebar": "#202327", "sidebar_border": "#2d3036", "nav_text": "#c9cdd3",
    "nav_hover": "#2b2f35", "nav_selected": "#7fb2f5", "menu_bg": "#2b2d31", "menu_border": "#3a3d43",
    "menu_selected": "#383c43", "menu_icon": "#aeb4bc", "switch_off": "#4a4f57", "selection": "#2f5a91",
    "divider": "#2e3238",
}


def set_style_state(widget: Any, name: str, value: Any) -> None:
    """Set a dynamic property matched by a QSS selector and re-polish so it applies at once."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


def palette(dark: bool, modern: bool = False) -> Dict[str, str]:
    """Return the colour palette for the requested mode and design."""
    if modern:
        return MODERN_DARK if dark else MODERN_LIGHT
    return DARK if dark else LIGHT


def sign_icon(color: str, plus: bool, size: int = 16) -> QIcon:
    """A bold "+" or "-" drawn in ``color`` (sharp at any scale; Qt greys it out when disabled)."""
    scale = 3  # drawn larger and scaled down by Qt for high-DPI screens
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 2.4 * scale)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    margin, middle, end = 3.5 * scale, size * scale / 2, (size - 3.5) * scale
    painter.drawLine(QPointF(margin, middle), QPointF(end, middle))
    if plus:
        painter.drawLine(QPointF(middle, margin), QPointF(middle, end))
    painter.end()
    return QIcon(pixmap)


def resolve_dark(mode: str, app: object) -> bool:
    """Whether the configured colour theme ('system'/'light'/'dark') renders dark."""
    if mode in ("light", "dark"):
        return mode == "dark"
    return is_dark_mode(app)


def is_dark_mode(app: object) -> bool:
    """Best-effort OS dark-mode detection (Qt 6.5 colour scheme, palette fallback)."""
    try:
        from PyQt6.QtCore import Qt
        scheme = app.styleHints().colorScheme()  # type: ignore[attr-defined]
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except Exception:
        pass
    try:
        from PyQt6.QtGui import QPalette
        col = app.palette().color(QPalette.ColorRole.Window)  # type: ignore[attr-defined]
        return col.lightness() < 128
    except Exception:
        return False


_QSS = """
* { outline: none; }

QWidget { background-color: %(bg)s; color: %(text)s; font-size: 13px; }
QLabel, QFrame, QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }
QScrollArea { border: none; }

QMenuBar { background: transparent; border: none; padding: 2px 4px; }
QMenuBar::item { background: transparent; padding: 5px 10px; border-radius: 6px; }
QMenuBar::item:selected { background: %(hover)s; }
/* An open menu: its menu-bar entry and the popup frame share the accent colour, so they read as one. */
QMenuBar::item:pressed { background: %(accent)s; color: %(on_accent)s; }
QMenu { background: %(panel)s; border: 1px solid %(accent)s; border-radius: 4px; padding: 5px; }
QMenu::item { padding: 8px 40px 8px 15px; border-radius: 6px; }
QMenu::icon { left: 12px; }
QMenu::item:selected { background: %(accent)s; color: %(on_accent)s; }
QMenu::item:disabled { color: %(muted)s; }

QTabWidget::pane { border: none; top: -1px; }
QTabBar::tab {
    background: transparent; color: %(muted)s; padding: 6px 14px; margin-right: 2px;
    border: none; border-bottom: 2px solid transparent;
}
QTabBar::tab:hover { color: %(text)s; }
QTabBar::tab:selected { color: %(accent)s; border-bottom: 2px solid %(accent)s; }

QGroupBox {
    background: %(panel)s; border: 1px solid %(group_border)s; border-radius: 8px;
    margin-top: 8px; padding: 10px 10px 6px 10px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 6px; color: %(accent)s; background: %(bg)s;
}

QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: %(field)s; color: %(text)s; border: 1px solid %(field_border)s;
    border-radius: 8px; padding: 3px 8px;
    selection-background-color: %(accent)s; selection-color: %(on_accent)s;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid %(accent)s; }
QLineEdit:hover, QComboBox:hover, QSpinBox:hover { border-color: %(muted)s; }
QComboBox { padding-right: 32px; }
/* An editable combo places its line edit left of the drop-down itself; extra padding would only add a gap. */
QComboBox:editable { padding-right: 4px; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: top right; width: 28px;
    border: none; border-left: 1px solid %(field_border)s;
    border-top-right-radius: 7px; border-bottom-right-radius: 7px; background: %(panel2)s;
}
QComboBox::drop-down:hover { background: %(hover)s; }
QComboBox QAbstractItemView {
    background: %(panel)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px;
    selection-background-color: %(accent)s; selection-color: %(on_accent)s; outline: none;
}

QPushButton {
    background: %(panel2)s; color: %(text)s; border: 1px solid %(border)s;
    border-radius: 8px; padding: 4px 12px;
}
QPushButton:hover { border-color: %(accent)s; color: %(accent)s; }
QPushButton:pressed { background: %(hover)s; }
QPushButton:disabled { color: %(muted)s; border-color: %(border)s; }
QPushButton#save_button {
    background: %(accent)s; color: %(on_accent)s; border: 1px solid %(accent)s;
    font-weight: 600; padding: 9px 18px;
}
QPushButton#save_button:hover { background: %(accent_hover)s; border-color: %(accent_hover)s; color: %(on_accent)s; }
QPushButton#save_button:pressed { background: %(accent_press)s; }
/* Prompt list "+" / "-" (icon-only): coloured so they stand out next to the Save button. */
QPushButton#post_rp_add_btn, QPushButton#post_rp_remove_btn { padding: 5px 14px; }
QPushButton#post_rp_add_btn { background: %(accent)s; border: 1px solid %(accent)s; }
QPushButton#post_rp_add_btn:hover { background: %(accent_hover)s; border-color: %(accent_hover)s; }
QPushButton#post_rp_add_btn:pressed { background: %(accent_press)s; }
QPushButton#post_rp_remove_btn { border: 1px solid %(warn)s; }
QPushButton#post_rp_remove_btn:hover { background: %(hover)s; border: 2px solid %(warn)s; }
QPushButton#post_rp_remove_btn:disabled { border-color: %(border)s; background: %(panel2)s; }
/* Small round "?" help badge next to the LivePrompting toggle (fixed 22px in code). */
QPushButton[link="true"] {
    background: transparent; border: none; padding: 0 4px; color: %(accent)s; text-decoration: underline;
}
QPushButton[link="true"]:hover { color: %(accent_hover)s; background: transparent; }
QLabel[key_status="ok"] { color: %(muted)s; }
QLabel[key_status="missing"] { color: %(warn)s; font-weight: 600; }
QPushButton#liveprompt_help_button {
    padding: 0; border-radius: 11px; font-weight: 700;
    background: %(panel2)s; color: %(muted)s; border: 1px solid %(border)s;
}
QPushButton#liveprompt_help_button:hover { background: %(accent)s; color: %(on_accent)s; border-color: %(accent)s; }
QPushButton#liveprompt_help_button:pressed { background: %(accent_press)s; }

QCheckBox { spacing: 8px; background: transparent; }
QCheckBox::indicator {
    width: 18px; height: 18px; border-radius: 5px;
    border: 1px solid %(field_border)s; background: %(field)s;
}
QCheckBox::indicator:hover { border-color: %(accent)s; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; }

QSlider { background: transparent; }
QSlider::groove:horizontal { height: 4px; background: %(border)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal {
    /* 12px + 2px border on each side = 16px wide; the 4px groove + 6px margins = 16px high:
       equal sides and a radius of half that make the handle a circle. */
    background: %(panel)s; border: 2px solid %(accent)s;
    width: 12px; margin: -6px 0; border-radius: 8px;
}
QSlider::handle:horizontal:hover { background: %(accent)s; }
/* Models without temperature disable the slider: grey track and handle make that obvious. */
QSlider::sub-page:horizontal:disabled { background: %(border)s; }
QSlider::handle:horizontal:disabled { background: %(panel2)s; border-color: %(border)s; }
QLabel:disabled { color: %(muted)s; }

/* Collapsible section headers (prompt, FFmpeg): flat full-width caption with a separator line. */
QWidget#collapsible_header, QWidget#collapsible_content { background: transparent; }
QLabel#collapsible_title { color: %(accent)s; font-weight: 600; }
QLabel#collapsible_title[hovered="true"] { color: %(accent_hover)s; }
QFrame#collapsible_line { border: none; background: %(group_border)s; max-height: 1px; min-height: 1px; }

QListWidget {
    background: %(field)s; color: %(text)s; border: 1px solid %(field_border)s;
    border-radius: 8px; padding: 4px;
}
QListWidget::item { padding: 4px 8px; border-radius: 6px; }
/* Before :selected, so a selected alternate row still shows the accent. */
QListWidget::item:alternate { background: %(panel2)s; }
QListWidget::item:selected { background: %(accent)s; color: %(on_accent)s; }
QTableWidget {
    background: %(panel)s; alternate-background-color: %(panel2)s;
    border: 1px solid %(border)s; border-radius: 8px;
    selection-background-color: %(hover)s; selection-color: %(text)s;
}
QTableWidget::item { padding: 0 8px; border-bottom: 1px solid %(border)s; }
QHeaderView::section {
    background: %(panel2)s; color: %(muted)s; padding: 5px 8px;
    border: none; border-bottom: 1px solid %(border)s; font-weight: 600;
}
QLabel#apiKeyPreview { color: %(muted)s; }
QGroupBox[incomplete="true"] { border: 2px solid %(warn)s; }
/* Muted lead-in above the prompt list. */
QLabel#transformations_tab_description_label { color: %(muted)s; font-style: italic; margin-bottom: 10px; }
QLabel#transformations_unavailable_label {
    color: %(text)s; background: %(panel)s; border: 1px solid %(warn)s;
    border-radius: 8px; padding: 8px 10px;
}
QLabel#prompt_token_label {
    background: %(panel2)s; border: 1px solid %(border)s; border-radius: 10px;
    padding: 2px 10px; color: %(muted)s;
}
QLabel#prompt_token_label[level="near"] { color: %(accent)s; }
QLabel#prompt_token_label[level="over"] { color: %(warn)s; }
QLabel#ffmpeg_status_label[found="true"] { color: %(accent)s; }
QLabel#ffmpeg_status_label[found="false"] { color: %(warn)s; }
QLabel#local_engine_status[level="good"] { color: %(accent)s; font-weight: 600; }
QLabel#local_engine_status[level="busy"] { color: %(warn)s; }
QLabel#local_engine_status[level="bad"] { color: %(danger)s; font-weight: 600; }
QLabel#local_engine_status[level="off"] { color: %(muted)s; }
QLineEdit[warning="true"] { border-color: %(warn)s; }
QSplitter::handle { background: %(border)s; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: %(border)s; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: %(muted)s; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: %(border)s; border-radius: 5px; min-width: 24px; }
QScrollBar::handle:horizontal:hover { background: %(muted)s; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

#brandHeader { background: %(panel)s; border: 1px solid %(border)s; border-radius: 12px; }
#brandTitle { font-size: 15px; font-weight: 700; color: %(text)s; background: transparent; }
#brandSub { color: %(muted)s; background: transparent; }
#hotkeyBadge {
    background: %(panel2)s; border: 1px solid %(border)s; border-radius: 10px;
    padding: 3px 12px; color: %(accent)s; font-weight: 600;
}
QPushButton#brandPlayButton { border-radius: 10px; padding: 3px 12px; }
"""


#: Added on top of ``_QSS`` in the modern design: sidebar navigation, page headers, cards and quiet menus.
_MODERN_QSS = """
QWidget { font-size: 10pt; }
QFrame#sidebar { background: %(sidebar)s; border: none; border-right: 1px solid %(sidebar_border)s; }
QLabel#brandName { font-size: 12pt; font-weight: 600; color: %(heading)s; }
QLabel#brandVersion { color: %(muted)s; font-size: 8pt; }
QListWidget#nav { background: transparent; border: none; padding: 0; }
QListWidget#nav::item {
    padding: 8px 10px; margin: 1px 0; border-radius: 8px; color: %(nav_text)s; border: 1px solid transparent;
}
QListWidget#nav::item:hover { background: %(nav_hover)s; }
QListWidget#nav::item:selected { background: %(panel)s; color: %(nav_selected)s; border-color: %(border)s; }
#sidebarButton {
    background: transparent; border: 1px solid transparent; border-radius: 8px;
    padding: 7px 10px; color: %(nav_text)s;
}
QToolButton#sidebarButton { padding-left: 17px; }
#sidebarButton:hover { background: %(nav_hover)s; color: %(nav_text)s; border-color: transparent; }
#sidebarButton:disabled { color: %(muted)s; }
QToolButton#sidebarButton::menu-indicator { image: none; width: 0; }
QLabel#pageTitle { font-size: 17pt; font-weight: 600; color: %(heading)s; }
QLabel#pageSubtitle { color: %(muted)s; font-size: 9pt; padding-bottom: 6px; }

QTabWidget#tabs::pane { border: none; }
QGroupBox {
    background: %(panel)s; border: 1px solid %(group_border)s; border-radius: 12px;
    margin-top: 26px; padding: 0; font-weight: 400;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left; left: 4px; padding: 0 0 6px 0;
    color: %(muted)s; background: transparent; font-weight: 600; font-size: 8pt;
}
QWidget#cardBody, QWidget#settingRow { background: transparent; }
QListWidget#promptList { background: transparent; border: none; padding: 0; }
QWidget#post_rp_instruction_options { background: transparent; }
QListWidget#promptList::item, QListWidget#promptList::item:selected { background: transparent; border: none; }
QLabel#cardNote { color: %(muted)s; font-size: 9pt; font-style: normal; margin: 0; }
QSplitter::handle { background: transparent; }
QFrame#rowDivider { background: %(divider)s; border: none; }
QLabel[rowTitle="true"] { font-weight: 600; color: %(text)s; }
QLabel[rowTitle="true"]:disabled { color: %(muted)s; }
QLabel#rowDescription { color: %(muted)s; font-size: 9pt; }
QLabel#trayTitle { font-size: 10pt; }
QLabel#trayStatus { font-size: 8pt; }
QLabel#trayVersion { font-size: 8pt; }
QLabel#collapsible_title { color: %(text)s; }
QFrame#collapsible_line { background: %(border)s; }

QPushButton {
    background: %(panel)s; color: %(text)s; border: 1px solid %(field_border)s; border-radius: 8px; padding: 5px 14px;
}
QPushButton:hover { background: %(hover)s; color: %(text)s; border-color: %(field_border)s; }
QPushButton:pressed { background: %(panel2)s; }
QPushButton#save_button { min-width: 110px; padding: 7px 18px; }
QFrame#footer { background: %(panel)s; border: none; border-top: 1px solid %(border)s; }
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    selection-background-color: %(selection)s; selection-color: %(text)s;
}
QListWidget::item:selected { background: %(selection)s; color: %(text)s; }
QCheckBox { spacing: 10px; }
QCheckBox::indicator, QCheckBox::indicator:checked, QCheckBox::indicator:hover {
    width: 34px; height: 20px; border: none; background: transparent; border-radius: 0;
}

QMenu { background: %(menu_bg)s; border: 1px solid %(menu_border)s; border-radius: 0; padding: 4px; font-size: 9pt; }
QMenu::item { padding: 7px 28px 7px 18px; border-radius: 4px; }
QMenu::icon { left: 10px; }
QMenu::item:selected { background: %(menu_selected)s; color: %(text)s; }
QMenu::item:disabled { color: %(muted)s; }
QMenu::separator { height: 1px; background: %(border)s; margin: 4px 10px; }
QMenu#tray { padding: 6px; }
QMenu#tray::item { padding: 7px 40px 7px 18px; border-radius: 6px; }
QWidget#trayHeader { background: transparent; }
QLabel#trayTitle { font-weight: 600; color: %(heading)s; }
QLabel#trayStatus { color: %(muted)s; }
QLabel#trayVersion { color: %(muted)s; }

QScrollBar:vertical { width: 8px; margin: 2px; }
QScrollBar:horizontal { height: 8px; margin: 2px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { border-radius: 3px; }
"""


def _switch_svg(track: str, knob: str, on: bool, opacity: float = 1.0) -> str:
    """SVG string for an on/off switch (34x20) that replaces the checkbox square in the modern design."""
    knob_x = 24 if on else 10
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 34 20' opacity='{opacity}'>"
            f"<rect x='0.5' y='1' width='33' height='18' rx='9' fill='{track}'/>"
            f"<circle cx='{knob_x}' cy='10' r='6.5' fill='{knob}'/></svg>")


def _check_svg(color: str) -> str:
    """SVG string for the checkbox checkmark in the given stroke colour."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
            f"<path fill='none' stroke='{color}' stroke-width='2.4' stroke-linecap='round' "
            f"stroke-linejoin='round' d='M3.5 8.5l3 3 6-7'/></svg>")


def _arrow_svg(color: str, up: bool) -> str:
    """SVG string for a spinbox up/down chevron in the given stroke colour."""
    d = "M2.5 8l3.5-4 3.5 4" if up else "M2.5 4l3.5 4 3.5-4"
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'>"
            f"<path fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' "
            f"stroke-linejoin='round' d='{d}'/></svg>")


def trash_svg(color: str) -> str:
    """SVG string for a trash can (discard) in the given stroke colour."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
            f"<g fill='none' stroke='{color}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'>"
            f"<path d='M2.75 4.5h10.5M6.25 4.5V2.75h3.5V4.5M4.25 4.5l.6 8.75h6.3l.6-8.75M6.75 7v4M9.25 7v4'/>"
            f"</g></svg>")


def restart_svg(color: str) -> str:
    """SVG string for a circular arrow (start over) in the given stroke colour."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
            f"<g fill='none' stroke='{color}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'>"
            f"<path d='M12.9 9.2A5 5 0 1 1 11.6 4.4'/><path d='M12.25 1.9v2.85H9.4'/>"
            f"</g></svg>")


def pause_svg(color: str) -> str:
    """SVG string for two vertical bars (pause) in the given stroke colour."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
            f"<g fill='none' stroke='{color}' stroke-width='1.75' stroke-linecap='round'>"
            f"<path d='M5.75 3.5v9M10.25 3.5v9'/>"
            f"</g></svg>")


def play_svg(color: str) -> str:
    """SVG string for a right-pointing triangle (resume) in the given stroke colour."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
            f"<g fill='none' stroke='{color}' stroke-width='1.5' stroke-linejoin='round'>"
            f"<path d='M5 3.25v9.5L12.75 8z'/>"
            f"</g></svg>")


def move_svg(color: str) -> str:
    """SVG string for a four-way arrow cross (drag to move) in the given stroke colour."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
            f"<g fill='none' stroke='{color}' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'>"
            f"<path d='M8 1.75v12.5M1.75 8h12.5M6 3.75l2-2 2 2M6 12.25l2 2 2-2M3.75 6l-2 2 2 2M12.25 6l2 2-2 2'/>"
            f"</g></svg>")


def write_icon_qss(dark: bool, cache_dir: str, modern: bool = False) -> str:
    """Write themed checkmark/arrow SVGs into cache_dir and return QSS referencing them.

    Qt renders QSS ``image:`` from SVG *files* but not from ``data:`` URIs for spinbox
    sub-controls, so the icons are written to disk and referenced by path.
    """
    import os
    os.makedirs(cache_dir, exist_ok=True)
    c = palette(dark, modern)
    suffix = ("m" if modern else "") + ("d" if dark else "l")
    assets = {
        "check": _check_svg(c["on_accent"]),
        "up": _arrow_svg(c["accent"], True),
        "down": _arrow_svg(c["accent"], False),
    }
    if modern:
        assets.update({
            "switch_off": _switch_svg(c["switch_off"], "#ffffff", False),
            "switch_on": _switch_svg(c["accent"], "#ffffff", True),
            "switch_off_disabled": _switch_svg(c["switch_off"], "#ffffff", False, 0.45),
            "switch_on_disabled": _switch_svg(c["accent"], "#ffffff", True, 0.45),
        })
    paths: Dict[str, str] = {}
    for name, svg in assets.items():
        fp = os.path.join(cache_dir, f"{name}_{suffix}.svg")
        # Skip the write when the cached file already holds this exact SVG — apply_theme
        # re-runs on every OS colour-scheme flip, and the icons only change with the palette.
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                unchanged = fh.read() == svg
        except OSError:
            unchanged = False
        if not unchanged:
            with open(fp, "w", encoding="utf-8") as fh:
                fh.write(svg)
        paths[name] = fp.replace("\\", "/")
    if modern:
        checkbox_qss = f"""
QCheckBox::indicator {{ image: url({paths['switch_off']}); }}
QCheckBox::indicator:checked {{ image: url({paths['switch_on']}); }}
QCheckBox::indicator:disabled {{ image: url({paths['switch_off_disabled']}); }}
QCheckBox::indicator:checked:disabled {{ image: url({paths['switch_on_disabled']}); }}"""
    else:
        checkbox_qss = f"""
QCheckBox::indicator:checked {{ image: url({paths['check']}); }}"""
    return checkbox_qss + f"""
QComboBox::down-arrow {{ image: url({paths['down']}); width: 16px; height: 16px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 26px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right;
    width: 24px; border-left: 1px solid {c['field_border']};
    border-bottom: 1px solid {c['field_border']};
    border-top-right-radius: 8px; background: {c['panel2']};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 24px; border-left: 1px solid {c['field_border']};
    border-bottom-right-radius: 8px; background: {c['panel2']};
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{ background: {c['hover']}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({paths['up']}); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({paths['down']}); width: 10px; height: 10px; }}
"""


_TOOLTIP_QSS = """
QToolTip {
    background-color: %(panel)s; color: %(text)s;
    border: 1px solid %(group_border)s; padding: 7px 9px; border-radius: 8px;
    font-size: 13px; font-weight: normal;
}
"""


def tooltip_stylesheet(dark: bool, modern: bool = False) -> str:
    """Tooltip QSS for the whole application.

    Set on the QApplication, not the window: Qt reuses an open tooltip when the pointer moves to
    the next widget and then drops the window's stylesheet, so its padding would vanish.
    """
    return _TOOLTIP_QSS % palette(dark, modern)


def build_stylesheet(dark: bool, modern: bool = False) -> str:
    """Build the base application QSS for light or dark mode (icons added by write_icon_qss)."""
    if modern:
        return (_QSS + _MODERN_QSS) % palette(dark, modern=True)
    return _QSS % palette(dark)
