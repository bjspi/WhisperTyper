"""Cross-platform Qt stylesheet (light/dark, teal accent) for the settings window.

Single responsibility: produce the application QSS for a light or dark palette and
detect the OS colour scheme. No app state.
"""

from __future__ import annotations

from typing import Dict

LIGHT: Dict[str, str] = {
    "bg": "#f5f8fa", "panel": "#ffffff", "panel2": "#eef3f6", "text": "#16212b",
    "muted": "#5a6b78", "border": "#d6dfe5", "field": "#ffffff", "field_border": "#cfd9e0",
    "accent": "#0e8aa8", "accent_hover": "#0b7690", "accent_press": "#095c74",
    "on_accent": "#ffffff", "warn": "#cf6a3f", "hover": "#eef3f6",
}

DARK: Dict[str, str] = {
    "bg": "#0e1620", "panel": "#16212c", "panel2": "#1b2733", "text": "#dce6ee",
    "muted": "#8598a5", "border": "#273744", "field": "#121c26", "field_border": "#2c3d4a",
    "accent": "#35c0dd", "accent_hover": "#4fcbe4", "accent_press": "#2aa6c1",
    "on_accent": "#08222b", "warn": "#e08b4a", "hover": "#1e2b37",
}


def palette(dark: bool) -> Dict[str, str]:
    """Return the colour palette for the requested mode."""
    return DARK if dark else LIGHT


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

QToolTip {
    background-color: %(panel)s; color: %(text)s;
    border: 1px solid %(border)s; padding: 5px 8px; border-radius: 6px;
}

QMenuBar { background: transparent; border: none; padding: 2px 4px; }
QMenuBar::item { background: transparent; padding: 5px 10px; border-radius: 6px; }
QMenuBar::item:selected { background: %(hover)s; }
QMenu { background: %(panel)s; border: 1px solid %(border)s; border-radius: 8px; padding: 5px; }
QMenu::item { padding: 8px 40px 8px 15px; border-radius: 6px; }
QMenu::icon { left: 12px; }
QMenu::item:selected { background: %(accent)s; color: %(on_accent)s; }
QMenu::item:disabled { color: %(muted)s; }

QTabWidget::pane { border: none; top: -1px; }
QTabBar::tab {
    background: transparent; color: %(muted)s; padding: 8px 16px; margin-right: 2px;
    border: none; border-bottom: 2px solid transparent;
}
QTabBar::tab:hover { color: %(text)s; }
QTabBar::tab:selected { color: %(accent)s; border-bottom: 2px solid %(accent)s; }

QGroupBox {
    background: %(panel)s; border: 1px solid %(border)s; border-radius: 12px;
    margin-top: 14px; padding: 14px 12px 10px 12px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 6px; color: %(accent)s; background: %(bg)s;
}

QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: %(field)s; color: %(text)s; border: 1px solid %(field_border)s;
    border-radius: 8px; padding: 6px 8px;
    selection-background-color: %(accent)s; selection-color: %(on_accent)s;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid %(accent)s; }
QLineEdit:hover, QComboBox:hover, QSpinBox:hover { border-color: %(muted)s; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: %(panel)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px;
    selection-background-color: %(accent)s; selection-color: %(on_accent)s; outline: none;
}

QPushButton {
    background: %(panel2)s; color: %(text)s; border: 1px solid %(border)s;
    border-radius: 8px; padding: 6px 14px;
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
/* Small round "?" help badge next to the LivePrompting toggle (fixed 22px in code). */
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

QSlider::groove:horizontal { height: 4px; background: %(border)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal {
    background: %(panel)s; border: 2px solid %(accent)s;
    width: 14px; height: 14px; margin: -6px 0; border-radius: 9px;
}
QSlider::handle:horizontal:hover { background: %(accent)s; }

QListWidget {
    background: %(field)s; color: %(text)s; border: 1px solid %(field_border)s;
    border-radius: 8px; padding: 4px;
}
QListWidget::item { padding: 6px 8px; border-radius: 6px; }
QListWidget::item:selected { background: %(accent)s; color: %(on_accent)s; }
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
"""


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


def write_icon_qss(dark: bool, cache_dir: str) -> str:
    """Write themed checkmark/arrow SVGs into cache_dir and return QSS referencing them.

    Qt renders QSS ``image:`` from SVG *files* but not from ``data:`` URIs for spinbox
    sub-controls, so the icons are written to disk and referenced by path.
    """
    import os
    os.makedirs(cache_dir, exist_ok=True)
    c = palette(dark)
    suffix = "d" if dark else "l"
    assets = {
        "check": _check_svg(c["on_accent"]),
        "up": _arrow_svg(c["accent"], True),
        "down": _arrow_svg(c["accent"], False),
    }
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
    return f"""
QCheckBox::indicator:checked {{ image: url({paths['check']}); }}
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
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({paths['up']}); width: 13px; height: 13px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({paths['down']}); width: 13px; height: 13px; }}
"""


def build_stylesheet(dark: bool) -> str:
    """Build the base application QSS for light or dark mode (icons added by write_icon_qss)."""
    return _QSS % palette(dark)
