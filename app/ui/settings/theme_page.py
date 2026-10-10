"""Light/dark stylesheet of the settings window, its branding header and OS colour-scheme tracking."""
from __future__ import annotations

import os
from typing import Any, Dict

from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.core.constants import APP_DATA_DIR
from app.core.hotkeys import pretty_hotkey
from app.ui import theme
from app.ui.settings.base import SettingsWindowBase
from app.ui.tray_icons import app_icon


class ThemePage(SettingsWindowBase):
    """Apply the cross-platform light/dark stylesheet and the branding header."""

    _brand_sub_label: QLabel
    _brand_hotkey_badge: QLabel
    _theme_watch_connected: bool = False

    def _install_brand_header(self) -> None:
        """Add a branding header (icon + name + current hotkey badge) above the tabs."""
        header = QWidget(self)
        header.setObjectName("brandHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(10)

        icon_label = QLabel(header)
        icon = app_icon()
        if not icon.isNull():
            icon_label.setPixmap(icon.pixmap(26, 26))
        row.addWidget(icon_label)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("WhisperTyper", header)
        title.setObjectName("brandTitle")
        self._brand_sub_label = QLabel("", header)
        self._brand_sub_label.setObjectName("brandSub")
        title_box.addWidget(title)
        title_box.addWidget(self._brand_sub_label)
        row.addLayout(title_box)
        row.addStretch()

        self._brand_hotkey_badge = QLabel("", header)
        self._brand_hotkey_badge.setObjectName("hotkeyBadge")
        row.addWidget(self._brand_hotkey_badge)

        # Below the File/Help menu bar, above the tabs.
        self.main_layout.insertWidget(1, header)
        self.update_brand_header()

    def update_brand_header(self) -> None:
        """Refresh the header's hotkey badge and tagline."""
        self._brand_hotkey_badge.setText("⌨  " + pretty_hotkey(self.config.get("hotkey", "")))
        self._brand_sub_label.setText(self.translator.tr("brand_tagline"))

    def theme_palette(self) -> Dict[str, str]:
        """Colours of the active theme (light until the first theme is applied)."""
        return self._theme_palette or theme.palette(False)

    def apply_theme(self) -> None:
        """Apply the light/dark teal stylesheet (config 'color_theme': system/light/dark)."""
        dark = theme.resolve_dark(self.config.get("color_theme", "system"), QApplication.instance())
        self._theme_palette = theme.palette(dark)
        qss = theme.build_stylesheet(dark)
        try:
            qss += theme.write_icon_qss(dark, os.path.join(APP_DATA_DIR, "theme_icons"))
        except Exception:
            pass  # icons are cosmetic; never let a write failure break theming
        self.setStyleSheet(qss)
        self._replacements_tab.highlighter.set_theme(dark)
        # State colours (incomplete sections, token counter, FFmpeg status) are QSS property
        # selectors and follow the new palette by themselves; only the header is drawn by hand.
        self.update_brand_header()
        self._connect_theme_watch()

    def _connect_theme_watch(self) -> None:
        """Subscribe once to OS colour-scheme changes to re-theme live."""
        if self._theme_watch_connected:
            return
        try:
            app = QApplication.instance()
            app.styleHints().colorSchemeChanged.connect(self._on_color_scheme_changed)  # type: ignore[union-attr]
            self._theme_watch_connected = True
        except Exception:
            self._theme_watch_connected = False

    def _on_color_scheme_changed(self, *_args: Any) -> None:
        """Re-apply the stylesheet when the OS switches between light and dark."""
        self.apply_theme()
