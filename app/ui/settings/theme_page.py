"""Light/dark stylesheet of the settings window, its branding header and OS colour-scheme tracking."""
from __future__ import annotations

import os
from typing import Any, Dict

from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStyle, QVBoxLayout, QWidget

from app.core.constants import APP_DATA_DIR
from app.core.hotkeys import pretty_hotkey
from app.ui import theme
from app.ui.settings.base import SettingsWindowBase
from app.ui.tray_icons import app_icon


class ThemePage(SettingsWindowBase):
    """Apply the cross-platform light/dark stylesheet and the branding header."""

    _brand_sub_label: QLabel
    _brand_hotkey_badge: QLabel
    _brand_header: QWidget
    _brand_row: QHBoxLayout
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

        # Plays the last recording; as tall as the hotkey badge beside it.
        self.brand_play_button = QPushButton(header)
        self.brand_play_button.setObjectName("brandPlayButton")
        self.brand_play_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        row.addWidget(self.brand_play_button)

        self._brand_hotkey_badge = QLabel("", header)
        self._brand_hotkey_badge.setObjectName("hotkeyBadge")
        row.addWidget(self._brand_hotkey_badge)

        # Below the File/Help menu bar, above the tabs.
        self.main_layout.insertWidget(1, header)
        self._brand_header, self._brand_row = header, row
        self.update_brand_header()

    def update_brand_header(self) -> None:
        """Refresh the header's hotkey badge and tagline."""
        self._brand_hotkey_badge.setText("⌨  " + pretty_hotkey(self.config.get("hotkey", "")))
        self._brand_sub_label.setText(self.translator.tr("brand_tagline"))
        self.set_playback_active(getattr(self, "_playback_active", False))

    def set_playback_active(self, active: bool) -> None:
        """Show Play, or Stop while the last recording plays."""
        self._playback_active = active
        icon = self._playback_icon(active)
        style = self.style()
        if icon is None and style is not None:
            pixmap = QStyle.StandardPixmap.SP_MediaStop if active else QStyle.StandardPixmap.SP_MediaPlay
            icon = style.standardIcon(pixmap)
        if icon is not None:
            self.brand_play_button.setIcon(icon)
        self.brand_play_button.setToolTip(
            self.translator.tr("playback_stop_tooltip" if active else "playback_play_tooltip"))

    def theme_palette(self) -> Dict[str, str]:
        """Colours of the active theme (light until the first theme is applied)."""
        return self._theme_palette or theme.palette(False)

    def apply_theme(self) -> None:
        """Apply the light/dark stylesheet (config 'color_theme': system/light/dark) in the configured design."""
        dark = theme.resolve_dark(self.config.get("color_theme", "system"), QApplication.instance())
        modern = self._modern
        self._theme_palette = theme.palette(dark, modern)
        self._apply_design(modern, self._theme_palette)
        qss = theme.build_stylesheet(dark, modern)
        try:
            qss += theme.write_icon_qss(dark, os.path.join(APP_DATA_DIR, "theme_icons"), modern)
        except Exception:
            pass  # icons are cosmetic; never let a write failure break theming
        self.setStyleSheet(qss)
        self._apply_prompt_list_icons()
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.setStyleSheet(theme.tooltip_stylesheet(dark, modern))
        self._replacements_tab.highlighter.set_theme(dark)
        # State colours (incomplete sections, token counter, FFmpeg status) are QSS property
        # selectors and follow the new palette by themselves; only the header is drawn by hand.
        self.update_brand_header()
        self._connect_theme_watch()
        self.appearance_changed.emit()

    def _apply_prompt_list_icons(self) -> None:
        """Icon-only "+" / "-" below the prompt list, coloured for the active theme."""
        colors = self.theme_palette()
        for button, color, plus in ((self.post_rp_add_btn, colors["on_accent"], True),
                                    (self.post_rp_remove_btn, colors["warn"], False)):
            button.setText("")
            button.setIcon(theme.sign_icon(color, plus))

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
