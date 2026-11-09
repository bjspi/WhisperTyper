"""Light/dark stylesheet of the settings window, its hotkey badge and play button, and OS colour-scheme tracking."""
from __future__ import annotations

import os
from typing import Any, Dict

from PyQt6.QtWidgets import QApplication

from app.core.constants import APP_DATA_DIR
from app.core.hotkeys import pretty_hotkey
from app.ui import theme
from app.ui.line_icons import line_icon
from app.ui.settings.base import SettingsWindowBase


class ThemePage(SettingsWindowBase):
    """Apply the cross-platform light/dark stylesheet; keep the sidebar's hotkey badge and play button current."""

    _theme_watch_connected: bool = False

    def update_hotkey_badge(self) -> None:
        """Refresh the sidebar's hotkey badge and the play button's icon and tooltip."""
        self._hotkey_badge.setText("⌨  " + pretty_hotkey(self.config.get("hotkey", "")))
        self.set_playback_active(getattr(self, "_playback_active", False))

    def set_playback_active(self, active: bool) -> None:
        """Show Play, or Stop while the last recording plays."""
        self._playback_active = active
        self.play_button.setIcon(line_icon("stop" if active else "play", self.theme_palette()["menu_icon"]))
        self.play_button.setToolTip(
            self.translator.tr("playback_stop_tooltip" if active else "playback_play_tooltip"))

    def theme_palette(self) -> Dict[str, str]:
        """Colours of the active theme (light until the first theme is applied)."""
        return self._theme_palette or theme.palette(False)

    def apply_theme(self) -> None:
        """Apply the light/dark stylesheet (config 'color_theme': system/light/dark)."""
        dark = theme.resolve_dark(self.config.get("color_theme", "system"), QApplication.instance())
        self._theme_palette = theme.palette(dark)
        self._apply_chrome_icons(self._theme_palette)
        qss = theme.build_stylesheet(dark)
        try:
            qss += theme.write_icon_qss(dark, os.path.join(APP_DATA_DIR, "theme_icons"))
        except Exception:
            pass  # icons are cosmetic; never let a write failure break theming
        self.setStyleSheet(qss)
        self._apply_prompt_list_icons()
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.setStyleSheet(theme.tooltip_stylesheet(dark))
        self._replacements_tab.highlighter.set_theme(dark)
        # State colours (incomplete sections, token counter, FFmpeg status) are QSS property
        # selectors and follow the new palette by themselves; only the play icon is tinted by hand.
        self.update_hotkey_badge()
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
