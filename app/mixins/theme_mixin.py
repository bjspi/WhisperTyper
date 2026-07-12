"""ThemeMixin — apply the light/dark teal stylesheet and follow OS colour-scheme changes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.core.hotkeys import pretty_hotkey
from app.ui import theme


class ThemeMixin:
    """Apply the cross-platform light/dark stylesheet and the branding header."""

    _theme_palette: Optional[Dict[str, str]]
    _theme_is_dark: bool
    _theme_watch_connected: bool

    def _install_brand_header(self) -> None:
        """Add a branding header (icon + name + current hotkey badge) above the tabs."""
        header = QWidget(self)  # type: ignore[arg-type]
        header.setObjectName("brandHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(10)

        icon_label = QLabel(header)
        app_icon = self._get_app_icon()  # type: ignore[attr-defined]
        if not app_icon.isNull():
            icon_label.setPixmap(app_icon.pixmap(26, 26))
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
        self.main_layout.insertWidget(1, header)  # type: ignore[attr-defined]
        self._update_brand_header()

    def _update_brand_header(self) -> None:
        """Refresh the header's hotkey badge and tagline (safe if the header is absent)."""
        badge = getattr(self, "_brand_hotkey_badge", None)
        if badge is not None:
            badge.setText("⌨  " + pretty_hotkey(getattr(self, "hotkey_str", "")))
        sub = getattr(self, "_brand_sub_label", None)
        if sub is not None:
            sub.setText(self.translator.tr("brand_tagline"))  # type: ignore[attr-defined]

    def apply_theme(self) -> None:
        """Apply the light/dark teal stylesheet (config 'color_theme': system/light/dark)."""
        app = QApplication.instance()
        mode = self.config.get("color_theme", "system")  # type: ignore[attr-defined]
        if mode == "light":
            dark = False
        elif mode == "dark":
            dark = True
        else:
            dark = theme.is_dark_mode(app)
        self._theme_is_dark = dark
        self._theme_palette = theme.palette(dark)
        qss = theme.build_stylesheet(dark)
        try:
            import os

            from app.core.constants import APP_DATA_DIR
            qss += theme.write_icon_qss(dark, os.path.join(APP_DATA_DIR, "theme_icons"))
        except Exception:
            pass  # icons are cosmetic; never let a write failure break theming
        self.setStyleSheet(qss)  # type: ignore[attr-defined]
        # Re-apply the per-group-box highlight so it matches the new palette.
        for name in ("_update_transcription_api_group_style", "_update_rephrase_api_group_style",
                     "_update_prompt_token_counter", "_update_brand_header"):
            fn = getattr(self, name, None)
            if callable(fn):
                fn()
        self._connect_theme_watch()

    def _connect_theme_watch(self) -> None:
        """Subscribe once to OS colour-scheme changes to re-theme live."""
        if getattr(self, "_theme_watch_connected", False):
            return
        try:
            app = QApplication.instance()
            app.styleHints().colorSchemeChanged.connect(self._on_color_scheme_changed)  # type: ignore[attr-defined]
            self._theme_watch_connected = True
        except Exception:
            self._theme_watch_connected = False

    def _on_color_scheme_changed(self, *_args: Any) -> None:
        """Re-apply the stylesheet when the OS switches between light and dark."""
        self.apply_theme()
