"""Application fonts registered with Qt at startup."""
from __future__ import annotations

import logging

from PyQt6.QtGui import QFontDatabase

from app.core.env import is_MACOS
from app.core.paths import resource_path

#: Noto Color Emoji (COLRv1, SIL Open Font License 1.1): colour emoji incl. regional and subdivision flags.
_EMOJI_FONT = resource_path("resources", "fonts", "NotoColorEmoji-COLRv1.ttf")


def install_emoji_font() -> None:
    """Make Qt draw emoji with the bundled colour font (requires a QGuiApplication).

    Windows' Segoe UI Emoji has no flags and Linux often lacks a colour emoji font; macOS ships
    Apple Color Emoji, so it keeps its system font.
    """
    if is_MACOS:
        return
    font_id = QFontDatabase.addApplicationFont(_EMOJI_FONT)
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id != -1 else []
    if not families:
        logging.warning("Bundled emoji font could not be loaded: %s", _EMOJI_FONT)
        return
    QFontDatabase.addApplicationEmojiFontFamily(families[0])
