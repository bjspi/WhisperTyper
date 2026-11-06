"""The spoken-language choices as people recognise them: every language under its own name."""
from __future__ import annotations

from typing import List, Tuple

from PyQt6.QtCore import QLocale

from app.core.constants import LANGUAGES

#: Qt names the regional default of these languages ("American English", "Español de España").
_PLAIN_NAMES = {"en": "English", "es": "Español", "pt": "Português"}
#: The stored value of "detect the language"; its caption is translated by whoever shows the list.
DETECT_LANGUAGE = ""


def _own_name(english_name: str, code: str) -> str:
    """The language's name in that language ("Deutsch"); the English name where Qt does not know it."""
    if code in _PLAIN_NAMES:
        return _PLAIN_NAMES[code]
    locale = QLocale(code)
    if locale.language() == QLocale.Language.C:
        return english_name
    native = locale.nativeLanguageName()
    return native[:1].upper() + native[1:] if native else english_name


def input_language_items() -> List[Tuple[str, str]]:
    """(caption, language code) pairs in the catalog's order."""
    return [(english_name if code == DETECT_LANGUAGE else _own_name(english_name, code), code)
            for english_name, code in LANGUAGES.items()]
