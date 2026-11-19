"""The spoken-language choices as people recognise them: every language under its own name."""
from __future__ import annotations

from typing import List, Tuple

from PyQt6.QtCore import QLocale

from app.core.input_languages import DETECT_LANGUAGE as DETECT_LANGUAGE
from app.core.input_languages import LANGUAGES

#: Qt names the regional default of these languages ("American English", "Español de España").
_PLAIN_NAMES = {"en": "English", "es": "Español", "pt": "Português"}
_ENGLISH_NAMES = {code: name for name, code in LANGUAGES.items()}


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
    return [(language_name(code), code) for code in _ENGLISH_NAMES]


def language_name(code: str) -> str:
    """The native caption of a language code, falling back to the code if unknown."""
    english_name = _ENGLISH_NAMES.get(code)
    if english_name is None or code == DETECT_LANGUAGE:
        return english_name or code
    return _own_name(english_name, code)
