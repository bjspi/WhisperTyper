"""Cross-platform system queries and process helpers (subprocess / OS APIs)."""
from __future__ import annotations

import os
import re
import subprocess

from app.core.env import is_MACOS, is_WINDOWS


def open_with_default_app(path: str) -> None:
    """Open a file with the OS default application (Explorer/Finder/xdg-open semantics)."""
    if is_WINDOWS:
        os.startfile(path)  # noqa: S606 - intended behavior
    elif is_MACOS:
        subprocess.call(["open", path])
    else:
        subprocess.call(["xdg-open", path])


def no_window_kwargs() -> dict:
    """Subprocess kwargs that suppress the transient console window on Windows.

    WhisperTyper is a GUI app, so any ``subprocess`` call (ffmpeg, git, …) would otherwise
    flash up a console window. Spread this into the call: ``subprocess.run(..., **no_window_kwargs())``.
    Returns an empty dict off Windows.
    """
    if is_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def system_language_2char() -> str:
    """Return a 2-character language code (e.g. 'en', 'de'); English variants map to 'en'."""
    lang = None
    if is_WINDOWS:
        try:
            import ctypes
            import locale
            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()  # type: ignore[attr-defined]
            lang = locale.windows_locale.get(lang_id, 'en')
        except Exception:
            lang = os.environ.get('LANG', 'en')
    elif is_MACOS:
        try:
            output = subprocess.check_output(
                ["defaults", "read", "-g", "AppleLanguages"],
                universal_newlines=True
            )
            match = re.search(r'"([a-zA-Z\-]+)"', output)
            if match:
                lang = match.group(1)
        except Exception:
            lang = os.environ.get('LANG', 'en')
    else:
        lang = os.environ.get('LANG', 'en')

    if not lang:
        return 'en'
    lang = lang.replace('_', '-').lower()
    if lang.startswith('en'):
        return 'en'
    return lang.split('-')[0]
