"""Platform flags + tiny platform helpers. Single source for is_MACOS / is_WINDOWS."""

import os
import subprocess
import sys

is_MACOS = sys.platform.startswith("darwin")
is_WINDOWS = sys.platform.startswith("win")


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
