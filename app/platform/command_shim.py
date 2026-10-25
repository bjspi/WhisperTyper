"""The ``whispertyper`` command in the user's bin folder, on PATH — for checkouts and package installs alike.

A clone started with ``python run.py`` has no command at all, and a package install keeps its
launchers in an environment no terminal may know. The launchers placed here start *this*
installation (its interpreter and its ``run.py``), so self-updates and moved virtual environments
are followed by rewriting them at the next start. Windows gets ``.exe`` launchers with the app icon
(``whispertyper`` for the terminal, ``whispertyperw`` without a console window) and the folder on
the user's PATH; macOS and Linux get an executable script in ``~/.local/bin``, which their shells
usually put on PATH already. Switching off removes only launchers of this installation.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path
from typing import List, Sequence

from app.core.env import is_WINDOWS
from app.core.icon_file import IconImage
from app.services.gitutil import project_root

COMMAND = "whispertyper"
#: Windows only: starts the tray app without a console window (Start menu entry, autostart).
WINDOWLESS_COMMAND = "whispertyperw"
#: Marks a launcher as written by WhisperTyper; with the project folder it tells this installation's apart.
_SIGNATURE = "# WhisperTyper command launcher"
_SCRIPT_MODE = 0o755
_ENVIRONMENT_KEY = "Environment"
_PATH_VALUE = "Path"
_WM_SETTINGCHANGE = 0x001A
_HWND_BROADCAST = 0xFFFF
_SMTO_ABORTIFHUNG = 0x0002
_BROADCAST_TIMEOUT_MS = 2000


def supported() -> bool:
    """A frozen (PyInstaller) build has no Python to launch and no console; it gets no command."""
    return not getattr(sys, "frozen", False)


def bin_dir() -> Path:
    """The user's command folder: ``$XDG_BIN_HOME`` or ``~/.local/bin`` (as pipx and ``uv tool`` use)."""
    configured = os.environ.get("XDG_BIN_HOME")
    return Path(configured) if configured else Path.home() / ".local" / "bin"


def launcher_paths() -> List[Path]:
    """The launcher files of this platform."""
    if is_WINDOWS:
        return [bin_dir() / f"{COMMAND}.exe", bin_dir() / f"{WINDOWLESS_COMMAND}.exe"]
    return [bin_dir() / COMMAND]


def is_installed() -> bool:
    """Whether every launcher exists and starts this installation."""
    return all(_belongs_here(path) for path in launcher_paths())


def shadowed_by() -> "Path | None":
    """Another ``whispertyper`` that a terminal finds first on PATH (e.g. an older pip install), if any."""
    found = shutil.which(COMMAND)
    if found is None or Path(found).parent == bin_dir():
        return None
    return Path(found)


def install(icon: Sequence[IconImage]) -> bool:
    """Write the launchers and put :func:`bin_dir` on PATH; True when a launcher changed.

    Unchanged launchers are not rewritten (a running one cannot be replaced on Windows).

    Raises:
        OSError: A launcher of another installation is in the way (it is left alone), the launcher
            could not be built, or a file could not be written.
    """
    foreign = [path for path in launcher_paths() if path.exists() and not _belongs_here(path)]
    if foreign:
        raise OSError(f"{foreign[0]} belongs to another WhisperTyper installation; remove it first")
    bin_dir().mkdir(parents=True, exist_ok=True)
    changed = False
    for path, content in zip(launcher_paths(), _launchers(icon)):
        if path.is_file() and path.read_bytes() == content:
            continue
        _replace(path, content)
        changed = True
    if is_WINDOWS:
        _add_to_user_path(bin_dir())
    return changed


def uninstall() -> None:
    """Remove the launchers of this installation; the PATH entry stays (other tools share the folder).

    Raises:
        OSError: A launcher could not be removed (e.g. it is running right now on Windows).
    """
    for path in launcher_paths():
        if _belongs_here(path):
            path.unlink()


# --- Launchers -------------------------------------------------------------------------------
def _start_script() -> str:
    """The Python source every launcher runs: this installation's ``run.py`` entry point."""
    root = project_root()
    return (f"{_SIGNATURE}\n# {root}\nimport sys\n\nsys.path.insert(0, {root!r})\n"
            "from run import main\n\nsys.exit(main())\n")


def _launchers(icon: Sequence[IconImage]) -> List[bytes]:
    """The content of each path in :func:`launcher_paths`."""
    if not is_WINDOWS:
        python = Path(sys.executable)
        script = f"#!{python}\n{_start_script()}"
        return [script.encode("utf-8")]
    from app.platform.windows_launcher import build_launcher

    interpreter_folder = Path(sys.executable).parent
    return [
        build_launcher(interpreter_folder / "python.exe", _start_script(), icon, windowless=False),
        build_launcher(_windowless_interpreter(interpreter_folder), _start_script(), icon, windowless=True),
    ]


def _windowless_interpreter(folder: Path) -> Path:
    """``pythonw.exe`` when the environment has one, else the console interpreter."""
    windowless = folder / "pythonw.exe"
    return windowless if windowless.is_file() else folder / "python.exe"


def _belongs_here(path: Path) -> bool:
    """A launcher written by WhisperTyper for this installation's project folder."""
    if not path.is_file():
        return False
    content = path.read_bytes()
    return _SIGNATURE.encode() in content and f"# {project_root()}\n".encode() in content


def _replace(path: Path, content: bytes) -> None:
    """Write ``path``; a running Windows launcher is renamed aside first (it can be renamed, not overwritten)."""
    if is_WINDOWS and path.exists():
        aside = path.with_suffix(".old")
        aside.unlink(missing_ok=True)
        path.replace(aside)
        path.write_bytes(content)
        try:
            aside.unlink()
        except OSError:
            pass  # still running; replaced again (and so removed) at the next change
        return
    path.write_bytes(content)
    if not is_WINDOWS:
        path.chmod(_SCRIPT_MODE)


# --- PATH (Windows) --------------------------------------------------------------------------
def _add_to_user_path(folder: Path) -> None:
    """Append ``folder`` to the user's PATH (registry) and tell running programs, if it is missing."""
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _ENVIRONMENT_KEY, 0,
                        winreg.KEY_READ | winreg.KEY_WRITE) as key:
        try:
            value, kind = winreg.QueryValueEx(key, _PATH_VALUE)
        except FileNotFoundError:
            value, kind = "", winreg.REG_EXPAND_SZ
        entries = [entry for entry in str(value).split(os.pathsep) if entry]
        if any(_same_folder(entry, folder) for entry in entries):
            return
        winreg.SetValueEx(key, _PATH_VALUE, 0, kind, os.pathsep.join([*entries, str(folder)]))
    _announce_environment_change()


def _same_folder(entry: str, folder: Path) -> bool:
    """Whether a PATH entry (possibly with ``%VARIABLES%``) names ``folder``."""
    expanded = os.path.expandvars(entry).rstrip("\\/")
    return os.path.normcase(os.path.normpath(expanded)) == os.path.normcase(os.path.normpath(str(folder)))


def _announce_environment_change() -> None:
    """Let Explorer pick up the new PATH, so terminals opened from now on find the command."""
    result = ctypes.c_ulong()
    ctypes.windll.user32.SendMessageTimeoutW(_HWND_BROADCAST, _WM_SETTINGCHANGE, 0, "Environment",
                                             _SMTO_ABORTIFHUNG, _BROADCAST_TIMEOUT_MS, ctypes.byref(result))
