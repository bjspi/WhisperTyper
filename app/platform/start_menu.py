"""The WhisperTyper entry in the system launcher: Start menu (Windows search) or the Linux application list.

The entry starts this installation's interpreter with its ``run.py`` (a frozen build: its own
executable), so it keeps working after self-updates; it is rewritten at every start. macOS lists
apps from bundles only and gets no entry.
"""
from __future__ import annotations

import base64
import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

from app.core.env import is_MACOS, is_WINDOWS
from app.platform.system import no_window_kwargs

APP_NAME = "WhisperTyper"
_DESCRIPTION = "Voice to text and live AI prompting"
_LINUX_ENTRY = Path.home() / ".local" / "share" / "applications" / "whispertyper.desktop"
#: Seconds PowerShell may take to write the link.
_LINK_TIMEOUT_S = 60


def supported() -> bool:
    """Whether this platform has such an entry (macOS needs an app bundle instead)."""
    return not is_MACOS


def entry_path() -> Path:
    """Where the entry lives: the Start menu link or the ``.desktop`` file."""
    if is_WINDOWS:
        return _start_menu_folder() / f"{APP_NAME}.lnk"
    return _LINUX_ENTRY


def exists() -> bool:
    """Whether the entry is there."""
    return entry_path().exists()


def create(command: Sequence[str], icon: Path, working_dir: Path) -> Path:
    """Create or replace the entry starting ``command``; returns its path.

    Raises:
        OSError: The entry could not be written.
    """
    if is_WINDOWS:
        return _create_link(command, icon, working_dir)
    return _create_desktop_file(command, icon)


def remove() -> None:
    """Remove the entry, if there is one.

    Raises:
        OSError: The entry could not be removed.
    """
    entry_path().unlink(missing_ok=True)


def _start_menu_folder() -> Path:
    """The user's Start menu ``Programs`` folder."""
    roaming = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(roaming) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def _create_link(command: Sequence[str], icon: Path, working_dir: Path) -> Path:
    """Write the ``.lnk`` through the Windows Script Host (PowerShell; no pywin32 needed)."""
    target = entry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    script = "\n".join([
        "$shell = New-Object -ComObject WScript.Shell",
        f"$link = $shell.CreateShortcut({_quoted(str(target))})",
        f"$link.TargetPath = {_quoted(command[0])}",
        f"$link.Arguments = {_quoted(subprocess.list2cmdline(list(command[1:])))}",
        f"$link.WorkingDirectory = {_quoted(str(working_dir))}",
        f"$link.IconLocation = {_quoted(str(icon))}",
        f"$link.Description = {_quoted(_DESCRIPTION)}",
        "$link.Save()",
    ])
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
                                timeout=_LINK_TIMEOUT_S, **no_window_kwargs())
    except subprocess.TimeoutExpired as error:
        raise OSError(f"could not create {target}: PowerShell did not finish") from error
    if result.returncode != 0 or not target.exists():
        raise OSError(f"could not create {target}: {result.stderr.strip() or result.stdout.strip()}")
    return target


def _quoted(text: str) -> str:
    """A PowerShell single-quoted string literal."""
    return "'" + text.replace("'", "''") + "'"


def _create_desktop_file(command: Sequence[str], icon: Path) -> Path:
    """Write the freedesktop ``.desktop`` entry."""
    target = entry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "[Desktop Entry]\nType=Application\n"
        f"Name={APP_NAME}\nComment={_DESCRIPTION}\n"
        f"Exec={shlex.join(command)}\nIcon={icon}\nTerminal=false\nCategories=Utility;Audio;\n",
        encoding="utf-8",
    )
    return target
