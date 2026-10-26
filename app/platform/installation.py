"""How WhisperTyper is installed, and how Python packages get into the environment it runs in.

A source checkout (a git clone run with any interpreter: venv, conda, system Python) installs its
``requirements.txt``, and a self-update can add requirements there. A packaged install (pipx,
``uv tool``, ``uvx``) gets its dependencies from ``pyproject.toml`` when it is installed, so a missing
package there means a broken environment that its tool has to reinstall. Environments made by uv
usually have no pip; uv then installs into them. Qt-free and free of third-party imports: the start
uses it when a package is missing.
"""
from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

from app.core.env import is_WINDOWS
from app.services.gitutil import project_root

PACKAGE_NAME = "whispertyper"
REQUIREMENTS_FILE = "requirements.txt"
#: pipx writes this file into each environment it manages.
_PIPX_MARKER = "pipx_metadata.json"
#: ``uv tool install`` writes this file into each tool environment.
_UV_TOOL_MARKER = "uv-receipt.toml"
#: ``uvx`` runs tools from cached environments below this folder of uv's cache.
_UVX_CACHE_FOLDER = "archive-v0"


class Installation(Enum):
    """The ways WhisperTyper can be installed; each one gets missing packages differently."""

    CHECKOUT = "checkout"
    PIPX = "pipx"
    UV_TOOL = "uv tool"
    UVX = "uvx"
    PACKAGE = "package"
    FROZEN = "frozen"


def requirements_file() -> Optional[str]:
    """The checkout's ``requirements.txt``; None for packaged installs and frozen builds."""
    if getattr(sys, "frozen", False):
        return None
    path = os.path.join(project_root(), REQUIREMENTS_FILE)
    return path if os.path.isfile(path) else None


def installation() -> Installation:
    """How this copy of WhisperTyper is installed."""
    if getattr(sys, "frozen", False):
        return Installation.FROZEN
    if requirements_file() is not None:
        return Installation.CHECKOUT
    prefix = Path(sys.prefix)
    if (prefix / _PIPX_MARKER).is_file():
        return Installation.PIPX
    if (prefix / _UV_TOOL_MARKER).is_file():
        return Installation.UV_TOOL
    if _UVX_CACHE_FOLDER in prefix.parts:
        return Installation.UVX
    return Installation.PACKAGE


def console_python() -> str:
    """This environment's interpreter with a console: under ``pythonw`` child processes get no output pipes."""
    folder, name = os.path.split(sys.executable)
    if name.lower() == "pythonw.exe":
        candidate = os.path.join(folder, "python.exe")
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def install_command(arguments: Sequence[str]) -> Optional[List[str]]:
    """Program and arguments that install ``arguments`` (requirements, or ``-r file``) into this environment.

    The interpreter's own pip when it has one, otherwise uv aimed at this interpreter; None when
    neither is available.
    """
    python = console_python()
    if importlib.util.find_spec("pip") is not None:
        return [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location",
                *arguments]
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", python, *arguments]
    return None


def requirements_install_command() -> Optional[List[str]]:
    """The command that installs the checkout's requirements; None without a checkout or an installer."""
    requirements = requirements_file()
    return install_command(["-r", requirements]) if requirements else None


def repair_command() -> Optional[str]:
    """The command a user runs by hand to get missing packages; None when it depends on how they ran uvx/pip."""
    kind = installation()
    if kind is Installation.CHECKOUT:
        command = requirements_install_command()
        return display_command(command) if command else display_command(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file() or REQUIREMENTS_FILE])
    if kind is Installation.PIPX:
        return f"pipx reinstall {PACKAGE_NAME}"
    if kind is Installation.UV_TOOL:
        return f"uv tool upgrade --reinstall {PACKAGE_NAME}"
    return None


def display_command(command: Sequence[str]) -> str:
    """A command line as the user's shell expects it, for copying from a message."""
    return subprocess.list2cmdline(command) if is_WINDOWS else shlex.join(command)
