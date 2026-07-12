"""Git self-update helpers (source checkouts only).

These are intentionally Qt-free so they can be imported anywhere. The tray menu uses
``find_git_root()`` to decide whether to surface the "Update" entry at all: a frozen
(PyInstaller) build is never a git checkout, so the entry only appears when the app is
run from a cloned source tree.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional

from app.core.env import no_window_kwargs
from app.core.paths import resource_path


def project_root() -> str:
    """Return the source-checkout root (the parent of the bundled ``app`` package)."""
    return os.path.dirname(resource_path())


def find_git_root() -> Optional[str]:
    """Return the enclosing git working-tree root, or ``None`` if this isn't a checkout.

    A frozen build can never be a git checkout, so we short-circuit there and the tray
    never shows the "Update" entry for it. Otherwise we walk up from the project root
    looking for a ``.git`` entry (a directory for a normal clone, a file for worktrees
    and submodules).
    """
    if getattr(sys, "frozen", False):
        return None
    path = project_root()
    while True:
        if os.path.exists(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:  # reached the filesystem root
            return None
        path = parent


def git_available() -> bool:
    """True if a ``git`` executable is resolvable on PATH."""
    return shutil.which("git") is not None


def current_head(root: str) -> Optional[str]:
    """Return the current commit SHA of the checkout at ``root``, or ``None`` on failure.

    A fast, offline, read-only call — used to tell "already up to date" from a real update
    by comparing the HEAD before and after a pull, which is robust against git's localized
    "Already up to date." wording.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def count_behind_upstream(root: str) -> Optional[int]:
    """Return how many commits the checkout at ``root`` is *behind* its upstream branch.

    Offline and instant — it reads the already-fetched remote-tracking ref (``@{u}``), so run
    a ``git fetch`` first to refresh it. Returns ``None`` when there's no upstream configured
    or git errors out (treated by callers as "no update known", never as an update).
    """
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..@{u}"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None
