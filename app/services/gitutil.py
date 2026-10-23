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
from typing import List, Optional

from app.core.paths import resource_path
from app.platform.system import no_window_kwargs


def project_root() -> str:
    """Return the source-checkout root (the parent of the bundled ``app`` package)."""
    return os.path.dirname(resource_path())


def find_git_root() -> Optional[str]:
    """Return the project root if it is a git working tree, or ``None`` if this isn't a checkout.

    A frozen build can never be a git checkout, so we short-circuit there and the tray
    never shows the "Update" entry for it. Otherwise only a ``.git`` entry directly in the
    project root counts (a directory for a normal clone, a file for worktrees and
    submodules): an unpacked copy inside some other repository must not pull that one.
    """
    if getattr(sys, "frozen", False):
        return None
    path = project_root()
    return path if os.path.exists(os.path.join(path, ".git")) else None


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


def changed_paths(root: str, old: str, new: str, paths: List[str]) -> List[str]:
    """Which of ``paths`` differ between commits ``old`` and ``new`` (empty on any git failure)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", old, new, "--", *paths],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


#: The changes shown before an update are cut off here; a full diff of a big release is unreadable anyway.
MAX_CHANGES_CHARS = 200_000


def release_changes(root: str, tag: str) -> Optional[str]:
    """What updating the checkout at ``root`` to the (already fetched) ``tag`` brings: commits, files, diff.

    ``None`` when git fails, e.g. because the tag was not fetched.
    """
    parts = []
    for arguments in (["log", "--oneline", "--no-decorate", f"HEAD..{tag}"],
                      ["diff", "--stat", "HEAD", tag],
                      ["diff", "HEAD", tag]):
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                **no_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        parts.append(result.stdout.strip())
    changes = "\n\n".join(part for part in parts if part)
    if len(changes) > MAX_CHANGES_CHARS:
        changes = changes[:MAX_CHANGES_CHARS] + "\n\n[…]"
    return changes
