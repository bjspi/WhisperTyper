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

#: Local git calls (no network) that take longer than this are hung.
GIT_LOCAL_TIMEOUT_S = 10


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


def _git(root: str, arguments: List[str]) -> Optional[str]:
    """Run a fast, offline git command in ``root``; its stdout, or ``None`` when git fails."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_LOCAL_TIMEOUT_S,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def current_head(root: str) -> Optional[str]:
    """Return the current commit SHA of the checkout at ``root``, or ``None`` on failure.

    A fast, offline, read-only call — used to tell "already up to date" from a real update
    by comparing the HEAD before and after a pull, which is robust against git's localized
    "Already up to date." wording.
    """
    output = _git(root, ["rev-parse", "HEAD"])
    if output is None:
        return None
    return output.strip() or None


def changed_paths(root: str, old: str, new: str, paths: List[str]) -> List[str]:
    """Which of ``paths`` differ between commits ``old`` and ``new`` (empty on any git failure)."""
    output = _git(root, ["diff", "--name-only", old, new, "--", *paths]) or ""
    return [line for line in output.splitlines() if line.strip()]


def is_pristine_release(root: str, tag: str) -> bool:
    """Whether the checkout holds exactly the (already fetched) ``tag``'s files and nothing else.

    Compares content, not commits: after the release history was rewritten (same files, new
    commit IDs) the checkout no longer fast-forwards, yet replacing it loses nothing. Local
    edits or untracked files count as a difference; ignored files such as the config do not.
    """
    trees = _git(root, ["rev-parse", "HEAD^{tree}", f"{tag}^{{tree}}"])
    if trees is None:
        return False
    head_tree, tag_tree = trees.split()
    return head_tree == tag_tree and _git(root, ["status", "--porcelain"]) == ""


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
        output = _git(root, arguments)
        if output is None:
            return None
        parts.append(output.strip())
    changes = "\n\n".join(part for part in parts if part)
    if len(changes) > MAX_CHANGES_CHARS:
        changes = changes[:MAX_CHANGES_CHARS] + "\n\n[…]"
    return changes
