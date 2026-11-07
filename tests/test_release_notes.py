"""Release notes gather the commits since the previous release, grouped by type, in a real repository."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from scripts import release_notes

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

GIT_TIMEOUT_S = 60


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git = _git(tmp_path)
    git("init", "--quiet", "--initial-branch=main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    return tmp_path


def _git(repository: Path) -> Callable[..., None]:
    def run(*arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True, timeout=GIT_TIMEOUT_S)

    return run


def _commit(repository: Path, message: str) -> None:
    _git(repository)("commit", "--quiet", "--allow-empty", "-m", message)


def _notes(repository: Path, tag: str) -> str:
    previous = release_notes.previous_tag(tag, repository)
    commits = release_notes.commits_since(previous, tag, repository)
    return release_notes.release_notes(tag, previous, commits, "owner/name")


def test_changes_are_grouped_by_type_with_their_descriptions(repository: Path) -> None:
    _commit(repository, "feat: first")
    _commit(repository, "chore: release 1.0.0")
    _git(repository)("tag", "-a", "v1.0.0", "-m", "1.0.0")
    _commit(repository, "fix(ui): keep the window\n\n- On top.\n- Sized\n  right.")
    _commit(repository, "docs: explain the settings [skip ci]")
    _commit(repository, "feat!: drop the old config")
    _commit(repository, "Plain subject")
    _commit(repository, "chore: release 1.1.0\n\nA cleaner window.")
    _git(repository)("tag", "-a", "v1.1.0", "-m", "1.1.0")
    assert _notes(repository, "v1.1.0") == (
        "A cleaner window.\n\n"
        "## Features\n\n"
        "- **Drop the old config** (breaking)\n\n"
        "## Fixes\n\n"
        "- **Keep the window** (ui)\n\n"
        "  - On top.\n"
        "  - Sized\n"
        "    right.\n\n"
        "## Other changes\n\n"
        "- **Explain the settings**\n\n"
        "- Plain subject\n\n"
        "**Full Changelog**: https://github.com/owner/name/compare/v1.0.0...v1.1.0\n"
    )


def test_the_first_release_has_every_commit_and_no_compare_link(repository: Path) -> None:
    _commit(repository, "feat: start")
    _commit(repository, "chore: release 0.1.0")
    _git(repository)("tag", "-a", "v0.1.0", "-m", "0.1.0")
    assert _notes(repository, "v0.1.0") == "## Features\n\n- **Start**\n"


@pytest.mark.parametrize("subject", ["chore: release 2.0.0", "chore(release): release 2.0.0", "Version 2.0.0"])
def test_release_commits_are_left_out(subject: str) -> None:
    assert release_notes.Commit(subject, "").is_release
