"""The self-update only treats the project root itself as a git checkout, and spots a rewritten release."""
from __future__ import annotations

import subprocess
import sys

import pytest

from app.services import gitutil


def test_a_git_directory_in_the_project_root_is_a_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "WhisperTyper"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(gitutil, "project_root", lambda: str(checkout))
    assert gitutil.find_git_root() == str(checkout)


def test_a_git_file_marks_a_worktree_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "WhisperTyper"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: ../.git/worktrees/WhisperTyper\n")
    monkeypatch.setattr(gitutil, "project_root", lambda: str(checkout))
    assert gitutil.find_git_root() == str(checkout)


def test_a_copy_inside_a_foreign_repository_is_no_checkout(tmp_path, monkeypatch):
    # An unpacked zip of the app inside the user's own repository must not offer to pull that one.
    (tmp_path / "theirs" / ".git").mkdir(parents=True)
    copy = tmp_path / "theirs" / "tools" / "WhisperTyper"
    copy.mkdir(parents=True)
    monkeypatch.setattr(gitutil, "project_root", lambda: str(copy))
    assert gitutil.find_git_root() is None


def test_a_frozen_build_is_never_a_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "WhisperTyper"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(gitutil, "project_root", lambda: str(checkout))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert gitutil.find_git_root() is None


def _git(cwd, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True)


def _release(checkout, tag: str, email: str) -> None:
    """Commit the checkout's files as ``tag`` with the given author (a history rewrite changes it)."""
    _git(checkout, "add", "-A")
    _git(checkout, "-c", "user.name=Maintainer", "-c", f"user.email={email}", "commit", "-q", "-m", tag)
    _git(checkout, "tag", "-f", tag)


@pytest.mark.skipif(not gitutil.git_available(), reason="git is not installed")
def test_only_an_untouched_checkout_of_a_rewritten_release_is_pristine(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    (checkout / ".gitignore").write_text("config.json\n")
    (checkout / "app.py").write_text("print('v1')\n")
    _release(checkout, "v1.0.0", "old@example.com")
    (checkout / "config.json").write_text("{}")  # Ignored files never count.
    # The rewritten tag: same files, a new commit without a common ancestor.
    _git(checkout, "checkout", "-q", "--orphan", "rewritten")
    _release(checkout, "v1.0.0-new", "new@example.com")
    _git(checkout, "checkout", "-q", "v1.0.0")
    root = str(checkout)

    assert gitutil.is_pristine_release(root, "refs/tags/v1.0.0-new")
    (checkout / "notes.txt").write_text("mine")
    assert not gitutil.is_pristine_release(root, "refs/tags/v1.0.0-new")
    (checkout / "notes.txt").unlink()
    (checkout / "app.py").write_text("print('edited')\n")
    assert not gitutil.is_pristine_release(root, "refs/tags/v1.0.0-new")
    assert not gitutil.is_pristine_release(root, "refs/tags/missing")
