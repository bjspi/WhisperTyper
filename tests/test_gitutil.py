"""The self-update only treats the project root itself as a git checkout."""
from __future__ import annotations

import sys

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
