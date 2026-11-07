"""Write a release's notes from the commits since the previous release.

The release workflow publishes them as the text of the GitHub release. Preview them with
``python scripts/release_notes.py v1.2.3``.

Commits follow Conventional Commits (``feat(ui): …``) and are grouped by their type, each with its
subject and description. The ``chore: release X.Y.Z`` commit itself is left out; a description it
carries goes on top as the release's summary.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
RELEASE_TAGS: Final = "v*"
RELEASE_COMMIT: Final = re.compile(r"chore(\(release\))?: release \d+\.\d+\.\d+.*|Version \d+\.\d+\.\d+.*")
CONVENTIONAL: Final = re.compile(r"(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<breaking>!)?:\s*(?P<text>.+)")
SECTIONS: Final = (("Features", ("feat",)), ("Fixes", ("fix",)), ("Performance", ("perf",)))
OTHER_SECTION: Final = "Other changes"
SKIP_CI: Final = "[skip ci]"
GIT_TIMEOUT_S: Final = 60
FIELD_END: Final = "\x1f"  # Never in a commit message.
RECORD_END: Final = "\x1e"
BODY_INDENT: Final = "  "  # Nests a commit's own list under its subject.


@dataclass(frozen=True, slots=True)
class Commit:
    """One commit's message, split as Git stores it."""

    subject: str
    body: str

    @property
    def is_release(self) -> bool:
        """Whether this is the commit that bumps the version."""
        return RELEASE_COMMIT.fullmatch(self.subject) is not None

    @property
    def kind(self) -> str:
        """The Conventional Commits type, such as ``feat``; empty for any other subject."""
        match = CONVENTIONAL.fullmatch(self.subject)
        return match["type"] if match else ""

    @property
    def headline(self) -> str:
        """The subject without its type, capitalized, with scope and a breaking change noted after it."""
        match = CONVENTIONAL.fullmatch(self.subject)
        if match is None:
            return self.subject
        text = match["text"]
        notes = [part for part in (match["scope"], "breaking" if match["breaking"] else "") if part]
        suffix = f" ({', '.join(notes)})" if notes else ""
        return f"**{text[:1].upper()}{text[1:]}**{suffix}"


def previous_tag(tag: str, repository: Path = ROOT) -> str | None:
    """The release tag before ``tag``; None for the first release."""
    found = _git(repository, "describe", "--tags", "--abbrev=0", "--match", RELEASE_TAGS, f"{tag}^")
    return found.strip() if found is not None else None


def commits_since(previous: str | None, tag: str, repository: Path = ROOT) -> list[Commit]:
    """The commits ``tag`` adds to ``previous``, oldest first."""
    revisions = f"{previous}..{tag}" if previous else tag
    log = _git(repository, "log", "--reverse", "--no-merges", f"--format=%s{FIELD_END}%b{RECORD_END}", revisions)
    if log is None:
        raise SystemExit(f"Git cannot list the commits of {revisions}.")
    commits = []
    for record in log.split(RECORD_END):
        subject, _, body = record.strip("\n").partition(FIELD_END)
        if subject.strip():
            commits.append(Commit(" ".join(subject.replace(SKIP_CI, "").split()), body.strip()))
    return commits


def release_notes(tag: str, previous: str | None, commits: list[Commit], github_repository: str = "") -> str:
    """Markdown: the release's summary, then its changes grouped by type, each with its description."""
    parts = [commit.body for commit in commits if commit.is_release and commit.body]
    changes = [commit for commit in commits if not commit.is_release]
    grouped = {kinds: [c for c in changes if c.kind in kinds] for _, kinds in SECTIONS}
    known = {kind for _, kinds in SECTIONS for kind in kinds}
    sections = [(title, grouped[kinds]) for title, kinds in SECTIONS]
    sections.append((OTHER_SECTION, [c for c in changes if c.kind not in known]))
    parts.extend(_section(title, entries) for title, entries in sections if entries)
    if previous and github_repository:
        parts.append(f"**Full Changelog**: https://github.com/{github_repository}/compare/{previous}...{tag}")
    return "\n\n".join(parts) + "\n" if parts else ""


def _section(title: str, commits: list[Commit]) -> str:
    return f"## {title}\n\n" + "\n\n".join(_entry(commit) for commit in commits)


def _entry(commit: Commit) -> str:
    if not commit.body:
        return f"- {commit.headline}"
    body = "\n".join(BODY_INDENT + line if line else "" for line in commit.body.splitlines())
    return f"- {commit.headline}\n\n{body}"


def _git(repository: Path, *arguments: str) -> str | None:
    """Git's output, or None when it fails, such as ``describe`` before the first tag."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    """Print the notes of the release tag given on the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("tag", help="the release tag, such as v1.2.3")
    parser.add_argument("--repository", default="", help="owner/name for the compare link")
    arguments = parser.parse_args(argv)
    previous = previous_tag(arguments.tag)
    commits = commits_since(previous, arguments.tag)
    notes = release_notes(arguments.tag, previous, commits, arguments.repository)
    sys.stdout.buffer.write(notes.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
