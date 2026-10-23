"""Which published release is newest. Updates follow release tags such as ``v0.15.2``, never single commits.

A tag exists a few minutes before its release: pushing it starts the workflow that runs CI and only
then publishes the GitHub release with its source ZIP. A new tag therefore counts once that ZIP can
be downloaded (see ``GitUpdater``). No Qt and no git calls here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

REPOSITORY_URL = "https://github.com/bjspi/WhisperTyper"
#: The source ZIP every release publishes (see .github/workflows/release.yml).
RELEASE_ZIP_URL = REPOSITORY_URL + "/releases/download/{tag}/WhisperTyper-{tag}.zip"
_RELEASE_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)")


@dataclass(frozen=True, order=True)
class Release:
    """A ``vX.Y.Z`` release; ordered by its version numbers."""

    major: int
    minor: int
    patch: int

    @classmethod
    def from_tag(cls, tag: str) -> Optional["Release"]:
        """The release a ``vX.Y.Z`` tag names, or None for any other tag."""
        match = _RELEASE_TAG.fullmatch(tag.strip())
        return cls(*map(int, match.groups())) if match else None

    @classmethod
    def from_version(cls, version: str) -> Optional["Release"]:
        """The release of an ``X.Y.Z`` version string."""
        return cls.from_tag(f"v{version}")

    @property
    def version(self) -> str:
        """``X.Y.Z``, as shown to the user."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def tag(self) -> str:
        """``vX.Y.Z``, the git tag."""
        return f"v{self.version}"

    @property
    def zip_url(self) -> str:
        """Downloadable only once the release workflow passed, minutes after the tag was pushed."""
        return RELEASE_ZIP_URL.format(tag=self.tag)


def newest_release(tags: Iterable[str]) -> Optional[Release]:
    """The highest ``vX.Y.Z`` tag; other tags such as ``v1.0-beta`` are ignored."""
    return max(filter(None, map(Release.from_tag, tags)), default=None)


def remote_tags(ls_remote_output: str) -> List[str]:
    """Tag names from ``git ls-remote --tags --refs`` output."""
    prefix = "refs/tags/"
    tags = []
    for line in ls_remote_output.splitlines():
        reference = line.rpartition("\t")[2].strip()
        if reference.startswith(prefix):
            tags.append(reference[len(prefix):])
    return tags
