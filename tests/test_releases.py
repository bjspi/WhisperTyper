"""Release tags: parsing, ordering and the newest published version."""
from __future__ import annotations

from app.services.releases import Release, newest_release, remote_tags


def test_release_tags_are_ordered_by_version_and_other_tags_ignored():
    assert Release.from_tag("v0.10.0") > Release.from_tag("v0.9.12")  # type: ignore[operator]
    assert Release.from_version("0.15.1") == Release(0, 15, 1)
    assert Release.from_tag("v1.0-beta") is None and Release.from_tag("0.15.1") is None
    assert newest_release(["v0.9.12", "v0.10.0", "v0.11.0-rc1", "latest"]) == Release(0, 10, 0)
    assert newest_release([]) is None


def test_release_urls_and_names():
    release = Release(0, 15, 2)
    assert release.tag == "v0.15.2" and release.version == "0.15.2"
    assert release.zip_url.endswith("/releases/download/v0.15.2/WhisperTyper-v0.15.2.zip")


def test_remote_tags_reads_ls_remote_output():
    output = "abc\trefs/tags/v0.15.1\ndef\trefs/tags/v0.15.2\nfff\trefs/heads/main\n"
    assert remote_tags(output) == ["v0.15.1", "v0.15.2"]
