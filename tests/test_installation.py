"""How the installation kind and the package installer are detected."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.platform import installation
from app.platform.installation import Installation


@pytest.fixture
def packaged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A packaged install (no requirements.txt) whose environment is ``tmp_path``."""
    monkeypatch.setattr(installation, "requirements_file", lambda: None)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    return tmp_path


def test_a_checkout_has_its_requirements_file():
    assert installation.installation() is Installation.CHECKOUT
    assert installation.requirements_file() is not None


def test_pipx_and_uv_tool_environments_are_told_by_their_marker_files(packaged: Path):
    assert installation.installation() is Installation.PACKAGE
    (packaged / "uv-receipt.toml").write_text("")
    assert installation.installation() is Installation.UV_TOOL
    (packaged / "pipx_metadata.json").write_text("{}")
    assert installation.installation() is Installation.PIPX


def test_uvx_runs_from_uvs_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(installation, "requirements_file", lambda: None)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "uv" / "cache" / "archive-v0" / "abc123"))
    assert installation.installation() is Installation.UVX
    assert installation.repair_command() is None


def test_packaged_installs_are_repaired_by_their_tool(packaged: Path):
    (packaged / "pipx_metadata.json").write_text("{}")
    assert installation.repair_command() == "pipx reinstall whispertyper"


def test_pip_installs_and_skips_the_path_warnings():
    command = installation.install_command(["x"])
    assert command is not None
    assert command[1:4] == ["-m", "pip", "install"] and "--no-warn-script-location" in command


def test_uv_installs_into_an_environment_without_pip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(installation.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(installation.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    command = installation.install_command(["-r", "requirements.txt"])
    assert command == ["/usr/bin/uv", "pip", "install", "--python", installation.console_python(),
                       "-r", "requirements.txt"]


def test_without_pip_and_uv_nothing_can_install(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(installation.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(installation.shutil, "which", lambda name: None)
    assert installation.install_command(["x"]) is None
    assert installation.requirements_install_command() is None
    assert "pip install -r" in (installation.repair_command() or "")
