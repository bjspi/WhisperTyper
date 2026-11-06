"""The terminal command launchers and the icon formats behind them."""
from __future__ import annotations

import struct

import pytest

import app.platform.command_shim as shim
from app.core.icon_file import IconImage, ico_file, icon_group

IMAGES = [IconImage(16, b"\x89PNG-small"), IconImage(256, b"\x89PNG-large-image")]


def test_ico_file_points_at_each_image_and_writes_256_as_zero():
    data = ico_file(IMAGES)
    assert struct.unpack_from("<HHH", data) == (0, 1, 2)
    first = struct.unpack_from("<BBBBHHII", data, 6)
    second = struct.unpack_from("<BBBBHHII", data, 22)
    assert first[:2] == (16, 16) and second[:2] == (0, 0)
    assert data[first[7]:first[7] + first[6]] == IMAGES[0].png
    assert data[second[7]:second[7] + second[6]] == IMAGES[1].png


def test_icon_group_numbers_the_images_from_the_first_id():
    group = icon_group(IMAGES, first_id=5)
    assert [struct.unpack_from("<BBBBHHIH", group, 6 + 14 * index)[7] for index in range(2)] == [5, 6]


def test_icon_sizes_above_256_are_rejected():
    with pytest.raises(ValueError):
        ico_file([IconImage(512, b"x")])


@pytest.fixture
def bin_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_BIN_HOME", str(tmp_path / "bin"))
    monkeypatch.setattr(shim, "_add_to_user_path", lambda folder: None)
    return tmp_path / "bin"


def test_install_writes_launchers_that_start_this_installation(bin_folder):
    assert shim.install(IMAGES) is True
    assert all(path.parent == bin_folder for path in shim.launcher_paths())
    assert shim.is_installed()
    assert shim.install(IMAGES) is False  # unchanged launchers are left alone


def test_a_launcher_of_another_installation_is_never_replaced(bin_folder):
    bin_folder.mkdir()
    foreign = shim.launcher_paths()[0]
    foreign.write_bytes(b"someone else's whispertyper")
    with pytest.raises(OSError, match="another WhisperTyper installation"):
        shim.install(IMAGES)
    shim.uninstall()
    assert foreign.read_bytes() == b"someone else's whispertyper"


def test_uninstall_removes_only_own_launchers(bin_folder):
    shim.install(IMAGES)
    shim.uninstall()
    assert not any(path.exists() for path in shim.launcher_paths())
    assert not shim.is_installed()


def test_a_run_with_a_throwaway_home_folder_leaves_path_and_start_menu_alone(monkeypatch):
    """Inherited by every process of such a run, so it also holds after the app restarts itself."""
    from unittest.mock import Mock

    system_integration = pytest.importorskip("app.controllers.system_integration")  # needs Qt
    monkeypatch.setenv(system_integration.SKIP_ENV, "1")
    render = Mock()
    monkeypatch.setattr(system_integration, "render_icon", render)
    system_integration.SystemIntegration(Mock(config={})).apply()
    render.assert_not_called()
