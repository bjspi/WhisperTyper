"""Reading the selection always hands the user's clipboard back, whatever ``restore_clipboard`` says."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

import app.controllers.text_output as text_output_module  # noqa: E402
from app.controllers.text_output import TextOutput  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("restore_clipboard", [True, False])
def test_get_selected_text_restores_the_clipboard(qapp, monkeypatch, restore_clipboard):
    clipboard = ["user's own clipboard"]
    snapshot = Mock()
    snapshot.restore.side_effect = lambda: clipboard.__setitem__(0, "user's own clipboard")
    monkeypatch.setattr(text_output_module.ClipboardSnapshot, "capture", staticmethod(lambda: snapshot))
    monkeypatch.setattr(text_output_module.copykitten, "copy", lambda text: clipboard.__setitem__(0, text))
    monkeypatch.setattr(text_output_module.copykitten, "paste", lambda: clipboard[0])
    monkeypatch.setattr(text_output_module.time, "sleep", lambda _seconds: None)
    output = TextOutput({"restore_clipboard": restore_clipboard}, lambda key, **_kw: key, Mock(), lambda _p: None)
    monkeypatch.setattr(output, "_simulate_key_combination",
                        lambda char: clipboard.__setitem__(0, "selected words") or True)

    assert output.get_selected_text() == "selected words"
    snapshot.restore.assert_called_once()
    assert clipboard[0] == "user's own clipboard"
