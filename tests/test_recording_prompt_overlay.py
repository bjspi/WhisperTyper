"""Recording prompt palette: wrapping chips, elapsed timer and selection reporting (Qt, not shown)."""
from __future__ import annotations

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
QtCore = pytest.importorskip("PyQt6.QtCore")
QtGui = pytest.importorskip("PyQt6.QtGui")
QtTest = pytest.importorskip("PyQt6.QtTest")

from app.ui import floating_buttons  # noqa: E402
from app.ui.floating_buttons import RecordingPromptOverlay  # noqa: E402
from app.ui.flow_layout import FlowLayout  # noqa: E402

PROMPTS = [
    {"caption": "✨ Instruction", "text": "RUN", "kind": "instruction"},
    {"caption": "Polish", "text": "POLISH", "auto_apply": True},
]


@pytest.fixture
def qapp(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(RecordingPromptOverlay, "show", lambda self: None)
    yield app
    RecordingPromptOverlay.close_current()


def overlay(selected, prompts=PROMPTS):
    return RecordingPromptOverlay(prompts, "Recording…", "None", selected.append, use_system_position=False)


def test_flow_layout_wraps_onto_more_lines_when_narrow(qapp):
    host = QtWidgets.QWidget()
    flow = FlowLayout(host, spacing=6)
    for caption in ("First chip", "Second chip", "Third chip"):
        flow.addWidget(QtWidgets.QPushButton(caption))
    one_line = flow.heightForWidth(flow.sizeHint().width())
    assert flow.heightForWidth(flow.sizeHint().width() // 2) > one_line
    assert flow.heightForWidth(10_000) == one_line


def test_auto_apply_prompt_is_preselected_without_reporting_a_choice(qapp):
    selected: list = []
    palette = overlay(selected)
    checked = [button.text() for button in palette._prompt_buttons if button.isChecked()]
    assert checked == ["Polish"] and selected == []


def test_clicks_report_the_entry_and_none(qapp):
    selected: list = []
    palette = overlay(selected)
    none_chip, instruction_chip, _polish = palette._prompt_buttons
    instruction_chip.click()
    none_chip.click()
    assert selected == [PROMPTS[0], None]
    assert none_chip.isChecked() and not instruction_chip.isChecked()


def test_timer_counts_up_from_opening(qapp, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(floating_buttons.time, "monotonic", lambda: now[0])
    palette = overlay([])
    assert palette.timer_label.text() == "0:00"
    now[0] += 83
    palette._update_elapsed()
    assert palette.timer_label.text() == "1:23"


def test_long_caption_is_elided_and_keeps_the_full_text_as_tooltip(qapp):
    caption = "Rewrite this as a very long and detailed instruction " * 4
    palette = overlay([], [{"caption": caption, "text": "x"}])
    chip = palette._prompt_buttons[1]
    assert chip.text().endswith("…") and chip.toolTip() == caption.strip()
    assert palette.card.width() <= RecordingPromptOverlay.MAX_CARD_WIDTH


def test_restart_counts_the_elapsed_time_from_zero_again(qapp, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(floating_buttons.time, "monotonic", lambda: now[0])
    palette = overlay([])
    now[0] += 42
    palette._update_elapsed()
    palette.restart_timer()
    assert palette.timer_label.text() == "0:00"
    now[0] += 5
    palette._update_elapsed()
    assert palette.timer_label.text() == "0:05"


def test_header_buttons_report_cancel_and_restart_without_taking_focus(qapp):
    calls: list[str] = []
    palette = RecordingPromptOverlay(PROMPTS, "Recording…", "None", lambda _prompt: None,
                                     use_system_position=False, on_cancel=lambda: calls.append("cancel"),
                                     on_restart=lambda: calls.append("restart"),
                                     cancel_tooltip="Cancel recording", restart_tooltip="Restart recording")
    assert palette.restart_button is not None and palette.cancel_button is not None
    palette.restart_button.click()
    palette.cancel_button.click()
    assert calls == ["restart", "cancel"]
    for button, tooltip in ((palette.restart_button, "Restart recording"), (palette.cancel_button, "Cancel recording")):
        assert button.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
        assert button.toolTip() == tooltip and not button.icon().isNull()


def test_header_buttons_are_left_out_without_callbacks(qapp):
    palette = overlay([])
    assert palette.restart_button is None and palette.cancel_button is None


def drag(widget, offset):
    """Press on ``widget``, move the mouse by ``offset`` and release."""
    press = widget.mapToGlobal(QtCore.QPoint(5, 5))
    QtTest.QTest.mousePress(widget, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(5, 5))
    move = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(5, 5),
                             QtCore.QPointF(press + offset), QtCore.Qt.MouseButton.NoButton,
                             QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, move)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(5, 5))


def test_palette_near_the_cursor_moves_by_dragging_its_header_or_handle(qapp):
    palette = RecordingPromptOverlay(PROMPTS, "Recording…", "None", lambda _prompt: None,
                                     use_system_position=False, move_tooltip="Drag to move")
    handle = palette.move_handle
    assert handle is not None and handle.toolTip() == "Drag to move"
    assert palette.card.rect().contains(handle.geometry())
    size_all, pointing = QtCore.Qt.CursorShape.SizeAllCursor, QtCore.Qt.CursorShape.PointingHandCursor
    assert palette.status_label.cursor().shape() == size_all == palette.timer_label.cursor().shape()
    assert palette._prompt_buttons[0].cursor().shape() == pointing
    start = palette.pos()
    drag(handle, QtCore.QPoint(40, 25))
    assert palette.pos() == start + QtCore.QPoint(40, 25)
    drag(palette.status_label, QtCore.QPoint(-10, 5))
    drag(palette.timer_label, QtCore.QPoint(-10, 5))
    assert palette.pos() == start + QtCore.QPoint(20, 35)
    drag(palette._prompt_buttons[0], QtCore.QPoint(30, 30))  # Chips keep their clicks.
    assert palette.pos() == start + QtCore.QPoint(20, 35)


def test_palette_by_the_status_area_has_no_move_handle(qapp, monkeypatch):
    monkeypatch.setattr(floating_buttons, "is_WINDOWS", True)
    palette = RecordingPromptOverlay(PROMPTS, "Recording…", "None", lambda _prompt: None,
                                     use_system_position=True, move_tooltip="Drag to move")
    assert palette.move_handle is None
    assert palette.status_label.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor


def test_pause_freezes_the_timer_and_offers_resume(qapp, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(floating_buttons.time, "monotonic", lambda: now[0])
    calls: list[str] = []
    palette = RecordingPromptOverlay(PROMPTS, "Recording…", "None", lambda _prompt: None,
                                     use_system_position=False, on_pause=lambda: calls.append("pause"),
                                     paused_text="Paused", pause_tooltip="Pause", resume_tooltip="Resume")
    assert palette.pause_button is not None and palette.pause_button.toolTip() == "Pause"
    palette.pause_button.click()
    assert calls == ["pause"]
    width = palette.card.width()
    now[0] += 10
    palette.set_paused(True)
    now[0] += 50
    palette._update_elapsed()
    assert palette.timer_label.text() == "0:10"
    assert palette.status_label.text() == "Paused" and palette.pause_button.toolTip() == "Resume"
    palette.set_paused(False)
    now[0] += 5
    palette._update_elapsed()
    assert palette.timer_label.text() == "0:15"
    assert palette.status_label.text() == "Recording" and palette.pause_button.toolTip() == "Pause"
    assert palette.card.width() == width

def test_digit_keys_are_taken_only_while_the_mouse_is_over_the_palette(qapp, monkeypatch):
    registered: dict[int, int] = {}
    monkeypatch.setattr(floating_buttons, "is_WINDOWS", True)
    monkeypatch.setattr(floating_buttons, "register_window_hotkey",
                        lambda _hwnd, hotkey_id, vk: registered.setdefault(hotkey_id, vk) is not None)
    monkeypatch.setattr(floating_buttons, "unregister_window_hotkey",
                        lambda _hwnd, hotkey_id: registered.pop(hotkey_id))
    selected: list = []
    palette = RecordingPromptOverlay(PROMPTS, "Recording…", "None", selected.append,
                                     use_system_position=False, digit_keys=True)
    assert registered == {}
    def enter():
        return QtGui.QEnterEvent(QtCore.QPointF(), QtCore.QPointF(), QtCore.QPointF())

    # Whether the test platform maps the window is not the point: visibility is set explicitly.
    monkeypatch.setattr(palette, "isVisible", lambda: False)
    QtWidgets.QApplication.sendEvent(palette, enter())
    assert registered == {}  # A late Enter of a closed palette takes no keys nothing would release.
    monkeypatch.setattr(palette, "isVisible", lambda: True)
    QtWidgets.QApplication.sendEvent(palette, enter())
    # "None" and two prompts: 1-3 on the number row and the number pad.
    assert sorted(registered.values()) == [0x31, 0x32, 0x33, 0x61, 0x62, 0x63]
    palette.select_by_digit(3)
    palette.select_by_digit(1)
    palette.select_by_digit(7)  # No chip: ignored.
    palette.select_by_digit(0)  # The tenth chip: there is none.
    assert selected == [PROMPTS[1], None]
    QtWidgets.QApplication.sendEvent(palette, QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert registered == {}


def test_digit_keys_stay_free_when_disabled(qapp, monkeypatch):
    monkeypatch.setattr(floating_buttons, "is_WINDOWS", True)
    monkeypatch.setattr(floating_buttons, "register_window_hotkey", lambda *_args: pytest.fail("registered"))
    palette = RecordingPromptOverlay(PROMPTS, "Recording…", "None", lambda _prompt: None,
                                     use_system_position=False, digit_keys=False)
    QtWidgets.QApplication.sendEvent(palette, QtGui.QEnterEvent(QtCore.QPointF(), QtCore.QPointF(), QtCore.QPointF()))


@pytest.mark.skipif(not floating_buttons.is_WINDOWS, reason="native Windows message")
def test_hotkey_id_is_read_from_a_native_wm_hotkey_message():
    import ctypes
    from ctypes import wintypes

    from app.platform.windows import hotkey_id_of
    msg = wintypes.MSG(message=0x0312, wParam=0xB031)
    address = ctypes.addressof(msg)
    assert hotkey_id_of(b"windows_generic_MSG", address) == 0xB031
    msg.message = 0x0200  # WM_MOUSEMOVE
    assert hotkey_id_of(b"windows_generic_MSG", address) is None



def test_texts_bundle_stands_in_for_the_single_string_parameters(qapp):
    from app.ui.floating_buttons import RecordingPaletteTexts
    texts = RecordingPaletteTexts(status_text="Recording…", none_text="None", paused_text="Paused",
                                  pause_tooltip="Pause", resume_tooltip="Resume", move_tooltip="Drag to move")
    palette = RecordingPromptOverlay(PROMPTS, on_selection_changed=lambda _prompt: None, use_system_position=False,
                                     on_pause=lambda: None, texts=texts)
    assert palette.status_label.text() == "Recording" and palette._prompt_buttons[0].text() == "None"
    assert palette.pause_button is not None and palette.pause_button.toolTip() == "Pause"
    assert palette.move_handle is not None and palette.move_handle.toolTip() == "Drag to move"
    palette.set_paused(True)
    assert palette.status_label.text() == "Paused" and palette.pause_button.toolTip() == "Resume"
    assert RecordingPaletteTexts().status_text == "" and RecordingPaletteTexts().device_tooltip == ""
