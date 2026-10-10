"""Recording prompt palette: wrapping chips, elapsed timer and selection reporting (Qt, not shown)."""
from __future__ import annotations

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

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
