"""Prompt editing: stable edit targets across reordering, limits, removal, automatic prompt, instruction entry."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QPalette  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QWidget,
)

from app.ui.transformations_tab import InstructionControls, TransformationsEditor  # noqa: E402

INSTRUCTION = {"kind": "instruction", "caption": "Go", "text": "Carry it out", "enabled": True,
               "trigger_words": "prompt", "scan_depth": 5}


class _Translator:
    def tr(self, key, **_kwargs):
        return f"<{key}>"


@pytest.fixture
def editor():
    app = QApplication.instance() or QApplication([])
    splitter = QSplitter()
    placeholder = QWidget()
    splitter.addWidget(placeholder)
    splitter.addWidget(QWidget())
    scan_depth = QSpinBox()
    scan_depth.setRange(1, 99)
    controls = InstructionControls(active=QCheckBox(), options=QWidget(), trigger_enabled=QCheckBox(),
                                   trigger_words=QLineEdit(), scan_depth=scan_depth, strip_trigger=QCheckBox(),
                                   selection_context=QCheckBox())
    widgets = dict(caption_edit=QLineEdit(), text_label=QLabel(), text_edit=QTextEdit(), show_during_recording=QCheckBox(),
                   auto_apply=QCheckBox(), add_button=QPushButton(), remove_button=QPushButton())
    entries = [INSTRUCTION, {"caption": "First", "text": "one", "show_during_recording": True},
               {"caption": "Second", "text": "two"}, "not a template"]
    instance = TransformationsEditor(entries, _Translator(), splitter=splitter, list_placeholder=placeholder,
                                     instruction=controls, **widgets)
    yield instance, {**widgets, "instruction": controls}
    splitter.deleteLater()
    app.processEvents()


def test_loads_valid_templates_and_selects_the_first(editor):
    instance, widgets = editor
    assert [(entry["kind"], entry["caption"]) for entry in instance.entries()] == [
        ("instruction", "Go"), ("prompt", "First"), ("prompt", "Second")]
    assert instance.entries()[1] == {"kind": "prompt", "caption": "First", "text": "one",
                                     "show_during_recording": True, "auto_apply": False}
    assert widgets["caption_edit"].text() == "Go"
    # The instruction shows its switch and LivePrompting block instead of "apply automatically".
    assert not widgets["instruction"].options.isHidden() and not widgets["instruction"].active.isHidden()
    assert widgets["auto_apply"].isHidden()
    assert widgets["text_label"].text() == "<liveprompt_system_prompt_label>"
    instance.list.setCurrentRow(1)
    assert widgets["instruction"].options.isHidden() and not widgets["auto_apply"].isHidden()


def test_edits_follow_the_template_after_reordering(editor):
    instance, widgets = editor
    instance.list.setCurrentRow(1)
    widgets["caption_edit"].setText("First edited")
    moved = instance.list.takeItem(1)
    instance.list.insertItem(2, moved)
    instance.list.setCurrentItem(moved)
    instance.list.reordered.emit()
    widgets["text_edit"].setPlainText("one edited")
    assert [entry["caption"] for entry in instance.entries()] == ["Go", "Second", "First edited"]
    assert instance.entries()[2]["text"] == "one edited"


def test_add_respects_the_limit_and_remove_selects_a_neighbour(editor):
    instance, widgets = editor
    while widgets["add_button"].isEnabled():
        widgets["add_button"].click()
    own = [entry for entry in instance.entries() if entry["kind"] == "prompt"]
    assert len(own) == instance.max_entries  # the instruction entry does not count
    assert instance.list.item(instance.list.count() - 1).text() == "<caption_placeholder>"
    instance.list.setCurrentRow(1)
    widgets["remove_button"].click()
    assert len(instance.entries()) == instance.max_entries
    assert widgets["caption_edit"].text() == "Second"
    assert widgets["add_button"].isEnabled()


def test_automatic_prompt_is_exclusive_and_always_shown(editor):
    instance, widgets = editor
    instance.list.setCurrentRow(1)  # First
    widgets["auto_apply"].setChecked(True)
    assert widgets["show_during_recording"].isChecked() and not widgets["show_during_recording"].isEnabled()
    instance.list.setCurrentRow(2)  # Second
    assert widgets["show_during_recording"].isEnabled() and not widgets["auto_apply"].isChecked()
    widgets["auto_apply"].setChecked(True)
    assert widgets["show_during_recording"].isChecked()
    assert [(entry["auto_apply"], entry["show_during_recording"]) for entry in instance.entries()[1:]] == [
        (False, True), (True, True)]
    widgets["auto_apply"].setChecked(False)
    assert widgets["show_during_recording"].isEnabled()
    assert not any(entry["auto_apply"] for entry in instance.entries())


def test_instruction_cannot_be_removed_but_can_be_moved(editor):
    instance, widgets = editor
    instance.list.setCurrentRow(0)
    assert not widgets["remove_button"].isEnabled()
    widgets["remove_button"].click()
    assert instance.entries()[0]["kind"] == "instruction"
    moved = instance.list.takeItem(0)
    instance.list.insertItem(2, moved)
    instance.list.setCurrentItem(moved)
    instance.list.reordered.emit()
    assert [entry["kind"] for entry in instance.entries()] == ["prompt", "prompt", "instruction"]
    assert not widgets["remove_button"].isEnabled()


def test_instruction_settings_are_stored_and_inactive_entry_is_greyed(editor):
    instance, widgets = editor
    controls = widgets["instruction"]
    controls.trigger_words.setText("befehl, ki")
    controls.scan_depth.setValue(8)
    controls.strip_trigger.setChecked(True)
    controls.selection_context.setChecked(True)
    controls.trigger_enabled.setChecked(False)
    assert not controls.trigger_words.isEnabled()  # LivePrompting off greys its trigger settings
    assert controls.selection_context.isEnabled()
    controls.active.setChecked(False)
    assert not widgets["text_edit"].isEnabled()  # a switched-off instruction greys the whole form
    instruction = instance.entries()[0]
    assert (instruction["enabled"], instruction["trigger_enabled"], instruction["trigger_words"],
            instruction["scan_depth"], instruction["strip_trigger"], instruction["use_selection_context"]) == (
        False, False, "befehl, ki", 8, True, True)
    disabled = instance.list.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
    assert instance.list.item(0).foreground().color() == disabled
