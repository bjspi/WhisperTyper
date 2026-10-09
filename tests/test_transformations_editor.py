"""Transformation template editing: stable edit targets across reordering, limits, removal."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QCheckBox, QLineEdit, QPushButton, QSplitter, QTextEdit, QWidget  # noqa: E402

from app.ui.transformations_tab import TransformationsEditor  # noqa: E402


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
    widgets = dict(caption_edit=QLineEdit(), text_edit=QTextEdit(), show_during_recording=QCheckBox(),
                   add_button=QPushButton(), remove_button=QPushButton())
    entries = [{"caption": "First", "text": "one", "show_during_recording": True},
               {"caption": "Second", "text": "two"}, "not a template"]
    instance = TransformationsEditor(entries, _Translator(), splitter=splitter, list_placeholder=placeholder, **widgets)
    yield instance, widgets
    splitter.deleteLater()
    app.processEvents()


def test_loads_valid_templates_and_selects_the_first(editor):
    instance, widgets = editor
    assert instance.entries() == [{"caption": "First", "text": "one", "show_during_recording": True},
                                  {"caption": "Second", "text": "two", "show_during_recording": False}]
    assert widgets["caption_edit"].text() == "First"


def test_edits_follow_the_template_after_reordering(editor):
    instance, widgets = editor
    widgets["caption_edit"].setText("First edited")
    moved = instance.list.takeItem(0)
    instance.list.insertItem(1, moved)
    instance.list.setCurrentItem(moved)
    instance.list.reordered.emit()
    widgets["text_edit"].setPlainText("one edited")
    assert [entry["caption"] for entry in instance.entries()] == ["Second", "First edited"]
    assert instance.entries()[1]["text"] == "one edited"


def test_add_respects_the_limit_and_remove_selects_a_neighbour(editor):
    instance, widgets = editor
    while widgets["add_button"].isEnabled():
        widgets["add_button"].click()
    assert len(instance.entries()) == instance.max_entries
    assert instance.list.item(instance.list.count() - 1).text() == "<caption_placeholder>"
    instance.list.setCurrentRow(0)
    widgets["remove_button"].click()
    assert len(instance.entries()) == instance.max_entries - 1
    assert widgets["caption_edit"].text() == "Second"
    assert widgets["add_button"].isEnabled()
