"""Tests for app/ui/tooltip_wrap.py."""
from __future__ import annotations

import pytest

QtGui = pytest.importorskip("PyQt6.QtGui")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from app.ui.tooltip_wrap import wrap_lines, wrapped  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
METRICS = QtGui.QFontMetrics(QtGui.QFont("Arial", 10))


def test_lines_stay_within_width_and_keep_every_word():
    text = "Wenn aktiviert, erscheint die Prompt-Auswahl während einer Aufnahme unter Windows beim Infobereich. " * 3
    lines = wrap_lines(text.strip(), METRICS, 300)
    assert len(lines) > 1
    assert all(METRICS.horizontalAdvance(line) <= 300 for line in lines)
    assert " ".join(lines).split() == text.split()


def test_explicit_newlines_and_html_are_kept_safe():
    assert wrap_lines("a\nb", METRICS, 300) == ["a", "b"]
    assert wrapped("a < b & c", METRICS) == '<p style="white-space:nowrap">a &lt; b &amp; c</p>'
