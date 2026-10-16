"""Tests for app/ui/tooltip_wrap.py."""
from __future__ import annotations

from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import QApplication

from app.ui.tooltip_wrap import wrap_lines, wrapped

_app = QApplication.instance() or QApplication([])
METRICS = QFontMetrics(QFont("Arial", 10))


def test_lines_stay_within_width_and_keep_every_word():
    text = "Wenn aktiviert, erscheint die Prompt-Auswahl während einer Aufnahme unter Windows beim Infobereich. " * 3
    lines = wrap_lines(text.strip(), METRICS, 300)
    assert len(lines) > 1
    assert all(METRICS.horizontalAdvance(line) <= 300 for line in lines)
    assert " ".join(lines).split() == text.split()


def test_explicit_newlines_and_html_are_kept_safe():
    assert wrap_lines("a\nb", METRICS, 300) == ["a", "b"]
    assert wrapped("a < b & c", METRICS) == '<p style="white-space:nowrap">a &lt; b &amp; c</p>'
