"""Small reusable behaviours for settings widgets."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QPushButton, QScrollArea, QSlider

#: Pixels scrolled per wheel notch, at least; trackpads report exact pixel deltas instead.
_MIN_WHEEL_STEP_PX = 18
_WHEEL_LINES_PER_NOTCH = 3


class SliderWheelToScrollArea(QObject):
    """Scroll the page instead of the slider on wheel gestures (macOS trackpads move sliders by accident)."""

    def __init__(self, slider: QSlider, scroll_area: QScrollArea) -> None:
        """Install the filter on ``slider``; wheel events then scroll ``scroll_area``."""
        super().__init__(slider)
        self._scroll_area = scroll_area
        slider.installEventFilter(self)

    def eventFilter(self, watched: Optional[QObject], event: Optional[QEvent]) -> bool:  # noqa: N802 - Qt API
        """Consume wheel events and forward their delta to the scroll area's vertical bar."""
        if not isinstance(event, QWheelEvent):
            return False
        scrollbar = self._scroll_area.verticalScrollBar()
        if scrollbar is None:
            return False
        pixel_delta = event.pixelDelta().y()
        angle_delta = event.angleDelta().y()
        if pixel_delta:
            delta = -pixel_delta
        elif angle_delta:
            notches = angle_delta / 120.0
            delta = int(-notches * max(scrollbar.singleStep(), _MIN_WHEEL_STEP_PX) * _WHEEL_LINES_PER_NOTCH)
        else:
            return True
        scrollbar.setValue(scrollbar.value() + delta)
        return True


def fit_button_to_captions(button: QPushButton, *captions: str) -> None:
    """Fix ``button`` to its widest caption, so switching texts never resizes its neighbours."""
    metrics = button.fontMetrics()
    padding = button.sizeHint().width() - metrics.horizontalAdvance(button.text())
    button.setFixedWidth(max(metrics.horizontalAdvance(caption) for caption in (button.text(), *captions)) + padding)
