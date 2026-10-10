"""Small reusable behaviours for settings widgets."""
from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QLabel, QPushButton, QScrollArea, QSlider

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


def show_temperature_support(slider: QSlider, value_label: QLabel, supported: bool, translator: Any) -> None:
    """Disable a temperature slider for models that take no temperature; the stored value is kept.

    While disabled, both the slider and its value say why, so the greyed-out control is not mistaken
    for a broken one.
    """
    slider.setEnabled(supported)
    value_label.setEnabled(supported)
    if supported:
        slider.setToolTip(translator.tr("temperature_tooltip"))
        value_label.setToolTip("")
    else:
        reason = translator.tr("temperature_unsupported_tooltip")
        slider.setToolTip(reason)
        value_label.setToolTip(reason)
