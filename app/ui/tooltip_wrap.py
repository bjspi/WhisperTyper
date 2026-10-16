"""App-wide tooltip behaviour: readable wrapping, a shorter delay and no roll-in animation.

Qt shows a plain-text tooltip on one line however long it is, and wraps rich text to a width
of its own choosing (very narrow for medium texts). An application event filter therefore
breaks every long tooltip itself, measured in pixels, so translations need no line breaks.
Tab tooltips are set per tab, so the tab under the pointer supplies the text.

Qt waits ``SH_ToolTip_WakeUpDelay`` (700 ms) before the first tooltip, after which further tooltips
appear at once until none was shown for a while; the wait is shortened here. Windows also rolls
tooltips in (``UI_AnimateTooltip``), which looked like the tooltip being built up; it is turned off.
"""
from __future__ import annotations

import html
from typing import List, Optional

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QFont, QFontMetrics, QHelpEvent
from PyQt6.QtWidgets import QApplication, QProxyStyle, QStyle, QStyleHintReturn, QStyleOption, QTabBar, QToolTip, QWidget

#: Longest tooltip line in pixels; shorter tooltips stay on one line.
MAX_LINE_PX = 460
#: Tooltip text size, as set for QToolTip in the theme stylesheet.
TOOLTIP_FONT_PX = 13
#: Hover time before the first tooltip appears (Qt's default is 700 ms).
WAKE_UP_DELAY_MS = 500


def wrap_lines(text: str, metrics: QFontMetrics, max_px: int = MAX_LINE_PX) -> List[str]:
    """Greedy word wrap of ``text`` to ``max_px``; explicit newlines stay line breaks."""
    lines: List[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            candidate = f"{line} {word}" if line else word
            if line and metrics.horizontalAdvance(candidate) > max_px:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def wrapped(text: str, metrics: QFontMetrics, max_px: int = MAX_LINE_PX) -> str:
    """``text`` as rich text broken into lines of at most ``max_px``; Qt does not re-wrap it."""
    body = "<br>".join(html.escape(line) for line in wrap_lines(text, metrics, max_px))
    return f'<p style="white-space:nowrap">{body}</p>'


def _tooltip_text(widget: QWidget, event: QHelpEvent) -> str:
    """The tooltip Qt would show: the hovered tab's for a tab bar, else the widget's."""
    if isinstance(widget, QTabBar):
        index = widget.tabAt(event.pos())
        return widget.tabToolTip(index) if index >= 0 else ""
    return widget.toolTip()


class _TooltipWrapper(QObject):
    """Application event filter that shows long plain-text tooltips wrapped."""

    def eventFilter(self, watched: Optional[QObject], event: Optional[QEvent]) -> bool:  # noqa: N802 - Qt API
        """Show a long plain-text tooltip wrapped; everything else passes through."""
        if event is None or event.type() != QEvent.Type.ToolTip or not isinstance(watched, QWidget):
            return False
        assert isinstance(event, QHelpEvent)
        text = _tooltip_text(watched, event)
        if not text or text.lstrip().startswith("<"):
            return False
        font = QFont(QApplication.font())
        font.setPixelSize(TOOLTIP_FONT_PX)
        metrics = QFontMetrics(font)
        if metrics.horizontalAdvance(text) <= MAX_LINE_PX and "\n" not in text:
            return False
        QToolTip.showText(event.globalPos(), wrapped(text, metrics), watched)
        return True


class _TooltipTimingStyle(QProxyStyle):
    """The platform style with a shorter tooltip wake-up delay."""

    def styleHint(self, hint: QStyle.StyleHint, option: Optional[QStyleOption] = None,  # noqa: N802 - Qt API
                  widget: Optional[QWidget] = None, returnData: Optional[QStyleHintReturn] = None) -> int:  # noqa: N803
        """Shorten ``SH_ToolTip_WakeUpDelay``; every other hint comes from the platform style."""
        if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return WAKE_UP_DELAY_MS
        return super().styleHint(hint, option, widget, returnData)


_wrapper: Optional[_TooltipWrapper] = None


def install(app: QApplication) -> None:
    """Install the tooltip behaviour on ``app`` (idempotent); call before any window is created."""
    global _wrapper
    if _wrapper is None:
        _wrapper = _TooltipWrapper(app)
        app.installEventFilter(_wrapper)
        base = app.style()
        if base is not None:
            app.setStyle(_TooltipTimingStyle(base.name()))
        app.setEffectEnabled(Qt.UIEffect.UI_AnimateTooltip, False)
        app.setEffectEnabled(Qt.UIEffect.UI_FadeTooltip, False)
