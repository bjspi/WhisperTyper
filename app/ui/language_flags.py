"""Small flags for spoken languages (``resources/flags``, from the MIT-licensed flag-icons set)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QIcon, QImage, QPainter, QPainterPath, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from app.core.paths import resource_path

#: Country (or region) flag per language code. Languages without one clear flag, or sharing
#: one with a language listed here (Arabic, Swahili, Tamil, ...), keep their short code instead.
_FLAG_FILES = {
    "en": "us", "de": "de", "fr": "fr", "es": "es", "it": "it", "nl": "nl", "af": "za", "hy": "am",
    "az": "az", "be": "by", "bs": "ba", "bg": "bg", "ca": "es-ct", "zh": "cn", "hr": "hr", "cs": "cz",
    "da": "dk", "et": "ee", "fi": "fi", "gl": "es-ga", "el": "gr", "he": "il", "hi": "in", "hu": "hu",
    "is": "is", "id": "id", "ja": "jp", "kk": "kz", "ko": "kr", "lv": "lv", "lt": "lt", "mk": "mk",
    "ms": "my", "mi": "nz", "ne": "np", "no": "no", "fa": "ir", "pl": "pl", "pt": "pt", "ro": "ro",
    "ru": "ru", "sr": "rs", "sk": "sk", "sl": "si", "sv": "se", "tl": "ph", "th": "th", "tr": "tr",
    "uk": "ua", "ur": "pk", "vi": "vn", "cy": "gb-wls",
}
#: Flag height in logical pixels; the files are drawn 4:3.
FLAG_HEIGHT = 12
FLAG_WIDTH = 16
#: Pixel density the flags are drawn at, sharp on high-DPI screens.
_SCALE = 3
_CORNER_RADIUS = 2.0
#: Opacity of the grey flag of a language that is not selected.
_MUTED_OPACITY = 0.6


def has_flag(code: str) -> bool:
    """Whether the language is shown by a flag rather than its short code."""
    return code in _FLAG_FILES


@lru_cache(maxsize=None)
def flag_pixmap(code: str, muted: bool = False) -> Optional[QPixmap]:
    """The language's flag with rounded corners; ``muted`` greys it out. None without a flag."""
    country = _FLAG_FILES.get(code)
    if country is None:
        return None
    renderer = QSvgRenderer(resource_path("resources", "flags", f"{country}.svg"))
    if not renderer.isValid():
        return None
    image = QImage(FLAG_WIDTH * _SCALE, FLAG_HEIGHT * _SCALE, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter, QRectF(image.rect()))
    painter.end()
    if muted:
        image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    pixmap = QPixmap(FLAG_WIDTH * _SCALE, FLAG_HEIGHT * _SCALE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(pixmap.rect()), _CORNER_RADIUS * _SCALE, _CORNER_RADIUS * _SCALE)
    painter.setClipPath(clip)
    if muted:
        painter.setOpacity(_MUTED_OPACITY)
    painter.drawImage(0, 0, image)
    painter.end()
    pixmap.setDevicePixelRatio(_SCALE)
    return pixmap


def flag_icon(code: str) -> QIcon:
    """A pill icon: the grey flag while unchecked, the coloured flag while checked. Empty without a flag."""
    icon = QIcon()
    colored, muted = flag_pixmap(code), flag_pixmap(code, muted=True)
    if colored is None or muted is None:
        return icon
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
        icon.addPixmap(muted, mode, QIcon.State.Off)
        icon.addPixmap(colored, mode, QIcon.State.On)
    return icon
