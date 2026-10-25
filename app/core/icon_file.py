"""Windows icon formats from PNG images — pure byte layout, no Qt, no I/O.

An ``.ico`` file and the icon resources of an executable hold the same images with two slightly
different directories: the file points at each image by offset, the resource group by resource
ID. PNG-compressed images are valid in both since Windows Vista, so no bitmap conversion is needed.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Sequence

#: ``ICONDIR``: reserved, type (1 = icon), image count.
_DIRECTORY = struct.Struct("<HHH")
#: ``ICONDIRENTRY`` of a file: width, height, colours, reserved, planes, bit depth, size, file offset.
_FILE_ENTRY = struct.Struct("<BBBBHHII")
#: ``GRPICONDIRENTRY`` of a resource group: as above, but a resource ID instead of the offset.
_GROUP_ENTRY = struct.Struct("<BBBBHHIH")
_ICON_TYPE = 1
_PLANES = 1
_BIT_DEPTH = 32
#: The directories store sizes in one byte; 0 stands for 256 pixels.
_LARGEST_SIZE = 256


@dataclass(frozen=True)
class IconImage:
    """One square PNG image of an icon, ``size`` pixels wide."""

    size: int
    png: bytes


def ico_file(images: Sequence[IconImage]) -> bytes:
    """The ``.ico`` file holding ``images``."""
    offset = _DIRECTORY.size + _FILE_ENTRY.size * len(images)
    entries: List[bytes] = []
    for image in images:
        entries.append(_FILE_ENTRY.pack(*_dimensions(image), _PLANES, _BIT_DEPTH, len(image.png), offset))
        offset += len(image.png)
    return _DIRECTORY.pack(0, _ICON_TYPE, len(images)) + b"".join(entries) + b"".join(image.png for image in images)


def icon_group(images: Sequence[IconImage], first_id: int) -> bytes:
    """The ``RT_GROUP_ICON`` resource for ``images`` stored as ``RT_ICON`` resources ``first_id``, ``first_id + 1``, …"""
    entries = [_GROUP_ENTRY.pack(*_dimensions(image), _PLANES, _BIT_DEPTH, len(image.png), first_id + index)
               for index, image in enumerate(images)]
    return _DIRECTORY.pack(0, _ICON_TYPE, len(images)) + b"".join(entries)


def _dimensions(image: IconImage) -> tuple[int, int, int, int]:
    """Width, height, colour count and reserved byte of a directory entry."""
    if not 0 < image.size <= _LARGEST_SIZE:
        raise ValueError(f"icon images are 1 to {_LARGEST_SIZE} pixels, not {image.size}")
    side = image.size % _LARGEST_SIZE
    return side, side, 0, 0
