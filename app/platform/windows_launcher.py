"""Windows ``.exe`` launchers for a Python script, with the app icon — the format pip uses.

A launcher is a small stub (the ``t``/``w`` executables pip ships in its vendored ``distlib``),
followed by a ``#!`` line naming the interpreter and a zip archive with ``__main__.py``. The stub
runs the interpreter on its own file, which Python executes as a zip application. The icon is
written into the stub's resources *before* the shebang and archive are appended, because
updating the resources rewrites the executable and would drop data after it.
"""
from __future__ import annotations

import ctypes
import importlib.util
import io
import os
import platform
import struct
import tempfile
import zipfile
from ctypes import wintypes
from pathlib import Path
from typing import Any, Sequence

from app.core.icon_file import IconImage, icon_group

#: Resource types and the IDs the icon is stored under (the lowest group ID is the file's icon).
_RT_ICON = 3
_RT_GROUP_ICON = 14
_GROUP_ID = 1
_FIRST_IMAGE_ID = 1
_LANG_NEUTRAL = 0
#: Where pip keeps the launcher stubs, and their names: console ("t") or windowless ("w").
_DISTLIB_PACKAGES = ("pip._vendor.distlib", "distlib")
_CONSOLE_KIND = "t"
#: A fixed time stamp keeps the launcher byte-identical, so an unchanged one is never rewritten.
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_WINDOWLESS_KIND = "w"


class LauncherError(OSError):
    """No launcher stub is available or its icon could not be written."""


def build_launcher(interpreter: Path, script: str, icon: Sequence[IconImage], *, windowless: bool) -> bytes:
    """The launcher that runs ``script`` (the ``__main__.py`` source) with ``interpreter``.

    Raises:
        LauncherError: pip's launcher stubs are missing, or the icon could not be added.
    """
    stub = _stub(_WINDOWLESS_KIND if windowless else _CONSOLE_KIND)
    if icon:
        stub = _with_icon(stub, icon)
    return stub + _shebang(interpreter) + _zipped_main(script)


def _stub(kind: str) -> bytes:
    """Pip's launcher stub of ``kind`` for this machine's architecture."""
    name = f"{kind}{_architecture_suffix()}.exe"
    for package in _DISTLIB_PACKAGES:
        folder = _package_folder(package)
        if folder is not None and (folder / name).is_file():
            return (folder / name).read_bytes()
    raise LauncherError("pip is not installed in this Python environment; it provides the launcher for the command")


def _architecture_suffix() -> str:
    """``64-arm``, ``64`` or ``32``, matching the stub names in ``distlib``."""
    if platform.machine().lower() in ("arm64", "aarch64"):
        return "64-arm"
    return "64" if struct.calcsize("P") == 8 else "32"


def _package_folder(package: str) -> "Path | None":
    """The folder of ``package`` without importing it (pip is never imported by the app)."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _shebang(interpreter: Path) -> bytes:
    """The ``#!`` line the stub reads; a path with spaces is quoted, as pip does."""
    path = str(interpreter)
    if " " in path:
        path = f'"{path}"'
    return b"#!" + path.encode("utf-8") + b"\n"


def _zipped_main(script: str) -> bytes:
    """A zip archive holding ``script`` as ``__main__.py``, uncompressed (its text stays readable in the file)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(zipfile.ZipInfo("__main__.py", date_time=_FIXED_ZIP_TIME), script)
    return buffer.getvalue()


def _with_icon(stub: bytes, images: Sequence[IconImage]) -> bytes:
    """``stub`` with ``images`` as its icon resources (written through the Windows resource API)."""
    handle, path = tempfile.mkstemp(suffix=".exe")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(stub)
        _update_icon_resources(path, images)
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def _update_icon_resources(path: str, images: Sequence[IconImage]) -> None:
    """Add the icon images and their group to the executable at ``path``."""
    kernel32 = _kernel32()
    update = kernel32.BeginUpdateResourceW(path, False)
    if not update:
        raise LauncherError(f"cannot open {path} for resource updates: {ctypes.WinError()}")
    try:
        for index, image in enumerate(images):
            _update(kernel32, update, _RT_ICON, _FIRST_IMAGE_ID + index, image.png)
        _update(kernel32, update, _RT_GROUP_ICON, _GROUP_ID, icon_group(images, _FIRST_IMAGE_ID))
    except LauncherError:
        kernel32.EndUpdateResourceW(update, True)  # discard
        raise
    if not kernel32.EndUpdateResourceW(update, False):
        raise LauncherError(f"cannot write the icon into {path}: {ctypes.WinError()}")


def _update(kernel32: Any, update: int, resource_type: int, resource_id: int, data: bytes) -> None:
    """Store one resource (type and name as integer IDs, ``MAKEINTRESOURCE``)."""
    buffer = ctypes.create_string_buffer(data, len(data))
    if not kernel32.UpdateResourceW(update, ctypes.c_void_p(resource_type), ctypes.c_void_p(resource_id),
                                    _LANG_NEUTRAL, buffer, len(data)):
        raise LauncherError(f"cannot add icon resource {resource_type}/{resource_id}: {ctypes.WinError()}")


def _kernel32() -> Any:
    """``kernel32`` with the prototypes of the resource-update functions."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=False)
    kernel32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
    kernel32.BeginUpdateResourceW.restype = wintypes.HANDLE
    kernel32.UpdateResourceW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, wintypes.WORD,
                                         ctypes.c_void_p, wintypes.DWORD]
    kernel32.UpdateResourceW.restype = wintypes.BOOL
    kernel32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]
    kernel32.EndUpdateResourceW.restype = wintypes.BOOL
    return kernel32
