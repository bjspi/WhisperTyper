"""Native Windows AAC encoding through Media Foundation; no external executable."""
from __future__ import annotations

import ctypes
import uuid
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import Any, Iterator

from app.core.env import is_WINDOWS

_P = ctypes.c_void_p
_U32 = ctypes.c_uint32
_I64 = ctypes.c_int64
_HRESULT = ctypes.c_int32


class _Guid(ctypes.Structure):
    """Native GUID layout."""

    _fields_ = [("data1", _U32), ("data2", ctypes.c_uint16),
                ("data3", ctypes.c_uint16), ("data4", ctypes.c_ubyte * 8)]


def _guid(value: str) -> _Guid:
    """Parse the little-endian Windows GUID representation."""
    return _Guid.from_buffer_copy(uuid.UUID(value).bytes_le)


_MAJOR = _guid("48eba18e-f8c9-4687-bf11-0a74c9f96a8f")
_SUBTYPE = _guid("f7e34c9a-42e8-4714-b74b-cb29d72c35e5")
_CHANNELS = _guid("37e48bf5-645e-4c5b-89de-ada9e29b696a")
_RATE = _guid("5faeeae7-0290-4c31-9e8a-c534f68d9dba")
_BYTES = _guid("1aab75c8-cfef-451c-ab95-ac034b8e1731")
_ALIGN = _guid("322de230-9eeb-43bd-ab7a-ff412251541d")
_BITS = _guid("f2deb57f-40fa-4764-aa33-ed4f2d1ff669")
_PAYLOAD = _guid("bfbabe79-7434-4d1c-94f0-72a3b9e17188")
_AUDIO = _guid("73647561-0000-0010-8000-00aa00389b71")
_PCM = _guid("00000001-0000-0010-8000-00aa00389b71")
_AAC = _guid("00001610-0000-0010-8000-00aa00389b71")


def _check(result: int) -> None:
    """Raise with the native HRESULT, never with audio or credential contents."""
    if result < 0:
        raise OSError(f"Media Foundation error 0x{result & 0xffffffff:08X}")


def _call(pointer: _P, index: int, types: tuple[Any, ...] = (), *args: Any) -> int:
    """Call the documented COM vtable slot with explicit native argument types."""
    address = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(_P)))[0][index]
    return ctypes.WINFUNCTYPE(_HRESULT, _P, *types)(address)(pointer, *args)


def _uint(pointer: _P, key: _Guid, value: int) -> None:
    """Set an IMFAttributes UINT32 value."""
    _check(_call(pointer, 21, (ctypes.POINTER(_Guid), _U32), ctypes.byref(key), value))


def _set_guid(pointer: _P, key: _Guid, value: _Guid) -> None:
    """Set an IMFAttributes GUID value."""
    _check(_call(pointer, 24, (ctypes.POINTER(_Guid), ctypes.POINTER(_Guid)), ctypes.byref(key), ctypes.byref(value)))


@contextmanager
def _foundation() -> Iterator[tuple[Any, Any]]:
    """Initialize Media Foundation on the caller's thread and balance native lifetime."""
    if not is_WINDOWS:
        raise OSError("Media Foundation is only available on Windows")
    ole = ctypes.WinDLL("ole32")
    mfplat = ctypes.WinDLL("mfplat")
    mf = ctypes.WinDLL("mf")
    ole.CoInitializeEx.argtypes = [_P, _U32]
    ole.CoInitializeEx.restype = _HRESULT
    initialized = ole.CoInitializeEx(None, 0)
    # Qt may have already initialized this thread in another apartment.
    if initialized != -2147417850:  # RPC_E_CHANGED_MODE
        _check(initialized)
    mfplat.MFStartup.argtypes = [_U32, _U32]
    mfplat.MFStartup.restype = _HRESULT
    started = False
    try:
        _check(mfplat.MFStartup(0x20070, 0))
        started = True
        yield mfplat, mf
    finally:
        if started:
            mfplat.MFShutdown()
        if initialized >= 0:
            ole.CoUninitialize()


def _create(stack: ExitStack, library: Any, name: str, types: tuple[Any, ...] = (), *args: Any) -> _P:
    """Create one COM object and release it even when later encoding fails."""
    pointer = _P()
    function = getattr(library, name)
    function.argtypes = [*types, ctypes.POINTER(_P)]
    function.restype = _HRESULT
    _check(function(*args, ctypes.byref(pointer)))
    stack.callback(_call, pointer, 2)
    return pointer


@lru_cache(maxsize=1)
def available_aac_bitrates() -> tuple[int, ...]:
    """Enumerate installed native AAC output types for mono 48 kHz audio."""
    with _foundation() as (_, mf), ExitStack() as stack:
        collection = _create(stack, mf, "MFTranscodeGetAudioOutputAvailableTypes",
                             (ctypes.POINTER(_Guid), _U32, _P), ctypes.byref(_AAC), 0x3F, None)
        count = _U32()
        _check(_call(collection, 3, (ctypes.POINTER(_U32),), ctypes.byref(count)))
        bitrates = set()
        for index in range(count.value):
            with ExitStack() as item_stack:
                media_type = _P()
                _check(_call(collection, 4, (_U32, ctypes.POINTER(_P)), index, ctypes.byref(media_type)))
                item_stack.callback(_call, media_type, 2)
                values = []
                for key in (_CHANNELS, _RATE, _BYTES):
                    value = _U32()
                    _check(_call(media_type, 7, (ctypes.POINTER(_Guid), ctypes.POINTER(_U32)), ctypes.byref(key), ctypes.byref(value)))
                    values.append(value.value)
                if values[:2] == [1, 48000]:
                    bitrates.add(values[2] * 8 // 1000)
        return tuple(sorted(bitrates))


def write_aac(path: str, pcm: bytes, samplerate: int, bitrate_kbps: int) -> None:
    """Encode mono signed 16-bit PCM as an AAC/M4A file using the native sink writer."""
    if not pcm or len(pcm) % 2 or samplerate <= 0 or bitrate_kbps not in available_aac_bitrates():
        raise ValueError("Invalid PCM or unsupported native AAC bitrate")
    with _foundation() as (mfplat, _), ExitStack() as stack:
        reader = ctypes.WinDLL("mfreadwrite")
        writer = _create(stack, reader, "MFCreateSinkWriterFromURL", (ctypes.c_wchar_p, _P, _P), path, None, None)
        output = _create(stack, mfplat, "MFCreateMediaType")
        input_type = _create(stack, mfplat, "MFCreateMediaType")
        for media_type, subtype, rate in ((output, _AAC, 48000), (input_type, _PCM, samplerate)):
            _set_guid(media_type, _MAJOR, _AUDIO)
            _set_guid(media_type, _SUBTYPE, subtype)
            for key, value in ((_CHANNELS, 1), (_RATE, rate), (_BITS, 16)):
                _uint(media_type, key, value)
        _uint(output, _BYTES, bitrate_kbps * 1000 // 8)
        _uint(output, _PAYLOAD, 0)
        _uint(input_type, _ALIGN, 2)
        _uint(input_type, _BYTES, samplerate * 2)
        stream = _U32()
        _check(_call(writer, 3, (_P, ctypes.POINTER(_U32)), output, ctypes.byref(stream)))
        _check(_call(writer, 4, (_U32, _P, _P), stream, input_type, None))
        _check(_call(writer, 5))
        # Feed bounded 100 ms buffers so long recordings do not duplicate the PCM allocation.
        chunk_bytes = max(2, samplerate // 10 * 2)
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset:offset + chunk_bytes]
            with ExitStack() as sample_stack:
                buffer = _create(sample_stack, mfplat, "MFCreateMemoryBuffer", (_U32,), len(chunk))
                data = _P()
                _check(_call(buffer, 3, (ctypes.POINTER(_P), _P, _P), ctypes.byref(data), None, None))
                try:
                    ctypes.memmove(data, chunk, len(chunk))
                finally:
                    _check(_call(buffer, 4))
                _check(_call(buffer, 6, (_U32,), len(chunk)))
                sample = _create(sample_stack, mfplat, "MFCreateSample")
                _check(_call(sample, 42, (_P,), buffer))
                start = offset // 2 * 10_000_000 // samplerate
                end = (offset + len(chunk)) // 2 * 10_000_000 // samplerate
                _check(_call(sample, 36, (_I64,), start))
                _check(_call(sample, 38, (_I64,), end - start))
                _check(_call(writer, 6, (_U32, _P), stream, sample))
        _check(_call(writer, 11))
