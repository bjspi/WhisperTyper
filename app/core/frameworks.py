"""Platform framework bindings: macOS Accessibility/Quartz/AVFoundation + Windows hotkey constants.

Single responsibility: import the optional native symbols once, exposing None where unavailable.
"""

from __future__ import annotations

from typing import Any, Dict

from app.core.env import is_MACOS

if is_MACOS:
    try:
        from HIServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
    except ImportError:
        AXIsProcessTrustedWithOptions = None
        kAXTrustedCheckOptionPrompt = None

    try:
        from Quartz import CGPreflightListenEventAccess, CGRequestListenEventAccess
    except ImportError:
        CGPreflightListenEventAccess = None
        CGRequestListenEventAccess = None

    try:
        import objc
        from Foundation import NSURL
    except ImportError:
        objc = None
        NSURL = None

    AVAudioRecorder = None
    if objc:
        try:
            _avfoundation_globals: Dict[str, Any] = {}
            objc.loadBundle(
                'AVFoundation',
                _avfoundation_globals,
                bundle_path=objc.pathForFramework('/System/Library/Frameworks/AVFoundation.framework'),
            )
            AVAudioRecorder = _avfoundation_globals.get("AVAudioRecorder")
        except Exception:
            AVAudioRecorder = None
else:
    AXIsProcessTrustedWithOptions = None
    kAXTrustedCheckOptionPrompt = None
    CGPreflightListenEventAccess = None
    CGRequestListenEventAccess = None
    objc = None
    NSURL = None
    AVAudioRecorder = None

# Windows hotkey constants (defined unconditionally; only used under is_WINDOWS at runtime).
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
