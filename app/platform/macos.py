"""macOS integration: optional Accessibility/Quartz/AVFoundation bindings and app activation.

The native symbols are imported once and exposed as None where unavailable.
"""

from __future__ import annotations

import subprocess
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


def frontmost_app_name() -> str:
    """Return the name of the frontmost macOS application."""
    script = 'tell application "System Events" to get name of first application process whose frontmost is true'
    return subprocess.check_output(['osascript', '-e', script]).decode().strip()


def activate_app(app_name: str) -> None:
    """Bring the named macOS application back to the foreground."""
    script = f'tell application "{app_name}" to activate'
    subprocess.call(['osascript', '-e', script])
