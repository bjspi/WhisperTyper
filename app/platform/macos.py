"""macOS integration: optional Accessibility/Quartz/AVFoundation bindings and app activation.

The native symbols are imported once and exposed as None where unavailable.
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict, Optional, Tuple

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
    AVCaptureDevice = None
    if objc:
        try:
            _avfoundation_globals: Dict[str, Any] = {}
            objc.loadBundle(
                'AVFoundation',
                _avfoundation_globals,
                bundle_path=objc.pathForFramework('/System/Library/Frameworks/AVFoundation.framework'),
            )
            AVAudioRecorder = _avfoundation_globals.get("AVAudioRecorder")
            AVCaptureDevice = _avfoundation_globals.get("AVCaptureDevice")
        except Exception:
            AVAudioRecorder = None
            AVCaptureDevice = None
else:
    AXIsProcessTrustedWithOptions = None
    kAXTrustedCheckOptionPrompt = None
    CGPreflightListenEventAccess = None
    CGRequestListenEventAccess = None
    objc = None
    NSURL = None
    AVAudioRecorder = None
    AVCaptureDevice = None


def frontmost_app_name() -> str:
    """Return the name of the frontmost macOS application."""
    script = 'tell application "System Events" to get name of first application process whose frontmost is true'
    return subprocess.check_output(['osascript', '-e', script]).decode().strip()


def make_nonactivating(widget: Any) -> None:
    """Make a Qt panel non-activating: clicks keep the user's app in front (Qt has no flag for it).

    Such a panel also shows on other apps' fullscreen spaces.
    """
    from PyQt6.QtGui import QGuiApplication

    if objc is None or QGuiApplication.platformName() != "cocoa":
        return  # e.g. the offscreen platform in tests has no NSView behind winId()
    try:
        window = objc.objc_object(c_void_p=int(widget.winId())).window()
        window.setStyleMask_(window.styleMask() | (1 << 7))  # NSWindowStyleMaskNonactivatingPanel
    except Exception:
        pass


def frontmost_window_center() -> Optional[Tuple[int, int]]:
    """Centre of the frontmost app's front window in global coordinates, or None."""
    try:
        from AppKit import NSWorkspace
        from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionOnScreenOnly

        pid = NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier()
        for info in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID):
            bounds = info.get("kCGWindowBounds", {})
            # Fullscreen apps also own a thin title-bar strip; skip such small windows.
            if info.get("kCGWindowOwnerPID") == pid and info.get("kCGWindowLayer") == 0 and bounds.get("Height", 0) >= 100:
                return int(bounds["X"] + bounds["Width"] / 2), int(bounds["Y"] + bounds["Height"] / 2)
    except Exception:
        pass
    return None


def activate_app(app_name: str) -> None:
    """Bring the named macOS application back to the foreground."""
    script = f'tell application "{app_name}" to activate'
    subprocess.call(['osascript', '-e', script])
