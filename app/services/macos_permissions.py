"""macOS privacy permissions used for global hotkeys and text insertion.

Each request shows the system prompt where macOS still allows it and reports whether the
permission is granted. Unavailable frameworks (other platforms, missing PyObjC) count as
granted, so callers never block on a check that cannot run.
"""
from __future__ import annotations

import logging

from app.platform.macos import (
    AXIsProcessTrustedWithOptions,
    CGPreflightListenEventAccess,
    CGRequestListenEventAccess,
    kAXTrustedCheckOptionPrompt,
)

#: Step-by-step permission walkthrough linked from the information dialogs.
PERMISSIONS_GUIDE_URL = "https://github.com/bjspi/WhisperTyper/blob/main/docs/INSTALL_MACOS.md"


def request_input_monitoring() -> bool:
    """Ask for Input Monitoring (global hotkeys); False if it is not granted (yet)."""
    if not CGPreflightListenEventAccess or not CGRequestListenEventAccess:
        return True
    try:
        if CGPreflightListenEventAccess():
            return True
        return bool(CGRequestListenEventAccess())
    except Exception as e:
        logging.warning("Failed to request macOS Input Monitoring permission: %s", e)
        return False


def request_accessibility() -> bool:
    """Ask for Accessibility (hotkeys, simulated paste); False if it is not granted (yet)."""
    if not AXIsProcessTrustedWithOptions or not kAXTrustedCheckOptionPrompt:
        return True
    try:
        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}))
    except Exception as e:
        logging.warning("Failed to request macOS Accessibility permission: %s", e)
        return False
