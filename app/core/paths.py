"""Resource path resolution for frozen (PyInstaller) and source runs."""

from __future__ import annotations

import os
import sys


def resource_path(*path_segments: str) -> str:
    """
    Get the absolute path to a resource, for both frozen and non-frozen apps.

    Resources live under the ``app`` directory. In source runs that is the parent
    of this ``core`` package; when frozen it is the bundled ``app`` folder.
    """
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            base_path = os.path.join(sys._MEIPASS, "app")
        else:
            base_path = os.path.join(os.path.dirname(sys.executable), "app")
    else:
        # .../app/core/paths.py -> .../app
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not path_segments:
        return base_path

    return os.path.join(base_path, *path_segments)
