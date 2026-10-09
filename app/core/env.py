"""Platform flags. Single source for is_MACOS / is_WINDOWS."""

import sys

is_MACOS = sys.platform.startswith("darwin")
is_WINDOWS = sys.platform.startswith("win")
