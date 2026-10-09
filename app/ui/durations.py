"""Named on-screen durations and input delays, so feedback timing is tuned in one place."""
from __future__ import annotations

# Tray balloon display times (ms), by message weight.
BALLOON_CONFIRM_MS = 1600   # brief "done ✓" after a successful insert
BALLOON_SHORT_MS = 2000     # short notices: too short, no speech, no audio
BALLOON_INFO_MS = 2500      # informational results, e.g. copied to clipboard
BALLOON_ERROR_MS = 3000     # recoverable errors
BALLOON_NOTICE_MS = 3500    # notices with an instruction to read
BALLOON_WARNING_MS = 4000   # warnings that need attention
BALLOON_LONG_MS = 4500      # long explanations
BALLOON_MAX_MS = 5000       # the longest regular message
# The "recording…" balloon stays until the stop/cancel path replaces it.
BALLOON_PERSISTENT_MS = 99_999_999

# Clipboard paste choreography.
PASTE_PREPARE_S = 0.1          # let the OS publish the new clipboard before Ctrl/Cmd+V
PASTE_SETTLE_S = 0.1           # let the target application consume the paste
CLIPBOARD_RESTORE_MS = 500     # restore the user's clipboard after a regular paste
CLIPBOARD_RESTORE_FAST_MS = 600  # fast paste skips the waits, so restore a little later

# Reading the selection through a simulated copy.
SELECT_ALL_SETTLE_S = 0.08     # let the focused field apply Ctrl/Cmd+A before copying
COPY_SETTLE_S = 0.1            # let the OS publish the copied selection
