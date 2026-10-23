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

# Paste and copy timings are expert settings in config.json (app/core/expert_settings.py).
