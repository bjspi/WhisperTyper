"""Expert mode: whether the settings window and the tray menu offer everything or only the common parts.

It only decides what is on screen; hidden settings keep their values and stay in effect. It is
unrelated to the ``expert`` block of config.json (see ``expert_settings``), which has no GUI.
"""
from __future__ import annotations

from typing import Any, Mapping

#: Config key of the sidebar switch.
EXPERT_MODE_KEY = "expert_mode"


def expert_mode_shown(config: Mapping[str, Any], modern_design: bool) -> bool:
    """Whether expert settings and tools are shown; the classic design has no switch and shows everything."""
    return not modern_design or bool(config.get(EXPERT_MODE_KEY, True))
