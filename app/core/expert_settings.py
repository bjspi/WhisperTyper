"""Expert settings: tuning values editable only in config.json's "expert" block, never in the GUI.

The defaults are the values the app has always used, so an untouched block changes nothing.
Missing keys are filled in on load, so the file always lists every available setting.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, NamedTuple, Tuple

#: Config key of the block.
EXPERT_KEY = "expert"


class ExpertInt(NamedTuple):
    """An integer setting with its default and allowed range (inclusive)."""

    default: int
    minimum: int
    maximum: int


EXPERT_SETTINGS: Dict[str, ExpertInt] = {
    # Clipboard paste choreography (milliseconds).
    "paste_prepare_ms": ExpertInt(100, 0, 2000),          # let the OS publish the new clipboard before Ctrl/Cmd+V
    "paste_settle_ms": ExpertInt(100, 0, 2000),           # let the target application consume the paste
    "clipboard_restore_ms": ExpertInt(500, 0, 10000),     # restore the user's clipboard after a regular paste
    "clipboard_restore_fast_ms": ExpertInt(600, 0, 10000),  # fast paste skips the waits, so restore a little later
    # Reading the selection through a simulated copy (milliseconds).
    "copy_settle_ms": ExpertInt(100, 0, 2000),            # let the OS publish the copied selection
    "select_all_settle_ms": ExpertInt(80, 0, 2000),       # let the focused field apply Ctrl/Cmd+A before copying
}


def expert_defaults() -> Dict[str, int]:
    """A fresh block holding every default."""
    return {key: spec.default for key, spec in EXPERT_SETTINGS.items()}


def _as_int(value: Any) -> Any:
    """``value`` as an int, or None when it is no whole number (booleans and text included)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def normalize_expert(block: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a loaded block: fill missing keys, reset invalid values, clamp out-of-range ones.

    Unknown keys are kept (a typo must not vanish silently) but reported. Returns the
    repaired block and one warning per problem.
    """
    if not isinstance(block, dict):
        return expert_defaults(), [f'Expert settings: "{EXPERT_KEY}" is not an object; using the defaults.']
    result = dict(block)
    warnings = [f'Expert settings: unknown key "{key}" is ignored.' for key in block if key not in EXPERT_SETTINGS]
    for key, spec in EXPERT_SETTINGS.items():
        if key not in result:
            result[key] = spec.default
            continue
        number = _as_int(result[key])
        if number is None:
            warnings.append(f"Expert settings: {key}={result[key]!r} is not a whole number; "
                            f"using the default {spec.default}.")
            result[key] = spec.default
        elif not spec.minimum <= number <= spec.maximum:
            clamped = max(spec.minimum, min(spec.maximum, number))
            warnings.append(f"Expert settings: {key}={number} is outside {spec.minimum}-{spec.maximum}; "
                            f"using {clamped}.")
            result[key] = clamped
        else:
            result[key] = number
    return result, warnings


def expert_value(config: Mapping[str, Any], key: str) -> int:
    """The validated value of ``key``; the default when the block or value is missing or invalid."""
    spec = EXPERT_SETTINGS[key]
    block = config.get(EXPERT_KEY)
    number = _as_int(block.get(key)) if isinstance(block, dict) else None
    if number is None:
        return spec.default
    return max(spec.minimum, min(spec.maximum, number))


def expert_seconds(config: Mapping[str, Any], key: str) -> float:
    """A millisecond setting in seconds, for ``time.sleep``."""
    return expert_value(config, key) / 1000
