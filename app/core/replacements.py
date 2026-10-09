"""Literal transcript corrections: whole words, flexible spacing and a single pass."""
from __future__ import annotations

import re


class ReplacementError(ValueError):
    """Identify an invalid rule without including its possibly private contents."""

    def __init__(self, line: int) -> None:
        """Keep the source line available for a translated UI error."""
        self.line = line
        super().__init__(f"Invalid replacement rule at line {line}")


class Replacements:
    """Compile saved rules once; replacement calls only scan the original transcript."""

    def __init__(self, raw: str) -> None:
        """Parse `source, alias ; replacement ; 1`, with an optional fixed-spelling flag."""
        terms: dict[str, tuple[str, str, bool, int]] = {}
        for number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(";")]
            if len(parts) not in (2, 3) or not parts[1] or (len(parts) == 3 and parts[2] not in ("", "0", "1")):
                raise ReplacementError(number)
            sources = [" ".join(source.split()) for source in parts[0].split(",") if source.strip()]
            if not sources:
                raise ReplacementError(number)
            fixed = len(parts) == 3 and parts[2] == "1"
            for source in sources:
                # As in Turbo-Type, a repeated source belongs to the later rule.
                terms[source.lower()] = (source, parts[1], fixed, number)
        self._terms = sorted(terms.values(), key=lambda term: len(term[0]), reverse=True)
        self.rule_count = len({term[3] for term in self._terms})
        self.term_count = len(self._terms)
        alternatives = "|".join("(" + r"\s+".join(re.escape(word) for word in source.split()) + ")"
                                for source, _replacement, _fixed, _line in self._terms)
        # Unicode letters/digits delimit words; underscores behave like punctuation, as in Turbo-Type.
        self._pattern = re.compile(r"(?<![^\W_])(?:" + alternatives + r")(?![^\W_])", re.IGNORECASE) if terms else None

    def apply(self, text: str) -> tuple[str, int, list[int]]:
        """Return corrected text, match count and matched source-line numbers."""
        if self._pattern is None or not text:
            return text, 0, []
        matched_lines: set[int] = set()

        def replace(match: re.Match[str]) -> str:
            """Use the matched alternative directly, including Unicode case variants."""
            assert match.lastindex is not None
            _source, replacement, fixed, line = self._terms[match.lastindex - 1]
            matched_lines.add(line)
            return replacement if fixed else _cased(match.group(), replacement)

        result, count = self._pattern.subn(replace, text)
        return result, count, sorted(matched_lines)


def _cased(found: str, replacement: str) -> str:
    """Carry upper/title casing from the match, preserving the replacement otherwise."""
    letters = [char for char in found if char.isalpha()]
    if len(letters) >= 2 and all(char.isupper() for char in letters):
        return replacement.upper()
    if letters and letters[0].isupper():
        for index, char in enumerate(replacement):
            if char.isalpha():
                return replacement[:index] + char.upper() + replacement[index + 1:]
    return replacement
