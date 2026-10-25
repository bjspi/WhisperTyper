"""What the command line prints: results on stdout, progress and errors on stderr.

People get coloured panels and tables; ``--json`` prints one JSON document for tools. When
stdout is redirected (``> file``, a pipe) results are written as plain text, so a script
receives exactly the text and nothing of the decoration.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Iterable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

#: Colours of the WhisperTyper accent and the message kinds.
THEME = Theme({
    "accent": "bold cyan",
    "muted": "grey62",
    "ok": "bold green",
    "warn": "bold yellow",
    "error": "bold red",
    "label": "bold",
})

#: Results; plain text when redirected.
stdout = Console(theme=THEME, highlight=False)
#: Progress, warnings and errors; never mixed into results.
stderr = Console(theme=THEME, highlight=False, stderr=True)


def print_json(value: Any) -> None:
    """Write ``value`` (dataclasses included) as one indented JSON document."""
    stdout.out(json.dumps(_jsonable(value), indent=2, ensure_ascii=False), highlight=False)


def print_plain(text: str) -> None:
    """Write ``text`` unchanged, without markup interpretation or wrapping."""
    stdout.out(text, highlight=False)


def decorated() -> bool:
    """Whether stdout is a terminal that gets panels and colours."""
    return stdout.is_terminal


def result_panel(text: str, title: str, subtitle: str = "") -> None:
    """Show one result text in an accent-framed panel."""
    stdout.print(Panel(Text(text), title=Text(title, style="accent"), subtitle=Text(subtitle, style="muted"),
                       title_align="left", subtitle_align="right", border_style="cyan", padding=(1, 2)))


def facts_table(title: str, rows: Iterable[tuple[str, str]]) -> None:
    """Show label/value pairs as a borderless two-column table under a heading."""
    table = Table.grid(padding=(0, 3))
    table.add_column(style="label", no_wrap=True)
    table.add_column()
    for label, value in rows:
        table.add_row(label, Text(value))
    stdout.print(Text(title, style="accent"))
    stdout.print(table)
    stdout.print()


def success(message: str) -> None:
    """A finished step, on stderr."""
    stderr.print(Text.assemble(("✔ ", "ok"), message))


def warning(message: str) -> None:
    """Something worth knowing that does not stop the command, on stderr."""
    stderr.print(Text.assemble(("! ", "warn"), message))


def error(message: str, hint: Optional[str] = None) -> None:
    """A failure, on stderr; ``hint`` says what to do about it. Messages are never read as markup."""
    stderr.print(Text.assemble(("✘ ", "error"), message))
    if hint:
        stderr.print(Text(f"  {hint}", style="muted"))


def _jsonable(value: Any) -> Any:
    """Dataclasses as dicts, recursively; everything else as it is."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
