"""Aligned plain-text tables for terminal output."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def format_cell(value: Any, spec: str | None = None) -> str:
    if value is None:
        return "-"
    if spec:
        return format(value, spec)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.3f}" if abs(value) < 1e6 else f"{value:.3e}"
    return str(value)


def format_table(rows: Sequence[Mapping[str, Any] | Sequence[Any]],
                 columns: Sequence[str] | None = None,
                 headers: Sequence[str] | None = None,
                 formats: Mapping[str, str] | None = None) -> str:
    """Render rows (dicts or sequences) with right-aligned numbers and left-aligned text."""
    formats = formats or {}
    if rows and isinstance(rows[0], Mapping):
        columns = list(columns or rows[0].keys())
        cells = [[row.get(column) for column in columns] for row in rows]  # type: ignore[union-attr]
    else:
        cells = [list(row) for row in rows]
        columns = list(columns or (headers or []))
    headers = list(headers or columns)
    if not headers and cells:
        headers = [""] * len(cells[0])
    keys = list(columns) if columns else [str(i) for i in range(len(headers))]
    text = [[format_cell(value, formats.get(keys[i]) if i < len(keys) else None)
             for i, value in enumerate(row)] for row in cells]
    numeric = [all(isinstance(row[i], (int, float)) and not isinstance(row[i], bool)
                   or row[i] is None for row in cells) and any(row[i] is not None for row in cells)
               for i in range(len(headers))] if cells else [False] * len(headers)
    widths = [max([len(headers[i])] + [len(row[i]) for row in text]) for i in range(len(headers))]

    def line(values: Sequence[str]) -> str:
        parts = [value.rjust(widths[i]) if numeric[i] else value.ljust(widths[i])
                 for i, value in enumerate(values)]
        return "  ".join(parts).rstrip()

    out = [line(headers), "  ".join("-" * width for width in widths)]
    out += [line(row) for row in text]
    return "\n".join(out) + "\n"
