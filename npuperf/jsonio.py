"""Strict JSON input and deterministic JSON output."""
from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 64 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError("non-finite JSON number: " + value)


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("JSON number outside the finite double range: " + text[:40])
    return value


def loads_strict(text: str) -> Any:
    """Parse JSON, rejecting duplicate keys, NaN/Infinity and numbers that overflow a double."""
    try:
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant, parse_float=_finite_float)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error}") from error


def load_strict(path: Path | str, max_bytes: int = MAX_INPUT_BYTES) -> Any:
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{path} exceeds the {max_bytes // (1024 * 1024)} MiB input limit")
    return loads_strict(path.read_text(encoding="utf-8"))


def dumps(value: Any, indent: int | None = 2, sort_keys: bool = True) -> str:
    """Deterministic text: sorted keys, no NaN, trailing newline."""
    return json.dumps(value, indent=indent, sort_keys=sort_keys, allow_nan=False) + "\n"


def write_text(path: Path | str, text: str, overwrite: bool = True) -> Path:
    """Write ``text``; with ``overwrite=False`` an existing file raises FileExistsError."""
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def write_json(path: Path | str, value: Any, overwrite: bool = True) -> Path:
    return write_text(path, dumps(value), overwrite)


def write_report(directory: Path | str, name: str, report: Any,
                 rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    """Write ``<directory>/<name>.json`` and ``<name>.csv``, replacing earlier runs."""
    if not rows:
        raise ValueError("no report rows")
    directory = Path(directory)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return (write_json(directory / (name + ".json"), report),
            write_text(directory / (name + ".csv"), stream.getvalue()))
