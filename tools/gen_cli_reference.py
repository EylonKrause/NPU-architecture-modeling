#!/usr/bin/env python3
"""Generate the npu-sim option tables of docs/usage.md from ``npu-sim --help``.

    python3 tools/gen_cli_reference.py            print the generated sections
    python3 tools/gen_cli_reference.py --write    replace them in docs/usage.md
    python3 tools/gen_cli_reference.py --check    exit 1 if docs/usage.md differs

``--help`` renders the run-control flags of src/main.cpp and the option table of
src/options.cpp, so the tables follow the build they are generated from. Each table sits
between ``<!-- generated:NAME -->`` and ``<!-- end:NAME -->``, the marker convention of
npuperf.reproduce; the text around the markers is left alone.
"""

from __future__ import annotations

import difflib
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf import cli  # noqa: E402

DOCUMENT = ROOT / "docs" / "usage.md"
RUN_SECTION = "npu-sim-run-options"
CONFIG_SECTION = "npu-sim-config-options"
MARKER = re.compile(r"<!-- generated:([A-Za-z0-9_.-]+) -->\n(.*?)<!-- end:\1 -->", re.S)
FLAG = re.compile(r"--[a-z0-9][a-z0-9-]*")
KEY = re.compile(r"[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*")
DEFAULT = re.compile(r"(.*) \(default (.+)\)")


@dataclass(frozen=True)
class Row:
    flag: str  # empty for config-file-only keys
    key: str  # empty for run-control flags
    value: str  # placeholder; empty for switches and file-only keys
    default: str
    help: str


def parse_row(section: str, head: str, text: str) -> Row:
    tokens = head.split()
    match = DEFAULT.fullmatch(text)
    help_text, default = (match.group(1), match.group(2)) if match else (text, "")
    if section == "run":
        if not FLAG.fullmatch(tokens[0]) or len(tokens) > 2:
            raise ValueError(f"unexpected run-control row {head!r}")
        return Row(tokens[0], "", tokens[1] if len(tokens) == 2 else "", default, help_text)
    if not FLAG.fullmatch(tokens[0]):
        if len(tokens) != 1 or not KEY.fullmatch(tokens[0]):
            raise ValueError(f"unexpected configuration row {head!r}")
        return Row("", tokens[0], "", default, help_text)
    if len(tokens) not in (2, 3) or not KEY.fullmatch(tokens[-1]):
        raise ValueError(f"unexpected configuration row {head!r}")
    return Row(tokens[0], tokens[-1], tokens[1] if len(tokens) == 3 else "", default, help_text)


def parse_help(text: str) -> dict[str, list[Row]]:
    """Run-control and configuration rows of ``npu-sim --help``. Each row is a two-space
    indented head (flag, placeholder, config key) and a six-space indented description."""
    rows: dict[str, list[Row]] = {"run": [], "config": []}
    section = None
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("Run control:"):
            section = "run"
        elif line.startswith("Configuration"):
            section = "config"
        elif section and line.startswith("  ") and not line.startswith("   "):
            following = lines[index + 1] if index + 1 < len(lines) else ""
            if not following.startswith("      ") or not following.strip():
                raise ValueError(f"help row without a description: {line.strip()!r}")
            rows[section].append(parse_row(section, line.strip(), following.strip()))
            index += 1
        index += 1
    if not rows["run"] or not rows["config"]:
        raise ValueError("npu-sim --help lacks the run-control or configuration rows")
    return rows


def cell(text: str) -> str:
    """Table-cell text: flags as code, Markdown metacharacters escaped."""
    text = text.replace("\\", "\\\\").replace("|", "\\|").replace("*", "\\*")
    text = text.replace("<", "&lt;")
    return FLAG.sub(lambda match: f"`{match.group(0)}`", text)


def code(text: str) -> str:
    """A code span; GitHub tables need the pipe escaped even inside one."""
    return "`" + text.replace("|", "\\|") + "`" if text else ""


def render(rows: dict[str, list[Row]]) -> dict[str, str]:
    run = ["| Flag | Value | Default | Description |", "|---|---|---|---|"]
    run += [
        f"| {code(r.flag)} | {code(r.value)} | {code(r.default)} | {cell(r.help)} |"
        for r in rows["run"]
    ]
    config = ["| Flag | Config key | Value | Default | Description |", "|---|---|---|---|---|"]
    config += [
        f"| {code(r.flag)} | {code(r.key)} | {code(r.value)} | {code(r.default)} | {cell(r.help)} |"
        for r in rows["config"]
    ]
    return {RUN_SECTION: "\n".join(run) + "\n", CONFIG_SECTION: "\n".join(config) + "\n"}


def sections(text: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in MARKER.finditer(text)}


def replace_sections(text: str, tables: dict[str, str]) -> str:
    missing = sorted(set(tables) - set(sections(text)))
    if missing:
        raise ValueError(f"{DOCUMENT.name} lacks the markers of: {', '.join(missing)}")

    def body(match: re.Match) -> str:
        name = match.group(1)
        return f"<!-- generated:{name} -->\n{tables.get(name, match.group(2))}<!-- end:{name} -->"

    return MARKER.sub(body, text)


def drift(text: str, tables: dict[str, str]) -> list[str]:
    """A unified diff per committed section that differs from ``tables``."""
    committed = sections(text)
    problems = []
    for name, content in tables.items():
        if name not in committed:
            problems.append(f"{DOCUMENT.name} has no generated:{name} section\n")
        elif committed[name] != content:
            problems.append(
                "".join(
                    difflib.unified_diff(
                        committed[name].splitlines(True),
                        content.splitlines(True),
                        f"committed {name}",
                        f"generated {name}",
                    )
                )
            )
    return problems


def help_text(binary: Path) -> str:
    process = subprocess.run(
        [str(binary), "--help"], capture_output=True, text=True, timeout=30, check=False
    )
    if process.returncode != 0:
        raise RuntimeError(f"{binary.name} --help exited with status {process.returncode}")
    return process.stdout


@cli.main_guard("gen_cli_reference")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("gen_cli_reference", __doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="update the document in place")
    mode.add_argument(
        "--check", action="store_true", help="fail when the document differs; writes nothing"
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=DOCUMENT,
        help="Markdown file holding the option tables (default: docs/usage.md)",
    )
    cli.add_binary_options(parser, timeout=False)
    args = parser.parse_args(argv)
    tables = render(parse_help(help_text(cli.simulator(args))))
    if not (args.write or args.check):
        for name, content in tables.items():
            print(f"<!-- generated:{name} -->\n{content}<!-- end:{name} -->")
        return 0
    text = args.document.read_text(encoding="utf-8")
    if args.check:
        problems = drift(text, tables)
        for problem in problems:
            print(problem, file=sys.stderr, end="")
        if problems:
            print(
                f"gen_cli_reference: {args.document} differs from npu-sim --help; run "
                "python3 tools/gen_cli_reference.py --write",
                file=sys.stderr,
            )
            return cli.EXIT_FAILED
        return 0
    updated = replace_sections(text, tables)
    if updated != text:
        args.document.write_text(updated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
