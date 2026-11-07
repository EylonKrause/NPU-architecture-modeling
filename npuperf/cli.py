"""Shared command-line plumbing: one parser style, one error policy.

Exit status: 0 success; 1 the run failed (simulator error, failed check, timeout, I/O);
2 the tool rejected its arguments or input (bad flag, invalid file, missing binary,
refusing to overwrite). Errors print one line, ``<tool>: <message>``, on stderr.
"""
from __future__ import annotations

import argparse
import functools
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable, Sequence

from . import binary as binaries

EXIT_FAILED = 1
EXIT_REJECTED = 2


class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


class Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors are a single ``<prog>: <message>`` line."""

    def __init__(self, prog: str, description: str | None = None, **kwargs):
        kwargs.setdefault("formatter_class", Formatter)
        super().__init__(prog=prog, description=description, **kwargs)

    def error(self, message: str):  # type: ignore[override]
        self.exit(EXIT_REJECTED, f"{self.prog}: {message} (see --help)\n")


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from error
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from error
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be nonnegative, got {value}")
    return value


def int_list(low: int, high: int) -> Callable[[str], list[int]]:
    def parse(text: str) -> list[int]:
        try:
            values = sorted({int(item) for item in text.split(",")})
        except ValueError as error:
            raise argparse.ArgumentTypeError("expected comma-separated integers") from error
        if not values or values[0] < low or values[-1] > high:
            raise argparse.ArgumentTypeError(f"values must be in {low}..{high}")
        return values
    return parse


def add_binary_options(parser: argparse.ArgumentParser, timeout: bool = True) -> None:
    parser.add_argument("--binary", type=Path, help="npu-sim executable or build directory "
                        "(default: $NPU_SIM, then build-live/, build/, build/<preset>/)")
    parser.add_argument("--build-dir", type=Path, help="CMake build directory to search first")
    if timeout:
        parser.add_argument("--timeout", type=positive_int, default=60,
                            help="seconds per simulator run")


def add_format_option(parser: argparse.ArgumentParser, default: str = "table") -> None:
    parser.add_argument("--format", choices=("table", "json"), default=default,
                        help="stdout format")


def simulator(args: argparse.Namespace, name: str = "npu-sim") -> Path:
    return binaries.resolve(getattr(args, "binary", None), name, getattr(args, "build_dir", None))


def report(tool: str, error: BaseException) -> str:
    text = " ".join(str(error).split()) or type(error).__name__
    return f"{tool}: {text[:400]}"


def exit_status(error: BaseException) -> int:
    if isinstance(error, (ValueError, FileNotFoundError, FileExistsError,
                          IsADirectoryError, NotADirectoryError)):
        return EXIT_REJECTED
    return EXIT_FAILED


def main_guard(tool: str) -> Callable[[Callable[..., int | None]], Callable[..., int]]:
    """Wrap ``main(argv)`` so expected failures exit with one stderr line, not a traceback."""
    def decorate(function: Callable[..., int | None]) -> Callable[..., int]:
        @functools.wraps(function)
        def guarded(argv: Sequence[str] | None = None) -> int:
            try:
                return int(function(argv) or 0)
            except KeyboardInterrupt:
                return 130
            except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                if isinstance(error, subprocess.TimeoutExpired):
                    error = RuntimeError(f"timed out after {error.timeout:g} s: "
                                         f"{' '.join(map(str, error.cmd))[:200]}")
                print(report(tool, error), file=sys.stderr)
                return exit_status(error)
        return guarded
    return decorate
