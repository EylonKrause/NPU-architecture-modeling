"""Locate the simulator executables that CMake builds.

Search order for ``find_simulator(name)``:
  1. the environment variable named after the executable (npu-sim -> $NPU_SIM),
  2. ``build_dir`` when given,
  3. ROOT/build-live (run.py), ROOT/build, then every ROOT/build/<preset>/ in name order.
Each directory is tried with and without a ``Release/`` subdirectory (multi-config
generators), and with the ``.exe`` suffix on Windows. Paths resolve against the repository
root, never the current directory.
"""
from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SIMULATORS = ("npu-sim", "npu-tlm-demo", "npu-transformer-demo", "npu-mobile-demo")


class BinaryNotFound(FileNotFoundError):
    """No simulator executable at the requested or default locations."""


def environment_variable(name: str) -> str:
    return name.upper().replace("-", "_")


def executable_names(name: str) -> tuple[str, ...]:
    return (name + ".exe", name) if os.name == "nt" else (name,)


def default_build_dirs() -> list[Path]:
    dirs = [ROOT / "build-live", ROOT / "build"]
    presets = ROOT / "build"
    if presets.is_dir():
        dirs += sorted(path for path in presets.iterdir()
                       if path.is_dir() and not path.name.startswith(("_", ".", "CMakeFiles")))
    return dirs


def candidates(name: str, build_dirs: Iterable[Path]) -> list[Path]:
    paths = []
    for directory in build_dirs:
        for sub in ("Release", ""):
            for executable in executable_names(name):
                paths.append(directory / sub / executable if sub else directory / executable)
    return paths


def find_in(build_dir: Path | str, name: str = "npu-sim") -> Path:
    """Find ``name`` in one build directory (with Release/ and .exe variants)."""
    directory = Path(build_dir).expanduser()
    for path in candidates(name, [directory]):
        if path.is_file():
            return path.resolve()
    raise BinaryNotFound(f"{executable_names(name)[0]} not found in {directory}")


def find_simulator(name: str = "npu-sim", build_dir: Path | str | None = None,
                   environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    variable = environment_variable(name)
    override = environ.get(variable)
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path.resolve()
        raise BinaryNotFound(f"${variable} names {path}, which is not a file")
    dirs = ([Path(build_dir).expanduser()] if build_dir is not None else []) + default_build_dirs()
    for path in candidates(name, dirs):
        if path.is_file():
            return path.resolve()
    searched = ", ".join(str(d) for d in dirs[:4]) + (", ..." if len(dirs) > 4 else "")
    raise BinaryNotFound(f"{name} not found (set ${variable}, pass --binary or --build-dir; "
                         f"searched {searched})")


def resolve(path: Path | str | None, name: str = "npu-sim",
            build_dir: Path | str | None = None) -> Path:
    """The given file or build directory, else the default search."""
    if path is None:
        return find_simulator(name, build_dir)
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        return find_in(candidate, name)
    if not candidate.is_file():
        raise BinaryNotFound(f"binary does not exist: {candidate.resolve()}")
    return candidate.resolve()
