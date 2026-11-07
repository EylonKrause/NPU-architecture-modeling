#!/usr/bin/env python3
"""Build npu-sim and open the live dashboard.

The dashboard port is bound before the build starts, so a busy port fails fast and the
port stays reserved while CMake runs.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def positive_jobs(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 128:
        raise argparse.ArgumentTypeError("jobs must be from 1 to 128")
    return value


def port_number(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be from 0 to 65535")
    return value


def cmake_executable() -> str:
    binary = shutil.which("cmake")
    if binary is None:
        raise RuntimeError("CMake 3.24 or newer is required; install it as described in "
                           "docs/QUICKSTART.md.")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             timeout=15, check=False)
    match = re.search(r"cmake version (\d+)\.(\d+)", version.stdout)
    if version.returncode or match is None or tuple(map(int, match.groups())) < (3, 24):
        raise RuntimeError("CMake 3.24 or newer is required; check the CMake on PATH "
                           "(docs/QUICKSTART.md).")
    return binary


def build(build_dir: Path, jobs: int) -> None:
    cmake = cmake_executable()
    print("Building npu-sim (the first build fetches SystemC 3.0.2)...", flush=True)
    commands = [
        [cmake, "-S", str(ROOT), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
         "-DBUILD_TESTING=OFF"],
        [cmake, "--build", str(build_dir), "--config", "Release", "--target", "npu-sim",
         "--parallel", str(jobs)],
    ]
    for command in commands:
        if subprocess.run(command, cwd=ROOT, check=False).returncode:
            raise RuntimeError("native build failed; see the compiler or CMake errors above "
                               "and docs/QUICKSTART.md")


def find_binary(build_dir: Path) -> Path:
    from npuperf import binary as binaries
    try:
        return binaries.find_in(build_dir, "npu-sim")
    except binaries.BinaryNotFound as error:
        # Still BinaryNotFound, so the exit status is 2 like every other npuperf tool.
        hint = f"{error}; run without --skip-build to build it"
        raise binaries.BinaryNotFound(hint) from error


def verify_binary(binary: Path) -> None:
    try:
        result = subprocess.run([str(binary), "--m", "4", "--n", "4", "--k", "8"],
                                capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"cannot run {binary}: {error}; run without --skip-build to rebuild "
                           "it for this computer") from error
    try:
        data = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"{binary} failed its startup check: {result.stderr[-1000:]}") from error
    if result.returncode or not isinstance(data, dict) or data.get("numerical_match") is not True:
        raise RuntimeError(f"{binary} failed numerical validation: {result.stderr[-1000:]}")


def main(argv=None) -> int:
    if sys.version_info < (3, 10):
        print("run.py: Python 3.10 or newer is required (docs/QUICKSTART.md)", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT))
    from npuperf import cli, live

    @cli.main_guard("run.py")
    def launch(arguments) -> int:
        parser = cli.Parser("run.py", __doc__)
        parser.add_argument("--build-dir", type=Path, default=ROOT / "build-live",
                            help="CMake build directory")
        parser.add_argument("--jobs", type=positive_jobs, default=2, help="parallel build jobs")
        parser.add_argument("--port", type=port_number, default=8765,
                            help="dashboard port on 127.0.0.1; 0 picks a free one")
        parser.add_argument("--profile", choices=live.PROFILES, default="balanced",
                            help="workload to start with")
        parser.add_argument("--no-browser", action="store_true", help="print the URL only")
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--build-only", action="store_true",
                          help="build and check the simulator, then exit")
        mode.add_argument("--skip-build", action="store_true",
                          help="use an existing build without running CMake")
        args = parser.parse_args(arguments)
        server = None
        if not args.build_only:
            try:
                server = live.DashboardServer(("127.0.0.1", args.port))
            except OSError as error:
                raise RuntimeError(f"cannot use local port {args.port}: "
                                   f"{error.strerror or error}; choose a free port, for example "
                                   "python run.py --port 8766") from error
        try:
            build_dir = args.build_dir.expanduser().resolve()
            if not args.skip_build:
                build(build_dir, args.jobs)
            binary = find_binary(build_dir)
            verify_binary(binary)
            if args.build_only:
                print(f"Build and numerical check passed: {binary}", flush=True)
                return 0
            bound, server = server, None
            return live.serve(binary, args.port, args.profile, autostart=True,
                              open_browser=not args.no_browser, bound=bound)
        finally:
            if server is not None:
                server.server_close()

    return launch(argv)


if __name__ == "__main__":
    sys.exit(main())
