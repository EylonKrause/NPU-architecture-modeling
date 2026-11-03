#!/usr/bin/env python3
"""Build and launch the live NPU simulator with one Python command."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def positive_jobs(text):
    value = int(text)
    if not 1 <= value <= 128:
        raise argparse.ArgumentTypeError("jobs must be from 1 to 128")
    return value


def port_number(text):
    value = int(text)
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be from 0 to 65535")
    return value


def available_port(port):
    # Fail before a costly first build. The dashboard binds authoritatively
    # afterwards; this probe deliberately does not reserve the port throughout
    # compilation or interfere with an existing process.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise RuntimeError(
            f"Cannot use local port {port}: {error}. "
            "Choose a free port, for example: python run.py --port 8766"
        ) from error


def cmake_executable():
    binary = shutil.which("cmake")
    if binary is None:
        raise RuntimeError("CMake 3.24 or newer is required. Install the one-time build tools described in docs/QUICKSTART.md.")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             timeout=15, check=False)
    match = re.search(r"cmake version (\d+)\.(\d+)", version.stdout)
    if version.returncode or match is None or tuple(map(int, match.groups())) < (3, 24):
        raise RuntimeError("CMake 3.24 or newer is required; check your CMake installation and PATH. See docs/QUICKSTART.md.")
    return binary


def build(build_dir, jobs):
    cmake = cmake_executable()
    print("Preparing the native NPU model. The first build downloads pinned SystemC; later builds reuse it.", flush=True)
    commands = [
        [cmake, "-S", str(ROOT), "-B", str(build_dir),
         "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF"],
        [cmake, "--build", str(build_dir), "--config", "Release",
         "--target", "npu-sim", "--parallel", str(jobs)],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError(
                "Native build failed; the simulator was not started. "
                "Check the compiler/CMake errors above and docs/QUICKSTART.md."
            )


def find_binary(build_dir):
    name = "npu-sim.exe" if os.name == "nt" else "npu-sim"
    # Visual Studio and multi-config Ninja put Release executables in a
    # configuration subdirectory; ordinary Unix generators use the build root.
    for candidate in (build_dir / "Release" / name, build_dir / name):
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        f"Simulator binary {name} not found in {build_dir}. "
        "Run without --skip-build to build it first."
    )


def verify_binary(binary):
    try:
        result = subprocess.run([str(binary), "--m", "4", "--n", "4", "--k", "8"],
                                capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"Cannot run simulator binary {binary}: {error}. "
            "Run without --skip-build to rebuild for this computer."
        ) from error
    try:
        data = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise RuntimeError("The built simulator failed its startup check. " + result.stderr[-1000:]) from error
    if result.returncode or not isinstance(data, dict) or data.get("numerical_match") is not True:
        raise RuntimeError("The built simulator failed numerical validation. " + result.stderr[-1000:])


def main(argv=None):
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required. See docs/QUICKSTART.md.")
    # Load our controller by its path: an unrelated installed package called
    # 'scripts' must not shadow the project's standard-library-only frontend.
    spec = importlib.util.spec_from_file_location("npu_live_dashboard", ROOT / "scripts/live_dashboard.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load scripts/live_dashboard.py; download the complete repository.")
    live_dashboard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = live_dashboard
    spec.loader.exec_module(live_dashboard)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-live")
    parser.add_argument("--jobs", type=positive_jobs, default=2, help="parallel native build jobs (default 2)")
    parser.add_argument("--port", type=port_number, default=8765, help="local dashboard port; 0 chooses an available port")
    parser.add_argument("--profile", choices=live_dashboard.PROFILES, default="balanced")
    parser.add_argument("--no-browser", action="store_true", help="print the URL without opening a browser")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true", help="build and validate the model, then exit")
    mode.add_argument("--skip-build", action="store_true", help="use an already built model without invoking CMake")
    args = parser.parse_args(argv)
    if not args.build_only:
        available_port(args.port)
    build_dir = args.build_dir.expanduser().resolve()
    if not args.skip_build:
        build(build_dir, args.jobs)
    binary = find_binary(build_dir)
    verify_binary(binary)
    if args.build_only:
        print(f"Build and numerical startup check passed: {binary}", flush=True)
        return 0
    print("Starting continuous validated inference. Keep this terminal open; press Ctrl+C to stop.", flush=True)
    options = ["--binary", str(binary), "--port", str(args.port),
               "--autostart", "--profile", args.profile]
    if not args.no_browser:
        options.append("--open-browser")
    return live_dashboard.main(options)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print("NPU launcher: " + str(error), file=sys.stderr)
        sys.exit(1)
