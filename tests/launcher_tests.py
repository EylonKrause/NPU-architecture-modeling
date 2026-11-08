#!/usr/bin/env python3
"""run.py: startup from another directory against a built npu-sim, a validated job on the
dashboard, clean shutdown, and one-line errors for bad arguments and environments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run.py"
BINARY: Path
BUILD_DIRECTORY: Path


class LauncherProcess:
    """Collect startup output without blocking indefinitely on a pipe."""

    def __init__(self, arguments: list[str], working_directory: Path):
        self.url = ""
        self.errors = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(LAUNCHER), *arguments],
            cwd=working_directory, stdout=subprocess.PIPE, stderr=self.errors,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)

        self.reader = threading.Thread(target=read_output, daemon=True)
        self.reader.start()

    def diagnostics(self) -> str:
        self.errors.flush()
        self.errors.seek(0)
        return self.errors.read()

    def read_url(self, timeout: float = 20) -> str:
        deadline = time.monotonic() + timeout
        output: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                break
            output.append(line)
            match = re.search(r"http://127\.0\.0\.1:\d+", line)
            if match:
                self.url = match.group(0)
                return self.url
        raise AssertionError(
            "Launcher did not announce its URL "
            f"(pid={self.process.pid}, returncode={self.process.poll()}, "
            f"stdout_reader_alive={self.reader.is_alive()}):\n"
            f"{''.join(output)}\nstderr:\n{self.diagnostics()}"
        )

    def close(self) -> None:
        if self.process.poll() is None:
            if os.name == "posix":
                self.process.send_signal(signal.SIGINT)
            else:
                # TerminateProcess skips Python signal handlers on Windows, so the
                # simulator child is stopped through /api/control first.
                if self.url:
                    request = Request(
                        self.url + "/api/control", data=b'{"action":"stop"}', method="POST",
                        headers={"Content-Type": "application/json", "Origin": self.url},
                    )
                    try:
                        with urlopen(request, timeout=3) as response:
                            response.read()
                    except (URLError, ConnectionError, TimeoutError, OSError):
                        pass
                self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.reader.join(timeout=2)
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.errors.close()


class LauncherTests(unittest.TestCase):
    def run_launcher(self, arguments: list[str], *, env: dict[str, str] | None = None):
        with tempfile.TemporaryDirectory(prefix="npu-launcher-cwd-") as directory:
            return subprocess.run(
                [sys.executable, str(LAUNCHER), *arguments], cwd=directory,
                env=env, capture_output=True, text=True, encoding="utf-8", timeout=20,
            )

    def assert_clean_failure(self, result, status: int, *terms: str) -> None:
        self.assertEqual(result.returncode, status, result.stderr)
        self.assertTrue(result.stderr.startswith("run.py: "), result.stderr)
        self.assertNotIn("Traceback (most recent call last)", result.stderr)
        self.assertTrue(result.stderr.strip(), "failed startup printed no diagnostic")
        for term in terms:
            self.assertIn(term.lower(), result.stderr.lower())

    def test_launches_from_another_directory_and_validates_a_job(self):
        with tempfile.TemporaryDirectory(prefix="npu-launcher-cwd-") as directory:
            launcher = LauncherProcess(
                ["--skip-build", "--build-dir", str(BUILD_DIRECTORY),
                 "--port", "0", "--no-browser"], Path(directory),
            )
            url = ""
            try:
                url = launcher.read_url()
                self.assertNotEqual(int(url.rsplit(":", 1)[1]), 0)
                with urlopen(url + "/", timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(b"SystemC", response.read())
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    with urlopen(url + "/api/state", timeout=3) as response:
                        state = json.load(response)
                    self.assertFalse(state["error"], state["error"])
                    self.assertEqual(state["profile"], "balanced")
                    self.assertEqual(state["state"], "running")
                    if state["jobs_completed"] > 0:
                        self.assertTrue(state["recent_jobs"][0]["telemetry"]["validated"])
                        self.assertGreater(state["recent_jobs"][0]["telemetry"]["macs"], 0)
                        break
                    time.sleep(0.05)
                else:
                    self.fail(f"No validated inference completed: {state}")
                self.assertIsNone(launcher.process.poll(), launcher.diagnostics())
            finally:
                launcher.close()
            if os.name == "posix":
                self.assertEqual(launcher.process.returncode, 0)
            self.assertTrue(url, "launcher printed no local URL")
            with self.assertRaises((URLError, ConnectionError, TimeoutError, OSError)):
                urlopen(url + "/api/state", timeout=1)

    def test_invalid_cli_inputs_fail_before_starting(self):
        for arguments, term in [
            (["--port", "-1"], "port"),
            (["--port", "65536"], "port"),
            (["--port", "not-a-number"], "port"),
            (["--profile", "not-a-profile"], "profile"),
            (["--build-only", "--skip-build"], "--skip-build"),
            (["--jobs", "0"], "jobs"),
        ]:
            with self.subTest(arguments=arguments):
                self.assert_clean_failure(self.run_launcher(arguments), 2, term)

    def test_missing_binary_has_actionable_error(self):
        with tempfile.TemporaryDirectory(prefix="npu-no-binary-") as directory:
            result = self.run_launcher(
                ["--skip-build", "--build-dir", directory, "--port", "0", "--no-browser"])
            self.assert_clean_failure(result, 2, "npu-sim", "without --skip-build")

    def test_corrupt_executable_is_rejected_before_serving(self):
        with tempfile.TemporaryDirectory(prefix="npu-bad-binary-") as directory:
            executable = Path(directory) / ("npu-sim.exe" if os.name == "nt" else "npu-sim")
            executable.write_bytes(b"plain text without an executable header\n")
            if os.name == "posix":
                executable.chmod(0o755)
            result = self.run_launcher(
                ["--skip-build", "--build-dir", directory, "--port", "0", "--no-browser"])
            self.assert_clean_failure(result, 1, "npu-sim")
            self.assertNotRegex(result.stdout, r"http://127\.0\.0\.1:\d+")

    def test_busy_port_is_reported_before_any_build(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with tempfile.TemporaryDirectory(prefix="npu-busy-port-") as directory:
                build = Path(directory) / "must-not-be-created"
                result = self.run_launcher(
                    ["--build-dir", str(build), "--port", str(port), "--no-browser"])
                self.assert_clean_failure(result, 1, "port", str(port))
                self.assertFalse(build.exists(), "busy port reported after configuring a build")

    def test_port_is_reusable_right_after_a_restart(self):
        # The server closes the connection first ("Connection: close"), so its side of the
        # connection sits in TIME_WAIT on the dashboard port. On POSIX the next bind
        # succeeds only with SO_REUSEADDR, the option DashboardServer sets.
        sys.path.insert(0, str(ROOT))
        from npuperf import live
        first = live.DashboardServer(("127.0.0.1", 0))
        port = first.server_address[1]
        thread = threading.Thread(target=first.serve_forever, daemon=True)
        thread.start()
        client = socket.create_connection(("127.0.0.1", port), timeout=3)
        try:
            client.sendall(b"GET /missing HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
                           b"Connection: close\r\n\r\n" % port)
            while client.recv(4096):  # read to EOF: the server has closed its end
                pass
        finally:
            first.shutdown()
            first.server_close()
            thread.join(timeout=3)
            client.close()
        second = live.DashboardServer(("127.0.0.1", port))
        second.server_close()

    def test_missing_cmake_has_setup_guidance(self):
        with tempfile.TemporaryDirectory(prefix="npu-no-cmake-") as directory:
            environment = dict(os.environ, PATH="")
            result = self.run_launcher(
                ["--build-only", "--build-dir", str(Path(directory) / "build"), "--no-browser"],
                env=environment,
            )
            self.assert_clean_failure(result, 1, "cmake")
            self.assertRegex(result.stderr.lower(), r"install|prerequisite|require|not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / "npu-sim")
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve(strict=True)
    BUILD_DIRECTORY = BINARY.parent
    if BINARY.suffix.lower() == ".exe" and BUILD_DIRECTORY.name.lower() == "release":
        BUILD_DIRECTORY = BUILD_DIRECTORY.parent
    unittest.main(argv=[__file__, *remaining])
