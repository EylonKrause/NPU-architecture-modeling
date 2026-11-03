#!/usr/bin/env python3
"""Loopback-only live controller for the project's actual SystemC simulator.

The web API selects bounded, built-in workloads. It never accepts a command,
executable, environment variable, network address, or filesystem path.
"""
from __future__ import annotations

import argparse
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import random
import signal
from socketserver import TCPServer
import subprocess
import sys
import threading
import time
import tempfile
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "balanced": {"label": "Balanced INT8", "description": "Overlapped load, matrix compute and store with two SRAM slots."},
    "compute-bound": {"label": "Compute pressure", "description": "A larger reduction stresses the compute pipeline."},
    "memory-bound": {"label": "Memory pressure", "description": "Reduced external-memory bandwidth exposes DMA contention."},
    "palette-weights": {"label": "Palettized weights", "description": "INT8 weights encoded as 2-bit lookup indices plus LUTs, with charged decode work."},
}
# These arguments are owned by the application, not supplied by HTTP clients.
PROFILE_ARGS = {
    "balanced": ["--m", "64", "--n", "64", "--k", "128", "--tile-m", "8", "--tile-n", "8", "--tile-k", "32", "--buffer-slots", "2"],
    "compute-bound": ["--m", "64", "--n", "64", "--k", "128", "--tile-m", "8", "--tile-n", "8", "--tile-k", "32", "--buffer-slots", "2", "--macs-per-cycle", "4", "--memory-bytes-per-cycle", "128"],
    "memory-bound": ["--m", "64", "--n", "64", "--k", "128", "--tile-m", "8", "--tile-n", "8", "--tile-k", "32", "--buffer-slots", "2", "--macs-per-cycle", "256", "--memory-bytes-per-cycle", "4", "--memory-latency", "24"],
    "palette-weights": ["--tile-m", "8", "--tile-n", "8", "--tile-k", "32", "--buffer-slots", "2", "--weight-bits", "8", "--weight-encoding", "palette", "--palette-bits", "2", "--input-zero", "0", "--weight-zero", "0", "--shift", "4"],
}
MAX_MESSAGE = 1_048_576


class LiveError(RuntimeError):
    pass


class Controller:
    """One owned simulator child, stepped by a bounded wall-clock scheduler."""

    def __init__(self, binary: Path, tick_seconds: float = 0.025, slice_cycles: int = 256):
        self.binary = binary.resolve(strict=True)
        self.tick_seconds = tick_seconds
        self.slice_cycles = slice_cycles
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.process: subprocess.Popen | None = None
        self.messages: queue.Queue = queue.Queue(maxsize=8)
        self.state = "idle"
        self.profile = "balanced"
        self.telemetry: dict = {}
        self.history: deque = deque(maxlen=8)
        self.jobs_completed = 0
        self.total_cycles = 0
        self.total_macs = 0
        self.error = ""
        self.started_at: float | None = None
        self.thread = threading.Thread(target=self._schedule, name="systemc-scheduler", daemon=True)
        self.thread.start()

    def _read_stdout(self, process: subprocess.Popen, messages: queue.Queue) -> None:
        try:
            assert process.stdout is not None
            while True:
                line = process.stdout.readline(MAX_MESSAGE + 1)
                if not line:
                    messages.put_nowait(LiveError("Simulator exited before sending a response."))
                    return
                if len(line) > MAX_MESSAGE:
                    raise LiveError("Simulator response exceeded the safety limit.")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LiveError("Simulator response must be a JSON object.")
                messages.put_nowait(value)
        except (ValueError, OSError, LiveError, queue.Full) as exc:
            try:
                messages.put_nowait(LiveError(f"Invalid simulator response: {exc}"))
            except queue.Full:
                pass

    def _receive(self) -> dict:
        try:
            value = self.messages.get(timeout=10)
        except queue.Empty as exc:
            raise LiveError("Simulator did not respond within ten seconds.") from exc
        if isinstance(value, Exception):
            raise value
        if "error" in value:
            raise LiveError(str(value["error"])[:300])
        self.telemetry = value
        return value

    def _launch(self) -> None:
        self._terminate()
        self.messages = queue.Queue(maxsize=8)
        seed = self.jobs_completed + 1
        # Only this fixed fixture profile needs a file. The application owns the
        # path and dimensions; HTTP clients cannot supply tensors or paths.
        fixture = tempfile.TemporaryDirectory(prefix="npu-live-palette-") if self.profile == "palette-weights" else None
        try:
            flags = list(PROFILE_ARGS[self.profile])
            if fixture is not None:
                rng = random.Random(seed)
                path = Path(fixture.name) / "weights.npu"
                values = [rng.randint(-8, 8) for _ in range(64 * 128)]
                values += [rng.choice((-3, -1, 1, 3)) for _ in range(128 * 64)]
                values += [rng.randint(-31, 31) for _ in range(64)]
                path.write_text("NPU1 64 64 128\n" + " ".join(map(str, values)) + "\n", encoding="ascii")
                flags += ["--input", str(path)]
            self.process = subprocess.Popen(
                [str(self.binary), "--live-stdio", *flags, "--seed", str(seed)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", bufsize=1,
            )
            threading.Thread(target=self._read_stdout, args=(self.process, self.messages),
                             name="systemc-json-reader", daemon=True).start()
            self._receive()  # Initial snapshot acknowledges all file reads.
        finally:
            if fixture is not None:
                fixture.cleanup()
        self.started_at = time.monotonic()

    def _advance(self, cycles: int) -> None:
        if self.process is None or self.process.stdin is None:
            raise LiveError("Start a workload before advancing the simulator.")
        try:
            self.process.stdin.write(f"advance {cycles}\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise LiveError("Simulator input channel closed unexpectedly.") from exc
        value = self._receive()
        if value.get("completed", False):
            if value.get("validated") is not True:
                raise LiveError("Completed inference did not match the scalar reference.")
            self.jobs_completed += 1
            self.total_cycles += value["cycles"]
            self.total_macs += value["macs"]
            self.history.appendleft({"job": self.jobs_completed, "profile": self.profile,
                                     "telemetry": value})

    def _terminate(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    assert process.stdin is not None
                    process.stdin.write('stop\n')
                    process.stdin.flush()
                    process.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

    def _schedule(self) -> None:
        while not self.closed.is_set():
            self.wake.wait(self.tick_seconds)
            self.wake.clear()
            with self.lock:
                if self.state != "running" or self.closed.is_set():
                    continue
                try:
                    # A completed inference is validated before the next actual
                    # SystemC process starts. No fabricated activity is emitted.
                    if self.telemetry.get("completed", False):
                        self._launch()
                    self._advance(self.slice_cycles)
                except (LiveError, OSError) as exc:
                    self.error = str(exc)[:300]
                    self.state = "error"
                    self._terminate()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "kind": "SystemC architectural simulator",
                "state": self.state, "profile": self.profile,
                "slice_cycles": self.slice_cycles, "jobs_completed": self.jobs_completed,
                "completed_job_cycles": self.total_cycles, "completed_job_macs": self.total_macs,
                "telemetry": self.telemetry, "recent_jobs": list(self.history),
                "error": self.error,
            }

    def control(self, command: dict) -> dict:
        if not isinstance(command, dict) or set(command) - {"action", "profile", "cycles"}:
            raise ValueError("Only action, profile and cycles are accepted.")
        action = command.get("action")
        if not isinstance(action, str) or action not in {"start", "pause", "resume", "step", "stop", "reset"}:
            raise ValueError("Unknown control action.")
        if "profile" in command and (action not in {"start", "reset"} or not isinstance(command["profile"], str) or command["profile"] not in PROFILES):
            raise ValueError("Choose a built-in workload profile for start or reset.")
        if "cycles" in command and action != "step":
            raise ValueError("Only step accepts a cycle count.")
        cycles = command.get("cycles", self.slice_cycles)
        if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= 10000:
            raise ValueError("Step cycles must be an integer from 1 to 10000.")
        with self.lock:
            try:
                if action in {"start", "reset"}:
                    self.profile = command.get("profile", self.profile)
                    self.error = ""
                    self.jobs_completed = 0
                    self.total_cycles = 0
                    self.total_macs = 0
                    self.history.clear()
                    self._launch()
                    self.state = "running" if action == "start" else "paused"
                elif action == "stop":
                    self.state = "stopped"
                    self._terminate()
                elif action == "pause":
                    if self.state != "running":
                        raise ValueError("Only a running model can be paused.")
                    self.state = "paused"
                elif action == "resume":
                    if self.state != "paused":
                        raise ValueError("Only a paused model can be resumed.")
                    self.state = "running"
                elif action == "step":
                    if self.state != "paused":
                        raise ValueError("Pause the model before stepping.")
                    if self.telemetry.get("completed", False):
                        self._launch()
                    self._advance(cycles)
            except (LiveError, OSError) as exc:
                self.error = str(exc)[:300]
                self.state = "error"
                self._terminate()
                raise LiveError(self.error) from exc
            return self.snapshot()

    def close(self) -> None:
        self.closed.set()
        self.wake.set()
        self.thread.join(timeout=12)
        with self.lock:
            self.state = "stopped"
            self._terminate()


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self) -> None:
        # HTTPServer normally performs synchronous reverse DNS here. This
        # loopback-only dashboard needs no host lookup; a slow/offline resolver
        # must not delay startup before it can announce its local URL.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def __init__(self, address: tuple[str, int], controller: Controller):
        self.controller = controller
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: DashboardServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        pass

    def _allowed_host(self) -> bool:
        port = self.server.server_address[1]
        host = self.headers.get("Host", "").lower()
        return host in {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    def _respond(self, status: int, value, content_type: str = "application/json") -> None:
        content = value if isinstance(value, bytes) else json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        if not self._allowed_host():
            self._respond(HTTPStatus.FORBIDDEN, {"error": "Loopback Host header required."})
            return
        path = urlsplit(self.path).path
        if path == "/api/state":
            self._respond(200, self.server.controller.snapshot())
        elif path == "/api/profiles":
            self._respond(200, PROFILES)
        elif path in {"/", "/index.html", "/app.js", "/style.css"}:
            name = "index.html" if path == "/" else path[1:]
            content_type = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8"}[name]
            self._respond(200, (ROOT / "web" / name).read_bytes(), content_type)
        else:
            self._respond(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if not self._allowed_host():
            self.close_connection = True
            self._respond(403, {"error": "Loopback Host header required."})
            return
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + self.headers.get("Host", ""):
            self.close_connection = True
            self._respond(403, {"error": "Cross-origin controls are disabled."})
            return
        if urlsplit(self.path).path != "/api/control":
            self.close_connection = True
            self._respond(404, {"error": "Not found."})
            return
        if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            self._respond(415, {"error": "Send a bounded application/json body."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                self.close_connection = True
                raise ValueError("Request body must contain 1 to 4096 bytes.")
            command = json.loads(self.rfile.read(length))
            self._respond(200, self.server.controller.control(command))
        except (ValueError, UnicodeDecodeError) as exc:
            self._respond(400, {"error": str(exc)[:300]})
        except LiveError as exc:
            self._respond(502, {"error": str(exc)[:300]})

    def do_OPTIONS(self) -> None:
        self._respond(403, {"error": "Cross-origin access is disabled."})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / "npu-sim")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--profile", choices=PROFILES, default="balanced")
    parser.add_argument("--autostart", action="store_true", help="start continuous inference immediately")
    parser.add_argument("--open-browser", action="store_true", help="open the local dashboard if a browser is available")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("Port must be from 0 to 65535.")
    controller = Controller(args.binary)
    server = None
    browser_timer = None
    previous_signals = {}
    try:
        server = DashboardServer(("127.0.0.1", args.port), controller)
        if args.autostart:
            controller.control({"action": "start", "profile": args.profile})
        url = f"http://127.0.0.1:{server.server_address[1]}"
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.signal(
                signum, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        print(f"Live SystemC NPU simulator: {url}", flush=True)
        if args.open_browser:
            def open_browser():
                import webbrowser
                try:
                    if not webbrowser.open(url, new=2):
                        print(f"Open {url} in your browser.", flush=True)
                except (OSError, webbrowser.Error):
                    print(f"Open {url} in your browser.", flush=True)
            # Let serve_forever begin before launching an external browser.
            browser_timer = threading.Timer(0.3, open_browser)
            browser_timer.daemon = True
            browser_timer.start()
        server.serve_forever(poll_interval=0.2)
    finally:
        if browser_timer is not None:
            browser_timer.cancel()
        if server is not None:
            server.server_close()
        controller.close()
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
