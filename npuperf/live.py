"""Serve the live dashboard on localhost and drive npu-sim over its --live-stdio protocol.

The HTTP API selects built-in workload profiles only; it never accepts a command, path,
executable or address. GET /api/state returns the last published snapshot and never waits
on the simulator process.
"""
from __future__ import annotations

import os
import random
import signal
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from urllib.parse import urlsplit
import json

from . import analytic, cli, sim
from .binary import ROOT
from .sim import LiveError

PROFILE_ARGS = {
    "balanced": ["--m", "64", "--n", "64", "--k", "128", "--tile-m", "8", "--tile-n", "8",
                 "--tile-k", "32", "--buffer-slots", "2"],
    "compute-bound": ["--m", "64", "--n", "64", "--k", "128", "--tile-m", "8", "--tile-n", "8",
                      "--tile-k", "32", "--buffer-slots", "2", "--macs-per-cycle", "4",
                      "--memory-bytes-per-cycle", "128"],
    "memory-bound": ["--m", "64", "--n", "64", "--k", "128", "--tile-m", "8", "--tile-n", "8",
                     "--tile-k", "32", "--buffer-slots", "2", "--macs-per-cycle", "256",
                     "--memory-bytes-per-cycle", "4", "--memory-latency", "24"],
    "palette-weights": ["--tile-m", "8", "--tile-n", "8", "--tile-k", "32", "--buffer-slots", "2",
                        "--weight-bits", "8", "--weight-encoding", "palette", "--palette-bits", "2",
                        "--input-zero", "0", "--weight-zero", "0", "--shift", "4"],
}
PROFILES = {
    "balanced": {"label": "Default machine",
                 "description": "64x64x128 INT8 GEMM, 8x8x32 tiles, 2 slots, 64 MAC/cycle, "
                                "16 B/cycle."},
    "compute-bound": {"label": "4 MAC/cycle",
                      "description": "The default GEMM with 4 MACs per cycle and 128 B/cycle."},
    "memory-bound": {"label": "4 B/cycle",
                     "description": "The default GEMM with 256 MACs per cycle, 4 B/cycle and "
                                    "24-cycle request latency."},
    "palette-weights": {"label": "2-bit palette weights",
                        "description": "Weights stored as 2-bit indices into a 4-entry LUT; "
                                       "decode cycles charged."},
}
MAX_STEP = 10000


def profile_bound(name: str) -> str | None:
    """Analytic bound class of a dense profile (compute, bandwidth or latency)."""
    try:
        return analytic.analyze(analytic.Design.from_flags(sim.from_flags(PROFILE_ARGS[name]))).bound
    except ValueError:
        return None


class Controller:
    """One owned simulator child, stepped by a wall-clock scheduler."""

    def __init__(self, binary: Path, tick_seconds: float = 0.025, slice_cycles: int = 256):
        self.binary = Path(binary).resolve(strict=True)
        self.tick_seconds = tick_seconds
        self.slice_cycles = slice_cycles
        self.lock = threading.RLock()
        self.view_lock = threading.Lock()
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.session: sim.LiveSession | None = None
        self.state = "idle"
        self.profile = "balanced"
        self.telemetry: dict = {}
        self.history: deque = deque(maxlen=8)
        self.jobs_completed = 0
        self.total_cycles = 0
        self.total_macs = 0
        self.error = ""
        self.started_at: float | None = None
        self._published: dict = {}
        self._publish()
        self.thread = threading.Thread(target=self._schedule, name="systemc-scheduler",
                                       daemon=True)
        self.thread.start()

    @property
    def process(self):
        return self.session.process if self.session is not None else None

    def _publish(self) -> None:
        view = {"kind": "SystemC architectural simulator", "state": self.state,
                "profile": self.profile, "slice_cycles": self.slice_cycles,
                "jobs_completed": self.jobs_completed, "completed_job_cycles": self.total_cycles,
                "completed_job_macs": self.total_macs, "telemetry": self.telemetry,
                "recent_jobs": list(self.history), "error": self.error}
        with self.view_lock:
            self._published = view

    def _launch(self) -> None:
        self._terminate()
        seed = self.jobs_completed + 1
        # The palette profile reads a fixed-shape tensor file that this process writes;
        # HTTP clients never supply tensors or paths.
        fixture = (tempfile.TemporaryDirectory(prefix="npu-live-palette-")
                   if self.profile == "palette-weights" else None)
        try:
            flags = list(PROFILE_ARGS[self.profile])
            if fixture is not None:
                rng = random.Random(seed)
                path = Path(fixture.name) / "weights.npu"
                values = [rng.randint(-8, 8) for _ in range(64 * 128)]
                values += [rng.choice((-3, -1, 1, 3)) for _ in range(128 * 64)]
                values += [rng.randint(-31, 31) for _ in range(64)]
                path.write_text("NPU1 64 64 128\n" + " ".join(map(str, values)) + "\n",
                                encoding="ascii")
                flags += ["--input", str(path)]
            self.session = sim.LiveSession(self.binary, [*flags, "--seed", str(seed)])
            self.telemetry = self.session.receive()  # first snapshot: all files read
        finally:
            if fixture is not None:
                fixture.cleanup()
        self.started_at = time.monotonic()

    def _advance(self, cycles: int) -> None:
        if self.session is None:
            raise LiveError("Start a workload before advancing the simulator.")
        value = self.session.advance(cycles)
        self.telemetry = value
        if value.get("completed", False):
            if value.get("validated") is not True:
                raise LiveError("Completed inference did not match the scalar reference.")
            self.jobs_completed += 1
            self.total_cycles += value["cycles"]
            self.total_macs += value["macs"]
            self.history.appendleft({"job": self.jobs_completed, "profile": self.profile,
                                     "telemetry": value})

    def _terminate(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            session.close()

    def _schedule(self) -> None:
        while not self.closed.is_set():
            self.wake.wait(self.tick_seconds)
            self.wake.clear()
            with self.lock:
                if self.state != "running" or self.closed.is_set():
                    continue
                try:
                    # The next job starts only after the previous one completed and its
                    # output matched the reference.
                    if self.telemetry.get("completed", False):
                        self._launch()
                    self._advance(self.slice_cycles)
                except (LiveError, OSError) as exc:
                    self.error = str(exc)[:300]
                    self.state = "error"
                    self._terminate()
                self._publish()

    def snapshot(self) -> dict:
        with self.view_lock:
            return dict(self._published)

    def control(self, command: dict) -> dict:
        if not isinstance(command, dict) or set(command) - {"action", "profile", "cycles"}:
            raise ValueError("Only action, profile and cycles are accepted.")
        action = command.get("action")
        if not isinstance(action, str) or action not in {"start", "pause", "resume", "step",
                                                         "stop", "reset"}:
            raise ValueError("Unknown control action.")
        if "profile" in command and (action not in {"start", "reset"}
                                     or not isinstance(command["profile"], str)
                                     or command["profile"] not in PROFILES):
            raise ValueError("Choose a built-in workload profile for start or reset.")
        if "cycles" in command and action != "step":
            raise ValueError("Only step accepts a cycle count.")
        cycles = command.get("cycles", self.slice_cycles)
        if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= MAX_STEP:
            raise ValueError(f"Step cycles must be an integer from 1 to {MAX_STEP}.")
        with self.lock:
            try:
                if action in {"start", "reset"}:
                    self.profile = command.get("profile", self.profile)
                    self.error = ""
                    self.jobs_completed = self.total_cycles = self.total_macs = 0
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
                self._publish()
                raise LiveError(self.error) from exc
            self._publish()
            return self.snapshot()

    def close(self) -> None:
        self.closed.set()
        self.wake.set()
        self.thread.join(timeout=12)
        with self.lock:
            self.state = "stopped"
            self._terminate()
            self._publish()


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    # POSIX: rebind while old connections sit in TIME_WAIT after a restart. Windows'
    # SO_REUSEADDR would let a second server take a port that is already listening.
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # HTTPServer.server_bind does a reverse DNS lookup; a loopback server needs none and
        # a slow resolver would delay startup.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address) -> None:
        # Browsers drop keep-alive connections at will; that is not an error worth a traceback.
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)

    def __init__(self, address: tuple[str, int], controller: Controller | None = None):
        # run.py binds before building, so the controller can arrive later (see serve()).
        self.controller = controller  # type: ignore[assignment]
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
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; img-src 'self'; object-src 'none'; "
                         "base-uri 'none'; frame-ancestors 'none'")
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
            self._respond(200, {name: {**info, "bound": profile_bound(name)}
                                for name, info in PROFILES.items()})
        elif path in {"/", "/index.html", "/app.js", "/style.css"}:
            name = "index.html" if path == "/" else path[1:]
            content_type = {"index.html": "text/html; charset=utf-8",
                            "app.js": "text/javascript; charset=utf-8",
                            "style.css": "text/css; charset=utf-8"}[name]
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
            self._respond(415, {"error": "Send an application/json body of at most 4096 bytes."})
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


def serve(binary: Path, port: int, profile: str = "balanced", autostart: bool = False,
          open_browser: bool = False, bound: DashboardServer | None = None) -> int:
    controller = Controller(binary)
    server = bound
    browser_timer = None
    previous_signals = {}
    try:
        if server is None:
            try:
                server = DashboardServer(("127.0.0.1", port), controller)
            except OSError as error:
                raise RuntimeError(f"cannot listen on 127.0.0.1:{port}: "
                                   f"{error.strerror or error}; choose another --port") from error
        else:
            server.controller = controller
        if autostart:
            controller.control({"action": "start", "profile": profile})
        url = f"http://127.0.0.1:{server.server_address[1]}"
        stopping = server
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.signal(
                signum, lambda *_: threading.Thread(target=stopping.shutdown, daemon=True).start())
        print(f"Dashboard: {url} (Ctrl+C to stop)", flush=True)
        if open_browser:
            def launch_browser() -> None:
                import webbrowser
                try:
                    if not webbrowser.open(url, new=2):
                        print(f"Open {url} in a browser.", flush=True)
                except (OSError, webbrowser.Error):
                    print(f"Open {url} in a browser.", flush=True)
            # Let serve_forever start before the browser connects.
            browser_timer = threading.Timer(0.3, launch_browser)
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


def port_number(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 65535:
        raise ValueError("port must be from 0 to 65535")
    return value


@cli.main_guard("live_dashboard")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("live_dashboard", __doc__)
    cli.add_binary_options(parser, timeout=False)
    parser.add_argument("--port", type=port_number, default=8765,
                        help="loopback port; 0 picks a free one")
    parser.add_argument("--profile", choices=PROFILES, default="balanced",
                        help="workload started by --autostart")
    parser.add_argument("--autostart", action="store_true", help="start running immediately")
    parser.add_argument("--open-browser", action="store_true", help="open the dashboard URL")
    args = parser.parse_args(argv)
    return serve(cli.simulator(args), args.port, args.profile, args.autostart, args.open_browser)


if __name__ == "__main__":
    raise SystemExit(main())
