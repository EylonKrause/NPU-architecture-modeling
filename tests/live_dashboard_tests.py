#!/usr/bin/env python3
"""Exercise the dashboard against the real compiled SystemC binary."""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
from pathlib import Path
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_dashboard", ROOT / "scripts" / "live_dashboard.py")
dashboard = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(dashboard)
BINARY: Path


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.controller = dashboard.Controller(BINARY, tick_seconds=0.01, slice_cycles=64)

    def tearDown(self):
        self.controller.close()

    def wait_for(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.controller.snapshot()
            if predicate(state):
                return state
            if state["error"]:
                self.fail(state["error"])
            time.sleep(0.01)
        self.fail("Model failed to reach the expected state before the deadline.")

    def test_real_simulation_pause_step_resume_stop(self):
        state = self.controller.control({"action": "start", "profile": "balanced"})
        self.assertEqual(state["state"], "running")
        state = self.wait_for(lambda s: s["telemetry"].get("cycles", 0) > 0)
        self.assertGreater(state["telemetry"]["cycles"], 0)
        state = self.controller.control({"action": "pause"})
        paused = state["telemetry"]["cycles"]
        time.sleep(0.06)
        self.assertEqual(self.controller.snapshot()["telemetry"]["cycles"], paused)
        state = self.controller.control({"action": "step", "cycles": 17})
        self.assertEqual(state["telemetry"]["cycles"], paused + 17)
        self.assertEqual(state["state"], "paused")
        self.controller.control({"action": "resume"})
        self.wait_for(lambda s: s["telemetry"]["cycles"] > paused + 17)
        process = self.controller.process
        state = self.controller.control({"action": "stop"})
        self.assertEqual(state["state"], "stopped")
        self.assertIsNone(self.controller.process)
        self.assertIsNotNone(process.poll())

    def test_full_output_validation_and_continuous_jobs(self):
        self.controller.control({"action": "reset", "profile": "balanced"})
        for _ in range(100):
            state = self.controller.control({"action": "step", "cycles": 10000})
            if state["jobs_completed"] == 2:
                break
        self.assertEqual(state["jobs_completed"], 2)
        first, second = state["recent_jobs"]
        for job in (first, second):
            result = job["telemetry"]
            self.assertTrue(result["completed"])
            self.assertTrue(result["validated"])
            self.assertEqual(result["tiles_completed"], result["tiles_total"])
            self.assertGreater(result["macs"], 0)
            self.assertGreater(result["read_bytes"], 0)
            self.assertGreater(result["write_bytes"], 0)
            self.assertIsInstance(result["output_hash"], str)
            self.assertTrue(result["output_hash"].isdigit())
        self.assertNotEqual(first["telemetry"]["output_hash"], second["telemetry"]["output_hash"])
        self.assertEqual(state["completed_job_cycles"], sum(j["telemetry"]["cycles"] for j in (first, second)))
        reset = self.controller.control({"action": "reset", "profile": "memory-bound"})
        self.assertEqual(reset["jobs_completed"], 0)
        self.assertEqual(reset["telemetry"]["cycles"], 0)
        self.assertEqual(reset["state"], "paused")

    def test_profiles_execute_and_validate(self):
        for profile in dashboard.PROFILES:
            self.controller.control({"action": "reset", "profile": profile})
            for _ in range(100):
                state = self.controller.control({"action": "step", "cycles": 10000})
                if state["jobs_completed"]:
                    break
            self.assertEqual(state["jobs_completed"], 1, profile)
            self.assertTrue(state["recent_jobs"][0]["telemetry"]["validated"], profile)

    def test_bounded_commands_and_state_transitions(self):
        for value in ({}, [], {"action": []}, {"action": "shell"},
                      {"action": "start", "profile": "../arbitrary"},
                      {"action": "start", "profile": []},
                      {"action": "start", "binary": "/bin/sh"},
                      {"action": "start", "cycles": 10},
                      {"action": "step", "cycles": True},
                      {"action": "step", "cycles": 10001},
                      {"action": "step", "cycles": -1},
                      {"action": "pause"}, {"action": "resume"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.controller.control(value)
        self.assertEqual(self.controller.snapshot()["state"], "idle")

    def test_http_assets_control_and_browser_boundaries(self):
        server = dashboard.DashboardServer(("127.0.0.1", 0), self.controller)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        def request(method, path, body=None, headers=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request(method, path, body, headers or {})
                response = connection.getresponse()
                return response.status, dict(response.getheaders()), response.read()
            finally:
                connection.close()

        try:
            for path in ("/", "/app.js", "/style.css", "/api/state", "/api/profiles"):
                status, headers, content = request("GET", path)
                self.assertEqual(status, 200, path)
                self.assertGreater(len(content), 0)
                self.assertNotIn("Access-Control-Allow-Origin", headers)
                self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertEqual(request("GET", "/../../etc/passwd")[0], 404)
            self.assertEqual(request("GET", "/api/state", headers={"Host": "attacker.invalid"})[0], 403)
            payload = json.dumps({"action": "reset", "profile": "balanced"})
            self.assertEqual(request("POST", "/api/control", payload,
                                     {"Content-Type": "application/json", "Origin": "https://attacker.invalid"})[0], 403)
            self.assertEqual(request("POST", "/api/control", payload,
                                     {"Content-Type": "text/plain"})[0], 415)
            self.assertEqual(request("OPTIONS", "/api/control")[0], 403)
            self.assertEqual(request("POST", "/api/control", "{" * 4097,
                                     {"Content-Type": "application/json"})[0], 400)
            self.assertEqual(request("POST", "/api/control", "not-json",
                                     {"Content-Type": "application/json"})[0], 400)
            status, _, content = request("POST", "/api/control", payload,
                                        {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(content)["state"], "paused")
            self.assertEqual(request("GET", "/api/control")[0], 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / "npu-sim")
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining])
