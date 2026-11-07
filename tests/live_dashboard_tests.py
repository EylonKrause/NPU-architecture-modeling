#!/usr/bin/env python3
"""Exercise the dashboard against the compiled npu-sim binary."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import http.client
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf import analytic
from npuperf import live as dashboard
from npuperf import sim
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

    def test_pause_step_resume_stop(self):
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
        self.assertEqual(state["completed_job_executed_macs"],
                         sum(j["telemetry"]["executed_macs"] for j in (first, second)))
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
            result = state["recent_jobs"][0]["telemetry"]
            self.assertTrue(result["validated"], profile)
            # A schema_version 2 snapshot echoes the complete design, so the machine panel
            # describes what actually ran rather than the profile's flags.
            self.assertEqual(state["machine"]["source"], "snapshot", profile)
            self.assertEqual(state["machine"]["cores"],
                             result["config"]["arch"]["cores"], profile)
            analysis = state["machine"]["analysis"]
            if analysis is None:
                continue
            # A lower bound of the simulated machine: no run may beat it.
            self.assertGreaterEqual(result["cycles"], analysis["lower_bound_cycles"], profile)
            if profile == "compute-bound":
                # The simulator meets the bound of the design its snapshot echoes (two panel
                # buffers per slot) exactly on the compute-bound profile.
                self.assertEqual(result["cycles"], analysis["lower_bound_cycles"])

    def test_palette_decode_varies_inputs_and_cleans_fixture(self):
        self.controller.control({"action": "reset", "profile": "palette-weights"})
        command = self.controller.process.args
        fixture = Path(command[command.index("--input") + 1])
        self.assertFalse(fixture.exists())  # First snapshot acknowledges all input reads.
        for _ in range(100):
            state = self.controller.control({"action": "step", "cycles": 10000})
            if state["jobs_completed"] == 2:
                break
        self.assertEqual(state["jobs_completed"], 2)
        first, second = [job["telemetry"] for job in state["recent_jobs"][:2]]
        for result in (first, second):
            self.assertTrue(result["validated"])
            self.assertEqual(sim.config_value(result["config"], "weight_encoding"), "palette")
            self.assertEqual(sim.config_value(result["config"], "weight_bits"), 8)
            self.assertEqual(sim.config_value(result["config"], "palette_bits"), 2)
            # schema_version 2 snapshots nest the weight format under "weights".
            self.assertEqual(result["schema_version"], 2)
            weights = result["config"]["weights"]
            self.assertEqual((weights["encoding"], weights["bits"], weights["palette_bits"]),
                             ("palette", 8, 2))
            self.assertLess(result["encoded_weight_storage_bytes"], result["dense_weight_storage_bytes"])
            self.assertGreater(result["weight_decode_cycles"], 0)
            self.assertEqual(result["executed_macs"], result["macs"])
            self.assertEqual(result["skipped_macs"], 0)
        self.assertNotEqual(first["input_preview"], second["input_preview"])
        self.assertNotEqual(first["output_hash"], second["output_hash"])
        workload = dict(state["workload"])
        self.assertIn(workload.pop("groups"), (1, None))  # None: a build without --groups
        self.assertEqual(workload, {"m": 64, "n": 64, "k": 128, "weight_bits": 8,
                                    "weight_encoding": "palette", "frequency_mhz": 200})

    def test_slice_utilisation_is_the_last_advance(self):
        state = self.controller.control({"action": "reset", "profile": "balanced"})
        self.assertIsNone(state["slice"])
        before = state["telemetry"]
        for cycles in (300, 17, 1):
            state = self.controller.control({"action": "step", "cycles": cycles})
            after = state["telemetry"]
            expected = {key: after[key] - before[key]
                        for key in ("cycles", "compute_busy_cycles", "dma_busy_cycles")}
            self.assertEqual({key: state["slice"][key] for key in expected}, expected)
            self.assertEqual(expected["cycles"], cycles)
            for name in ("compute", "dma"):
                self.assertEqual(state["slice"][f"{name}_busy_fraction"],
                                 expected[f"{name}_busy_cycles"] / cycles)
            before = after
        # Paused: every poll returns the last step's slice, not the whole job so far.
        time.sleep(0.05)
        self.assertEqual(self.controller.snapshot()["slice"], state["slice"])
        whole_job = after["compute_busy_cycles"] / after["cycles"]
        self.assertNotEqual(state["slice"]["compute_busy_fraction"], whole_job)
        self.assertEqual(state["workload"]["weight_encoding"], "dense")

    def test_slice_delta_and_workload_summary(self):
        before = {"cycles": 10, "compute_busy_cycles": 4, "dma_busy_cycles": 10}
        after = {"cycles": 30, "compute_busy_cycles": 19, "dma_busy_cycles": 20}
        self.assertEqual(dashboard.slice_delta(before, after),
                         {"cycles": 20, "compute_busy_cycles": 15, "dma_busy_cycles": 10,
                          "compute_busy_fraction": 0.75, "dma_busy_fraction": 0.5})
        for stale in (before, {**after, "cycles": 5}, {"cycles": 40}):
            with self.subTest(stale=stale):
                self.assertIsNone(dashboard.slice_delta(before, stale))
        nested = {"shape": {"m": 4, "n": 5, "k": 6, "groups": 2, "seed": 1},
                  "arch": {"frequency_mhz": 500},
                  "weights": {"bits": 4, "encoding": "sparse", "palette_bits": 4}}
        flat = {"m": 4, "n": 5, "k": 6, "groups": 2, "frequency_mhz": 500, "weight_bits": 4,
                "weight_encoding": "sparse"}
        expected = {"m": 4, "n": 5, "k": 6, "groups": 2, "weight_bits": 4,
                    "weight_encoding": "sparse", "frequency_mhz": 500}
        self.assertEqual(dashboard.workload_summary(nested), expected)
        self.assertEqual(dashboard.workload_summary(flat), expected)
        self.assertEqual(set(dashboard.workload_summary(None).values()), {None})

    def test_page_reads_only_published_fields(self):
        # Every controller field app.js reads must be one the controller publishes.
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("configValue", script)
        read = set(re.findall(r"\bworkload[.](\w+)", script))
        for unpacked in re.findall(r"const [{]([^}]*)[}] = workload", script):
            read.update(name.strip() for name in unpacked.split(","))
        self.assertTrue(read)
        self.assertLessEqual(read, set(dashboard.WORKLOAD_KEYS))
        machine = set(re.findall(r"\bmachine[.](\w+)", script))
        self.assertIn("macs_per_cycle", machine)
        self.assertLessEqual(machine, set(dashboard.MACHINE_KEYS))
        analysis = set(re.findall(r"\banalysis[.](\w+)", script))
        self.assertIn("lower_bound_cycles", analysis)
        self.assertNotIn("resource_bound_cycles", analysis)
        self.assertLessEqual(analysis, set(dashboard.ANALYSIS_KEYS))

        self.controller.control({"action": "reset", "profile": "balanced"})
        state = self.controller.control({"action": "step", "cycles": 300})
        slice_keys = set(state["slices"][-1])
        series = set(re.findall(r'key: "(\w+)", className', script))
        self.assertEqual(series, {"compute_busy_fraction", "dma_busy_fraction"})
        self.assertLessEqual(series | set(re.findall(r"\bslice[.](\w+)", script)), slice_keys)
        self.assertEqual(set(re.findall(r"\btransition[.](\w+)", script)),
                         {"job", "cycle", "unsampled_from"})
        self.assertIn('const STAGES = ["load", "compute", "store"];', script)
        self.assertLessEqual({"job", "cycle", "load", "compute", "store"},
                             set(state["timeline"]["transitions"][-1]))
        for key in ("slices", "timeline", "machine", "max_step", "completed_job_executed_macs"):
            self.assertIn(f"value.{key}", script)
            self.assertIn(key, state)
        # "% of peak" divides executed MACs by the array peak; the roofline keeps all MACs.
        self.assertIn("executedRate: ratio(t.executed_macs, t.cycles)", script)
        self.assertEqual(len(re.findall(r"ratio\(\w+[.]executedRate, machine[.]macs_per_cycle\)",
                                        script)), 2)

    def test_machine_slices_and_timeline(self):
        state = self.controller.control({"action": "reset", "profile": "balanced"})
        self.assertEqual(state["slices"], [])
        self.assertEqual(state["timeline"]["transitions"],
                         [{"job": 1, "cycle": 0, "load": None, "compute": None, "store": None}])
        self.assertEqual(state["max_step"], dashboard.MAX_STEP)
        machine = state["machine"]
        self.assertEqual(set(machine), set(dashboard.MACHINE_KEYS))
        self.assertEqual(set(machine["analysis"]), set(dashboard.ANALYSIS_KEYS))
        # The schema_version 2 snapshot config echoes the complete design, so the machine
        # comes from the snapshot, with the simulator's default of two panel buffers per slot.
        self.assertEqual(machine["source"], "snapshot")
        self.assertEqual(machine["panel_buffers"],
                         state["telemetry"]["config"]["mapping"]["panel_buffers"])
        design = analytic.Design.from_flags({**sim.from_flags(dashboard.PROFILE_ARGS["balanced"]),
                                             "panel_buffers": 2})
        expected = analytic.analyze(design)
        self.assertEqual((machine["macs_per_cycle"], machine["bytes_per_cycle"], machine["ridge"]),
                         (64, 16.0, 4.0))
        for key in ("lower_bound_cycles", "resource_bound_cycles", "estimate_cycles"):
            self.assertEqual(machine["analysis"][key], getattr(expected, key), key)
        self.assertEqual(machine["analysis"]["bound"], expected.bound)
        self.assertEqual(machine["analysis"]["effective_bytes_per_cycle"],
                         expected.dram_bytes / expected.memory_cycles)

        self.assertEqual(state["timeline"]["stride_cycles"], dashboard.TIMELINE_STRIDE)
        for cycles in (300, 1, 2000):
            before = state["telemetry"]
            state = self.controller.control({"action": "step", "cycles": cycles})
            last = state["slices"][-1]
            self.assertEqual(last["job"], 1)
            self.assertEqual(last["end_cycle"], state["telemetry"]["cycles"])
            self.assertEqual(last["cycles"], cycles)
            for key in ("executed_macs", "read_bytes", "write_bytes"):
                self.assertEqual(last[key], state["telemetry"][key] - before[key])
            # The snapshot has no per-direction DMA counters, so a slice carries none.
            self.assertFalse({"dma_read_busy_cycles", "dma_write_busy_cycles"} & set(last))
            # Sampled transitions sit on a stride grid from the start of the sampled window
            # (the step's last cycle may close a shorter sample); the cycles before that
            # window are one unsampled interval.
            window_start = max(before["cycles"], state["telemetry"]["cycles"]
                               - dashboard.TIMELINE_WINDOW)
            new = [t for t in state["timeline"]["transitions"] if t["cycle"] > before["cycles"]]
            gaps = [t for t in new if "unsampled_from" in t]
            if cycles > dashboard.TIMELINE_WINDOW:
                self.assertEqual([(t["unsampled_from"], t["cycle"]) for t in gaps],
                                 [(before["cycles"], window_start)])
            else:
                self.assertEqual(gaps, [])
            end = state["telemetry"]["cycles"]
            for transition in new:
                if "unsampled_from" not in transition and transition["cycle"] != end:
                    self.assertEqual((transition["cycle"] - window_start)
                                     % dashboard.TIMELINE_STRIDE, 0, transition)
        self.assertEqual(len(state["slices"]), 3)

        transitions = state["timeline"]["transitions"]
        self.assertGreater(len(transitions), 10)
        tiles = state["telemetry"]["tiles_total"]
        stages = ("load", "compute", "store")
        for previous, current in zip(transitions, transitions[1:]):
            self.assertLess(previous["cycle"], current["cycle"])
            if "unsampled_from" in current:
                # The stages held from the previous transition up to the last sample.
                self.assertLessEqual(previous["cycle"], current["unsampled_from"])
                self.assertLess(current["unsampled_from"], current["cycle"])
            else:  # a sampled transition is recorded only when a stage changed
                self.assertNotEqual([previous[k] for k in stages], [current[k] for k in stages])
            for stage in stages:
                self.assertTrue(current[stage] is None or 0 <= current[stage] < tiles)
            self.assertFalse(current["load"] is not None and current["store"] is not None)
        self.assertTrue(any(t["compute"] is not None for t in transitions))
        self.assertTrue(any(t["store"] is not None for t in transitions))

        for _ in range(dashboard.SLICE_HISTORY + 5):
            state = self.controller.control({"action": "step", "cycles": 1})
        self.assertEqual(len(state["slices"]), dashboard.SLICE_HISTORY)

        state = self.controller.control({"action": "reset", "profile": "palette-weights"})
        self.assertIsNone(state["machine"]["analysis"])
        self.assertEqual(state["machine"]["macs_per_cycle"], 64)
        self.assertEqual(len(state["timeline"]["transitions"]), 1)

    def test_machine_summary(self):
        # SNAPSHOT_DEFAULTS are the values from_result assumes for the keys it lets a
        # result config omit.
        partial = {key: value for key, value in dataclasses.asdict(analytic.Design()).items()
                   if key not in dashboard.SNAPSHOT_DEFAULTS}
        design = analytic.Design.from_result({"config": partial})
        self.assertEqual({key: getattr(design, key) for key in dashboard.SNAPSHOT_DEFAULTS},
                         dashboard.SNAPSHOT_DEFAULTS)

        full = dataclasses.asdict(analytic.Design(macs_per_cycle=128, memory_bytes_per_cycle=32))
        machine = dashboard.machine_summary("balanced", {"config": full})
        self.assertEqual(machine["source"], "snapshot")
        self.assertEqual((machine["macs_per_cycle"], machine["bytes_per_cycle"], machine["ridge"]),
                         (128, 32.0, 4.0))
        older = {"config": {"m": 32, "n": 64, "k": 128, "tile_m": 8, "tile_n": 8, "tile_k": 32,
                            "weight_bits": 8, "frequency_mhz": 500, "weight_encoding": "dense"}}
        machine = dashboard.machine_summary("memory-bound", older)
        self.assertEqual(machine["source"], "profile")
        self.assertEqual((machine["macs_per_cycle"], machine["bytes_per_cycle"],
                          machine["memory_latency"], machine["frequency_mhz"]), (256, 4.0, 24, 500))
        flags = {**sim.from_flags(dashboard.PROFILE_ARGS["memory-bound"]), "m": 32,
                 "frequency_mhz": 500, "panel_buffers": 1}
        expected = analytic.analyze(analytic.Design.from_flags(flags))
        self.assertEqual(machine["analysis"]["lower_bound_cycles"], expected.lower_bound_cycles)
        # A config that echoes panel_buffers overrides the single-buffer default.
        echoed = {"config": {**older["config"], "panel_buffers": 2}}
        design = dashboard.profile_design("balanced", echoed["config"])
        self.assertEqual((design.panel_buffers, design.m), (2, 32))
        self.assertEqual(dashboard.profile_design("balanced").panel_buffers, 1)

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
        # Startup must work even if the machine's reverse resolver is broken.
        # HTTPServer's default bind performs this unnecessary lookup and can
        # otherwise hang before the launcher can announce its loopback URL.
        with mock.patch("socket.getfqdn", side_effect=OSError("DNS unavailable")) as reverse_dns:
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
            reverse_dns.assert_not_called()
            self.assertEqual(server.server_name, "127.0.0.1")
            self.assertEqual(server.server_port, port)
            self.assertGreater(port, 0)
            for path in ("/", "/app.js", "/style.css", "/api/state", "/api/profiles"):
                status, headers, content = request("GET", path)
                self.assertEqual(status, 200, path)
                self.assertGreater(len(content), 0)
                self.assertNotIn("Access-Control-Allow-Origin", headers)
                self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
                if not path.startswith("/api/"):
                    # No CDNs, fonts or other hosts: the page works offline. The SVG
                    # namespace is an identifier, not a fetch.
                    local = content.replace(b"http://www.w3.org/2000/svg", b"")
                    self.assertIsNone(re.search(rb"https?://", local), path)
            # The profile list comes from /api/profiles, not from options in the page.
            profiles = json.loads(request("GET", "/api/profiles")[2])
            self.assertEqual(list(profiles), list(dashboard.PROFILES))
            for name, info in profiles.items():
                self.assertEqual(set(info), {"label", "description", "bound"}, name)
                self.assertEqual(info["bound"], dashboard.profile_bound(name))
            page = request("GET", "/")[2].decode()
            select = re.search(r'<select id="profile"[^>]*>(.*?)</select>', page, re.S).group(1)
            self.assertEqual(select.strip(), "")
            self.assertIn('fetch("/api/profiles"', request("GET", "/app.js")[2].decode())
            idle = json.loads(request("GET", "/api/state")[2])
            self.assertLessEqual({"machine", "slices", "timeline", "max_step"}, set(idle))
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


class DashboardCliTests(unittest.TestCase):
    def run_main(self, argv):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            status = dashboard.main(argv)
        return status, stderr.getvalue()

    def test_missing_binary_is_one_line_error(self):
        status, stderr = self.run_main(["--binary", str(ROOT / "does-not-exist" / "npu-sim")])
        self.assertEqual(status, 2)
        self.assertTrue(stderr.startswith("live_dashboard: "), stderr)
        self.assertEqual(len(stderr.strip().splitlines()), 1)
        self.assertNotIn("Traceback", stderr)

    def test_busy_port_is_one_line_error(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            status, stderr = self.run_main(["--binary", str(BINARY), "--port", str(port)])
        self.assertEqual(status, 1)
        self.assertTrue(stderr.startswith(f"live_dashboard: cannot listen on 127.0.0.1:{port}"),
                        stderr)
        self.assertNotIn("Traceback", stderr)

    def test_script_rejects_bad_port_with_status_two(self):
        run = subprocess.run([sys.executable, str(ROOT / "scripts" / "live_dashboard.py"),
                              "--port", "70000"], capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 2)
        self.assertTrue(run.stderr.startswith("live_dashboard: "), run.stderr)

    def test_state_does_not_wait_for_the_simulator_lock(self):
        controller = dashboard.Controller(BINARY, tick_seconds=0.01, slice_cycles=64)
        try:
            controller.control({"action": "reset", "profile": "balanced"})
            result = {}
            with controller.lock:  # the scheduler holds this while it talks to the child
                reader = threading.Thread(target=lambda: result.update(controller.snapshot()))
                reader.start()
                reader.join(timeout=2)
                self.assertFalse(reader.is_alive())
            self.assertEqual(result["state"], "paused")
        finally:
            controller.close()

    def test_profiles_report_their_analytic_bound(self):
        self.assertEqual(dashboard.profile_bound("compute-bound"), "compute")
        self.assertIn(dashboard.profile_bound("memory-bound"), ("bandwidth", "latency"))
        self.assertIsNone(dashboard.profile_bound("palette-weights"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / "npu-sim")
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining])
