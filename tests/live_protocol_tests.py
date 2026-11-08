#!/usr/bin/env python3
"""npu-sim --live-stdio: command protocol, pause and step semantics against the SystemC
kernel, output files, and a final state equal to the batch run's result."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

BINARY: Path
WORKLOAD = [
    "--m",
    "4",
    "--n",
    "5",
    "--k",
    "7",
    "--tile-m",
    "2",
    "--tile-n",
    "3",
    "--tile-k",
    "3",
    "--buffer-slots",
    "2",
]


class LiveChild:
    """npu-sim --live-stdio child with timed reads; a reader thread makes the timeout work on
    Windows anonymous pipes."""

    def __init__(self, arguments: list[str]):
        self.error_file = tempfile.TemporaryFile(mode="w+t")  # noqa: SIM115
        self.process = subprocess.Popen(
            [str(BINARY), *WORKLOAD, *arguments, "--live-stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.error_file,
            text=True,
            bufsize=1,
        )
        self.responses: queue.Queue[str | None] = queue.Queue()

        def read_lines() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)

        self.reader = threading.Thread(target=read_lines, daemon=True)
        self.reader.start()

    def errors(self) -> str:
        self.error_file.flush()
        self.error_file.seek(0)
        return self.error_file.read()

    def read(self) -> dict:
        try:
            line = self.responses.get(timeout=5)
        except queue.Empty as error:
            raise AssertionError(
                f"live process did not respond within 5 seconds: {self.errors()}"
            ) from error
        if line is None:
            raise AssertionError(f"live process exited before a response: {self.errors()}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AssertionError(f"response is not an object: {line!r}")
        return value

    def command(self, text: str) -> dict:
        assert self.process.stdin is not None
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        return self.read()

    def __enter__(self) -> LiveChild:
        return self

    def __exit__(self, *_) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.reader.join(timeout=1)
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.error_file.close()


class LiveProtocolChecks(unittest.TestCase):
    def test_output_paths_cannot_alias_other_outputs_or_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "same.vcd"
            destination.write_text("existing contents\n", encoding="utf-8")
            combinations = [
                ["--vcd", destination, "--progress-jsonl", destination],
                ["--vcd", destination, "--trace", destination],
                ["--progress-jsonl", destination, "--trace", destination],
                ["--progress-jsonl", destination, "--input", destination],
                ["--progress-jsonl", destination, "--quantization-file", destination],
            ]
            alias = Path(directory) / "hardlink.vcd"
            os.link(destination, alias)
            combinations.append(["--vcd", destination, "--progress-jsonl", alias])
            for arguments in combinations:
                with self.subTest(arguments=arguments):
                    run = subprocess.run(
                        [str(BINARY), *map(str, arguments)],
                        text=True,
                        capture_output=True,
                        timeout=10,
                    )
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("conflict", run.stderr)
                    self.assertEqual(destination.read_text(), "existing contents\n")

    def batch(self, arguments: list[str] | None = None) -> dict:
        process = subprocess.run(
            [str(BINARY), *WORKLOAD, "--dump-output", *(arguments or [])],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def assert_final(self, current: dict, batch: dict) -> None:
        self.assertTrue(current["simulation"])
        self.assertTrue(current["completed"])
        self.assertTrue(current["validated"])
        self.assertIs(current["numerical_match"], True)
        self.assertEqual(str(current["output_hash"]), str(batch["output_hash"]))
        self.assertEqual(current["output_preview"], batch["outputs"][:16])
        self.assertEqual(current["tiles_completed"], current["tiles_total"])
        self.assertEqual(current["panels_completed"], current["panels_total"])
        self.assertEqual(current["loaded_panels"], current["panels_total"])
        self.assertEqual(current["inflight_tiles"], 0)
        self.assertEqual(current["active_stages"], [])
        for name in (
            "cycles",
            "macs",
            "read_bytes",
            "write_bytes",
            "compute_busy_cycles",
            "dma_busy_cycles",
        ):
            self.assertEqual(current[name], batch[name], name)

    def test_pause_step_and_completion_follow_the_kernel(self):
        batch = self.batch()
        with LiveChild([]) as child:
            initial = child.read()
            self.assertEqual(initial["type"], "snapshot")
            self.assertEqual(initial["cycles"], 0)
            self.assertFalse(initial["completed"])
            self.assertIsNone(initial["numerical_match"])
            self.assertIsNone(initial["output_hash"])
            self.assertEqual(initial["output_preview"], [])
            time.sleep(0.08)
            self.assertEqual(
                child.command("snapshot"),
                initial,
                "host waiting must not advance the SystemC kernel",
            )
            current = child.command("advance 1")
            self.assertEqual(current["cycles"], 1)
            self.assertFalse(current["completed"])
            self.assertIsNone(current["numerical_match"])
            time.sleep(0.08)
            self.assertEqual(child.command("snapshot"), current)
            seen_dma = seen_compute = False
            deadline = time.monotonic() + 15
            previous = current
            while not current["completed"]:
                self.assertLess(time.monotonic(), deadline, "live run did not complete in 15 s")
                current = child.command("advance 3")
                self.assertGreaterEqual(current["cycles"], previous["cycles"])
                for name in (
                    "macs",
                    "read_bytes",
                    "write_bytes",
                    "tiles_completed",
                    "loaded_panels",
                    "panels_completed",
                    "compute_busy_cycles",
                    "dma_busy_cycles",
                ):
                    self.assertGreaterEqual(current[name], previous[name], name)
                self.assertLessEqual(current["inflight_tiles"], 2)
                # A loaded panel holds one of its slot's two panel buffers
                # until compute finishes it.
                self.assertLessEqual(
                    current["loaded_panels"] - current["panels_completed"],
                    2 * current["inflight_tiles"],
                )
                seen_dma |= current["dma_busy"]
                seen_compute |= current["compute_busy"]
                if not current["completed"]:
                    self.assertIsNone(current["numerical_match"])
                previous = current
            self.assertTrue(seen_dma)
            self.assertTrue(seen_compute)
            self.assert_final(current, batch)
            self.assertEqual(
                child.command("advance 100"), current, "a completed model must not keep advancing"
            )
            stopped = child.command("stop")
            self.assertEqual(stopped, dict(current, type="stopped"))
            self.assertEqual(child.process.wait(timeout=3), 0, child.errors())

    def test_single_panel_buffer_steps_to_its_batch_result(self):
        single, double = self.batch(["--panel-buffers", "1"]), self.batch()
        # K = 7 in panels of 3: one buffer serializes each panel load behind
        # the previous panel's compute.
        self.assertGreater(single["cycles"], double["cycles"])
        self.assertGreater(single["panel_wait_cycles"], 0)
        with LiveChild(["--panel-buffers", "1"]) as child:
            current = child.read()
            self.assertEqual(current["config"]["mapping"]["panel_buffers"], 1)
            deadline = time.monotonic() + 15
            while not current["completed"]:
                self.assertLess(time.monotonic(), deadline, "live run did not complete in 15 s")
                current = child.command("advance 5")
                self.assertLessEqual(
                    current["loaded_panels"] - current["panels_completed"],
                    current["inflight_tiles"],
                )
            self.assert_final(current, single)
            self.assertEqual(child.command("stop")["type"], "stopped")
            self.assertEqual(child.process.wait(timeout=3), 0, child.errors())

    def test_invalid_commands_do_not_advance_or_terminate(self):
        with LiveChild([]) as child:
            initial = child.read()
            for command in (
                "",
                "nonsense",
                "snapshot extra",
                "stop extra",
                "advance",
                "advance 0",
                "advance -1",
                "advance 10000001",
                "advance 1x",
                "advance 1 trailing",
                "x" * 257,
                "x" * 256,
            ):
                with self.subTest(command=command[:40]):
                    response = child.command(command)
                    self.assertEqual(response["type"], "error")
                    self.assertTrue(response["error"])
                    self.assertEqual(child.command("snapshot"), initial)
            self.assertEqual(child.command("stop")["type"], "stopped")
            self.assertEqual(child.process.wait(timeout=3), 0, child.errors())

    def test_clean_eof_while_paused(self):
        with LiveChild([]) as child:
            self.assertFalse(child.read()["completed"])
            assert child.process.stdin is not None
            child.process.stdin.close()
            self.assertEqual(child.process.wait(timeout=3), 0, child.errors())

    def test_automatic_progress_and_native_vcd_preserve_batch_result(self):
        baseline = self.batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            progress, waveform = path / "progress.jsonl", path / "npu.vcd"
            automatic = self.batch(
                ["--progress-jsonl", str(progress), "--slice-cycles", "7", "--vcd", str(waveform)]
            )
            for key in baseline:
                if key != "simulation_wall_seconds":
                    self.assertEqual(automatic[key], baseline[key], key)
            snapshots = [json.loads(line) for line in progress.read_text().splitlines()]
            self.assertGreater(len(snapshots), 2)
            self.assertEqual(snapshots[0]["cycles"], 0)
            self.assertEqual(
                [s["cycles"] for s in snapshots], sorted(s["cycles"] for s in snapshots)
            )
            self.assertTrue(all(s["type"] == "snapshot" for s in snapshots))
            self.assertTrue(all(s["numerical_match"] is None for s in snapshots[:-1]))
            self.assert_final(snapshots[-1], baseline)
            self.assertIn("$enddefinitions", waveform.read_text())

    def test_live_vcd_and_progress_files_follow_commands(self):
        baseline = self.batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            progress, waveform = path / "live.jsonl", path / "live.vcd"
            with LiveChild(["--progress-jsonl", str(progress), "--vcd", str(waveform)]) as child:
                initial = child.read()
                current = child.command("advance 10000000")
                self.assert_final(current, baseline)
                stopped = child.command("stop")
                self.assertEqual(child.process.wait(timeout=3), 0, child.errors())
            self.assertEqual(
                [json.loads(line) for line in progress.read_text().splitlines()],
                [initial, current, stopped],
            )
            self.assertIn("$enddefinitions", waveform.read_text())

    def test_invalid_slice_and_waveform_arguments_fail(self):
        for arguments in (
            ["--slice-cycles", "0"],
            ["--slice-cycles", "10000001"],
            ["--vcd", "incorrect-extension.txt"],
            ["--vcd", ""],
        ):
            with self.subTest(arguments=arguments):
                process = subprocess.run(
                    [str(BINARY), *arguments], capture_output=True, text=True, timeout=30
                )
                self.assertNotEqual(process.returncode, 0)
                self.assertNotIn('"numerical_match":true', process.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
