#!/usr/bin/env python3
"""Black-box checks of real SystemC signal traces and passive instrumentation.

The VCD reader uses only the standard library. Expected final counters come
from a separate untraced simulator invocation; arithmetic/timing oracles are
covered by the existing integration and architecture-contract suites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

BINARY: Path


def read_vcd(path: Path) -> tuple[dict[str, list[tuple[int, int]]], dict[str, int]]:
    """Read scalar/vector integer changes without depending on tool-generated IDs."""
    identifiers: dict[str, str] = {}
    widths: dict[str, int] = {}
    values: dict[str, list[tuple[int, int]]] = {}
    timestamp = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        declaration = re.fullmatch(r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\[.*\])?\s+\$end", line)
        if declaration:
            width, identifier, name = declaration.groups()
            if name in values:
                raise AssertionError(f"duplicate leaf signal: {name}")
            identifiers[identifier] = name
            widths[name] = int(width)
            values[name] = []
        elif line.startswith("#"):
            timestamp = int(line[1:])
        elif line.startswith("b"):
            bits, identifier = line[1:].split()
            if identifier in identifiers and all(bit in "01" for bit in bits):
                values[identifiers[identifier]].append((timestamp, int(bits, 2)))
        elif len(line) > 1 and line[0] in "01":
            identifier = line[1:]
            if identifier in identifiers:
                values[identifiers[identifier]].append((timestamp, int(line[0])))
    return values, widths


def states(changes: dict[str, list[tuple[int, int]]], names: set[str]):
    """Yield the value of every named signal after each VCD timestamp."""
    events = sorted((time, name, value) for name in names for time, value in changes[name])
    current = {name: 0 for name in names}
    for index, (time, name, value) in enumerate(events):
        current[name] = value
        if index + 1 == len(events) or events[index + 1][0] != time:
            yield time, dict(current)


class SystemCObservability(unittest.TestCase):
    def run_simulator(self, arguments: list[str]) -> dict:
        process = subprocess.run([str(BINARY), *arguments], capture_output=True,
                                 text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertNotIn("NPU/invariant", process.stderr)
        self.assertNotIn("multiple drivers", process.stderr)
        # Waveform setup messages must not corrupt the single result object.
        result = json.loads(process.stdout)
        self.assertTrue(result["numerical_match"])
        return result

    def verify_waveform(self, path: Path, result: dict, slots: int, panel_buffers: int) -> dict:
        self.assertTrue(path.is_file(), f"VCD missing: {path}")
        self.assertGreater(path.stat().st_size, 1000)
        text = path.read_text()
        self.assertIn("$enddefinitions", text)
        self.assertIn("$timescale", text)
        changes, widths = read_vcd(path)
        required = {
            "done", "dma_busy", "compute_busy", "dma_read", "cycles",
            "inflight_tiles", "loaded_tiles", "computed_tiles",
            "read_queue_depth", "write_queue_depth", "read_bytes", "write_bytes",
            "macs", "dma_tile_id", "dma_block_id", "compute_tile_id",
            "compute_block_id", "dma_busy_cycles", "compute_busy_cycles",
            "dma_requests", "dma_bursts",
            "executed_macs", "skipped_macs", "weight_decode_cycles",
        }
        self.assertTrue(required <= changes.keys(), required - changes.keys())
        for name in required:
            self.assertTrue(changes[name], name)
            expected_width = 1 if name in {"done", "dma_busy", "compute_busy", "dma_read"} else 64
            self.assertEqual(widths[name], expected_width, name)
        for name in {"cycles", "read_bytes", "write_bytes", "macs",
                     "dma_busy_cycles", "compute_busy_cycles", "dma_requests", "dma_bursts",
                     "executed_macs", "skipped_macs", "weight_decode_cycles"}:
            self.assertEqual(changes[name][-1][1], result[name], name)
            self.assertEqual([value for _, value in changes[name]],
                             sorted(value for _, value in changes[name]), name)
        self.assertEqual(changes["done"][-1][1], 1)
        self.assertEqual(changes["inflight_tiles"][-1][1], 0)
        for name in {"dma_busy", "compute_busy"}:
            self.assertIn(1, [value for _, value in changes[name]], name)
            self.assertEqual(changes[name][-1][1], 0, name)
            self.assertGreater(len(changes[name]), 2, name)
        for name in {"inflight_tiles", "computed_tiles"}:
            self.assertTrue(all(0 <= value <= slots for _, value in changes[name]), name)
        # Loaded panels waiting for compute each hold a panel buffer.
        self.assertTrue(all(0 <= value <= slots * panel_buffers
                            for _, value in changes["loaded_tiles"]))
        self.assertEqual(max(value for _, value in changes["inflight_tiles"]),
                         result["peak_inflight_tiles"])
        return changes

    def test_event_waveform_is_passive_and_contains_real_activity(self):
        arguments = ["--m", "4", "--n", "5", "--k", "7", "--tile-m", "2",
                     "--tile-n", "3", "--tile-k", "3", "--buffer-slots", "2",
                     "--dma-burst-bytes", "5", "--weight-cache-bytes", "256",
                     "--dump-output"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accelerator.vcd"
            baseline = self.run_simulator(arguments)
            traced = self.run_simulator([*arguments, "--vcd", str(path)])
            for key in baseline:
                if key != "simulation_wall_seconds":
                    self.assertEqual(traced[key], baseline[key], key)
            changes = self.verify_waveform(path, traced, 2, 2)
            self.assertGreater(max(value for _, value in changes["dma_tile_id"]), 0)
            self.assertEqual(max(value for _, value in changes["dma_block_id"]), 2)
            self.assertEqual(max(value for _, value in changes["compute_block_id"]), 2)

    def test_cycle_and_event_waveforms_have_same_architecture_result(self):
        arguments = ["--m", "3", "--n", "5", "--k", "5", "--tile-m", "1",
                     "--tile-n", "3", "--tile-k", "2", "--buffer-slots", "1",
                     "--array-rows", "2", "--array-cols", "2", "--weight-bits", "4",
                     "--dump-output"]
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for timing in ("event", "cycle"):
                path = Path(directory) / (timing + ".vcd")
                result = self.run_simulator([*arguments, "--timing", timing, "--vcd", str(path)])
                self.verify_waveform(path, result, 1, 2)
                results.append(result)
        for key in results[0]:
            if key not in {"simulation_wall_seconds", "config"}:
                self.assertEqual(results[0][key], results[1][key], key)

    def test_compressed_waveforms_match_real_decoder_and_skipped_work(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "weights.npu"
            source.write_text("NPU1 2 3 4\n1 2 3 4 5 6 7 8\n"
                              "0 0 0 1 0 0 0 0 0 1 -1 0\n0 0 0\n", encoding="ascii")
            for codec in ("palette", "sparse"):
                with self.subTest(codec=codec):
                    path = Path(directory) / (codec + ".vcd")
                    result = self.run_simulator(["--input", str(source), "--tile-m", "1",
                        "--tile-n", "2", "--tile-k", "2", "--input-zero", "0",
                        "--weight-zero", "0", "--weight-encoding", codec, "--palette-bits", "2",
                        "--sparse-block", "2", "--vcd", str(path)])
                    self.verify_waveform(path, result, 2, 2)
                    self.assertGreater(result["weight_decode_cycles"], 0)
                    self.assertEqual(result["executed_macs"] + result["skipped_macs"], 24)
                    if codec == "sparse":
                        self.assertGreater(result["skipped_macs"], 0)

    def test_waveform_shows_panel_prefetch_under_compute(self):
        # One slot: any load during a compute belongs to the same tile, so it
        # can only be the next panel going into the second panel buffer.
        arguments = ["--m", "4", "--n", "4", "--k", "12", "--tile-m", "4", "--tile-n", "4",
                     "--tile-k", "3", "--buffer-slots", "1", "--macs-per-cycle", "4",
                     "--memory-latency", "1"]
        names = {"dma_busy", "dma_read", "compute_busy", "dma_tile_id", "compute_tile_id",
                 "dma_block_id", "compute_block_id"}
        with tempfile.TemporaryDirectory() as directory:
            for buffers in (1, 2):
                with self.subTest(panel_buffers=buffers):
                    path = Path(directory) / f"panels{buffers}.vcd"
                    result = self.run_simulator([*arguments, "--panel-buffers", str(buffers),
                                                 "--vcd", str(path)])
                    changes = self.verify_waveform(path, result, 1, buffers)
                    overlap = [state for _, state in states(changes, names)
                               if state["dma_busy"] and state["dma_read"] and state["compute_busy"]]
                    if buffers == 1:
                        self.assertEqual(overlap, [])
                        continue
                    self.assertTrue(overlap)
                    for state in overlap:
                        self.assertEqual(state["dma_tile_id"], state["compute_tile_id"])
                        self.assertEqual(state["dma_block_id"], state["compute_block_id"] + 1)

    def test_vcd_in_unwritable_directory_fails_without_false_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "does-not-exist" / "accelerator.vcd"
            process = subprocess.run([str(BINARY), "--vcd", str(path)],
                                     capture_output=True, text=True, timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
