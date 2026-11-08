#!/usr/bin/env python3
"""Memory-hierarchy traffic and timing from integer formulas, checked against npu-sim traces.

derive() computes each panel's traffic and DMA and compute cycles, the LRU weight cache and
the SRAM footprint from the configuration alone. With one tile slot and one panel buffer the
run is serial and its cycles are the sum of those durations. Other mappings feed the same
durations to the timestamp scheduler of integration.py, which shares no code with src/, and
the trace is checked for causal order, resource exclusion and equality of the event and
cycle timing modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from integration import schedule  # noqa: E402

BINARY: Path
DEFAULT = dict(
    m=3,
    n=5,
    k=5,
    tile_m=1,
    tile_n=3,
    tile_k=2,
    weight_bits=4,
    macs_per_cycle=7,
    memory_bytes_per_cycle=4,
    memory_latency=3,
    pipeline_cycles=2,
    buffer_slots=1,
    sram_bytes=262144,
    dma_burst_bytes=0,
    weight_cache_bytes=0,
)


def last_stderr_line(proc):
    lines = proc.stderr.strip().splitlines()
    return lines[-1] if lines else ""


def divup(a, b):
    return (a + b - 1) // b


def dma_cycles(c, size):
    limit = c["dma_burst_bytes"] or size
    return sum(
        c["memory_latency"] + divup(min(limit, size - offset), c["memory_bytes_per_cycle"])
        for offset in range(0, size, limit)
    )


def panel_buffers(c):
    """Panel buffers per slot: the configured count (default 2), at most one per panel."""
    return min(c.get("panel_buffers", 2), divup(c["k"], c["tile_k"] or c["k"]))


def derive(c, per_channel=False):
    """Independent traffic/LRU/SRAM arithmetic; all counters use Python integers."""
    tile_k = c["tile_k"] or c["k"]
    panels = list(range(0, c["k"], tile_k))
    jobs, stores = [], []
    cache = OrderedDict()
    cache_bytes = peak = hits = misses = saved = 0
    # Tile ids run group-major, then row, then column.
    origins = [
        (group, row, col)
        for group in range(c.get("groups", 1))
        for row in range(0, c["m"], c["tile_m"])
        for col in range(0, c["n"], c["tile_n"])
    ]
    for group, row, col in origins:
        rows, cols = min(c["tile_m"], c["m"] - row), min(c["tile_n"], c["n"] - col)
        tile = len(stores)
        for block, inner in enumerate(panels):
            k = min(tile_k, c["k"] - inner)
            weights = divup(k * cols * c["weight_bits"], 8)
            metadata = (4 * cols + (12 * cols if per_channel else 0)) if block == 0 else 0
            size = rows * k + weights + metadata
            # One cache line per (group, column panel, K panel).
            key = (group, col, block)
            if c["weight_cache_bytes"]:
                if key in cache:
                    cache.move_to_end(key)
                    hits += 1
                    saved += weights
                    size -= weights
                else:
                    misses += 1
                    if weights <= c["weight_cache_bytes"]:
                        while cache_bytes + weights > c["weight_cache_bytes"]:
                            _, old_size = cache.popitem(last=False)
                            cache_bytes -= old_size
                        cache[key] = weights
                        cache_bytes += weights
                        peak = max(peak, cache_bytes)
            if c.get("array_rows", 0):
                ar, ac = c["array_rows"], c["array_cols"]
                skew = ar + ac - 2
                if c.get("dataflow", "os") == "os":
                    # Output waves of k cycles back to back, skew exposed once.
                    compute = divup(rows, ar) * divup(cols, ac) * k + skew
                else:
                    # Weight folds: the first preload (ar cycles) is exposed, a
                    # fold occupies max(rows, ar) while the next preload shifts
                    # in, and the last fold streams rows.
                    folds = divup(k, ar) * divup(cols, ac)
                    compute = ar + (folds - 1) * max(rows, ar) + rows + skew
                compute += c["pipeline_cycles"]
            else:
                compute = divup(rows * cols * k, c["macs_per_cycle"]) + c["pipeline_cycles"]
            jobs.append(
                dict(
                    tile=tile,
                    block=block,
                    load_bytes=size,
                    compute_cycles=compute,
                    dma_cycles=dma_cycles(c, size),
                )
            )
        stores.append(rows * cols)
    r, n = min(c["m"], c["tile_m"]), min(c["n"], c["tile_n"])
    max_k = min(c["k"], tile_k)
    # Bias and channel metadata stay for the tile; each panel buffer holds the
    # deepest panel; partial sums and outputs take 5 bytes per element.
    panel = r * max_k + divup(max_k * n * c["weight_bits"], 8)
    slot = 4 * n + panel_buffers(c) * panel + 5 * r * n
    if per_channel:
        slot += 12 * n
    reads, writes = sum(j["load_bytes"] for j in jobs), sum(stores)
    compute = sum(j["compute_cycles"] for j in jobs)
    dma = sum(j["dma_cycles"] for j in jobs) + sum(dma_cycles(c, size) for size in stores)
    sizes = [j["load_bytes"] for j in jobs] + stores
    return dict(
        jobs=jobs,
        stores=stores,
        read_bytes=reads,
        write_bytes=writes,
        compute_busy_cycles=compute,
        dma_busy_cycles=dma,
        serial_cycles=compute + dma,
        dma_requests=len(sizes),
        dma_bursts=sum(
            divup(size, c["dma_burst_bytes"]) if c["dma_burst_bytes"] else 1 for size in sizes
        ),
        weight_cache_hits=hits,
        weight_cache_misses=misses,
        weight_bytes_saved=saved,
        peak_cached_weight_bytes=peak,
        scratchpad_bytes=c["buffer_slots"] * slot + c["weight_cache_bytes"],
        reduction_blocks=len(panels),
    )


def invoke(c, *, timing="event", per_channel=False):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        trace_path = directory / "trace.csv"
        command = [str(BINARY), "--dump-output", "--trace", str(trace_path), "--timing", timing]
        for name, value in c.items():
            if name == "macs_per_cycle" and c.get("array_rows", 0):
                continue
            command += ["--" + name.replace("_", "-"), str(value)]
        if per_channel:
            qpath = directory / "channel.npuq"
            channels = c.get("groups", 1) * c["n"]
            qpath.write_text(f"NPUQ1 {channels}\n" + "0 1 7\n" * channels)
            command += ["--quantization-file", str(qpath)]
        proc = subprocess.run(command, text=True, capture_output=True, timeout=30)
        if proc.returncode:
            raise AssertionError(f"simulator failed: {proc.stderr}")
        result = json.loads(proc.stdout)
        with trace_path.open(newline="") as stream:
            trace = [
                {key: value if key == "stage" else int(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
        return result, trace


class ArchitectureChecks(unittest.TestCase):
    def assert_metrics(self, actual, expected):
        for key, value in expected.items():
            if key not in {"jobs", "stores", "serial_cycles"}:
                self.assertEqual(actual[key], value, key)
        shape = actual["config"]["shape"]
        self.assertEqual(actual["macs"], shape["groups"] * shape["m"] * shape["n"] * shape["k"])

    def assert_trace(self, c, actual, trace, expected):
        self.assertEqual(len(trace), len(expected["jobs"]) * 2 + len(expected["stores"]))
        by_key = {(s["tile_id"], s["block_id"], s["stage"]): s for s in trace}
        self.assertEqual(len(by_key), len(trace), "unique stage per panel")
        blocks = expected["reduction_blocks"]
        buffers = panel_buffers(c)
        for job in expected["jobs"]:
            key = job["tile"], job["block"]
            load, compute = by_key[(*key, "load")], by_key[(*key, "compute")]
            self.assertEqual(load["bytes"], job["load_bytes"])
            self.assertEqual(load["end_cycle"] - load["start_cycle"], job["dma_cycles"])
            self.assertEqual(compute["end_cycle"] - compute["start_cycle"], job["compute_cycles"])
            self.assertGreaterEqual(compute["start_cycle"], load["end_cycle"])
            self.assertEqual(compute["bytes"], 0)
            if job["block"]:
                previous = by_key[(job["tile"], job["block"] - 1, "compute")]
                self.assertGreaterEqual(compute["start_cycle"], previous["end_cycle"])
            if job["block"] >= buffers:
                # Panel b reuses the buffer of panel b - buffers.
                released = by_key[(job["tile"], job["block"] - buffers, "compute")]
                self.assertGreaterEqual(
                    load["start_cycle"],
                    released["end_cycle"],
                    "panel buffer overwritten before compute released it",
                )
        for tile, size in enumerate(expected["stores"]):
            store = by_key[(tile, blocks - 1, "store")]
            self.assertEqual(store["bytes"], size)
            self.assertEqual(store["end_cycle"] - store["start_cycle"], dma_cycles(c, size))
            self.assertGreaterEqual(
                store["start_cycle"], by_key[(tile, blocks - 1, "compute")]["end_cycle"]
            )
        for stages in ({"load", "store"}, {"compute"}):
            spans = sorted(
                (s for s in trace if s["stage"] in stages), key=lambda s: s["start_cycle"]
            )
            for previous, current in zip(spans, spans[1:], strict=False):
                self.assertGreaterEqual(
                    current["start_cycle"], previous["end_cycle"], "resource exclusion"
                )
        self.assertEqual(actual["cycles"], max(s["end_cycle"] for s in trace))
        self.assertGreaterEqual(
            actual["cycles"], max(actual["compute_busy_cycles"], actual["dma_busy_cycles"])
        )
        self.assertLessEqual(actual["cycles"], expected["serial_cycles"])
        self.assertLessEqual(actual["peak_inflight_tiles"], c["buffer_slots"])

    def assert_schedule(self, c, actual, trace, expected):
        """Cycles, waits and spans from the timestamp scheduler of integration.py."""
        blocks = expected["reduction_blocks"]
        panels = [
            [
                (job["load_bytes"], job["dma_cycles"], job["compute_cycles"])
                for job in expected["jobs"][tile * blocks : (tile + 1) * blocks]
            ]
            for tile in range(len(expected["stores"]))
        ]
        stores = [(size, dma_cycles(c, size)) for size in expected["stores"]]
        oracle = schedule(panels, stores, c["buffer_slots"], panel_buffers(c))
        for key in ("cycles", "peak_inflight_tiles", "slot_wait_cycles", "panel_wait_cycles"):
            self.assertEqual(actual[key], oracle[key], key)
        self.assertEqual(trace, oracle["trace"], "trace differs from the oracle schedule")

    def test_serial_exact_integer_oracle(self):
        rng = random.Random(8921)
        for trial in range(14):
            c = dict(
                DEFAULT,
                m=rng.randint(1, 5),
                n=rng.randint(1, 7),
                k=rng.randint(1, 13),
                tile_m=rng.randint(1, 4),
                tile_n=rng.randint(1, 4),
                tile_k=rng.randint(1, 6),
                panel_buffers=1,
                weight_bits=4 if trial % 2 else 8,
                dma_burst_bytes=rng.choice([0, 1, 3, 5, 16]),
                weight_cache_bytes=rng.choice([0, 1, 5, 256]),
            )
            if trial % 3:
                c.update(
                    array_rows=rng.randint(1, 4),
                    array_cols=rng.randint(1, 4),
                    dataflow=("os", "ws")[trial % 2],
                )
            if trial % 4 == 1:
                c.update(groups=rng.randint(2, 4))
            q = bool(trial % 2)
            expected = derive(c, q)
            with self.subTest(trial=trial, config=c):
                actual, trace = invoke(c, per_channel=q)
                self.assert_metrics(actual, expected)
                self.assertEqual(actual["cycles"], expected["serial_cycles"])
                self.assert_trace(c, actual, trace, expected)
                self.assert_schedule(c, actual, trace, expected)
                # The same mapping with two panel buffers per slot.
                double = dict(c, panel_buffers=2)
                expected = derive(double, q)
                actual, trace = invoke(double, per_channel=q)
                self.assert_metrics(actual, expected)
                self.assert_trace(double, actual, trace, expected)
                self.assert_schedule(double, actual, trace, expected)

    def test_overlap_and_timing_modes(self):
        for slots in (2, 3):
            c = dict(
                DEFAULT,
                m=4,
                n=5,
                k=7,
                tile_m=2,
                tile_n=3,
                tile_k=3,
                buffer_slots=slots,
                array_rows=2,
                array_cols=2,
                dma_burst_bytes=5,
                weight_cache_bytes=256,
            )
            expected = derive(c, True)
            event, event_trace = invoke(c, per_channel=True)
            cycle, cycle_trace = invoke(c, timing="cycle", per_channel=True)
            self.assert_metrics(event, expected)
            self.assert_trace(c, event, event_trace, expected)
            self.assert_schedule(c, event, event_trace, expected)
            self.assertEqual(event_trace, cycle_trace)
            for key in event:
                if key not in {"simulation_wall_seconds", "config"}:
                    self.assertEqual(event[key], cycle[key], key)
            self.assertLess(event["cycles"], expected["serial_cycles"])
            self.assertGreater(event["peak_inflight_tiles"], 1)

    def test_int4_padding_metadata_and_array_known_values(self):
        c = dict(
            DEFAULT,
            m=1,
            n=3,
            k=3,
            tile_m=1,
            tile_n=3,
            tile_k=1,
            array_rows=2,
            array_cols=2,
            pipeline_cycles=2,
            dma_burst_bytes=5,
        )
        plain, plain_trace = invoke(c)
        channels, channels_trace = invoke(c, per_channel=True)
        self.assertEqual(plain["read_bytes"], 21)  # 3 * (1 A + ceil(3 * 4 / 8) W) + 12 bias
        self.assertEqual(channels["read_bytes"], 57)  # + 3 channels * 3 INT32 vectors
        self.assertEqual(plain["compute_busy_cycles"], 18)  # 3 * (2 waves * 1 + 2 skew + 2 pipe)
        self.assertEqual(channels["compute_busy_cycles"], 18)
        for actual, trace, quant in [(plain, plain_trace, False), (channels, channels_trace, True)]:
            expected = derive(c, quant)
            self.assert_metrics(actual, expected)
            self.assert_trace(c, actual, trace, expected)
            self.assert_schedule(c, actual, trace, expected)
        # ws: ceil(1/2) * ceil(3/2) = 2 folds of one row; the second 2-cycle
        # preload is exposed: 3*(2 preload + 2 fold + 1 row + 2 skew + 2 pipeline).
        ws = dict(c, dataflow="ws")
        actual, trace = invoke(ws)
        self.assertEqual(actual["compute_busy_cycles"], 27)
        expected = derive(ws)
        self.assert_metrics(actual, expected)
        self.assert_trace(ws, actual, trace, expected)
        self.assert_schedule(ws, actual, trace, expected)

    def test_cache_lru_and_payload_invariance(self):
        outputs = []
        for capacity in (0, 1, 5, 12, 13, 256):
            c = dict(DEFAULT, buffer_slots=2, weight_cache_bytes=capacity)
            actual, trace = invoke(c)
            expected = derive(c)
            with self.subTest(cache_capacity=capacity):
                self.assert_metrics(actual, expected)
                self.assert_trace(c, actual, trace, expected)
                self.assert_schedule(c, actual, trace, expected)
                outputs.append((actual["accumulators"], actual["outputs"], actual["output_hash"]))
                self.assertLessEqual(actual["peak_cached_weight_bytes"], capacity)
                if capacity == 13:
                    self.assertEqual(actual["weight_cache_misses"], 6)
                    self.assertEqual(actual["weight_cache_hits"], 12)
                    self.assertEqual(actual["weight_bytes_saved"], 26)
        self.assertTrue(all(result == outputs[0] for result in outputs))

    def test_grouped_weight_cache_lines_are_per_group(self):
        # Each group's two row tiles share that group's weight panels; no group
        # may hit another group's line for the same column panel.
        c = dict(
            DEFAULT,
            groups=3,
            m=4,
            n=3,
            k=5,
            tile_m=2,
            tile_n=3,
            tile_k=2,
            buffer_slots=1,
            weight_cache_bytes=256,
        )
        expected = derive(c)
        actual, trace = invoke(c)
        self.assert_metrics(actual, expected)
        self.assert_trace(c, actual, trace, expected)
        self.assert_schedule(c, actual, trace, expected)
        self.assertEqual((actual["weight_cache_misses"], actual["weight_cache_hits"]), (9, 9))

    def test_sram_reservation_boundary(self):
        c = dict(DEFAULT, weight_cache_bytes=13, buffer_slots=2)
        expected = derive(c)
        c["sram_bytes"] = expected["scratchpad_bytes"]
        result, _ = invoke(c)
        self.assertEqual(result["scratchpad_bytes"], c["sram_bytes"])
        c["sram_bytes"] -= 1
        command = [str(BINARY)]
        for key, value in c.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertTrue(last_stderr_line(proc).startswith("npu-sim: "), proc.stderr)
        self.assertIn("SRAM", last_stderr_line(proc))

    def test_bad_architecture_flags(self):
        for flags in [
            ["--tile-k", "-1"],
            ["--tile-k", "1048577"],
            ["--weight-bits", "3"],
            ["--groups", "0"],
            ["--groups", "-1"],
            ["--groups", "1048577"],
            ["--array-rows", "2"],
            ["--array-cols", "2"],
            ["--array-rows", "257", "--array-cols", "2"],
            ["--array-rows", "2", "--array-cols", "2", "--macs-per-cycle", "4"],
            ["--dataflow", "ws"],
            ["--dataflow", "os"],
            ["--dataflow", "is"],
            ["--panel-buffers", "0"],
            ["--panel-buffers", "3"],
            ["--dma-burst-bytes", "-1"],
            ["--weight-cache-bytes", "-1"],
            ["--dma-burst-bytes", "18446744073709551616"],
            ["--weight-cache-bytes", "18446744073709551615"],
        ]:
            with self.subTest(flags=flags):
                proc = subprocess.run(
                    [str(BINARY), *flags], capture_output=True, text=True, timeout=30
                )
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertTrue(last_stderr_line(proc).startswith("npu-sim: "), proc.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
