#!/usr/bin/env python3
"""Independent black-box arithmetic and architecture-contract checks.

Uses only Python's standard library.  It never imports simulator implementation
code: fixture generation, quantization, traffic, and serial timing are derived
from docs/IMPLEMENTATION-CONTRACT.md with unbounded Python integers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
BINARY: Path

DEFAULTS = {
    "m": 4, "n": 5, "k": 7,
    "tile_m": 3, "tile_n": 2,
    "macs_per_cycle": 8, "memory_bytes_per_cycle": 4,
    "memory_latency": 3, "pipeline_cycles": 2,
    "buffer_slots": 2, "sram_bytes": 262144,
    "seed": 1, "frequency_mhz": 200,
    "input_zero": -3, "weight_zero": 2, "output_zero": 0,
    "multiplier": 1, "shift": 7, "relu": False,
}


def ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def groups(c: dict) -> int:
    return c.get("groups", 1)


def fixtures(c: dict) -> tuple[list[int], list[int], list[int]]:
    """Portable fixture stream, without C/C++ random libraries.

    Activations are M x (G*K) row-major, weights G blocks of K x N, bias G*N.
    """
    state = c["seed"] & MASK32
    g = groups(c)

    def values(count: int, modulus: int, offset: int) -> list[int]:
        nonlocal state
        result = []
        for _ in range(count):
            state = (1664525 * state + 1013904223) & MASK32
            result.append(state % modulus - offset)
        return result

    return (
        values(c["m"] * g * c["k"], 31, 15),
        values(g * c["k"] * c["n"], 31, 15),
        values(g * c["n"], 127, 63),
    )


def native_requantize(c: dict, acc: int) -> int:
    product = acc * c["multiplier"]
    if c["shift"]:
        magnitude = (abs(product) + (1 << (c["shift"] - 1))) // (1 << c["shift"])
        rounded = magnitude if product >= 0 else -magnitude
    else:
        rounded = product
    quantized = rounded + c["output_zero"]
    if c["relu"]:
        quantized = max(quantized, c["output_zero"])
    return min(127, max(-128, quantized))


def output_value(c: dict, problem, row: int, channel: int) -> tuple[int, int]:
    """Accumulator and INT8 output at row, global channel G*N index."""
    activations, weights, biases = problem
    g, n, k = groups(c), c["n"], c["k"]
    group, col = divmod(channel, n)
    acc = biases[channel] + sum(
        (activations[row * g * k + group * k + inner] - c["input_zero"])
        * (weights[(group * k + inner) * n + col] - c["weight_zero"])
        for inner in range(k)
    )
    if not -(1 << 31) <= acc < (1 << 31):
        raise OverflowError("reference accumulator is outside INT32")
    return acc, native_requantize(c, acc)


def expected_outputs(c: dict, problem: tuple[list[int], list[int], list[int]] | None = None) -> tuple[list[int], list[int]]:
    problem = fixtures(c) if problem is None else problem
    accumulators, outputs = [], []
    for row in range(c["m"]):
        for channel in range(groups(c) * c["n"]):
            acc, out = output_value(c, problem, row, channel)
            accumulators.append(acc)
            outputs.append(out)
    return accumulators, outputs


def expected_hash(accumulators: list[int], outputs: list[int]) -> int:
    stream = bytearray()
    for accumulator in accumulators:
        stream.extend((accumulator & MASK32).to_bytes(4, "little"))
    stream.extend(value & 255 for value in outputs)
    value = 14695981039346656037
    for byte in stream:
        value = ((value ^ byte) * 1099511628211) & MASK64
    return value


def tiles(c: dict) -> list[tuple[int, int]]:
    """Tile extents in id order: group-major, then row, then column."""
    return [
        (min(c["tile_m"], c["m"] - row), min(c["tile_n"], c["n"] - col))
        for _ in range(groups(c))
        for row in range(0, c["m"], c["tile_m"])
        for col in range(0, c["n"], c["tile_n"])
    ]


def panel_depths(c: dict) -> list[int]:
    """Reduction panel depths; tile_k 0 is one panel of depth K."""
    depth = c.get("tile_k", 0) or c["k"]
    return [min(depth, c["k"] - start) for start in range(0, c["k"], depth)]


def panel_buffers(c: dict) -> int:
    """Panel buffers a slot uses: the configured count (default 2), at most one per panel."""
    return min(c.get("panel_buffers", 2), len(panel_depths(c)))


def transfer_cycles(c: dict, size: int) -> int:
    return c["memory_latency"] + ceildiv(size, c["memory_bytes_per_cycle"])


def array_cycles(c: dict, rows: int, cols: int, depth: int) -> int:
    """MAC cycles of one panel on an A_r x A_c array, stepped wave by wave (os)
    or fold by fold (ws)."""
    ar, ac = c["array_rows"], c["array_cols"]
    skew = ar + ac - 2
    clock = 0
    if c.get("dataflow", "os") == "os":
        # Each output wave holds an A_r x A_c block and streams the panel's K.
        for _ in range(ceildiv(rows, ar) * ceildiv(cols, ac)):
            clock += depth
        return clock + skew
    # Each fold holds an A_r x A_c weight block and streams the tile's rows.
    # The next block shifts in, one array row per cycle, during the stream.
    weights_ready = ar
    for _ in range(ceildiv(depth, ar) * ceildiv(cols, ac)):
        start = max(clock, weights_ready)
        clock, weights_ready = start + rows, start + ar
    return clock + skew


def peak_macs(c: dict) -> int:
    return c["array_rows"] * c["array_cols"] if c.get("array_rows") else c["macs_per_cycle"]


def compute_cycles(c: dict, rows: int, cols: int, depth: int) -> int:
    if c.get("array_rows"):
        return array_cycles(c, rows, cols, depth) + c["pipeline_cycles"]
    return ceildiv(rows * cols * depth, c["macs_per_cycle"]) + c["pipeline_cycles"]


def tile_work(c: dict) -> tuple[list[list[tuple[int, int, int]]], list[tuple[int, int]]]:
    """Per tile: (load bytes, load cycles, compute cycles) for each reduction
    panel, and (store bytes, store cycles). The first panel carries the bias."""
    panels, stores = [], []
    for rows, cols in tiles(c):
        work = []
        for block, depth in enumerate(panel_depths(c)):
            size = rows * depth + depth * cols + (4 * cols if block == 0 else 0)
            work.append((size, transfer_cycles(c, size), compute_cycles(c, rows, cols, depth)))
        panels.append(work)
        stores.append((rows * cols, transfer_cycles(c, rows * cols)))
    return panels, stores


def expected_metrics(c: dict) -> dict[str, int]:
    panels, stores = tile_work(c)
    reads = sum(size for work in panels for size, _, _ in work)
    writes = sum(size for size, _ in stores)
    compute = sum(cycles for work in panels for _, _, cycles in work)
    dma = sum(cycles for work in panels for _, cycles, _ in work) + \
        sum(cycles for _, cycles in stores)
    max_rows, max_cols = min(c["m"], c["tile_m"]), min(c["n"], c["tile_n"])
    deepest = max(panel_depths(c))
    # A slot keeps bias, its panel buffers, INT32 partial sums and INT8 outputs.
    slot = 4 * max_cols + panel_buffers(c) * (max_rows * deepest + deepest * max_cols) + \
        5 * max_rows * max_cols
    return {
        "macs": groups(c) * c["m"] * c["n"] * c["k"],
        "read_bytes": reads, "write_bytes": writes,
        "compute_busy_cycles": compute, "dma_busy_cycles": dma,
        "scratchpad_bytes": c["buffer_slots"] * slot, "serial_cycles": compute + dma,
    }


def schedule(panels: list[list[tuple[int, int, int]]], stores: list[tuple[int, int]],
             slots: int, buffers: int) -> dict:
    """Timestamp scheduler independent of SystemC threads and delta cycles.

    panels[t][b] is (load bytes, load cycles, compute cycles) of tile t's
    reduction panel b; stores[t] is (bytes, cycles).  Complete every resource
    finishing at T first, close all zero-time queue transitions, then schedule
    work at T.  One loader requests panels in tile order: a tile's first panel
    needs a free slot, and panel b needs panel b - buffers of the same tile
    computed.  Compute takes loaded panels in order; a tile's store follows its
    last panel and frees the slot.  The shared non-preemptive DMA serves a
    ready request at once, alternating load and store when both wait.
    """
    count, blocks = len(panels), len(panels[0])
    computed = [0] * count
    ready: list[tuple[int, int]] = []
    unstored: list[int] = []
    spans = []
    time = inflight = peak = slot_wait = panel_wait = finished = 0
    next_tile = next_block = 0
    loader_free_at = 0
    loader_busy = store_busy = False
    read_request = write_request = None
    dma = compute = None
    prefer_read = True

    def span(stage: str, tile: int, block: int, duration: int, size: int) -> None:
        spans.append({"stage": stage, "tile_id": tile, "start_cycle": time,
                      "end_cycle": time + duration, "bytes": size, "block_id": block})

    while finished < count:
        # Every completion at the same timestamp is visible before arbitration.
        if dma is not None and dma[0] == time:
            _, stage, tile, block = dma
            dma = None
            if stage == "load":
                ready.append((tile, block))
                loader_busy, loader_free_at = False, time
            else:
                store_busy = False
                inflight -= 1
                finished += 1
        if compute is not None and compute[0] == time:
            _, tile, block = compute
            compute = None
            computed[tile] += 1
            if computed[tile] == blocks:
                unstored.append(tile)

        if compute is None and ready:
            tile, block = ready.pop(0)
            duration = panels[tile][block][2]
            compute = time + duration, tile, block
            span("compute", tile, block, duration, 0)
        if not store_busy and unstored:
            write_request, store_busy = unstored.pop(0), True
        if not loader_busy and next_tile < count:
            if next_block == 0:
                issue = inflight < slots
                if issue:
                    inflight += 1
                    peak = max(peak, inflight)
                    slot_wait += time - loader_free_at
            else:
                issue = next_block < buffers or computed[next_tile] > next_block - buffers
                if issue:
                    panel_wait += time - loader_free_at
            if issue:
                read_request, loader_busy = (next_tile, next_block), True
                next_block += 1
                if next_block == blocks:
                    next_tile, next_block = next_tile + 1, 0

        if dma is None and (read_request is not None or write_request is not None):
            read = read_request is not None and (write_request is None or prefer_read)
            if read:
                (tile, block), read_request = read_request, None
                size, duration, _ = panels[tile][block]
            else:
                tile, block, write_request = write_request, blocks - 1, None
                size, duration = stores[tile]
            stage = "load" if read else "store"
            dma = time + duration, stage, tile, block
            prefer_read = not read
            span(stage, tile, block, duration, size)

        if finished == count:
            break
        next_events = [event[0] for event in (dma, compute) if event is not None]
        if not next_events:
            raise AssertionError("independent timing scheduler deadlocked")
        time = min(next_events)
    return {
        "cycles": time, "peak_inflight_tiles": peak, "slot_wait_cycles": slot_wait,
        "panel_wait_cycles": panel_wait,
        "trace": sorted(spans, key=lambda x: (x["start_cycle"], x["stage"], x["tile_id"],
                                              x["block_id"])),
    }


def timing_oracle(c: dict) -> dict:
    panels, stores = tile_work(c)
    return schedule(panels, stores, c["buffer_slots"], panel_buffers(c))


ORACLE_KEYS = ("cycles", "peak_inflight_tiles", "slot_wait_cycles", "panel_wait_cycles")


def command(c: dict, *, timing: str = "event", dump: bool = True, trace: Path | None = None, input_path: Path | None = None) -> list[str]:
    result = [str(BINARY)]
    for key, value in c.items():
        if input_path is not None and key in {"m", "n", "k"}:
            continue
        if key == "macs_per_cycle" and c.get("array_rows"):
            continue
        if key in {"relu", "timing_only"}:
            if value:
                result.append("--" + key.replace("_", "-"))
        else:
            result.extend(["--" + key.replace("_", "-"), str(value)])
    result.extend(["--timing", timing])
    if dump:
        result.append("--dump-output")
    if trace is not None:
        result.extend(["--trace", str(trace)])
    if input_path is not None:
        result.extend(["--input", str(input_path)])
    return result


def invoke(c: dict, *, timeout: float = 30, **kwargs) -> tuple[dict, subprocess.CompletedProcess]:
    proc = subprocess.run(command(c, **kwargs), capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise AssertionError(f"simulator failed ({proc.returncode}): {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
    except (ValueError, TypeError) as exc:
        raise AssertionError(f"stdout was not one JSON object: {proc.stdout[:1000]!r}") from exc
    if not isinstance(payload, dict):
        raise AssertionError("stdout JSON is not an object")
    return payload, proc


class SimulatorIntegration(unittest.TestCase):
    def assert_rejected(self, proc: subprocess.CompletedProcess, why: str) -> None:
        # Rejected input exits 2 with a one-line "npu-sim: " diagnostic, the
        # last stderr line after any SystemC banner.
        self.assertEqual(proc.returncode, 2, f"{why}: {proc.stderr}")
        lines = proc.stderr.strip().splitlines()
        self.assertTrue(lines and lines[-1].startswith("npu-sim: "), f"{why}: {proc.stderr}")

    def check_trace(self, text: str, c: dict, result: dict) -> None:
        panels, stores = tile_work(c)
        blocks, buffers = len(panel_depths(c)), panel_buffers(c)
        records = list(csv.DictReader(text.splitlines()))
        self.assertEqual(len(records), len(tiles(c)) * (2 * blocks + 1))
        self.assertEqual(
            set(records[0]), {"stage", "tile_id", "start_cycle", "end_cycle", "bytes", "block_id"}
        )
        stages: dict[tuple[str, int, int], tuple[int, int, int]] = {}
        for record in records:
            stage, tile_id, block = record["stage"], int(record["tile_id"]), int(record["block_id"])
            self.assertIn(stage, {"load", "compute", "store"})
            self.assertTrue(0 <= tile_id < len(tiles(c)))
            self.assertTrue(0 <= block < blocks)
            key = stage, tile_id, block
            self.assertNotIn(key, stages, f"duplicate stage for panel: {key}")
            start, end, size = (int(record[k]) for k in ("start_cycle", "end_cycle", "bytes"))
            self.assertTrue(0 <= start < end <= result["cycles"])
            if stage == "load":
                expected_bytes, duration, _ = panels[tile_id][block]
            elif stage == "store":
                self.assertEqual(block, blocks - 1, "a store is recorded on the last panel")
                expected_bytes, duration = stores[tile_id]
            else:
                expected_bytes, duration = 0, panels[tile_id][block][2]
            self.assertEqual(size, expected_bytes, f"incorrect bytes: {key}")
            self.assertEqual(end - start, duration, f"incorrect duration: {key}")
            stages[key] = start, end, size
        last = blocks - 1
        for tile_id in range(len(tiles(c))):
            for block in range(blocks):
                load, compute = stages["load", tile_id, block], stages["compute", tile_id, block]
                self.assertLessEqual(load[1], compute[0])
                if block:
                    self.assertLessEqual(stages["compute", tile_id, block - 1][1], compute[0])
                if block >= buffers:
                    # Panel b reuses the buffer of panel b - buffers.
                    self.assertLessEqual(stages["compute", tile_id, block - buffers][1], load[0],
                                         "panel buffer overwritten before compute released it")
            self.assertLessEqual(stages["compute", tile_id, last][1],
                                 stages["store", tile_id, last][0])
        for resource in ({"load", "store"}, {"compute"}):
            intervals = sorted((start, end) for (stage, _, _), (start, end, _) in stages.items()
                               if stage in resource)
            for previous, current in zip(intervals, intervals[1:]):
                self.assertLessEqual(previous[1], current[0], f"overlap on exclusive resource {resource}")
        self.assertEqual(max(end for _, end, _ in stages.values()), result["cycles"])
        # An acquired slot may wait for DMA before its first load span starts,
        # so this is a lower bound on in-flight occupancy.
        occupancy_events = []
        for tile_id in range(len(tiles(c))):
            occupancy_events.extend([(stages["load", tile_id, 0][0], 1),
                                     (stages["store", tile_id, last][1], -1)])
        occupancy = peak = 0
        for _, change in sorted(occupancy_events):
            occupancy += change
            peak = max(peak, occupancy)
        self.assertEqual(occupancy, 0)
        self.assertLessEqual(peak, c["buffer_slots"])
        self.assertLessEqual(peak, result["peak_inflight_tiles"])

    def assert_schedule(self, c: dict, path: Path) -> dict:
        """Run c with a trace; cycles, waits and every span must equal the oracle."""
        oracle = timing_oracle(c)
        actual, _ = invoke(c, trace=path)
        for key in ORACLE_KEYS:
            self.assertEqual(actual[key], oracle[key], key)
        records = list(csv.DictReader(path.read_text().splitlines()))
        parsed = [{key: value if key == "stage" else int(value) for key, value in row.items()}
                  for row in records]
        self.assertEqual(parsed, oracle["trace"], "independent schedule trace differs")
        self.check_trace(path.read_text(), c, actual)
        return actual

    def check_case(self, overrides: dict, timing: str = "event") -> dict:
        c = {**DEFAULTS, **overrides}
        result, _ = invoke(c, timing=timing)
        accumulators, outputs = expected_outputs(c)
        self.assertEqual(result["accumulators"], accumulators)
        self.assertEqual(result["outputs"], outputs)
        self.assertIs(result["numerical_match"], True)
        self.assertEqual(result["numerics"], "bit-exact")
        self.assertEqual(result["output_hash"], expected_hash(accumulators, outputs))
        metrics = expected_metrics(c)
        for key, value in metrics.items():
            if key != "serial_cycles":
                self.assertEqual(result[key], value, key)
        self.assertGreater(result["cycles"], 0)
        self.assertLessEqual(result["cycles"], metrics["serial_cycles"])
        self.assertGreaterEqual(result["cycles"], max(metrics["compute_busy_cycles"], metrics["dma_busy_cycles"]))
        self.assertGreaterEqual(result["peak_inflight_tiles"], 1)
        self.assertLessEqual(result["peak_inflight_tiles"], min(c["buffer_slots"], len(tiles(c))))
        self.assertGreaterEqual(result["slot_wait_cycles"], 0)
        self.assertGreaterEqual(result["simulation_wall_seconds"], 0)
        self.assertTrue(math.isfinite(result["simulation_wall_seconds"]))
        self.assertAlmostEqual(result["compute_utilization"], metrics["compute_busy_cycles"] / result["cycles"])
        self.assertAlmostEqual(result["memory_utilization"], metrics["dma_busy_cycles"] / result["cycles"])
        self.assertAlmostEqual(result["mac_utilization"], metrics["macs"] / (peak_macs(c) * result["cycles"]))
        self.assertEqual(result["effective_macs_per_cycle"], peak_macs(c))
        total_bytes = metrics["read_bytes"] + metrics["write_bytes"]
        self.assertAlmostEqual(result["dma_payload_utilization"], total_bytes / (c["memory_bytes_per_cycle"] * result["cycles"]))
        self.assertAlmostEqual(result["arithmetic_intensity_macs_per_byte"], metrics["macs"] / total_bytes)
        self.assertEqual(result["resource_lower_bound_cycles"], max(metrics["compute_busy_cycles"], metrics["dma_busy_cycles"]))
        self.assertAlmostEqual(result["modeled_latency_ns"], result["cycles"] * 1000 / c["frequency_mhz"])
        return result

    def test_independent_arithmetic_and_edge_tiles(self):
        cases = [
            {},
            {"m": 1, "n": 1, "k": 1, "tile_m": 9, "tile_n": 7, "seed": 0},
            {"m": 5, "n": 7, "k": 13, "tile_m": 3, "tile_n": 4, "seed": MASK32},
            {"m": 3, "n": 9, "k": 5, "tile_m": 1, "tile_n": 8, "seed": 246813579},
            {"m": 7, "n": 3, "k": 19, "tile_m": 4, "tile_n": 2, "seed": 1 << 40},
        ]
        for case in cases:
            with self.subTest(config=case):
                self.check_case(case)

    def test_quantization_extremes_saturation_and_relu(self):
        cases = [
            {"input_zero": -128, "weight_zero": 127, "output_zero": -128, "multiplier": 2147483647, "shift": 31},
            {"input_zero": 127, "weight_zero": -128, "output_zero": 127, "multiplier": 2147483647, "shift": 0},
            {"input_zero": -128, "weight_zero": -128, "output_zero": 5, "shift": 0, "relu": True},
            {"input_zero": 127, "weight_zero": 127, "output_zero": -123, "multiplier": 0, "shift": 31, "relu": True},
            {"multiplier": 3, "shift": 1},
            {"multiplier": 7, "shift": 3, "relu": True, "output_zero": -16},
            {"multiplier": 1, "shift": 0},
        ]
        for case in cases:
            with self.subTest(quantization=case):
                self.check_case(case)

    def test_exact_serial_timing(self):
        for overrides in [
            {"m": 2, "n": 3, "k": 5, "tile_m": 2, "tile_n": 3, "buffer_slots": 3},
            {"m": 5, "n": 7, "k": 11, "tile_m": 2, "tile_n": 3, "buffer_slots": 1},
            {"m": 4, "n": 3, "k": 2, "tile_m": 1, "tile_n": 1, "buffer_slots": 1, "memory_latency": 0, "pipeline_cycles": 0},
        ]:
            with self.subTest(config=overrides):
                result = self.check_case(overrides)
                c = {**DEFAULTS, **overrides}
                self.assertEqual(result["cycles"], expected_metrics(c)["serial_cycles"])

    def test_event_and_cycle_agree_including_traces(self):
        with tempfile.TemporaryDirectory(prefix="npu-integration-") as temp:
            for index, overrides in enumerate([
                {"buffer_slots": 1},
                {"buffer_slots": 2},
                {"buffer_slots": 4, "memory_latency": 0, "pipeline_cycles": 0, "macs_per_cycle": 1000},
            ]):
                with self.subTest(config=overrides):
                    c = {**DEFAULTS, **overrides}
                    event_trace, cycle_trace = Path(temp) / f"event{index}.trace", Path(temp) / f"cycle{index}.trace"
                    event, _ = invoke(c, timing="event", trace=event_trace)
                    cycle, _ = invoke(c, timing="cycle", trace=cycle_trace)
                    # The host stepping mode and the trace path are run options,
                    # not architecture outcomes; they legitimately differ here.
                    def architecture_result(value: dict) -> dict:
                        copy = json.loads(json.dumps(value))
                        copy.pop("simulation_wall_seconds", None)
                        copy.pop("timing", None)
                        copy["config"]["sim"].pop("timing")
                        copy["config"]["sim"].pop("trace_path")
                        return copy
                    self.assertEqual(
                        architecture_result(event), architecture_result(cycle)
                    )
                    self.assertTrue(event_trace.is_file())
                    self.assertTrue(cycle_trace.is_file())
                    self.assertTrue(event_trace.read_text().strip())
                    self.assertEqual(event_trace.read_text(), cycle_trace.read_text())
                    self.check_trace(event_trace.read_text(), c, event)

    def test_independent_multislot_round_robin_timing(self):
        cases = [
            {"m": 4, "n": 5, "k": 7, "memory_latency": 0, "pipeline_cycles": 0, "memory_bytes_per_cycle": 10000, "macs_per_cycle": 10000, "buffer_slots": slots}
            for slots in (1, 2, 3, 6)
        ]
        cases.extend({
            "m": 1 + (i * 3) % 9, "n": 1 + (i * 5) % 11, "k": 1 + (i * 7) % 13,
            "tile_m": 1 + (i * 2) % 5, "tile_n": 1 + (i * 3) % 6,
            "macs_per_cycle": 1 + (i * 11) % 47,
            "memory_bytes_per_cycle": 1 + (i * 5) % 17,
            "memory_latency": i % 4, "pipeline_cycles": (i * 2) % 5,
            "buffer_slots": 1 + i % 5,
        } for i in range(16))
        with tempfile.TemporaryDirectory(prefix="npu-oracle-") as directory:
            for index, overrides in enumerate(cases):
                with self.subTest(config=overrides):
                    self.assert_schedule({**DEFAULTS, **overrides},
                                         Path(directory) / f"case{index}.csv")

    def test_double_buffered_panels_match_the_independent_schedule(self):
        cases = [
            {"m": 5, "n": 6, "k": 13, "tile_m": 2, "tile_n": 4, "tile_k": depth,
             "buffer_slots": slots, "panel_buffers": buffers}
            for depth in (2, 5) for slots in (1, 2, 3) for buffers in (1, 2)
        ]
        # The default is two panel buffers.
        cases.append({"m": 7, "n": 5, "k": 17, "tile_m": 3, "tile_n": 2, "tile_k": 4})
        rng = random.Random(3101)
        cases.extend({
            "m": rng.randint(1, 9), "n": rng.randint(1, 9), "k": rng.randint(2, 23),
            "tile_m": rng.randint(1, 5), "tile_n": rng.randint(1, 5), "tile_k": rng.randint(1, 8),
            "macs_per_cycle": rng.randint(1, 40), "memory_bytes_per_cycle": rng.randint(1, 12),
            "memory_latency": rng.randint(0, 6), "pipeline_cycles": rng.randint(0, 5),
            "buffer_slots": rng.randint(1, 4), "panel_buffers": rng.choice([1, 2]),
            "groups": rng.choice([1, 1, 2]),
        } for _ in range(20))
        with tempfile.TemporaryDirectory(prefix="npu-panels-") as directory:
            for index, overrides in enumerate(cases):
                with self.subTest(config=overrides):
                    self.check_case(overrides)
                    self.assert_schedule({**DEFAULTS, **overrides},
                                         Path(directory) / f"p{index}.csv")

    def test_single_panel_buffer_and_slot_is_serial(self):
        c = {"m": 5, "n": 7, "k": 11, "tile_m": 2, "tile_n": 3, "tile_k": 4,
             "buffer_slots": 1, "panel_buffers": 1}
        result = self.check_case(c)
        self.assertEqual(result["cycles"], expected_metrics({**DEFAULTS, **c})["serial_cycles"])
        # Serial: the loader waits out the compute of every non-final panel,
        # and the last panel's compute plus the store of every tile but the last.
        panels, stores = tile_work({**DEFAULTS, **c})
        self.assertEqual(result["panel_wait_cycles"],
                         sum(cycles for work in panels for _, _, cycles in work[:-1]))
        self.assertEqual(result["slot_wait_cycles"],
                         sum(work[-1][2] + store for work, (_, store) in zip(panels, stores[:-1])))
        overlapped = self.check_case({**c, "panel_buffers": 2})
        self.assertLess(overlapped["cycles"], result["cycles"])

    def test_panel_double_buffering_reaches_the_resource_bound(self):
        # With one panel buffer per slot the load of panel b+1 waits for the
        # compute of panel b, so the DMA idles while slots are free.
        c = {**DEFAULTS, "m": 64, "n": 64, "k": 512, "tile_m": 32, "tile_n": 32, "tile_k": 128,
             "macs_per_cycle": 256, "memory_bytes_per_cycle": 16, "memory_latency": 0}
        panels, _ = tile_work(c)
        # Prologue and epilogue: one panel load before compute starts and one
        # panel compute after the last load.
        slack = max(load for work in panels for _, load, _ in work) + \
            max(compute for work in panels for _, _, compute in work)
        for slots in (2, 4):
            with self.subTest(slots=slots):
                single = {**c, "buffer_slots": slots, "panel_buffers": 1}
                double = {**c, "buffer_slots": slots}
                serialised, _ = invoke(single, dump=False)
                overlapped, _ = invoke(double, dump=False)
                for config, result in ((single, serialised), (double, overlapped)):
                    oracle = timing_oracle(config)
                    for key in ORACLE_KEYS:
                        self.assertEqual(result[key], oracle[key], key)
                bound = overlapped["resource_lower_bound_cycles"]
                self.assertEqual(bound, serialised["resource_lower_bound_cycles"])
                self.assertLessEqual(overlapped["cycles"], bound + slack)
                self.assertGreater(serialised["cycles"], bound + slack)
                self.assertEqual(overlapped["scratchpad_bytes"] - serialised["scratchpad_bytes"],
                                 slots * (32 * 128 + 128 * 32))

    def test_panel_buffers_reserve_sram_and_reject_bad_counts(self):
        c = {**DEFAULTS, "m": 6, "n": 4, "k": 12, "tile_m": 3, "tile_n": 4, "tile_k": 5}
        single = expected_metrics({**c, "panel_buffers": 1})["scratchpad_bytes"]
        double = expected_metrics(c)["scratchpad_bytes"]
        # Each slot gains one panel: activations and weights of the deepest panel.
        self.assertEqual(double - single, c["buffer_slots"] * (3 * 5 + 5 * 4))
        fitted, _ = invoke({**c, "sram_bytes": single, "panel_buffers": 1}, dump=False)
        self.assertEqual(fitted["scratchpad_bytes"], single)
        proc = subprocess.run(command({**c, "sram_bytes": single}, dump=False),
                              capture_output=True, text=True, timeout=30)
        self.assert_rejected(proc, "the second panel buffer needs SRAM")
        self.assertIn("tile buffers exceed configured SRAM capacity", proc.stderr)
        # One reduction panel needs one buffer whatever the setting.
        whole = {**c, "tile_k": 0}
        self.assertEqual(expected_metrics(whole)["scratchpad_bytes"],
                         expected_metrics({**whole, "panel_buffers": 1})["scratchpad_bytes"])
        self.assertEqual(invoke(whole, dump=False)[0]["scratchpad_bytes"],
                         expected_metrics(whole)["scratchpad_bytes"])
        for value, fragment in (("0", "panel buffers must be 1 or 2"),
                                ("3", "panel buffers must be 1 or 2"),
                                ("-1", "--panel-buffers: expected an unsigned 64-bit integer"),
                                ("two", "--panel-buffers: expected an unsigned 64-bit integer")):
            with self.subTest(panel_buffers=value):
                proc = subprocess.run([str(BINARY), "--tile-k", "4", "--panel-buffers", value],
                                      capture_output=True, text=True, timeout=15)
                self.assert_rejected(proc, f"--panel-buffers {value}")
                self.assertIn(fragment, proc.stderr.strip().splitlines()[-1])

    def test_array_dataflows_match_the_independent_schedule(self):
        rng = random.Random(5150)
        cases = [{
            "m": rng.randint(1, 11), "n": rng.randint(1, 11), "k": rng.randint(1, 19),
            "tile_m": rng.randint(1, 6), "tile_n": rng.randint(1, 6),
            "tile_k": rng.choice([0, rng.randint(1, 7)]),
            "array_rows": rng.randint(1, 5), "array_cols": rng.randint(1, 5),
            "dataflow": ("os", "ws")[index % 2],
            "memory_bytes_per_cycle": rng.randint(1, 12), "memory_latency": rng.randint(0, 5),
            "pipeline_cycles": rng.randint(0, 4), "buffer_slots": rng.randint(1, 3),
            "panel_buffers": rng.choice([1, 2]),
        } for index in range(24)]
        with tempfile.TemporaryDirectory(prefix="npu-dataflow-") as directory:
            for index, overrides in enumerate(cases):
                with self.subTest(config=overrides):
                    self.check_case(overrides)
                    self.assert_schedule({**DEFAULTS, **overrides},
                                         Path(directory) / f"d{index}.csv")

    def test_dataflow_cycles_of_one_panel(self):
        # One tile and one panel, so compute_busy_cycles is that panel's time.
        for ar, ac, rows, cols, depth in [(4, 4, 8, 8, 16), (4, 4, 2, 8, 16), (3, 5, 7, 11, 13),
                                          (8, 2, 1, 9, 30), (1, 1, 3, 2, 5)]:
            base = {"m": rows, "n": cols, "k": depth, "tile_m": rows, "tile_n": cols,
                    "array_rows": ar, "array_cols": ac, "pipeline_cycles": 3}
            skew = ar + ac - 2
            waves = ceildiv(rows, ar) * ceildiv(cols, ac)
            folds = ceildiv(depth, ar) * ceildiv(cols, ac)
            # A fold of at least A_r rows hides the next weight preload.
            ws = folds * rows + ar + skew if rows >= ar else folds * ar + rows + skew
            for dataflow, cycles in (("os", waves * depth + skew), ("ws", ws)):
                with self.subTest(array=(ar, ac), tile=(rows, cols, depth), dataflow=dataflow):
                    result = self.check_case({**base, "dataflow": dataflow})
                    self.assertEqual(result["compute_busy_cycles"], cycles + 3)
                    self.assertEqual(result["config"]["arch"]["compute"]["dataflow"], dataflow)

    def test_each_dataflow_quantizes_a_different_dimension(self):
        # os rounds the tile's rows up to A_r; ws rounds the panel's K up to A_r.
        array = {"array_rows": 4, "array_cols": 4, "pipeline_cycles": 0, "n": 4, "tile_n": 4}
        ragged_rows = {**array, "m": 5, "tile_m": 5, "k": 64}
        shallow_k = {**array, "m": 64, "tile_m": 64, "k": 5}
        busy = {}
        for name, shape in (("ragged_rows", ragged_rows), ("shallow_k", shallow_k)):
            for dataflow in ("os", "ws"):
                busy[name, dataflow] = self.check_case(
                    {**shape, "dataflow": dataflow})["compute_busy_cycles"]
        self.assertLess(busy["ragged_rows", "ws"], busy["ragged_rows", "os"])
        self.assertLess(busy["shallow_k", "os"], busy["shallow_k", "ws"])

    def test_dataflow_requires_an_array(self):
        for arguments, fragment in [
            (["--dataflow", "ws"], "--dataflow requires --array-rows and --array-cols"),
            (["--dataflow", "os"], "--dataflow requires --array-rows and --array-cols"),
            (["--dataflow", "is", "--array-rows", "2", "--array-cols", "2"],
             "--dataflow: expected os|ws, got 'is'"),
            (["--dataflow", "ws", "--array-rows", "2", "--array-cols", "2", "--weight-encoding",
              "sparse"], "sparse zero-skipping is supported only by the throughput datapath"),
        ]:
            with self.subTest(arguments=arguments):
                proc = subprocess.run([str(BINARY), *arguments], capture_output=True, text=True,
                                      timeout=15)
                self.assert_rejected(proc, f"dataflow: {arguments}")
                self.assertIn(fragment, proc.stderr.strip().splitlines()[-1])
        with tempfile.TemporaryDirectory(prefix="npu-dataflow-config-") as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"arch": {"compute": {"dataflow": "ws"}}}')
            proc = subprocess.run([str(BINARY), "--config", str(path)], capture_output=True,
                                  text=True, timeout=15)
            self.assert_rejected(proc, "ws without an array in a config file")
            self.assertIn("weight-stationary dataflow requires --array-rows and --array-cols",
                          proc.stderr.strip().splitlines()[-1])
            # The echoed config of a vector run carries dataflow os and reloads.
            echoed = subprocess.run([str(BINARY), "--describe-config"], capture_output=True,
                                    text=True, timeout=15)
            self.assertEqual(echoed.returncode, 0, echoed.stderr)
            self.assertEqual(json.loads(echoed.stdout)["arch"]["compute"]["dataflow"], "os")
            path.write_text(echoed.stdout)
            proc = subprocess.run([str(BINARY), "--config", str(path)], capture_output=True,
                                  text=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            # An array from a file and the dataflow from a flag combine.
            path.write_text('{"arch": {"compute": {"array_rows": 2, "array_cols": 2}}}')
            proc = subprocess.run([str(BINARY), "--config", str(path), "--dataflow", "ws"],
                                  capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["config"]["arch"]["compute"]["dataflow"], "ws")

    def test_resource_scaling_and_memory_bound(self):
        base = {**DEFAULTS, "m": 16, "n": 16, "k": 32, "tile_m": 8, "tile_n": 8, "buffer_slots": 3}
        slower, _ = invoke({**base, "memory_bytes_per_cycle": 2})
        faster, _ = invoke({**base, "memory_bytes_per_cycle": 16})
        self.assertLess(faster["cycles"], slower["cycles"])
        self.assertEqual(faster["output_hash"], slower["output_hash"])
        serial, _ = invoke({**base, "buffer_slots": 1})
        overlap, _ = invoke(base)
        self.assertLess(overlap["cycles"], serial["cycles"])
        memory_bound = {**base, "memory_bytes_per_cycle": 1, "memory_latency": 20, "macs_per_cycle": 64}
        normal, _ = invoke(memory_bound)
        doubled, _ = invoke({**memory_bound, "macs_per_cycle": 128})
        self.assertLessEqual(doubled["cycles"], normal["cycles"])
        self.assertGreater(doubled["cycles"], normal["cycles"] * 0.8)
        self.assertGreaterEqual(doubled["cycles"], doubled["dma_busy_cycles"])
        self.assertEqual(normal["read_bytes"], doubled["read_bytes"])
        self.assertEqual(normal["output_hash"], doubled["output_hash"])
        compute_bound = {**base, "memory_bytes_per_cycle": 100000, "memory_latency": 0, "macs_per_cycle": 1}
        compute_slow, _ = invoke(compute_bound)
        compute_fast, _ = invoke({**compute_bound, "macs_per_cycle": 2})
        self.assertLess(compute_fast["cycles"], compute_slow["cycles"] * 0.65)

    def test_grouped_gemm_numerics_traffic_and_schedule(self):
        cases = [
            {"groups": 2},
            {"groups": 3, "m": 5, "n": 3, "k": 4, "tile_m": 2, "tile_n": 2},
            # Depthwise 3x3: N = 1, K = 9, one group per channel.
            {"groups": 12, "m": 7, "n": 1, "k": 9, "tile_m": 4, "tile_n": 1, "buffer_slots": 3},
            {"groups": 4, "m": 3, "n": 5, "k": 2, "tile_m": 3, "tile_n": 4, "buffer_slots": 1},
        ]
        with tempfile.TemporaryDirectory(prefix="npu-groups-") as directory:
            for index, overrides in enumerate(cases):
                with self.subTest(config=overrides):
                    result = self.check_case(overrides)
                    c = {**DEFAULTS, **overrides}
                    self.assertEqual(result["config"]["shape"]["groups"], c["groups"])
                    # The independent scheduler walks tiles group-major, then
                    # row, then column; trace ids must follow the same order.
                    self.assert_schedule(c, Path(directory) / f"groups{index}.csv")

    def test_timing_only_reproduces_the_bit_exact_schedule(self):
        cases = [
            {},
            {"groups": 3, "m": 9, "n": 5, "k": 11, "tile_m": 4, "tile_n": 2},
            {"m": 33, "n": 17, "k": 65, "tile_m": 8, "tile_n": 8, "tile_k": 16,
             "weight_cache_bytes": 2048, "dma_burst_bytes": 32},
        ]
        for overrides in cases:
            with self.subTest(config=overrides):
                c = {**DEFAULTS, **overrides}
                exact, _ = invoke(c, dump=False)
                timed, _ = invoke({**c, "timing_only": True}, dump=False)
                self.assertEqual(exact["numerics"], "bit-exact")
                self.assertEqual(timed["numerics"], "skipped")
                self.assertIsNone(timed["output_hash"])
                self.assertIsNone(timed["numerical_match"])
                self.assertIs(timed["config"]["sim"]["timing_only"], True)
                # Dense timing never reads tensor values.
                for key, value in exact.items():
                    if key not in {"numerics", "output_hash", "numerical_match",
                                   "simulation_wall_seconds", "config"}:
                        self.assertEqual(timed[key], value, key)

    def test_timing_only_beyond_the_bit_exact_limit(self):
        # Llama-3.2-1B FFN up-projection over a 2048-token prompt: 3.4e10 MACs.
        c = {**DEFAULTS, "m": 2048, "n": 8192, "k": 2048, "tile_m": 128, "tile_n": 128,
             "sram_bytes": 4 << 20, "macs_per_cycle": 1024, "memory_bytes_per_cycle": 64}
        rejected = subprocess.run(command(c, dump=False), capture_output=True, text=True,
                                  timeout=30)
        self.assert_rejected(rejected, "bit-exact runs stop at 4e9 MACs")
        self.assertIn("workload exceeds 4000000000 MACs for a bit-exact run", rejected.stderr)
        result, _ = invoke({**c, "timing_only": True}, dump=False, timeout=120)
        self.assertEqual(result["numerics"], "skipped")
        self.assertEqual(result["macs"], 2048 * 8192 * 2048)
        self.assertGreaterEqual(result["cycles"], result["macs"] // 1024)
        self.assertGreaterEqual(result["cycles"], result["resource_lower_bound_cycles"])
        for extra, fragment in ((["--dump-output"], "--timing-only runs without tensors"),
                                (["--weight-encoding", "palette"], "dense weights only"),
                                (["--k", "10001", "--m", "1000000", "--n", "1000",
                                  "--tile-m", "1000", "--tile-n", "1000"],
                                 "the --timing-only limit")):
            with self.subTest(extra=extra):
                proc = subprocess.run([str(BINARY), "--timing-only", *extra],
                                      capture_output=True, text=True, timeout=30)
                self.assert_rejected(proc, "timing-only misuse")
                self.assertIn(fragment, proc.stderr)
        with tempfile.TemporaryDirectory(prefix="npu-timing-only-") as directory:
            path = Path(directory) / "t.npu"
            path.write_text("NPU1 1 1 1\n0\n0\n0\n")
            proc = subprocess.run([str(BINARY), "--timing-only", "--input", str(path)],
                                  capture_output=True, text=True, timeout=30)
            self.assert_rejected(proc, "--input needs a bit-exact run")
            self.assertIn("--input and --dump-output need a bit-exact run", proc.stderr)

    def test_resnet50_conv1_gemm_is_bit_exact(self):
        # conv1 lowered to GEMM: 112*112 output pixels, 64 filters, 7*7*3 taps.
        c = {**DEFAULTS, "m": 12544, "n": 64, "k": 147, "tile_m": 64, "tile_n": 64,
             "macs_per_cycle": 256, "memory_bytes_per_cycle": 32}
        result, _ = invoke(c, timeout=120)
        self.assertIs(result["numerical_match"], True)
        self.assertEqual(result["numerics"], "bit-exact")
        self.assertEqual(result["macs"], 12544 * 64 * 147)
        accumulators, outputs = result["accumulators"], result["outputs"]
        self.assertEqual(len(outputs), 12544 * 64)
        self.assertEqual(result["output_hash"], expected_hash(accumulators, outputs))
        problem = fixtures(c)
        rng = random.Random(147)
        for _ in range(512):
            row, channel = rng.randrange(12544), rng.randrange(64)
            index = row * 64 + channel
            self.assertEqual((accumulators[index], outputs[index]),
                             output_value(c, problem, row, channel), (row, channel))
        # A different tiling, panel depth and slot count computes the same tensor.
        other, _ = invoke({**c, "tile_m": 8, "tile_n": 16, "tile_k": 32, "buffer_slots": 3},
                          dump=False, timeout=120)
        self.assertEqual(other["output_hash"], result["output_hash"])

    def test_llama_gemv_runs(self):
        # Llama-3.2-1B decode: one token through a 2048 -> 8192 projection.
        c = {**DEFAULTS, "m": 1, "n": 8192, "k": 2048, "tile_m": 1, "tile_n": 256,
             "sram_bytes": 4 << 20, "memory_bytes_per_cycle": 64}
        result, _ = invoke(c, dump=False, timeout=120)
        self.assertIs(result["numerical_match"], True)
        self.assertEqual(result["macs"], 8192 * 2048)
        # Every weight byte is read once: a GEMV is bound by weight traffic.
        self.assertGreaterEqual(result["read_bytes"], 8192 * 2048)
        timed, _ = invoke({**c, "timing_only": True}, dump=False, timeout=120)
        self.assertEqual(timed["cycles"], result["cycles"])

    def test_dimension_and_group_limits(self):
        for arguments, fragment in [
            (["--m", "1048577"], "matrix dimensions must be in 1..1048576"),
            (["--groups", "0"], "groups must be in 1..1048576"),
            (["--groups", "1024", "--n", "1025"], "groups*N and groups*K must be at most 1048576"),
            (["--groups", "2", "--n", "1", "--k", "524289"], "groups*N and groups*K"),
            (["--tile-m", "1048577"], "tile dimensions must be in 1..1048576"),
        ]:
            with self.subTest(arguments=arguments):
                proc = subprocess.run([str(BINARY), *arguments], capture_output=True, text=True,
                                      timeout=15)
                self.assert_rejected(proc, f"limit: {arguments}")
                self.assertIn(fragment, proc.stderr)

    def test_frequency_changes_latency_not_cycles(self):
        slow, _ = invoke({**DEFAULTS, "frequency_mhz": 100})
        fast, _ = invoke({**DEFAULTS, "frequency_mhz": 400})
        self.assertEqual(slow["cycles"], fast["cycles"])
        self.assertEqual(slow["output_hash"], fast["output_hash"])
        self.assertAlmostEqual(slow["modeled_latency_ns"], fast["modeled_latency_ns"] * 4)

    def test_external_workloads_extreme_operands_and_known_rounding(self):
        cases = [
            (
                {"m": 2, "n": 3, "k": 2, "shift": 8},
                ([-128, 127, 0, -1], [127, -128, 1, -1, 0, 127], [-100000, 100000, 23]),
            ),
            (
                {"m": 1, "n": 2, "k": 1, "multiplier": 2147483647, "shift": 31},
                ([0], [0, 0], [-(1 << 31), (1 << 31) - 1]),
            ),
            (
                {"m": 1, "n": 4, "k": 1, "multiplier": 1, "shift": 1},
                ([0], [0, 0, 0, 0], [-3, -1, 1, 3]),
            ),
        ]
        with tempfile.TemporaryDirectory(prefix="npu-input-") as directory:
            for index, (overrides, data) in enumerate(cases):
                with self.subTest(case=index):
                    c = {**DEFAULTS, "input_zero": 0, "weight_zero": 0, **overrides}
                    path = Path(directory) / f"case{index}.npu"
                    values = ["NPU1", str(c["m"]), str(c["n"]), str(c["k"])]
                    values.extend(str(value) for array in data for value in array)
                    path.write_text(" ".join(values) + "\n")
                    result, _ = invoke(c, input_path=path)
                    expected_acc, expected_quant = expected_outputs(c, data)
                    self.assertEqual(result["accumulators"], expected_acc)
                    self.assertEqual(result["outputs"], expected_quant)
                    self.assertEqual(result["output_hash"], expected_hash(expected_acc, expected_quant))
                    self.assertIs(result["numerical_match"], True)
                    for name in ("m", "n", "k"):
                        self.assertEqual(result["config"]["shape"][name], c[name])
                    if index == 2:
                        self.assertEqual(result["outputs"], [-2, -1, 1, 2])

    def test_reject_malformed_or_overflowing_workload_files(self):
        malformed = [
            "", "NPU2 1 1 1 0 0 0", "NPU1 0 1 1", "NPU1 -1 1 1",
            "NPU1 1 1 1 0 0", "NPU1 1 1 1 128 0 0", "NPU1 1 1 1 -129 0 0",
            "NPU1 1 1 1 0 -129 0", "NPU1 1 1 1 0 0 2147483648",
            "NPU1 1 1 1 0 0 0 unexpected", "NPU1 1 1 1 x 0 0",
            "NPU1 18446744073709551615 18446744073709551615 2",
            # Long zero-padded values are numerically tiny: this checks token
            # length limits independently of ordinary integer-overflow checks.
            "NPU1 " + "0" * 40 + "1 1 1 0 0 0",
            "NPU1 1 1 1 " + "0" * 40 + " 0 0",
            "NPU1 1 1 1 0 " + "0" * 40 + " 0",
            "NPU1 1 1 1 0 0 " + "0" * 40,
            "NPU1 1 1 1 0 0 0 " + "0" * 40,
        ]
        with tempfile.TemporaryDirectory(prefix="npu-invalid-input-") as directory:
            path = Path(directory) / "input.npu"
            for index, contents in enumerate(malformed):
                with self.subTest(case=index):
                    path.write_text(contents)
                    proc = subprocess.run([str(BINARY), "--input", str(path)], capture_output=True, text=True, timeout=15)
                    self.assert_rejected(proc, "bad workload must be rejected without crashing")
            path.write_text("NPU1 1 1 1 1 1 2147483647\n")
            proc = subprocess.run(
                [str(BINARY), "--input", str(path), "--input-zero", "0", "--weight-zero", "0"],
                capture_output=True, text=True, timeout=15,
            )
            self.assert_rejected(proc, "INT32 overflow must be rejected without crashing")
            path.write_text("NPU1 1 1 1 0 0 0\n")
            proc = subprocess.run([str(BINARY), "--input", str(path), "--m", "1"], capture_output=True, text=True, timeout=15)
            self.assert_rejected(proc, "ambiguous input and CLI dimensions must be rejected without crashing")

    def test_empty_explicit_paths_are_rejected(self):
        for arguments in [
            ["--input", ""],
            ["--input", "", "--m", "1", "--n", "1", "--k", "1"],
            ["--trace", ""],
        ]:
            with self.subTest(arguments=arguments):
                proc = subprocess.run([str(BINARY), *arguments], capture_output=True, text=True, timeout=5)
                self.assert_rejected(proc, "an explicit empty path must not be silently skipped")

    def test_cycle_step_budget_rejects_huge_wait_event_mode_remains_usable(self):
        c = {
            **DEFAULTS, "m": 1, "n": 1, "k": 1, "tile_m": 1, "tile_n": 1,
            "buffer_slots": 1, "memory_latency": 50000000, "pipeline_cycles": 0,
            "memory_bytes_per_cycle": 1, "macs_per_cycle": 1,
        }
        proc = subprocess.run(command(c, timing="cycle", dump=False), capture_output=True, text=True, timeout=5)
        self.assert_rejected(proc, "cycle mode must reject excessive host stepping work before simulation")
        # Event mode takes a handful of scheduling events despite the large
        # physical cycle count.  No trace, input file or output dump is needed.
        event, _ = invoke(c, timing="event", dump=False)
        self.assertEqual(event["cycles"], 100000008)
        self.assertIs(event["numerical_match"], True)
        self.assertNotIn("accumulators", event)
        self.assertNotIn("outputs", event)

    def test_reject_invalid_and_unsafe_inputs(self):
        arguments = [
            ["--m", "0"], ["--n", "-1"], ["--k", "18446744073709551616"],
            ["--tile-m", "0"], ["--tile-n", "0"], ["--macs-per-cycle", "0"],
            ["--memory-bytes-per-cycle", "0"], ["--buffer-slots", "0"],
            ["--sram-bytes", "1"], ["--frequency-mhz", "0"],
            ["--frequency-mhz", "nan"], ["--frequency-mhz", "inf"],
            ["--frequency-mhz", "abc"], ["--frequency-mhz", "1e999"], ["--frequency-mhz", " 200"],
            ["--input-zero", "-129"], ["--weight-zero", "128"],
            ["--output-zero", "128"], ["--shift", "32"],
            ["--multiplier", "2147483648"], ["--multiplier", "-1"],
            ["--timing", "wrong"], ["--m", "4junk"], ["--m"], ["--unknown-option"],
            ["--m", "18446744073709551615", "--n", "18446744073709551615"],
            ["--m", "1", "--n", "1", "--k", "200000", "--input-zero", "-128", "--weight-zero", "-128", "--sram-bytes", "10000000"],
        ]
        for args in arguments:
            with self.subTest(arguments=args):
                proc = subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=15)
                self.assert_rejected(proc, f"invalid input must be rejected without crashing: {args}")


def main() -> int:
    global BINARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    BINARY = args.binary.resolve()
    if not BINARY.is_file():
        parser.error(f"simulator binary not found: {BINARY}")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SimulatorIntegration)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
