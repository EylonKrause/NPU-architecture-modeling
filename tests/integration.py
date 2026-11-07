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


def fixtures(c: dict) -> tuple[list[int], list[int], list[int]]:
    """Explicit portable fixture stream, without C/C++ random libraries."""
    state = c["seed"] & MASK32

    def values(count: int, modulus: int, offset: int) -> list[int]:
        nonlocal state
        result = []
        for _ in range(count):
            state = (1664525 * state + 1013904223) & MASK32
            result.append(state % modulus - offset)
        return result

    return (
        values(c["m"] * c["k"], 31, 15),
        values(c["k"] * c["n"], 31, 15),
        values(c["n"], 127, 63),
    )


def expected_outputs(c: dict, problem: tuple[list[int], list[int], list[int]] | None = None) -> tuple[list[int], list[int]]:
    activations, weights, biases = fixtures(c) if problem is None else problem
    accumulators, outputs = [], []
    for row in range(c["m"]):
        for col in range(c["n"]):
            acc = biases[col] + sum(
                (activations[row * c["k"] + inner] - c["input_zero"])
                * (weights[inner * c["n"] + col] - c["weight_zero"])
                for inner in range(c["k"])
            )
            if not -(1 << 31) <= acc < (1 << 31):
                raise OverflowError("reference accumulator is outside INT32")
            accumulators.append(acc)
            product = acc * c["multiplier"]
            if c["shift"]:
                magnitude = (abs(product) + (1 << (c["shift"] - 1))) // (1 << c["shift"])
                rounded = magnitude if product >= 0 else -magnitude
            else:
                rounded = product
            quantized = rounded + c["output_zero"]
            if c["relu"]:
                quantized = max(quantized, c["output_zero"])
            outputs.append(min(127, max(-128, quantized)))
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
    return [
        (min(c["tile_m"], c["m"] - row), min(c["tile_n"], c["n"] - col))
        for row in range(0, c["m"], c["tile_m"])
        for col in range(0, c["n"], c["tile_n"])
    ]


def expected_metrics(c: dict) -> dict[str, int]:
    reads = writes = compute = dma = serial_cycles = 0
    for rows, cols in tiles(c):
        read = rows * c["k"] + c["k"] * cols + 4 * cols
        write = rows * cols
        stage = ceildiv(rows * cols * c["k"], c["macs_per_cycle"]) + c["pipeline_cycles"]
        transfer = 2 * c["memory_latency"] + ceildiv(read, c["memory_bytes_per_cycle"]) + ceildiv(write, c["memory_bytes_per_cycle"])
        reads += read
        writes += write
        compute += stage
        dma += transfer
        serial_cycles += stage + transfer
    max_rows, max_cols = min(c["m"], c["tile_m"]), min(c["n"], c["tile_n"])
    scratchpad = c["buffer_slots"] * (
        max_rows * c["k"] + c["k"] * max_cols + 4 * max_cols + 5 * max_rows * max_cols
    )
    return {
        "macs": c["m"] * c["n"] * c["k"],
        "read_bytes": reads, "write_bytes": writes,
        "compute_busy_cycles": compute, "dma_busy_cycles": dma,
        "scratchpad_bytes": scratchpad, "serial_cycles": serial_cycles,
    }


def timing_oracle(c: dict) -> dict:
    """Timestamp scheduler independent of SystemC threads and delta cycles.

    Complete every resource finishing at T first.  Close all zero-time FIFO
    transitions, then schedule work at T.  The shared non-preemptive DMA must
    select a ready request immediately, using load/store round robin.  Model
    state is per-tile rather than a copy of the simulator's coroutine code.
    """
    shape = tiles(c)
    state = ["unissued"] * len(shape)
    sizes = [(r * c["k"] + c["k"] * col + 4 * col, r * col) for r, col in shape]
    compute_cycles = [ceildiv(r * col * c["k"], c["macs_per_cycle"]) + c["pipeline_cycles"] for r, col in shape]
    spans = []
    time = issued = finished = inflight = peak = wait_cycles = 0
    loader = store = dma = compute = None
    next_loader_attempt = 0
    prefer_read = True

    def span(stage: str, tile: int, duration: int, size: int) -> None:
        spans.append({"stage": stage, "tile_id": tile, "start_cycle": time, "end_cycle": time + duration, "bytes": size})

    while finished < len(shape):
        # Every completion at the same physical timestamp is visible before
        # arbitration.  Resource completion order therefore cannot select a
        # stale winner when load and store become ready simultaneously.
        if dma is not None and dma[0] == time:
            _, stage, tile = dma
            dma = None
            if stage == "load":
                state[tile], loader = "loaded", None
                next_loader_attempt = time
            else:
                state[tile], store = "done", None
                inflight -= 1
                finished += 1
        if compute is not None and compute[0] == time:
            _, tile = compute
            state[tile], compute = "computed", None

        if compute is None:
            ready = next((i for i, s in enumerate(state) if s == "loaded"), None)
            if ready is not None:
                state[ready] = "computing"
                compute = time + compute_cycles[ready], ready
                span("compute", ready, compute_cycles[ready], 0)
        if store is None:
            ready = next((i for i, s in enumerate(state) if s == "computed"), None)
            if ready is not None:
                state[ready], store = "store_wait", ready
        if loader is None and issued < len(shape) and inflight < c["buffer_slots"]:
            loader = issued
            issued += 1
            state[loader] = "load_wait"
            inflight += 1
            peak = max(peak, inflight)
            wait_cycles += time - next_loader_attempt

        if dma is None:
            read_ready = loader is not None and state[loader] == "load_wait"
            write_ready = store is not None and state[store] == "store_wait"
            if read_ready or write_ready:
                read = read_ready and (not write_ready or prefer_read)
                stage, tile = ("load", loader) if read else ("store", store)
                size = sizes[tile][0 if read else 1]
                duration = c["memory_latency"] + ceildiv(size, c["memory_bytes_per_cycle"])
                dma = time + duration, stage, tile
                state[tile] = "loading" if read else "storing"
                prefer_read = not read
                span(stage, tile, duration, size)

        if finished == len(shape):
            break
        next_events = [event[0] for event in (dma, compute) if event is not None]
        if not next_events:
            raise AssertionError("independent timing scheduler deadlocked")
        time = min(next_events)
    return {
        "cycles": time, "peak_inflight_tiles": peak, "slot_wait_cycles": wait_cycles,
        "trace": [{**entry, "block_id": 0} for entry in sorted(spans, key=lambda x: (x["start_cycle"], x["stage"], x["tile_id"]))],
    }


def command(c: dict, *, timing: str = "event", dump: bool = True, trace: Path | None = None, input_path: Path | None = None) -> list[str]:
    result = [str(BINARY)]
    for key, value in c.items():
        if input_path is not None and key in {"m", "n", "k"}:
            continue
        if key == "relu":
            if value:
                result.append("--relu")
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


def invoke(c: dict, **kwargs) -> tuple[dict, subprocess.CompletedProcess]:
    proc = subprocess.run(command(c, **kwargs), capture_output=True, text=True, timeout=30)
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
        records = list(csv.DictReader(text.splitlines()))
        self.assertEqual(len(records), 3 * len(tiles(c)))
        self.assertEqual(
            set(records[0]), {"stage", "tile_id", "start_cycle", "end_cycle", "bytes", "block_id"}
        )
        stages: dict[tuple[str, int], tuple[int, int, int]] = {}
        for record in records:
            stage, tile_id = record["stage"], int(record["tile_id"])
            self.assertIn(stage, {"load", "compute", "store"})
            self.assertTrue(0 <= tile_id < len(tiles(c)))
            key = stage, tile_id
            self.assertNotIn(key, stages, f"duplicate stage for tile: {key}")
            start, end, size = (int(record[k]) for k in ("start_cycle", "end_cycle", "bytes"))
            self.assertTrue(0 <= start < end <= result["cycles"])
            rows, cols = tiles(c)[tile_id]
            if stage == "load":
                expected_bytes = rows * c["k"] + c["k"] * cols + 4 * cols
                duration = c["memory_latency"] + ceildiv(expected_bytes, c["memory_bytes_per_cycle"])
                self.assertEqual(size, expected_bytes)
            elif stage == "store":
                expected_bytes = rows * cols
                duration = c["memory_latency"] + ceildiv(expected_bytes, c["memory_bytes_per_cycle"])
                self.assertEqual(size, expected_bytes)
            else:
                duration = ceildiv(rows * cols * c["k"], c["macs_per_cycle"]) + c["pipeline_cycles"]
                self.assertEqual(size, 0)
            self.assertEqual(end - start, duration, f"incorrect duration: {key}")
            stages[key] = start, end, size
        for tile_id in range(len(tiles(c))):
            self.assertLessEqual(stages["load", tile_id][1], stages["compute", tile_id][0])
            self.assertLessEqual(stages["compute", tile_id][1], stages["store", tile_id][0])
        for resource in ({"load", "store"}, {"compute"}):
            intervals = sorted((start, end) for (stage, _), (start, end, _) in stages.items() if stage in resource)
            for previous, current in zip(intervals, intervals[1:]):
                self.assertLessEqual(previous[1], current[0], f"overlap on exclusive resource {resource}")
        self.assertEqual(max(end for _, end, _ in stages.values()), result["cycles"])
        # An acquired slot may wait for DMA before its load span starts, so this
        # is an independent lower bound on actual in-flight occupancy.
        occupancy_events = []
        for tile_id in range(len(tiles(c))):
            occupancy_events.extend([(stages["load", tile_id][0], 1), (stages["store", tile_id][1], -1)])
        occupancy = peak = 0
        for _, change in sorted(occupancy_events):
            occupancy += change
            peak = max(peak, occupancy)
        self.assertEqual(occupancy, 0)
        self.assertLessEqual(peak, c["buffer_slots"])
        self.assertLessEqual(peak, result["peak_inflight_tiles"])

    def check_case(self, overrides: dict, timing: str = "event") -> dict:
        c = {**DEFAULTS, **overrides}
        result, _ = invoke(c, timing=timing)
        accumulators, outputs = expected_outputs(c)
        self.assertEqual(result["accumulators"], accumulators)
        self.assertEqual(result["outputs"], outputs)
        self.assertIs(result["numerical_match"], True)
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
        self.assertAlmostEqual(result["mac_utilization"], metrics["macs"] / (c["macs_per_cycle"] * result["cycles"]))
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
                    c = {**DEFAULTS, **overrides}
                    oracle = timing_oracle(c)
                    path = Path(directory) / f"case{index}.csv"
                    actual, _ = invoke(c, trace=path)
                    for key in ("cycles", "peak_inflight_tiles", "slot_wait_cycles"):
                        self.assertEqual(actual[key], oracle[key], key)
                    records = list(csv.DictReader(path.read_text().splitlines()))
                    parsed = [{key: value if key == "stage" else int(value) for key, value in row.items()} for row in records]
                    self.assertEqual(parsed, oracle["trace"], "independent schedule trace differs")

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
