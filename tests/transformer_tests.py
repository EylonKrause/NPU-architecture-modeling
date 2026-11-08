#!/usr/bin/env python3
"""npu-transformer-demo: outputs against an independent scalar decoder in each storage
precision; cycles, traffic and SRAM against the stage schedule; KV-cache lifecycle,
progress events and CLI errors."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

BINARY: Path
# SystemC prints its banner on stderr before sc_main; suppressing it leaves only the demo's
# diagnostics on stderr.
ENV = {**os.environ, "SYSTEMC_DISABLE_COPYRIGHT_MESSAGE": "1"}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def unword(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def stored(value: float, precision: str) -> float:
    value = f32(value)
    if precision == "fp32":
        return value
    if precision == "fp16":
        try:
            return struct.unpack("<e", struct.pack("<e", value))[0]
        except OverflowError:
            return math.copysign(math.inf, value)
    # bfloat16 by nearest-neighbor search with ties to even, independent of the C++
    # bit-add rounding. Float64 holds every binary32-to-bfloat16 distance exactly.
    if not math.isfinite(value) or value == 0:
        return value
    sign = -1 if value < 0 else 1
    positive = abs(value)
    lower_word = word(positive) >> 16
    low = unword(lower_word << 16)
    high = math.ldexp(1.0, 128) if lower_word == 0x7F7F else unword((lower_word + 1) << 16)
    distance_low, distance_high = positive - low, high - positive
    pick_high = distance_high < distance_low or (distance_high == distance_low and lower_word & 1)
    rounded = high if pick_high else low
    if rounded == math.ldexp(1.0, 128):
        rounded = math.inf
    return math.copysign(rounded, sign)


def add(lhs: float, rhs: float) -> float:
    return f32(lhs + rhs)


def mul(lhs: float, rhs: float) -> float:
    return f32(lhs * rhs)


def reference(
    *, dim=16, heads=4, kv_heads=2, ff_dim=32, layers=1, tokens=6, seed=1, precision="fp16"
) -> list[list[float]]:
    """Full-sequence tensor path; all K/V of a layer computed before a causal masked read.

    The simulator runs the prompt as one batch and each decode token against an
    incremental cache. Every reduction result and materialized tensor passes through
    a binary32 or storage-format cast.
    """

    def cast(x):
        return stored(x, precision)

    head_dim, kv_dim = dim // heads, dim // heads * kv_heads

    def weights(rows, cols, matrix):
        # 1/64 up to 255 rows, then 1/sqrt(rows) rounded to a power of two, over 8.
        divisor = 8 * 2 ** max(3, (rows.bit_length() - 1) // 2)
        return [
            [
                cast(((row * 17 + col * 13 + seed % 101 + matrix * 7) % 31 - 15) / divisor)
                for col in range(cols)
            ]
            for row in range(rows)
        ]

    def dense(vector, matrix):
        output = []
        for col in range(len(matrix[0])):
            accumulated = 0.0
            for row, element in enumerate(vector):
                accumulated = add(accumulated, mul(element, matrix[row][col]))
            output.append(cast(accumulated))
        return output

    def norm(vector, which):
        square = 0.0
        for element in vector:
            square = add(square, mul(element, element))
        inv = f32(1 / f32(math.sqrt(add(f32(square / dim), f32(1e-5)))))
        gamma = [cast(1 + ((i * 3 + seed % 101 + which * 11) % 9 - 4) / 32) for i in range(dim)]
        return [cast(mul(mul(x, inv), g)) for x, g in zip(vector, gamma, strict=True)]

    def rotate(vector, count, position):
        result = list(vector)
        for head in range(count):
            for pair in range(0, head_dim, 2):
                angle = position / (10000 ** (pair / head_dim))
                co, si = f32(math.cos(angle)), f32(math.sin(angle))
                at = head * head_dim + pair
                left, right = vector[at : at + 2]
                result[at] = cast(add(mul(left, co), -mul(right, si)))
                result[at + 1] = cast(add(mul(left, si), mul(right, co)))
        return result

    def block(inputs, layer):
        base = 7 * layer
        matrices = [
            weights(dim, dim, base + 1),
            weights(dim, kv_dim, base + 2),
            weights(dim, kv_dim, base + 3),
            weights(dim, dim, base + 4),
            weights(dim, ff_dim, base + 5),
            weights(dim, ff_dim, base + 6),
            weights(ff_dim, dim, base + 7),
        ]
        normalized = [norm(vector, 2 * layer + 1) for vector in inputs]
        queries = [
            rotate(dense(vector, matrices[0]), heads, pos) for pos, vector in enumerate(normalized)
        ]
        keys = [
            rotate(dense(vector, matrices[1]), kv_heads, pos)
            for pos, vector in enumerate(normalized)
        ]
        values = [dense(vector, matrices[2]) for vector in normalized]
        scale = f32(1 / f32(math.sqrt(head_dim)))
        outputs = []
        for position, query in enumerate(queries):
            context = []
            for head in range(heads):
                kv_head = head // (heads // kv_heads)
                scores = []
                for past in range(position + 1):  # causal mask over the full tensors
                    score = 0.0
                    for lane in range(head_dim):
                        score = add(
                            score,
                            mul(
                                query[head * head_dim + lane], keys[past][kv_head * head_dim + lane]
                            ),
                        )
                    scores.append(mul(score, scale))
                maximum = max(scores)
                exponentials = [f32(math.exp(add(score, -maximum))) for score in scores]
                denominator = 0.0
                for value in exponentials:
                    denominator = add(denominator, value)
                probabilities = [f32(value / denominator) for value in exponentials]
                for lane in range(head_dim):
                    accumulated = 0.0
                    for past, probability in enumerate(probabilities):
                        accumulated = add(
                            accumulated, mul(probability, values[past][kv_head * head_dim + lane])
                        )
                    context.append(cast(accumulated))
            first = [
                cast(add(lhs, rhs))
                for lhs, rhs in zip(inputs[position], dense(context, matrices[3]), strict=True)
            ]
            normalized_first = norm(first, 2 * layer + 2)
            gate, up = dense(normalized_first, matrices[4]), dense(normalized_first, matrices[5])
            activated = [
                cast(mul(f32(g / add(1.0, f32(math.exp(-g)))), u))
                for g, u in zip(gate, up, strict=True)
            ]
            outputs.append(
                [
                    cast(add(lhs, rhs))
                    for lhs, rhs in zip(first, dense(activated, matrices[6]), strict=True)
                ]
            )
        return outputs

    hidden = [
        [cast(((pos * 19 + col * 7 + seed % 97) % 37 - 18) / 16) for col in range(dim)]
        for pos in range(tokens)
    ]
    for layer in range(layers):
        hidden = block(hidden, layer)
    return hidden


def buffers(
    *,
    dim=16,
    heads=4,
    kv_heads=2,
    ff_dim=32,
    context=32,
    prefill=3,
    width=2,
    sram_bytes=1 << 20,
    **_,
) -> dict:
    """SRAM buffers: the prompt batch's live rows at the peak of each half of a
    layer (3d + 2kv during attention, 2d + 2ff during the FFN), FP32 scores and
    probabilities of every head for one query row, one layer's K and V rows,
    and a weight buffer taking the rest of --sram-bytes, up to the largest
    weight matrix (d x max(d, ff))."""
    kv = dim // heads * kv_heads
    sizes = {
        "activation_buffer_bytes": prefill * width * max(3 * dim + 2 * kv, 2 * dim + 2 * ff_dim),
        "score_buffer_bytes": 4 * 2 * heads * context,
        "kv_buffer_bytes": 2 * context * kv * width,
    }
    sizes["weight_buffer_bytes"] = min(
        dim * max(dim, ff_dim) * width, sram_bytes - sum(sizes.values())
    )
    sizes["reserved_sram_bytes"] = sum(sizes.values())
    return sizes


def schedule(
    *,
    dim=16,
    heads=4,
    kv_heads=2,
    ff_dim=32,
    layers=1,
    prefill=3,
    decode=3,
    width=2,
    context=32,
    sram_bytes=1 << 20,
    tensor_rate=64,
    vector_rate=16,
    dma_rate=32,
    dma_startup=2,
) -> list[list[tuple]]:
    """Stage list of every request, from the documented schedule.

    The prompt is one request of `prefill` rows; each decode token is a request
    of one row. Per layer: RMSNorm (gamma DMA, vector), Q/K/V, RoPE on Q and K,
    K/V rows written to DRAM, earlier K/V rows read back, attention over p + 1
    keys at position p, O, residual, RMSNorm, gate, up, SiLU, down, residual.
    A projection of depth D and C columns streams its weights in column panels
    of floor(weight buffer / (D * width)) columns, the last one narrower; each
    panel is a weight DMA followed by its tensor stage. Stages run back to
    back: a DMA costs startup + ceil(bytes / rate), tensor and vector work
    ceil(work / rate). Each stage is (kind, amount, traffic class, cycles).
    """
    kv = dim // heads * kv_heads
    weight_buffer = buffers(
        dim=dim,
        heads=heads,
        kv_heads=kv_heads,
        ff_dim=ff_dim,
        context=context,
        prefill=prefill,
        width=width,
        sram_bytes=sram_bytes,
    )["weight_buffer_bytes"]

    def cycles(kind, amount):
        if kind == "dma":
            return dma_startup + math.ceil(amount / dma_rate)
        return math.ceil(amount / (tensor_rate if kind == "tensor" else vector_rate))

    def dense(rows, depth, columns):
        panel = min(columns, weight_buffer // (depth * width))
        stages = []
        for first in range(0, columns, panel):
            span = min(panel, columns - first)
            stages += [
                ("dma", depth * span * width, "weight"),
                ("tensor", rows * depth * span, None),
            ]
        return stages

    def request(start, rows):
        raw = [("dma", rows * dim * width, "input")]
        visible = [start + row + 1 for row in range(rows)]
        for _ in range(layers):
            raw += [("dma", dim * width, "weight"), ("vector", rows * (5 * dim + 2), None)]
            for cols in (dim, kv, kv):
                raw += dense(rows, dim, cols)
            raw += [("vector", rows * 5 * dim, None), ("vector", rows * 5 * kv, None)]
            raw += [("dma", rows * kv * width, "kv_write")] * 2
            if start:
                raw += [("dma", start * kv * width, "kv_read")] * 2
            raw += [
                ("tensor", sum(2 * dim * v for v in visible), None),
                ("vector", sum(heads * (6 * v + 1) for v in visible), None),
            ]
            raw += dense(rows, dim, dim) + [("vector", rows * dim, None)]
            raw += [("dma", dim * width, "weight"), ("vector", rows * (5 * dim + 2), None)]
            raw += dense(rows, dim, ff_dim) * 2
            raw += [("vector", rows * 5 * ff_dim, None)]
            raw += dense(rows, ff_dim, dim) + [("vector", rows * dim, None)]
        raw += [("dma", rows * dim * width, "output")]
        return [(kind, amount, traffic, cycles(kind, amount)) for kind, amount, traffic in raw]

    return [request(0, prefill)] + [request(prefill + i, 1) for i in range(decode)]


def stage_list(requests: list[list[tuple]]) -> list[tuple[str, int]]:
    """(direction or kind, amount) per stage, as progress_stages() reports them."""
    reads = ("input", "weight", "kv_read")
    return [
        (("dma_read" if traffic in reads else "dma_write") if kind == "dma" else kind, amount)
        for request in requests
        for kind, amount, traffic, _ in request
    ]


def progress_stages(path: Path) -> list[tuple[str, int]]:
    """(direction or kind, amount) of every completed stage, from the counter
    deltas between consecutive stage_complete events."""
    fields = ("tensor_macs", "vector_ops", "dma_read_bytes", "dma_write_bytes")
    previous = dict.fromkeys(fields, 0)
    stages = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["event"] != "stage_complete":
            continue
        delta = {field: event[field] - previous[field] for field in fields}
        previous = {field: event[field] for field in fields}
        if event["stage"] == 1:
            direction = "dma_read" if delta["dma_read_bytes"] else "dma_write"
            stages.append((direction, delta[direction + "_bytes"]))
        else:
            kind, field = (
                ("tensor", "tensor_macs") if event["stage"] == 2 else ("vector", "vector_ops")
            )
            stages.append((kind, delta[field]))
    return stages


def totals(requests: list[list[tuple]]) -> dict:
    stages = [stage for request in requests for stage in request]

    def sum_of(kind=None, traffic=None, field=1):
        return sum(
            stage[field]
            for stage in stages
            if (kind is None or stage[0] == kind) and (traffic is None or stage[2] == traffic)
        )

    return {
        "ttft_cycles": sum(stage[3] for stage in requests[0]),
        "tpot_cycles": [sum(stage[3] for stage in request) for request in requests[1:]],
        "tensor_macs": sum_of("tensor"),
        "vector_ops": sum_of("vector"),
        "tensor_cycles": sum_of("tensor", field=3),
        "vector_cycles": sum_of("vector", field=3),
        "dma_cycles": sum_of("dma", field=3),
        "dma_transactions": sum(1 for stage in stages if stage[0] == "dma"),
        "weight_read_bytes": sum_of("dma", "weight"),
        "kv_dram_read_bytes": sum_of("dma", "kv_read"),
        "kv_dram_write_bytes": sum_of("dma", "kv_write"),
        "dma_read_bytes": sum_of("dma", "input")
        + sum_of("dma", "weight")
        + sum_of("dma", "kv_read"),
        "dma_write_bytes": sum_of("dma", "output") + sum_of("dma", "kv_write"),
        "stages": len(stages),
    }


def failed_request_cycles(rows: int, width: int, dim: int = 16) -> int:
    """The lifecycle request whose last row is FLT_MAX: its input rows and the
    first RMSNorm gamma arrive, the RMSNorm vector stage runs, and the
    overflowing square sum stops the request (default rates)."""

    def dma(size):
        return 2 + math.ceil(size / 32)

    return dma(rows * dim * width) + dma(dim * width) + math.ceil(rows * (5 * dim + 2) / 16)


def invoke(
    *arguments: str, stdin: str | None = None, timeout: int = 30
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(BINARY), *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=ENV,
        input=stdin,
    )


def run(*arguments: str, timeout: int = 30) -> dict:
    completed = invoke(*arguments, timeout=timeout)
    if completed.returncode:
        raise AssertionError(f"{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout)


def round_words(words: list[int], precision: str) -> list[int]:
    # Over stdin: an argv this long fails on Windows (WinError 206, 32767-character limit).
    completed = invoke(
        "--precision", precision, "--round-f32-bits", "-", stdin=",".join(map(str, words))
    )
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)["rounded_f32_bits"]


class TransformerTests(unittest.TestCase):
    def assert_sequence(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for row, (result, oracle) in enumerate(zip(actual, expected, strict=True)):
            self.assertEqual(len(result), len(oracle))
            for col, (left, right) in enumerate(zip(result, oracle, strict=True)):
                self.assertTrue(math.isfinite(left))
                self.assertAlmostEqual(
                    left, right, delta=3e-6 * max(1, abs(right)), msg=f"output[{row}][{col}]"
                )

    def test_outputs_match_the_float_oracle_in_every_precision_and_kv_grouping(self):
        for precision in ("fp32", "fp16", "bf16"):
            for kv_heads in (1, 2, 4):
                with self.subTest(precision=precision, kv_heads=kv_heads):
                    result = run(
                        "--precision", precision, "--kv-heads", str(kv_heads), "--dump-output"
                    )
                    self.assert_sequence(
                        result["outputs"], reference(precision=precision, kv_heads=kv_heads)
                    )
                    self.assertTrue(result["partition_match"])

    def test_nondefault_shape_and_unsigned_seed(self):
        result = run(
            "--dim",
            "24",
            "--heads",
            "3",
            "--kv-heads",
            "1",
            "--ff-dim",
            "19",
            "--prefill",
            "2",
            "--decode",
            "3",
            "--seed",
            "4294967295",
            "--precision",
            "fp32",
            "--dump-output",
        )
        expected = reference(
            dim=24, heads=3, kv_heads=1, ff_dim=19, tokens=5, seed=4294967295, precision="fp32"
        )
        self.assert_sequence(result["outputs"], expected)

    def test_weight_scale_follows_the_reduction_depth(self):
        # Projections of depth 256 (weights over 128) and a down projection of
        # depth 1024 (over 256) against the oracle, in the storage formats where
        # an unscaled fixture loses range or precision first.
        for flags, shape, precisions in (
            (
                ("--dim", "256", "--heads", "4", "--kv-heads", "2", "--ff-dim", "32"),
                {"dim": 256, "heads": 4, "kv_heads": 2, "ff_dim": 32},
                ("fp16",),
            ),
            (
                (
                    "--ff-dim",
                    "1024",
                ),
                {"ff_dim": 1024},
                ("fp16", "bf16"),
            ),
        ):
            for precision in precisions:
                with self.subTest(flags=flags, precision=precision):
                    result = run(
                        *flags,
                        "--precision",
                        precision,
                        "--prefill",
                        "2",
                        "--decode",
                        "1",
                        "--dump-output",
                    )
                    self.assert_sequence(
                        result["outputs"], reference(**shape, tokens=3, precision=precision)
                    )
                    self.assertTrue(result["partition_match"])

    def test_stacked_layers_match_the_oracle(self):
        for precision in ("fp32", "bf16"):
            with self.subTest(precision=precision):
                result = run(
                    "--layers",
                    "3",
                    "--prefill",
                    "5",
                    "--decode",
                    "2",
                    "--precision",
                    precision,
                    "--dump-output",
                )
                self.assert_sequence(
                    result["outputs"], reference(layers=3, tokens=7, precision=precision)
                )
                self.assertTrue(result["partition_match"])
                width = 4 if precision == "fp32" else 2
                self.assertEqual(result["kv_dram_capacity_bytes"], 2 * 3 * 32 * 8 * width)

    def test_prefill_incremental_partitions_and_no_future_leakage(self):
        whole = run("--prefill", "8", "--decode", "0", "--dump-output")
        for prefill in (1, 3, 6):
            with self.subTest(prefill=prefill):
                chunked = run(
                    "--prefill", str(prefill), "--decode", str(8 - prefill), "--dump-output"
                )
                self.assertEqual(chunked["outputs"], whole["outputs"])
                self.assertEqual(chunked["requests_completed"], 1 + 8 - prefill)
        prefix = run("--prefill", "2", "--decode", "0", "--dump-output")
        self.assertEqual(prefix["outputs"], whole["outputs"][:2])

    def test_full_kv_cache_rejects_atomically_and_replays_after_reset(self):
        result = run("--context", "6", "--verify-lifecycle", "--dump-output")
        self.assertTrue(result["lifecycle_verified"])
        self.assertEqual(result["cache_resets"], 1)
        self.assertEqual(result["rejected_requests"], 5)
        self.assertEqual(result["rejected_work_cycles"], failed_request_cycles(2, 2))
        self.assertEqual(result["tokens_completed"], result["tokens_executed"])
        self.assertEqual(result["kv_live_bytes"], result["kv_dram_capacity_bytes"])
        self.assertGreater(result["simulation_cycles"], result["workload_cycles"])
        self.assert_sequence(result["outputs"], reference())

    def test_cycle_traffic_and_sram_accounting_match_the_schedule(self):
        cases = [
            ((), {}),
            (("--precision", "fp32"), {"width": 4}),
            (
                (
                    "--layers",
                    "2",
                    "--prefill",
                    "5",
                    "--decode",
                    "4",
                    "--dma-startup",
                    "7",
                    "--dma-bytes-per-cycle",
                    "5",
                    "--tensor-macs-per-cycle",
                    "3",
                    "--vector-ops-per-cycle",
                    "7",
                ),
                {
                    "layers": 2,
                    "prefill": 5,
                    "decode": 4,
                    "dma_startup": 7,
                    "dma_rate": 5,
                    "tensor_rate": 3,
                    "vector_rate": 7,
                },
            ),
            (
                (
                    "--dim",
                    "24",
                    "--heads",
                    "6",
                    "--kv-heads",
                    "3",
                    "--ff-dim",
                    "40",
                    "--prefill",
                    "7",
                    "--decode",
                    "1",
                    "--context",
                    "9",
                ),
                {
                    "dim": 24,
                    "heads": 6,
                    "kv_heads": 3,
                    "ff_dim": 40,
                    "prefill": 7,
                    "decode": 1,
                    "context": 9,
                },
            ),
            # 96 bytes past the 2624 of fixed buffers: 3-column panels of
            # the depth-16 projections and single columns of down (depth 32).
            (("--sram-bytes", "2720"), {"sram_bytes": 2720}),
        ]
        for flags, shape in cases:
            with self.subTest(flags=flags):
                report = run(*flags)
                expected = totals(schedule(**shape))
                for field, value in expected.items():
                    if field != "stages":
                        self.assertEqual(report[field], value, field)
                for field, value in buffers(**shape).items():
                    self.assertEqual(report[field], value, field)
                self.assertLessEqual(report["reserved_sram_bytes"], report["sram_budget_bytes"])
                self.assertEqual(report["prefill_cycles"], report["ttft_cycles"])
                self.assertEqual(report["decode_cycles"], sum(report["tpot_cycles"]))
                mean = sum(report["tpot_cycles"]) / max(1, len(report["tpot_cycles"]))
                self.assertAlmostEqual(report["tpot_mean_cycles"], mean, delta=1e-6 * max(1, mean))
                self.assertEqual(
                    report["workload_cycles"], report["prefill_cycles"] + report["decode_cycles"]
                )
                self.assertEqual(
                    report["workload_cycles"],
                    sum(
                        report[field] for field in ("tensor_cycles", "vector_cycles", "dma_cycles")
                    ),
                )
                self.assertEqual(report["simulation_cycles"], report["workload_cycles"])

    def test_closed_form_work_traffic_and_sram(self):
        d, h, f, kv, layers, width, context = 16, 4, 32, 8, 2, 2, 32
        prefill, decode = 5, 4
        report = run("--layers", str(layers), "--prefill", str(prefill), "--decode", str(decode))
        tokens = prefill + decode
        per_layer = 2 * d * d + 2 * d * kv + 3 * d * f
        # QK^T and PV at position p touch p + 1 keys: 2 * d * (p + 1) MACs.
        self.assertEqual(
            report["tensor_macs"], layers * (tokens * per_layer + d * tokens * (tokens + 1))
        )
        # Weights and both RMSNorm gammas stream once per layer per request.
        self.assertEqual(
            report["weight_read_bytes"], (1 + decode) * layers * (per_layer + 2 * d) * width
        )
        self.assertEqual(report["kv_dram_write_bytes"], 2 * layers * tokens * kv * width)
        self.assertEqual(
            report["kv_dram_read_bytes"],
            2 * layers * kv * width * sum(prefill + i for i in range(decode)),
        )
        self.assertEqual(
            report["dma_read_bytes"],
            tokens * d * width + report["weight_read_bytes"] + report["kv_dram_read_bytes"],
        )
        self.assertEqual(
            report["dma_write_bytes"], tokens * d * width + report["kv_dram_write_bytes"]
        )
        self.assertEqual(report["kv_dram_capacity_bytes"], 2 * layers * context * kv * width)
        self.assertEqual(report["kv_live_bytes"], 2 * layers * tokens * kv * width)
        self.assertEqual(
            report["activation_buffer_bytes"], prefill * width * max(3 * d + 2 * kv, 2 * d + 2 * f)
        )
        self.assertEqual(report["score_buffer_bytes"], 4 * 2 * h * context)
        self.assertEqual(report["kv_buffer_bytes"], 2 * context * kv * width)
        # The default budget holds the largest matrix (gate/up, d x f) whole.
        self.assertEqual(report["weight_buffer_bytes"], d * f * width)
        self.assertEqual(
            report["reserved_sram_bytes"],
            sum(
                report[field]
                for field in (
                    "activation_buffer_bytes",
                    "score_buffer_bytes",
                    "kv_buffer_bytes",
                    "weight_buffer_bytes",
                )
            ),
        )
        # KV traffic and capacity are DRAM quantities, reported as kv_dram_*.
        for stale in ("kv_read_bytes", "kv_write_bytes", "kv_capacity_bytes"):
            self.assertNotIn(stale, report)
        fp32 = run(
            "--layers",
            str(layers),
            "--prefill",
            str(prefill),
            "--decode",
            str(decode),
            "--precision",
            "fp32",
        )
        self.assertEqual(fp32["dma_read_bytes"], 2 * report["dma_read_bytes"])
        self.assertEqual(fp32["kv_dram_capacity_bytes"], 2 * report["kv_dram_capacity_bytes"])

    def test_lifecycle_check_leaves_the_sram_reservation_unchanged(self):
        # The lifecycle replay reuses the workload's partition, so the batch
        # buffer still holds the prompt only: a budget that fits the run
        # exactly also fits it with --verify-lifecycle.
        flags = ("--prefill", "32", "--decode", "32", "--context", "64")
        exact = run(*flags)["reserved_sram_bytes"]
        self.assertEqual(exact, buffers(prefill=32, context=64)["reserved_sram_bytes"])
        for extra in ((), ("--verify-lifecycle",)):
            with self.subTest(extra=extra):
                report = run(*flags, "--sram-bytes", str(exact), *extra)
                for field, value in buffers(prefill=32, context=64, sram_bytes=exact).items():
                    self.assertEqual(report[field], value, field)
                self.assertEqual(report["lifecycle_verified"], bool(extra))
        # One byte less does not hold the largest matrix whole: Q, K, V and O
        # still fit, gate and up split into 31 + 1 columns and down into 15 + 1.
        tight = run(*flags, "--sram-bytes", str(exact - 1), "--verify-lifecycle")
        shape = {"prefill": 32, "decode": 32, "context": 64, "sram_bytes": exact - 1}
        self.assertEqual(tight["ttft_cycles"], totals(schedule(**shape))["ttft_cycles"])
        self.assertEqual(tight["dma_transactions"], totals(schedule(**shape))["dma_transactions"])

    def test_batched_prefill_streams_weights_once_per_layer(self):
        runs = {
            rows: run("--prefill", str(rows), "--decode", "0", "--layers", "2")
            for rows in (1, 4, 16)
        }
        per_layer = 2 * 16 * 16 + 2 * 16 * 8 + 3 * 16 * 32 + 2 * 16
        for report in runs.values():
            self.assertEqual(report["weight_read_bytes"], 2 * per_layer * 2)
        per_token = [runs[rows]["ttft_cycles"] / rows for rows in (1, 4, 16)]
        self.assertEqual(per_token, sorted(per_token, reverse=True))
        self.assertLess(per_token[-1], per_token[0] / 2)

    def test_decode_reads_the_dram_kv_cache_every_token(self):
        report = run("--prefill", "4", "--decode", "12", "--context", "16")
        tpot = report["tpot_cycles"]
        self.assertEqual(len(tpot), 12)
        self.assertEqual(tpot, sorted(tpot))
        self.assertGreater(tpot[-1], tpot[0])
        self.assertEqual(report["kv_dram_read_bytes"], 2 * 8 * 2 * sum(range(4, 16)))
        self.assertGreater(report["tpot_mean_cycles"], report["ttft_cycles"] / 4)

    def test_kv_head_reduction_reduces_cache_and_work(self):
        multi = run("--kv-heads", "4")
        grouped = run("--kv-heads", "1")
        self.assertEqual(multi["kv_dram_capacity_bytes"], 4 * grouped["kv_dram_capacity_bytes"])
        self.assertGreater(multi["tensor_macs"], grouped["tensor_macs"])
        self.assertGreater(multi["dma_read_bytes"], grouped["dma_read_bytes"])
        self.assertEqual(multi["kv_dram_read_bytes"], 4 * grouped["kv_dram_read_bytes"])

    def test_largest_dimensions_stream_weights_in_panels_that_fit_sram(self):
        # Every stage, read back from the progress stream, matches the
        # schedule: no weight DMA exceeds the weight buffer, and the buffers fit
        # --sram-bytes. A 2048-wide matrix (8 MiB in 16 bits) cannot sit in the
        # 4 MiB budget and streams in panels; the 128 x 8192 one (2 MiB) fits
        # whole. The 2048 x 8192 layer runs at the default fp16 precision.
        common = {"prefill": 2, "decode": 1, "context": 3, "sram_bytes": 4194304}
        for flags, shape, split in (
            (
                (
                    "--dim",
                    "2048",
                    "--heads",
                    "32",
                    "--kv-heads",
                    "8",
                    "--ff-dim",
                    "64",
                    "--precision",
                    "bf16",
                ),
                {"dim": 2048, "heads": 32, "kv_heads": 8, "ff_dim": 64},
                True,
            ),
            (
                (
                    "--dim",
                    "128",
                    "--heads",
                    "64",
                    "--kv-heads",
                    "64",
                    "--ff-dim",
                    "8192",
                    "--precision",
                    "bf16",
                ),
                {"dim": 128, "heads": 64, "kv_heads": 64, "ff_dim": 8192},
                False,
            ),
            (
                ("--dim", "2048", "--heads", "32", "--kv-heads", "8", "--ff-dim", "8192"),
                {"dim": 2048, "heads": 32, "kv_heads": 8, "ff_dim": 8192},
                True,
            ),
        ):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "progress.jsonl"
                report = run(
                    *flags,
                    "--prefill",
                    "2",
                    "--decode",
                    "1",
                    "--context",
                    "3",
                    "--sram-bytes",
                    "4194304",
                    "--progress-jsonl",
                    str(path),
                    timeout=120,
                )
                requests = schedule(**shape, **common)
                self.assertEqual(progress_stages(path), stage_list(requests))
                expected = totals(requests)
                for field in (
                    "tensor_macs",
                    "ttft_cycles",
                    "tpot_cycles",
                    "dma_transactions",
                    "weight_read_bytes",
                ):
                    self.assertEqual(report[field], expected[field], field)
                sizes = buffers(**shape, **common)
                for field, value in sizes.items():
                    self.assertEqual(report[field], value, field)
                self.assertLessEqual(report["reserved_sram_bytes"], report["sram_budget_bytes"])
                largest = shape["dim"] * max(shape["dim"], shape["ff_dim"]) * 2
                self.assertEqual(sizes["weight_buffer_bytes"] < largest, split)
                weight_dmas = [
                    stage[1] for request in requests for stage in request if stage[2] == "weight"
                ]
                self.assertLessEqual(max(weight_dmas), sizes["weight_buffer_bytes"])
                self.assertTrue(report["partition_match"])

    def test_native_systemc_waveform(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "transformer"
            result = run("--vcd", str(base))
            trace = base.with_suffix(".vcd").read_text()
            for signal in ("decoder_stage", "decoder_busy", "tokens_completed", "kv_tokens"):
                self.assertIn(signal, trace)
            self.assertGreater(trace.count("#"), result["tokens_completed"] * 15)
            self.assertIn("$timescale", trace)

    def test_progress_records_stage_batch_and_commit_events(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            result = run("--progress-jsonl", str(path))
            events = [json.loads(line) for line in path.read_text().splitlines()]
            stages = [event for event in events if event["event"] == "stage_complete"]
            self.assertEqual(len(stages), totals(schedule())["stages"])
            cycles = [event["cycles"] for event in events]
            self.assertEqual(cycles, sorted(cycles))
            batches = [event for event in events if event["event"] == "batch_executed"]
            self.assertEqual([event["tokens_executed"] for event in batches], [3, 4, 5, 6])
            self.assertEqual([event["tokens_completed"] for event in batches], [0, 3, 4, 5])
            commits = [event for event in events if event["event"] == "request_commit"]
            self.assertEqual([event["tokens_completed"] for event in commits], [3, 4, 5, 6])
            self.assertEqual([event["kv_tokens"] for event in commits], [3, 4, 5, 6])
            self.assertEqual(commits[0]["cycles"], result["ttft_cycles"])
            self.assertEqual(events[-1]["cycles"], result["workload_cycles"])
            for field in ("tensor_macs", "vector_ops", "dma_read_bytes", "dma_write_bytes"):
                self.assertEqual(events[-1][field], result[field])

    def test_failed_request_keeps_its_work_but_commits_nothing(self):
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "progress.jsonl"
                result = run(
                    "--precision", precision, "--verify-lifecycle", "--progress-jsonl", str(path)
                )
                self.assertTrue(result["lifecycle_verified"])
                width = 4 if precision == "fp32" else 2
                self.assertEqual(result["rejected_work_cycles"], failed_request_cycles(2, width))
                events = [json.loads(line) for line in path.read_text().splitlines()]
                batches = [e for e in events if e["event"] == "batch_executed"]
                # Prompt, three decode tokens, then the same partition replayed after
                # the reset; the request that overflows mid-layer never completes a batch.
                self.assertEqual(
                    [e["tokens_executed"] for e in batches], [3, 4, 5, 6, 9, 10, 11, 12]
                )
                commits = [e for e in events if e["event"] == "request_commit"]
                self.assertEqual(
                    [e["tokens_completed"] for e in commits], [3, 4, 5, 6, 9, 10, 11, 12]
                )
                after = [
                    e for e in events if commits[3]["cycles"] < e["cycles"] < commits[4]["cycles"]
                ]
                self.assertTrue(
                    all(e["tokens_completed"] == 6 and e["kv_tokens"] == 0 for e in after)
                )
        # Capacity one: the failing request holds only the overflowing row.
        one = run("--context", "1", "--prefill", "1", "--decode", "0", "--verify-lifecycle")
        self.assertTrue(one["lifecycle_verified"])
        self.assertEqual(one["rejected_work_cycles"], failed_request_cycles(1, 2))

    def test_injected_output_fault_is_reported_and_exits_2(self):
        clean = run("--dump-output")["outputs"]
        completed = invoke("--inject-output-fault", "4", "--dump-output")
        self.assertEqual(completed.returncode, 2)
        self.assertTrue(
            completed.stderr.startswith(
                "npu-transformer-demo: batched and one-row execution disagree"
            ),
            completed.stderr,
        )
        report = json.loads(completed.stdout)
        self.assertFalse(report["partition_match"])
        outputs = report["outputs"]
        self.assertEqual(outputs[:4] + outputs[5:], clean[:4] + clean[5:])
        self.assertEqual(outputs[4][1:], clean[4][1:])
        self.assertEqual(abs(word(outputs[4][0]) - word(clean[4][0])), 1)

    def test_aliased_artifact_destinations_fail_before_truncating_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            victim = directory / "wave.vcd"
            sentinel = "existing file must survive rejected output aliases\n"
            victim.write_text(sentinel)
            pairs = [
                (directory / "wave", victim),
                (victim, victim),
                (directory / "wave", directory / "." / "wave.vcd"),
            ]
            if os.name == "posix":
                linked = directory / "alias.jsonl"
                linked.symlink_to(victim)
                pairs.append((directory / "wave", linked))
                hardlink = directory / "hardlink.jsonl"
                os.link(victim, hardlink)
                pairs.append((directory / "wave", hardlink))
                parent = directory / "linked-directory"
                parent.symlink_to(directory, target_is_directory=True)
                pairs.append((directory / "wave", parent / "wave.vcd"))
            for vcd, progress in pairs:
                with self.subTest(vcd=vcd, progress=progress):
                    result = invoke(
                        "--vcd", str(vcd), "--progress-jsonl", str(progress), timeout=10
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertTrue(
                        result.stderr.startswith("npu-transformer-demo: "), result.stderr
                    )
                    self.assertIn("output paths conflict", result.stderr)
                    self.assertEqual(victim.read_text(), sentinel)

    def test_fp16_all_finite_encodings_roundtrip(self):
        words = [
            word(struct.unpack("<e", struct.pack("<H", half))[0])
            for half in range(65536)
            if half & 0x7C00 != 0x7C00
        ]
        self.assertEqual(round_words(words, "fp16"), words)

    def test_storage_rounding_ties_subnormals_overflow_and_random_binary32(self):
        rng = random.Random(927)
        words = [
            0,
            0x80000000,
            0x3F800000,
            0x7F800000,
            0xFF800000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x33000000,
            0x33800000,
            0x387FC000,
            0x38800000,
            0x477FE000,
            0x477FF000,
            0x3F801000,
            0x3F803000,
            0x3F808000,
            0x3F818000,
            0x7F7F8000,
            0x7F7F7FFF,
        ]
        words += [rng.randrange(1 << 32) for _ in range(1800)]
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision):
                results = round_words(words, precision)
                for source, result in zip(words, results, strict=True):
                    number = unword(source)
                    if math.isnan(number):
                        self.assertTrue(math.isnan(unword(result)))
                        self.assertEqual(source >> 31, result >> 31)
                    else:
                        expected = (
                            source if precision == "fp32" else word(stored(number, precision))
                        )
                        self.assertEqual(result, expected, (hex(source), precision))
        # A short list can also be passed inline on the command line.
        inline = run("--precision", "bf16", "--round-f32-bits", f"{0x3F808000},{0x3F818000}")
        self.assertEqual(inline["rounded_f32_bits"], [0x3F800000, 0x3F820000])

    def test_help_anywhere_prints_usage(self):
        for arguments in (("--help",), ("--dim", "0", "--help"), ("--what", "--help")):
            with self.subTest(arguments=arguments):
                completed = invoke(*arguments)
                self.assertEqual(completed.returncode, 0)
                self.assertTrue(completed.stdout.startswith("Usage: npu-transformer-demo"))

    def test_invalid_dimensions_capacity_budget_and_cli(self):
        llama_layer = ("--dim", "2048", "--heads", "32", "--kv-heads", "8", "--ff-dim", "8192")
        invalid = [
            ("--dim", "0"),
            ("--dim", "2049"),
            ("--dim", "15"),
            ("--heads", "0"),
            ("--heads", "3"),
            ("--heads", "65"),
            ("--kv-heads", "3"),
            ("--ff-dim", "0"),
            ("--ff-dim", "8193"),
            ("--layers", "0"),
            ("--layers", "17"),
            ("--context", "5"),
            ("--prefill", "0"),
            ("--decode", "-1"),
            ("--seed", "4294967296"),
            ("--sram-bytes", "1"),
            ("--precision", "int4"),
            ("--dma-bytes-per-cycle", "0"),
            ("--tensor-macs-per-cycle", "nan"),
            ("--vcd", ""),
            ("--round-f32-bits", "1,"),
            ("--round-f32-bits", "4294967296"),
            ("--what", "1"),
            ("--dim",),
            ("--inject-output-fault", "6"),
            ("--inject-output-fault", "x"),
            ("--layers", "16", *llama_layer),
            ("--prefill", "64", "--context", "128", *llama_layer),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                result = invoke(*arguments, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertTrue(result.stderr.startswith("npu-transformer-demo: "), result.stderr)
                self.assertIn(arguments[0], result.stderr)
        empty = invoke("--round-f32-bits", "-", stdin="")
        self.assertEqual(empty.returncode, 2)
        self.assertTrue(
            empty.stderr.startswith("npu-transformer-demo: --round-f32-bits"), empty.stderr
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
