#!/usr/bin/env python3
"""Black-box checks of implicit-convolution traffic, the depthwise datapath and the
activation cache against independent oracles.

The oracles follow the model description in docs/architecture.md and share no code with
src/. The input footprint enumerates every kernel tap of every output pixel in a panel and
takes, per input row, the hull of the columns read; padding reads nothing. Its runs follow
the NHWC layout: one per input row when the panel holds every channel, else one per pixel of
the hull. Bursts walk the AXI beat slots. Numerics come from a direct convolution over the
NHWC tensor without an im2col matrix. The caches are replayed as LRUs over their lines in
each core's load order, which the stage trace gives.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

BINARY: Path

ACTIVATIONS, WEIGHTS, PARAMS, OUTPUTS = 0x8000_0000, 0x9000_0000, 0xA000_0000, 0xB000_0000
PAGE = 4096
MASK32 = (1 << 32) - 1
NO_CONV = {
    "input_h": 0,
    "input_w": 0,
    "input_c": 0,
    "kernel_h": 1,
    "kernel_w": 1,
    "stride_h": 1,
    "stride_w": 1,
    "dilation_h": 1,
    "dilation_w": 1,
    "pad_top": 0,
    "pad_left": 0,
    "pad_bottom": 0,
    "pad_right": 0,
}


def ceildiv(value: int, divisor: int) -> int:
    return -(-value // divisor)


# ------------------------------------------------------------------- configs --


def base_config(**changes) -> dict:
    c = {
        "m": 4,
        "n": 3,
        "k": 5,
        "groups": 1,
        "seed": 1,
        "conv": None,
        "tile_m": 2,
        "tile_n": 2,
        "tile_k": 0,
        "buffer_slots": 2,
        "panel_buffers": 2,
        "macs_per_cycle": 16,
        "pipeline_cycles": 3,
        "depthwise": 0,
        "weight_bits": 8,
        "weight_cache": 0,
        "activation_cache": 0,
        "input_zero": -3,
        "weight_zero": 2,
        "memory": "serial",
        "cores": 1,
        "data_bytes": 16,
        "burst_beats": 16,
    }
    c.update(changes)
    return c


def flags(c: dict) -> list[str]:
    out = [
        "--m",
        c["m"],
        "--n",
        c["n"],
        "--k",
        c["k"],
        "--groups",
        c["groups"],
        "--seed",
        c["seed"],
        "--tile-m",
        c["tile_m"],
        "--tile-n",
        c["tile_n"],
        "--tile-k",
        c["tile_k"],
        "--buffer-slots",
        c["buffer_slots"],
        "--panel-buffers",
        c["panel_buffers"],
        "--macs-per-cycle",
        c["macs_per_cycle"],
        "--pipeline-cycles",
        c["pipeline_cycles"],
        "--depthwise-macs-per-cycle",
        c["depthwise"],
        "--weight-bits",
        c["weight_bits"],
        "--weight-cache-bytes",
        c["weight_cache"],
        "--activation-cache-bytes",
        c["activation_cache"],
        "--input-zero",
        c["input_zero"],
        "--weight-zero",
        c["weight_zero"],
        "--sram-bytes",
        1 << 24,
        "--memory-model",
        c["memory"],
    ]
    if c["memory"] == "axi":
        out += [
            "--axi-data-bytes",
            c["data_bytes"],
            "--axi-burst-beats",
            c["burst_beats"],
            "--cores",
            c["cores"],
        ]
    if c["conv"]:
        for key, value in c["conv"].items():
            out += ["--conv-" + key.replace("_", "-"), value]
    return [str(value) for value in out]


def conv_shape(conv: dict) -> tuple[int, int]:
    """Output height and width: floor((in + pads - reach) / stride) + 1 per axis."""

    def extent(size, kernel, stride, dilation, low, high):
        return (size + low + high - dilation * (kernel - 1) - 1) // stride + 1

    return (
        extent(
            conv["input_h"],
            conv["kernel_h"],
            conv["stride_h"],
            conv["dilation_h"],
            conv["pad_top"],
            conv["pad_bottom"],
        ),
        extent(
            conv["input_w"],
            conv["kernel_w"],
            conv["stride_w"],
            conv["dilation_w"],
            conv["pad_left"],
            conv["pad_right"],
        ),
    )


def random_conv_config(rng: random.Random, depthwise: bool, size: int = 8) -> dict:
    while True:
        conv = dict(
            NO_CONV,
            input_h=rng.randint(1, size),
            input_w=rng.randint(1, size),
            kernel_h=rng.randint(1, 3),
            kernel_w=rng.randint(1, 3),
            stride_h=rng.randint(1, 3),
            stride_w=rng.randint(1, 3),
            dilation_h=rng.randint(1, 2),
            dilation_w=rng.randint(1, 2),
            pad_top=rng.randint(0, 2),
            pad_left=rng.randint(0, 2),
            pad_bottom=rng.randint(0, 2),
            pad_right=rng.randint(0, 2),
        )
        group_channels = rng.randint(1, 3 if size <= 8 else 8)
        groups = rng.randint(2, 6) if depthwise else rng.choice([1, 1, 2, 3])
        conv["input_c"] = groups * group_channels
        if (
            conv["dilation_h"] * (conv["kernel_h"] - 1) + 1
            > conv["input_h"] + conv["pad_top"] + conv["pad_bottom"]
            or conv["dilation_w"] * (conv["kernel_w"] - 1) + 1
            > conv["input_w"] + conv["pad_left"] + conv["pad_right"]
        ):
            continue
        out_h, out_w = conv_shape(conv)
        m, row_k = out_h * out_w, conv["kernel_w"] * group_channels
        return base_config(
            m=m,
            n=1 if depthwise else rng.randint(1, 3),
            k=conv["kernel_h"] * row_k,
            groups=groups,
            conv=conv,
            seed=rng.randint(1, 10**6),
            tile_m=rng.randint(1, m),
            tile_n=rng.randint(1, 4),
            tile_k=rng.randint(0, conv["kernel_h"]) * row_k,
            depthwise=rng.randint(1, 40) if depthwise else 0,
            input_zero=rng.randint(-10, 10),
        )


def random_memory(rng: random.Random, c: dict) -> dict:
    if rng.random() < 0.5:
        return dict(
            c,
            memory="axi",
            data_bytes=rng.choice([8, 16, 32]),
            burst_beats=rng.choice([1, 2, 16]),
            cores=rng.choice([1, 1, 2, 3]),
        )
    return c


# ------------------------------------------------------------------ workload --


def channel_wise(c: dict) -> bool:
    return c["depthwise"] > 0 and c["groups"] > 1 and c["n"] == 1


def grid(c: dict) -> tuple[int, int]:
    """(groups, columns) of the output tile grid."""
    return (1, c["groups"]) if channel_wise(c) else (c["groups"], c["n"])


def tile_list(c: dict) -> list[tuple[int, int, int, int, int]]:
    """(group, row, col, rows, cols) by tile id: group, then row tile, then column tile. A
    depthwise-datapath tile spans tile_n channels of one group of G columns."""
    groups, columns = grid(c)
    return [
        (g, r, col, min(c["tile_m"], c["m"] - r), min(c["tile_n"], columns - col))
        for g in range(groups)
        for r in range(0, c["m"], c["tile_m"])
        for col in range(0, columns, c["tile_n"])
    ]


def depths(c: dict) -> list[int]:
    size = c["tile_k"] or c["k"]
    return [min(size, c["k"] - start) for start in range(0, c["k"], size)]


def tile_groups(c: dict, tile) -> range:
    g, _, col, _, cols = tile
    return range(col, col + cols) if channel_wise(c) else range(g, g + 1)


def needed_pixels(c: dict, tile, block: int) -> dict[int, set[int]]:
    """Input row -> columns the panel's output pixels read over the block's kernel rows."""
    conv = c["conv"]
    _, out_w = conv_shape(conv)
    group_channels = conv["input_c"] // c["groups"]
    rows_per_block = (c["tile_k"] or c["k"]) // (conv["kernel_w"] * group_channels)
    first_ky = block * rows_per_block
    _, row, _, rows, _ = tile
    pixels: dict[int, set[int]] = {}
    for m in range(row, row + rows):
        oy, ox = divmod(m, out_w)
        for ky in range(first_ky, min(conv["kernel_h"], first_ky + rows_per_block)):
            for kx in range(conv["kernel_w"]):
                y = oy * conv["stride_h"] - conv["pad_top"] + ky * conv["dilation_h"]
                x = ox * conv["stride_w"] - conv["pad_left"] + kx * conv["dilation_w"]
                if 0 <= y < conv["input_h"] and 0 <= x < conv["input_w"]:
                    pixels.setdefault(y, set()).add(x)
    return pixels


def activation_runs(c: dict, tile, block: int) -> list[tuple[int, int]]:
    _, row, _, rows, _ = tile
    groups = tile_groups(c, tile)
    if c["conv"]:
        conv = c["conv"]
        width, channels = conv["input_w"], conv["input_c"]
        group_channels = channels // c["groups"]
        first, count = groups.start * group_channels, len(groups) * group_channels
        runs = []
        for y, columns in sorted(needed_pixels(c, tile, block).items()):
            low, high = min(columns), max(columns)
            if count == channels:
                runs.append(
                    (ACTIVATIONS + (y * width + low) * channels, (high - low + 1) * channels)
                )
            else:
                runs += [
                    (ACTIVATIONS + (y * width + x) * channels + first, count)
                    for x in range(low, high + 1)
                ]
        return runs
    k, stride = c["k"], c["groups"] * c["k"]
    depth, start = depths(c)[block], block * (c["tile_k"] or k)
    runs = []
    for r in range(row, row + rows):
        # Each group's K_p slice of the row, merged where slices touch.
        row_runs: list[list[int]] = []
        for g in groups:
            base = ACTIVATIONS + r * stride + g * k + start
            if row_runs and row_runs[-1][0] + row_runs[-1][1] == base:
                row_runs[-1][1] += depth
            else:
                row_runs.append([base, depth])
        runs += [tuple(run) for run in row_runs]
    return runs


def weight_bytes(c: dict, depth: int, cols: int) -> int:
    return ceildiv(depth * cols * c["weight_bits"], 8)


def weight_offsets(c: dict) -> dict[tuple[int, int, int], int]:
    """Dense weight panels packed back to back in the order the loader first reads them."""
    groups, columns = grid(c)
    offsets, cursor = {}, 0
    for g in range(groups):
        for col in range(0, columns, c["tile_n"]):
            cols = min(c["tile_n"], columns - col)
            for block, depth in enumerate(depths(c)):
                offsets[g, col, block] = cursor
                cursor += weight_bytes(c, depth, cols)
    return offsets


def scratchpad(c: dict) -> int:
    """SRAM of the slots and caches: per slot the tile's bias records, its panel buffers
    (one per reduction panel, at most panel_buffers), each holding the largest activation
    panel of any tile and the deepest weight panel, then INT32 sums and INT8 outputs."""
    _, columns = grid(c)
    rows, cols = min(c["tile_m"], c["m"]), min(c["tile_n"], columns)
    depth, blocks = max(depths(c)), len(depths(c))
    if c["conv"]:
        widest = (cols if channel_wise(c) else 1) * (c["conv"]["input_c"] // c["groups"])
        largest = (
            max(
                sum(max(xs) - min(xs) + 1 for xs in needed_pixels(c, tile, block).values())
                for tile in tile_list(c)
                for block in range(blocks)
            )
            * widest
        )
    else:
        largest = rows * depth * (cols if channel_wise(c) else 1)
    panel = largest + weight_bytes(c, depth, cols)
    slot = 4 * cols + min(c["panel_buffers"], blocks) * panel + 5 * rows * cols
    return c["buffer_slots"] * slot + c["weight_cache"] + c["activation_cache"]


def first_channel(c: dict, tile) -> int:
    g, _, col, _, _ = tile
    return col if channel_wise(c) else g * c["n"] + col


class Lru:
    def __init__(self, capacity: int):
        self.capacity, self.lines = capacity, {}
        self.hits = self.misses = self.saved = 0

    def lookup(self, key, size: int) -> bool:
        if not self.capacity:
            return False
        if key in self.lines:
            self.lines[key] = self.lines.pop(key)
            self.hits += 1
            self.saved += size
            return True
        self.misses += 1
        if size > self.capacity:
            return False
        while sum(self.lines.values()) + size > self.capacity:
            del self.lines[next(iter(self.lines))]
        self.lines[key] = size
        return False


def bursts(run: tuple[int, int], data_bytes: int, max_beats: int) -> list[int]:
    """Beats of each burst of a run: walk the beat slots it touches, closing a burst at
    max_beats or where the next slot opens a 4 KiB page."""
    base, length = run
    end = base + length
    groups: list[int] = [0]
    for slot in range(base // data_bytes, (end - 1) // data_bytes + 1):
        if groups[-1] and (groups[-1] == max_beats or slot * data_bytes % PAGE == 0):
            groups.append(0)
        groups[-1] += 1
    return groups


# ------------------------------------------------------------------ numerics --


def fixtures(c: dict) -> tuple[list[int], list[int], list[int]]:
    """The simulator's generator stream: the NHWC input (or the activation operand), then
    the weights, then the bias."""
    state = c["seed"] & MASK32

    def values(count: int, modulus: int, offset: int) -> list[int]:
        nonlocal state
        out = []
        for _ in range(count):
            state = (1664525 * state + 1013904223) & MASK32
            out.append(state % modulus - offset)
        return out

    conv = c["conv"]
    inputs = (
        conv["input_h"] * conv["input_w"] * conv["input_c"]
        if conv
        else c["m"] * c["groups"] * c["k"]
    )
    first = values(inputs, 31, 15)
    weights = (
        values(c["groups"] * c["k"] * c["n"], 16, 8)
        if c["weight_bits"] == 4
        else values(c["groups"] * c["k"] * c["n"], 31, 15)
    )
    return first, weights, values(c["groups"] * c["n"], 127, 63)


def requantize(acc: int) -> int:
    """Native rescale at the defaults: multiplier 1, shift 7, ties away from zero."""
    magnitude = (abs(acc) + 64) >> 7
    return max(-128, min(127, magnitude if acc >= 0 else -magnitude))


def expected_outputs(c: dict) -> tuple[list[int], list[int]]:
    """Direct convolution over the NHWC input (or the grouped GEMM without shape.conv),
    M x (G*N) row-major."""
    inputs, weights, bias = fixtures(c)
    g, n, k, iz, wz = c["groups"], c["n"], c["k"], c["input_zero"], c["weight_zero"]
    accumulators = []
    conv = c["conv"]
    for m in range(c["m"]):
        for channel in range(g * n):
            group, out = divmod(channel, n)
            acc = bias[channel]
            if conv:
                _, out_w = conv_shape(conv)
                oy, ox = divmod(m, out_w)
                cg = conv["input_c"] // g
                for ky in range(conv["kernel_h"]):
                    for kx in range(conv["kernel_w"]):
                        y = oy * conv["stride_h"] - conv["pad_top"] + ky * conv["dilation_h"]
                        x = ox * conv["stride_w"] - conv["pad_left"] + kx * conv["dilation_w"]
                        if not (0 <= y < conv["input_h"] and 0 <= x < conv["input_w"]):
                            continue  # padding is the input zero point: no contribution
                        for ci in range(cg):
                            value = inputs[
                                (y * conv["input_w"] + x) * conv["input_c"] + group * cg + ci
                            ]
                            weight = weights[
                                (group * k + (ky * conv["kernel_w"] + kx) * cg + ci) * n + out
                            ]
                            acc += (value - iz) * (weight - wz)
            else:
                acc += sum(
                    (inputs[m * g * k + group * k + i] - iz)
                    * (weights[(group * k + i) * n + out] - wz)
                    for i in range(k)
                )
            accumulators.append(acc)
    return accumulators, [requantize(acc) for acc in accumulators]


# ------------------------------------------------------------------- running --


def run(c: dict, *extra: str) -> tuple[dict, list[dict]]:
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.csv"
        command = [str(BINARY), *flags(c), "--trace", str(trace), "--dump-output", *extra]
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if process.returncode:
            raise AssertionError(f"npu-sim failed: {' '.join(command)}\n{process.stderr}")
        rows = list(csv.DictReader(io.StringIO(trace.read_text())))
    for row in rows:
        for key in ("tile_id", "start_cycle", "end_cycle", "bytes", "block_id", "core"):
            row[key] = int(row[key])
    return json.loads(process.stdout), rows


def rejected(arguments: list[str]) -> str:
    process = subprocess.run(
        [str(BINARY), *[str(a) for a in arguments]], capture_output=True, text=True, timeout=30
    )
    if process.returncode != 2:
        raise AssertionError(f"expected exit 2, got {process.returncode}: {process.stderr}")
    return process.stderr


class ConvTraffic(unittest.TestCase):
    def replay(self, c: dict, result: dict, spans: list[dict]) -> dict:
        """Replays every core's panel loads against the oracle; returns the totals."""
        tiles = tile_list(c)
        offsets = weight_offsets(c)
        blocks = len(depths(c))
        totals = dict(
            read=0,
            activation=0,
            bursts=0,
            beats=0,
            loads=0,
            act_hits=0,
            act_misses=0,
            act_saved=0,
            weight_hits=0,
        )
        loads = [span for span in spans if span["stage"] == "load"]
        for core in sorted({span["core"] for span in loads}):
            weights, activations = Lru(c["weight_cache"]), Lru(c["activation_cache"])
            mine = sorted((s for s in loads if s["core"] == core), key=lambda s: s["start_cycle"])
            order = [(s["tile_id"], s["block_id"]) for s in mine]
            self.assertEqual(order, sorted(order), "a core loads its tiles in tile-id order")
            for span in mine:
                tile, block = tiles[span["tile_id"]], span["block_id"]
                g, row, col, rows, cols = tile
                depth = depths(c)[block]
                runs = []
                act = activation_runs(c, tile, block)
                act_bytes = sum(length for _, length in act)
                unit = col // c["tile_n"] if channel_wise(c) else g
                if not activations.lookup((unit, row // c["tile_m"], block), act_bytes):
                    runs += act
                    totals["activation"] += act_bytes
                size = weight_bytes(c, depth, cols)
                if not weights.lookup((first_channel(c, tile), block), size):
                    runs.append((WEIGHTS + offsets[g, col, block], size))
                if block == 0:
                    runs.append((PARAMS + first_channel(c, tile) * 4, cols * 4))
                payload = sum(length for _, length in runs)
                self.assertEqual(
                    span["bytes"], payload, f"load of tile {span['tile_id']} block {block}"
                )
                totals["read"] += payload
                totals["loads"] += 1
                for each in runs:
                    if each[1]:
                        beats = bursts(each, c["data_bytes"], c["burst_beats"])
                        totals["bursts"] += len(beats)
                        totals["beats"] += sum(beats)
            totals["act_hits"] += activations.hits
            totals["act_misses"] += activations.misses
            totals["act_saved"] += activations.saved
            totals["weight_hits"] += weights.hits
        self.assertEqual(totals["loads"], len(tiles) * blocks, "every panel loads once")
        self.assertEqual(result["read_bytes"], totals["read"])
        self.assertEqual(result["activation_read_bytes"], totals["activation"])
        self.assertEqual(result["activation_cache_hits"], totals["act_hits"])
        self.assertEqual(result["activation_cache_misses"], totals["act_misses"])
        self.assertEqual(result["activation_bytes_saved"], totals["act_saved"])
        self.assertEqual(result["weight_cache_hits"], totals["weight_hits"])
        self.assertEqual(result["memory"]["read"]["requests"], totals["loads"])
        self.assertEqual(result["scratchpad_bytes"], scratchpad(c))
        if c["memory"] == "axi":
            self.assertEqual(result["memory"]["read"]["bursts"], totals["bursts"])
            self.assertEqual(result["memory"]["read"]["beats"], totals["beats"])
        # Stores write each tile's rows x columns.
        stores = {s["tile_id"]: s["bytes"] for s in spans if s["stage"] == "store"}
        self.assertEqual(stores, {i: t[3] * t[4] for i, t in enumerate(tiles)})
        # The closed-form estimate predicts the bytes the plan reads.
        predicted = result["cluster"]["predicted"][result["cluster"]["partition"]]
        self.assertEqual(predicted["read_bytes"], result["read_bytes"])
        self.assertEqual(predicted["write_bytes"], result["write_bytes"])
        return totals

    def check_numerics(self, c: dict, result: dict) -> None:
        accumulators, outputs = expected_outputs(c)
        self.assertEqual(result["numerics"], "bit-exact")
        self.assertEqual(result["accumulators"], accumulators)
        self.assertEqual(result["outputs"], outputs)

    def test_convolution_footprint_traffic_and_numerics(self):
        rng = random.Random(20261107)
        for trial in range(240):
            # Every fifth input is wide enough for its rows to cross 4 KiB pages.
            size = 24 if trial % 5 == 4 else 8
            c = random_memory(rng, random_conv_config(rng, trial % 3 == 0, size))
            if rng.random() < 0.4:
                c["activation_cache"] = rng.choice([1, 40, 200, 4096])
            if rng.random() < 0.3:
                c["weight_cache"] = rng.choice([8, 64, 1024])
            if rng.random() < 0.2:
                c["weight_bits"] = 4
            with self.subTest(trial=trial, flags=" ".join(flags(c))):
                result, spans = run(c)
                self.replay(c, result, spans)
                self.check_numerics(c, result)

    def test_footprint_counts_halo_once_per_panel(self):
        # One tile over the whole image: a 3x3 stride-1 kernel with one pixel of padding
        # reads every input byte exactly once, against 9x for the im2col matrix.
        conv = dict(
            NO_CONV,
            input_h=6,
            input_w=7,
            input_c=5,
            kernel_h=3,
            kernel_w=3,
            pad_top=1,
            pad_left=1,
            pad_bottom=1,
            pad_right=1,
        )
        c = base_config(m=42, n=4, k=45, conv=conv, tile_m=42, tile_n=4)
        result, _ = run(c)
        self.assertEqual(result["activation_read_bytes"], 6 * 7 * 5)
        plain = dict(c, conv=None)
        self.assertEqual(run(plain)[0]["activation_read_bytes"], 42 * 45)
        # Row tiles of one output row read three input rows each (two at the edges).
        result, spans = run(dict(c, tile_m=7))
        self.assertEqual(
            [s["bytes"] - 4 * 45 - 16 for s in spans if s["stage"] == "load"],
            [2 * 7 * 5] + [3 * 7 * 5] * 4 + [2 * 7 * 5],
        )
        # Panels of whole kernel rows: each block reads its own input rows.
        result, _ = run(dict(c, tile_k=15))
        self.assertEqual(result["activation_read_bytes"], 3 * 6 * 7 * 5 - 2 * 7 * 5)

    def test_depthwise_datapath(self):
        rng = random.Random(77)
        for trial in range(60):
            if trial % 2:
                c = random_conv_config(rng, depthwise=True)
            else:
                g = rng.randint(2, 12)
                c = base_config(
                    m=rng.randint(1, 30),
                    n=1,
                    k=rng.randint(1, 20),
                    groups=g,
                    tile_m=rng.randint(1, 12),
                    tile_n=rng.randint(1, g + 2),
                    depthwise=rng.randint(1, 64),
                    seed=rng.randint(1, 10**6),
                )
                c["tile_k"] = rng.choice([0, 0, rng.randint(1, c["k"])])
            c = random_memory(rng, c)
            c["pipeline_cycles"] = rng.randint(0, 5)
            with self.subTest(trial=trial, flags=" ".join(flags(c))):
                result, spans = run(c)
                tiles = tile_list(c)
                for span in spans:
                    if span["stage"] != "compute":
                        continue
                    _, _, _, rows, cols = tiles[span["tile_id"]]
                    depth = depths(c)[span["block_id"]]
                    self.assertEqual(
                        span["end_cycle"] - span["start_cycle"],
                        ceildiv(rows * cols * depth, c["depthwise"]) + c["pipeline_cycles"],
                    )
                macs = c["m"] * c["groups"] * c["k"]
                cores = c["cores"] if c["memory"] == "axi" else 1
                self.assertEqual(result["macs"], macs)
                self.assertEqual(result["depthwise_macs"], macs)
                self.assertEqual(result["effective_macs_per_cycle"], c["depthwise"] * cores)
                self.assertEqual(result["depthwise_peak_macs_per_cycle"], c["depthwise"] * cores)
                self.assertAlmostEqual(
                    result["depthwise_mac_utilization"],
                    macs / (c["depthwise"] * cores * result["cycles"]),
                )
                self.assertEqual(
                    result["compute_busy_cycles"],
                    sum(
                        s["end_cycle"] - s["start_cycle"] for s in spans if s["stage"] == "compute"
                    ),
                )
                self.replay(c, result, spans)
                self.check_numerics(c, result)
        # Rate 0, or N > 1, keeps grouped GEMM tiles on the array.
        c = base_config(m=6, n=1, k=9, groups=4, tile_m=3, tile_n=2)
        result, spans = run(c)
        self.assertEqual(result["depthwise_macs"], 0)
        self.assertEqual(result["effective_macs_per_cycle"], 16)
        self.assertEqual(len({s["tile_id"] for s in spans}), 4 * 2)
        result, _ = run(dict(c, n=2, depthwise=8))
        self.assertEqual(result["depthwise_macs"], 0)

    def test_activation_cache_reuses_row_tiles(self):
        # 3 row tiles x 4 column tiles x 2 reduction panels of 4 x 6 bytes.
        c = base_config(m=12, n=8, k=12, tile_m=4, tile_n=2, tile_k=6)
        plain, _ = run(c)
        fits, spans = run(dict(c, activation_cache=48))
        self.assertEqual(fits["activation_cache_hits"], 3 * 3 * 2)
        self.assertEqual(fits["activation_cache_misses"], 3 * 2)
        self.assertEqual(fits["activation_bytes_saved"], 3 * 3 * 2 * 24)
        self.assertEqual(fits["read_bytes"], plain["read_bytes"] - 3 * 3 * 2 * 24)
        self.assertEqual(fits["peak_cached_activation_bytes"], 48)
        self.replay(dict(c, activation_cache=48), fits, spans)
        # One panel short: the cyclic sweep evicts each panel before its reuse.
        thrash, spans = run(dict(c, activation_cache=47))
        self.assertEqual(thrash["activation_cache_hits"], 0)
        self.assertEqual(thrash["read_bytes"], plain["read_bytes"])
        self.replay(dict(c, activation_cache=47), thrash, spans)
        # The capacity counts in the SRAM budget.
        self.assertEqual(fits["scratchpad_bytes"], plain["scratchpad_bytes"] + 48)
        stderr = rejected(
            [*flags(dict(c, activation_cache=48)), "--sram-bytes", plain["scratchpad_bytes"] + 47]
        )
        self.assertIn("tile buffers exceed configured SRAM capacity", stderr)

    def test_timing_only_matches_bit_exact_traffic(self):
        rng = random.Random(5)
        for trial in range(10):
            c = random_memory(rng, random_conv_config(rng, depthwise=trial % 2 == 0))
            with self.subTest(trial=trial):
                exact, _ = run(c)
                command = [str(BINARY), *flags(c), "--timing-only"]
                timed = json.loads(
                    subprocess.run(
                        command, capture_output=True, text=True, timeout=60, check=True
                    ).stdout
                )
                for key in ("cycles", "read_bytes", "write_bytes", "activation_read_bytes"):
                    self.assertEqual(timed[key], exact[key], key)
                self.assertEqual(timed["numerics"], "skipped")

    def test_rejections(self):
        conv = [
            "--conv-input-h",
            8,
            "--conv-input-w",
            8,
            "--conv-input-c",
            4,
            "--conv-kernel-h",
            3,
            "--conv-kernel-w",
            3,
            "--conv-pad-top",
            1,
            "--conv-pad-left",
            1,
            "--conv-pad-bottom",
            1,
            "--conv-pad-right",
            1,
        ]
        good = ["--m", 64, "--n", 2, "--k", 36, *conv]
        rejected_cases = [
            (
                ["--m", 63, "--n", 2, "--k", 36, *conv],
                "M must equal output height x width = 8 x 8 = 64, got 63",
            ),
            (
                ["--m", 64, "--n", 2, "--k", 35, *conv],
                "K must equal kernel_h x kernel_w x input_c / groups = 36, got 35",
            ),
            (
                [*good, "--tile-k", 16],
                "--tile-k must be 0 or a multiple of kernel_w x input_c / groups = 12",
            ),
            ([*good, "--groups", 3], "input channels 4 must be a multiple of groups 3"),
            (["--conv-kernel-h", 3], "need the input extents"),
            (
                ["--m", 64, "--n", 2, "--k", 36, "--conv-input-h", 8, "--conv-input-w", 8],
                "input height, width and channels must all be in 1..1048576",
            ),
            (
                [*good, "--conv-dilation-h", 9],
                "kernel height extent 19 exceeds the padded input 10",
            ),
            (
                [
                    "--m",
                    4,
                    "--n",
                    1,
                    "--k",
                    9,
                    "--groups",
                    8,
                    "--depthwise-macs-per-cycle",
                    8,
                    "--weight-encoding",
                    "palette",
                ],
                "the depthwise datapath reads dense weights",
            ),
        ]
        for arguments, message in rejected_cases:
            with self.subTest(arguments=arguments):
                self.assertIn(message, rejected(arguments))
        with tempfile.TemporaryDirectory() as directory:
            tensors = Path(directory) / "t.npu"
            tensors.write_text("NPU1 64 2 36\n" + "0 " * (64 * 36 + 72 + 2))
            self.assertIn(
                "--input holds a GEMM operand", rejected(["--input", str(tensors), *conv])
            )
        # Accepted: tile_k on whole kernel rows, and the echo reloads.
        result = json.loads(
            subprocess.run(
                [str(BINARY), *[str(a) for a in good], "--tile-k", "24"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout
        )
        self.assertEqual(result["reduction_blocks"], 2)
        self.assertEqual(result["config"]["shape"]["conv"]["kernel_h"], 3)
        with tempfile.TemporaryDirectory() as directory:
            echo = Path(directory) / "echo.json"
            echo.write_text(json.dumps(result["config"]))
            again = json.loads(
                subprocess.run(
                    [str(BINARY), "--config", str(echo)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                ).stdout
            )
            self.assertEqual(again["cycles"], result["cycles"])
            self.assertEqual(again["output_hash"], result["output_hash"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
