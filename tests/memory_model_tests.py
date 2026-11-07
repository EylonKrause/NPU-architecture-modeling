#!/usr/bin/env python3
"""Black-box checks of the axi memory model against an independent oracle.

The oracle below is written from the memory-system description (SPEC section
4.3), not from the C++: it walks beats to form bursts, finds DRAM gaps by
fixpoint iteration over every reservation ever made, and replays the load,
compute and store pipeline as a timestamp simulation. Closed forms check
single-burst latency, steady-state bandwidth, 4 KiB splitting, misaligned
beats and the DRAM cap independently of the oracle.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import unittest


BINARY: Path

ACTIVATIONS, WEIGHTS, PARAMS, OUTPUTS = 0x8000_0000, 0x9000_0000, 0xA000_0000, 0xB000_0000
PAGE = 4096

BASE = {
    "m": 4, "n": 5, "k": 7, "groups": 1, "tile_m": 2, "tile_n": 3, "tile_k": 0,
    "buffer_slots": 2, "panel_buffers": 2, "macs_per_cycle": 8, "pipeline_cycles": 2,
    "weight_bits": 8, "axi_data_bytes": 16, "axi_burst_beats": 16,
    "axi_read_outstanding": 4, "axi_write_outstanding": 4,
    "axi_read_latency": 12, "axi_write_latency": 12, "dram_bytes_per_cycle": 0,
}


def ceildiv(value: int, divisor: int) -> int:
    return -(-value // divisor)


# ---------------------------------------------------------------- workload --

def depths(c: dict) -> list[int]:
    size = c["tile_k"] or c["k"]
    return [min(size, c["k"] - start) for start in range(0, c["k"], size)]


def tile_list(c: dict) -> list[tuple[int, int, int, int, int]]:
    """(group, row, col, rows, cols) in load order: group, then row, then column."""
    return [(g, r, col, min(c["tile_m"], c["m"] - r), min(c["tile_n"], c["n"] - col))
            for g in range(c["groups"])
            for r in range(0, c["m"], c["tile_m"])
            for col in range(0, c["n"], c["tile_n"])]


def weight_bytes(c: dict, depth: int, cols: int) -> int:
    return ceildiv(depth * cols * c["weight_bits"], 8)


def weight_offsets(c: dict) -> dict[tuple[int, int, int], int]:
    """Weight panels packed back to back in the order the loader first reads them."""
    offsets, cursor = {}, 0
    for g in range(c["groups"]):
        for col in range(0, c["n"], c["tile_n"]):
            cols = min(c["tile_n"], c["n"] - col)
            for block, depth in enumerate(depths(c)):
                offsets[g, col, block] = cursor
                cursor += weight_bytes(c, depth, cols)
    return offsets


def read_runs(c: dict, tile, block: int, offsets) -> list[tuple[int, int]]:
    g, row, col, rows, cols = tile
    k, groups, n = c["k"], c["groups"], c["n"]
    depth = depths(c)[block]
    start = block * (c["tile_k"] or k)
    runs = [(ACTIVATIONS + (row + r) * groups * k + g * k + start, depth) for r in range(rows)]
    runs.append((WEIGHTS + offsets[g, col, block], weight_bytes(c, depth, cols)))
    if block == 0:
        runs.append((PARAMS + (g * n + col) * 4, cols * 4))
    return runs


def store_runs(c: dict, tile) -> list[tuple[int, int]]:
    g, row, col, rows, cols = tile
    stride = c["groups"] * c["n"]
    return [(OUTPUTS + (row + r) * stride + g * c["n"] + col, cols) for r in range(rows)]


def compute_cycles(c: dict, tile, block: int) -> int:
    _, _, _, rows, cols = tile
    return ceildiv(rows * cols * depths(c)[block], c["macs_per_cycle"]) + c["pipeline_cycles"]


# ------------------------------------------------------------------ bursts --

def bursts(run: tuple[int, int], data_bytes: int, max_beats: int) -> list[tuple[int, int]]:
    """(beats, payload bytes) of each burst: walk the beat slots the run touches
    and close a burst when it holds max_beats or the next slot opens a page."""
    base, length = run
    end = base + length
    groups: list[list[int]] = [[]]
    for slot in range(base // data_bytes, (end - 1) // data_bytes + 1):
        if groups[-1] and (len(groups[-1]) == max_beats or slot * data_bytes % PAGE == 0):
            groups.append([])
        groups[-1].append(slot)
    return [(len(slots), min(end, (slots[-1] + 1) * data_bytes) - max(base, slots[0] * data_bytes))
            for slots in groups]


# ------------------------------------------------------------------ memory --

class Dram:
    def __init__(self, rate: int):
        self.rate, self.reserved = rate, []

    def reserve(self, earliest: int, size: int) -> int:
        start = earliest
        while True:
            clashes = [end for begin, end in self.reserved if begin < start + size and start < end]
            if not clashes:
                self.reserved.append((start, start + size))
                return start
            start = max(clashes)


class Channel:
    def __init__(self, limit: int, lead: int, tail: int):
        self.limit, self.lead, self.tail = limit, lead, tail
        self.request = None
        self.last_issue = None
        self.data_free = 0
        self.completions: list[int] = []
        self.stats = dict(requests=0, bursts=0, beats=0, payload_bytes=0, busy_cycles=0,
                          latency=0, max_latency_cycles=0, peak_outstanding=0)

    def submit(self, now: int, runs, tag, c: dict) -> None:
        pieces = [b for run in runs if run[1] for b in bursts(run, c["axi_data_bytes"],
                                                                  c["axi_burst_beats"])]
        self.request = dict(arrival=now, pieces=pieces, next=0, completion=now, tag=tag,
                            bytes=sum(length for _, length in runs), beats=0, busy=0)

    def pending(self) -> bool:
        return self.request is not None and self.request["next"] < len(self.request["pieces"])

    def next_issue(self) -> int:
        issue = self.request["arrival"]
        if self.last_issue is not None:
            issue = max(issue, self.last_issue + 1)
        if len(self.completions) >= self.limit:
            issue = max(issue, self.completions[-self.limit])
        return issue

    def issue(self, now: int, dram: Dram, data_bytes: int) -> None:
        request = self.request
        beats, _ = request["pieces"][request["next"]]
        earliest = max(now + self.lead, self.data_free)
        if dram.rate:
            size = beats * data_bytes
            begin = dram.reserve(earliest * dram.rate, size)
            start = ceildiv(begin, dram.rate)
            end = max(start + beats, ceildiv(begin + size, dram.rate))
        else:
            start, end = earliest, earliest + beats
        self.data_free = end
        self.completions.append(end + self.tail)
        self.last_issue = now
        outstanding = sum(done > now for done in self.completions)
        self.stats["peak_outstanding"] = max(self.stats["peak_outstanding"], outstanding)
        request["beats"] += beats
        request["busy"] += end - start
        request["completion"] = end + self.tail
        request["next"] += 1

    def finished(self, now: int) -> bool:
        return (self.request is not None and not self.pending() and
                self.request["completion"] == now)

    def retire(self) -> dict:
        request, s = self.request, self.stats
        latency = request["completion"] - request["arrival"]
        s["requests"] += 1
        s["bursts"] += len(request["pieces"])
        s["beats"] += request["beats"]
        s["payload_bytes"] += request["bytes"]
        s["busy_cycles"] += request["busy"]
        s["latency"] += latency
        s["max_latency_cycles"] = max(s["max_latency_cycles"], latency)
        self.request = None
        return request


def simulate(c: dict) -> dict:
    """Timestamp replay of the load, compute and store threads over the axi port."""
    tiles, blocks = tile_list(c), len(depths(c))
    buffers = min(c["panel_buffers"], blocks)
    offsets = weight_offsets(c)
    read = Channel(c["axi_read_outstanding"], c["axi_read_latency"], 0)
    write = Channel(c["axi_write_outstanding"], 0, c["axi_write_latency"])
    dram = Dram(c["dram_bytes_per_cycle"])
    now, free_slots, computed_panels, stored = 0, c["buffer_slots"], 0, 0
    load = dict(tile=0, block=0, state="slot")
    loaded, finished = [], []
    compute = None
    spans = []
    while stored < len(tiles):
        progressed = True
        while progressed:  # everything that happens in zero time at `now`
            progressed = False
            if load["state"] == "slot" and load["tile"] < len(tiles) and free_slots:
                free_slots -= 1
                load["state"] = "buffer"
                progressed = True
            if load["state"] == "buffer":
                first = load["tile"] * blocks
                released = first + load["block"] - buffers + 1
                if load["block"] < buffers or computed_panels >= released:
                    tile = tiles[load["tile"]]
                    read.submit(now, read_runs(c, tile, load["block"], offsets),
                                (load["tile"], load["block"]), c)
                    load["state"] = "reading"
                    progressed = True
            if load["state"] == "reading" and read.finished(now):
                request = read.retire()
                spans.append(("load", *request["tag"][:1], request["arrival"], now,
                              request["bytes"], request["tag"][1]))
                loaded.append(request["tag"])
                load["block"] += 1
                if load["block"] == blocks:
                    load.update(tile=load["tile"] + 1, block=0, state="slot")
                else:
                    load["state"] = "buffer"
                progressed = True
            if compute is not None and compute[2] == now:
                (tile_id, block), start, _ = compute
                spans.append(("compute", tile_id, start, now, 0, block))
                computed_panels += 1
                if block + 1 == blocks:
                    finished.append(tile_id)
                compute = None
                progressed = True
            if compute is None and loaded:
                tile_id, block = loaded.pop(0)
                compute = ((tile_id, block), now,
                           now + compute_cycles(c, tiles[tile_id], block))
                progressed = True
            if write.request is not None and write.finished(now):
                request = write.retire()
                spans.append(("store", request["tag"], request["arrival"], now,
                              request["bytes"], blocks - 1))
                stored += 1
                free_slots += 1
                progressed = True
            if write.request is None and finished:
                tile_id = finished.pop(0)
                write.submit(now, store_runs(c, tiles[tile_id]), tile_id, c)
                progressed = True
        # Grants after every same-cycle arrival: read before write.
        for channel in (read, write):
            if channel.pending() and channel.next_issue() == now:
                channel.issue(now, dram, c["axi_data_bytes"])
        if stored == len(tiles):
            break
        upcoming = [ch.next_issue() for ch in (read, write) if ch.pending()]
        upcoming += [ch.request["completion"] for ch in (read, write)
                     if ch.request is not None and not ch.pending()]
        if compute is not None:
            upcoming.append(compute[2])
        upcoming = [t for t in upcoming if t > now]
        assert upcoming, "oracle deadlock"
        now = min(upcoming)
    intervals = sorted((s[2], s[3]) for s in spans if s[0] != "compute")
    busy, reach = 0, 0
    for start, end in intervals:
        busy += max(0, end - max(start, reach))
        reach = max(reach, end)
    return dict(cycles=now, read=read.stats, write=write.stats, spans=sorted(spans),
                dma_busy_cycles=busy, dram_bytes=(read.stats["beats"] + write.stats["beats"]) *
                c["axi_data_bytes"])


# --------------------------------------------------------------- simulator --

def flags(c: dict) -> list[str]:
    result = ["--memory-model", "axi", "--sram-bytes", str(1 << 26)]
    for key, value in c.items():
        result += ["--" + key.replace("_", "-"), str(value)]
    return result


def run(c: dict, *extra: str, trace: bool = True) -> tuple[dict, list]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.csv"
        command = [str(BINARY), *flags(c), *extra] + (["--trace", str(path)] if trace else [])
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if process.returncode:
            raise AssertionError(f"simulator failed ({process.returncode}): {process.stderr}")
        result = json.loads(process.stdout)
        spans = []
        if trace:
            for row in csv.DictReader(io.StringIO(path.read_text())):
                spans.append((row["stage"], int(row["tile_id"]), int(row["start_cycle"]),
                              int(row["end_cycle"]), int(row["bytes"]), int(row["block_id"])))
    return result, sorted(spans)


class AxiMemoryModel(unittest.TestCase):
    def assert_matches_oracle(self, c: dict, *extra: str) -> dict:
        result, spans = run(c, *extra)
        expected = simulate(c)
        self.assertIs(result["numerical_match"], True)
        self.assertEqual(result["cycles"], expected["cycles"], c)
        self.assertEqual(result["dma_busy_cycles"], expected["dma_busy_cycles"], c)
        memory = result["memory"]
        self.assertEqual(memory["model"], "axi")
        for direction in ("read", "write"):
            got, want = memory[direction], expected[direction]
            for key in ("requests", "bursts", "beats", "payload_bytes", "busy_cycles",
                        "max_latency_cycles", "peak_outstanding"):
                self.assertEqual(got[key], want[key], f"{direction}.{key} {c}")
            self.assertAlmostEqual(got["mean_latency_cycles"], want["latency"] / want["requests"])
            self.assertAlmostEqual(got["burst_efficiency"],
                                   want["payload_bytes"] / (want["beats"] * c["axi_data_bytes"]))
            self.assertAlmostEqual(got["achieved_bytes_per_cycle"],
                                   want["payload_bytes"] / expected["cycles"])
            limit = c["axi_read_outstanding" if direction == "read" else "axi_write_outstanding"]
            self.assertLessEqual(got["peak_outstanding"], limit)
        self.assertEqual(result["read_bytes"], expected["read"]["payload_bytes"])
        self.assertEqual(result["write_bytes"], expected["write"]["payload_bytes"])
        self.assertEqual(result["dma_bursts"],
                         expected["read"]["bursts"] + expected["write"]["bursts"])
        self.assertEqual(memory["dram"]["bytes"], expected["dram_bytes"])
        self.assertEqual(spans, expected["spans"], c)
        return result

    def test_single_burst_latency(self):
        # One output: a 16-byte activation run, a 16-byte weight run and a
        # 4-byte bias record, each one beat at its own region base; the store
        # is one beat.
        base = dict(BASE, m=1, n=1, k=16, tile_m=1, tile_n=1, pipeline_cycles=0,
                    macs_per_cycle=16)
        for outstanding in (1, 2, 3, 4):
            for latency in (0, 1, 7, 100):
                c = dict(base, axi_read_outstanding=outstanding, axi_read_latency=latency,
                         axi_write_latency=latency)
                result, spans = run(c)
                load = next(s for s in spans if s[0] == "load")
                store = next(s for s in spans if s[0] == "store")
                # Burst i issues at max(i, completion of burst i - N); each
                # takes latency + 1 beat and the data channel is free by then.
                issues, done = [], []
                for i in range(3):
                    issue = i if i < outstanding else max(i, done[i - outstanding])
                    issues.append(issue)
                    done.append(max(issue + latency, done[-1] if done else 0) + 1)
                self.assertEqual(load[3] - load[2], done[-1], c)
                self.assertEqual(store[3] - store[2], 1 + latency, c)
                self.assertEqual(result["memory"]["write"]["max_latency_cycles"], 1 + latency)
                self.assertEqual(result["cycles"], done[-1] + 1 + 1 + latency)

    def test_steady_state_bandwidth(self):
        # A 1x64x4096 GEMV tile reads 4096 activation bytes, a 262144-byte
        # contiguous weight panel and a 256-byte bias run: 1041 full 256-byte
        # bursts. Away from the first latency, bytes per cycle approach
        # min(data_bytes, N * beats * data_bytes / (latency + beats)).
        latency, beats, data = 100, 16, 16
        c = dict(BASE, m=1, n=64, k=4096, tile_m=1, tile_n=64, axi_data_bytes=data,
                 axi_burst_beats=beats, axi_read_latency=latency)
        for outstanding in (1, 2, 4, 8, 16):
            with self.subTest(outstanding=outstanding):
                case = dict(c, axi_read_outstanding=outstanding)
                result = self.assert_matches_oracle(case)
                read = result["memory"]["read"]
                self.assertEqual(read["bursts"], 1041)
                self.assertEqual(read["peak_outstanding"], min(outstanding, 1041))
                steady = min(data, outstanding * beats * data / (latency + beats))
                achieved = read["payload_bytes"] / (read["max_latency_cycles"] - latency)
                self.assertAlmostEqual(achieved / steady, 1.0, delta=0.01)

    def test_four_kib_split(self):
        # Rows of 3000 bytes start at 3000 * r: with 4096-byte bursts each run
        # takes one burst per 4 KiB page it touches.
        c = dict(BASE, m=8, n=1, k=3000, tile_m=8, tile_n=1, axi_burst_beats=256,
                 axi_data_bytes=16, macs_per_cycle=64)
        result = self.assert_matches_oracle(c)
        activation_bursts = sum((3000 * r + 2999) // PAGE - (3000 * r) // PAGE + 1
                                for r in range(8))
        weight_bursts = ceildiv(3000, PAGE)  # a 3000-byte panel at the 4 KiB-aligned base
        self.assertEqual(activation_bursts, 13)
        self.assertEqual(result["memory"]["read"]["bursts"], activation_bursts + weight_bursts + 1)
        # Below 4 KiB, max_burst_beats splits the same runs first.
        short = self.assert_matches_oracle(dict(c, axi_burst_beats=16))
        self.assertGreater(short["memory"]["read"]["bursts"], result["memory"]["read"]["bursts"])

    def test_misaligned_beats(self):
        # Rows of 37 bytes: each run starts at 37 * r and touches
        # ceil(end / 16) - floor(start / 16) beats, partial at both ends.
        c = dict(BASE, m=16, n=1, k=37, tile_m=16, tile_n=1, axi_data_bytes=16)
        result = self.assert_matches_oracle(c)
        activation_beats = sum(ceildiv(37 * r + 37, 16) - 37 * r // 16 for r in range(16))
        weight_beats, bias_beats = ceildiv(37, 16), 1
        read = result["memory"]["read"]
        self.assertEqual(read["beats"], activation_beats + weight_beats + bias_beats)
        self.assertEqual(read["payload_bytes"], 16 * 37 + 37 + 4)
        self.assertAlmostEqual(read["burst_efficiency"],
                               read["payload_bytes"] / (16 * read["beats"]))
        self.assertLess(read["burst_efficiency"], 0.75)
        # The same rows aligned to the beat move with full beats.
        aligned = self.assert_matches_oracle(dict(c, k=48))
        self.assertEqual(aligned["memory"]["read"]["beats"], 16 * 3 + 3 + 1)

    def test_reads_and_writes_overlap(self):
        c = dict(BASE, m=32, n=32, k=64, tile_m=8, tile_n=8, buffer_slots=3,
                 macs_per_cycle=64, axi_read_latency=20, axi_write_latency=20)
        result, spans = run(c)
        loads = [s for s in spans if s[0] == "load"]
        stores = [s for s in spans if s[0] == "store"]
        overlapping = [(l, s) for l in loads for s in stores if l[2] < s[3] and s[2] < l[3]]
        self.assertTrue(overlapping, "a store must run while a later tile loads")
        self.assert_matches_oracle(c)
        # Union busy time is less than the summed request time once they overlap.
        summed = sum(s[3] - s[2] for s in loads + stores)
        self.assertLess(result["dma_busy_cycles"], summed)
        self.assertLessEqual(result["dma_busy_cycles"], result["cycles"])

    def test_dram_rate_caps_both_channels(self):
        c = dict(BASE, m=64, n=64, k=256, tile_m=16, tile_n=16, macs_per_cycle=4096,
                 axi_read_outstanding=16, axi_write_outstanding=16, axi_read_latency=30,
                 axi_write_latency=30, buffer_slots=4)
        unlimited = self.assert_matches_oracle(c)
        for rate in (1, 3, 8, 16, 24, 32):
            with self.subTest(rate=rate):
                capped = self.assert_matches_oracle(dict(c, dram_bytes_per_cycle=rate))
                dram = capped["memory"]["dram"]
                self.assertEqual(dram["bytes_per_cycle"], rate)
                # DRAM moves every beat at full width: no faster than the cap.
                self.assertLessEqual(dram["bytes"], rate * capped["cycles"])
                self.assertGreaterEqual(capped["cycles"], unlimited["cycles"])
                self.assertEqual(capped["peak_memory_bytes_per_cycle"], min(32, rate))
                if rate < 16:
                    self.assertGreater(capped["cycles"], unlimited["cycles"])
                    self.assertGreater(dram["utilization"], 0.9)
        # A rate at the combined bus width never binds.
        wide = self.assert_matches_oracle(dict(c, dram_bytes_per_cycle=32))
        self.assertEqual(wide["cycles"], unlimited["cycles"])

    def test_random_mappings_match_the_oracle(self):
        rng = random.Random(4)
        for case in range(60):
            m, n, k = rng.randint(1, 24), rng.randint(1, 24), rng.randint(1, 70)
            c = dict(BASE, m=m, n=n, k=k, groups=rng.choice([1, 1, 2, 3]),
                     tile_m=rng.randint(1, 9), tile_n=rng.randint(1, 9),
                     tile_k=rng.choice([0, rng.randint(1, k)]),
                     buffer_slots=rng.randint(1, 4), panel_buffers=rng.choice([1, 2]),
                     macs_per_cycle=rng.choice([1, 4, 16, 64]),
                     pipeline_cycles=rng.randint(0, 5), weight_bits=rng.choice([4, 8]),
                     axi_data_bytes=rng.choice([8, 16, 32, 64]),
                     axi_burst_beats=rng.choice([1, 2, 3, 16, 256]),
                     axi_read_outstanding=rng.randint(1, 8),
                     axi_write_outstanding=rng.randint(1, 8),
                     axi_read_latency=rng.randint(0, 40), axi_write_latency=rng.randint(0, 40),
                     dram_bytes_per_cycle=rng.choice([0, 0, 1, 5, 8, 16, 33]))
            with self.subTest(case=case, config=c):
                self.assert_matches_oracle(c)

    def test_event_and_cycle_timing_agree(self):
        rng = random.Random(9)
        for case in range(12):
            c = dict(BASE, m=rng.randint(1, 12), n=rng.randint(1, 12), k=rng.randint(1, 40),
                     tile_m=rng.randint(1, 5), tile_n=rng.randint(1, 5),
                     axi_read_outstanding=rng.randint(1, 4), axi_read_latency=rng.randint(0, 9),
                     axi_write_latency=rng.randint(0, 9),
                     dram_bytes_per_cycle=rng.choice([0, 4, 16]))
            with self.subTest(case=case):
                event, event_spans = run(c, "--timing", "event")
                cycle, cycle_spans = run(c, "--timing", "cycle")
                self.assertEqual(event_spans, cycle_spans)
                for key in event:
                    if key not in {"config", "simulation_wall_seconds"}:
                        self.assertEqual(event[key], cycle[key], key)

    def test_traffic_matches_the_serial_model(self):
        # Both models move the same bytes, including palette weights and
        # weight-cache hits, which drop the weight run from a panel load.
        common = ["--m", "24", "--n", "16", "--k", "40", "--tile-m", "6", "--tile-n", "4",
                  "--tile-k", "16", "--weight-cache-bytes", "4096", "--weight-encoding",
                  "palette", "--palette-bits", "8", "--sram-bytes", str(1 << 22)]
        results = {}
        for model in ("serial", "axi"):
            process = subprocess.run([str(BINARY), *common, "--memory-model", model],
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            results[model] = json.loads(process.stdout)
        serial, axi = results["serial"], results["axi"]
        self.assertGreater(serial["weight_cache_hits"], 0)
        for key in ("read_bytes", "write_bytes", "weight_read_bytes", "weight_cache_hits",
                    "output_hash", "macs", "dma_requests"):
            self.assertEqual(serial[key], axi[key], key)
        self.assertEqual(serial["memory"]["read"]["payload_bytes"], serial["read_bytes"])
        self.assertEqual(axi["memory"]["read"]["payload_bytes"], axi["read_bytes"])
        self.assertEqual(serial["memory"]["model"], "serial")
        self.assertNotIn("dram", serial["memory"])

    def test_snapshots_carry_channel_state(self):
        c = dict(BASE, m=16, n=16, k=48, tile_m=4, tile_n=4, tile_k=16,
                 axi_read_outstanding=3, axi_read_latency=9, dram_bytes_per_cycle=12)
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            process = subprocess.run([str(BINARY), *flags(c), "--progress-jsonl", str(progress),
                                      "--slice-cycles", "5"],
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            batch = json.loads(process.stdout)
            snapshots = [json.loads(line) for line in progress.read_text().splitlines()]
        self.assertGreater(len(snapshots), 10)
        seen_outstanding = seen_both = False
        for snap in snapshots:
            memory = snap["memory"]
            self.assertEqual(snap["config"]["arch"]["memory"]["axi"]["read_outstanding"], 3)
            read, write = memory["read"], memory["write"]
            self.assertLessEqual(read["outstanding"], 3)
            self.assertLessEqual(write["outstanding"], c["axi_write_outstanding"])
            self.assertEqual(snap["dma_busy"], read["active"] or write["active"])
            self.assertEqual(snap["dma_write"], write["active"])
            self.assertEqual("load" in snap["active_stages"], read["active"])
            self.assertEqual("store" in snap["active_stages"], write["active"])
            seen_outstanding |= read["outstanding"] > 1
            seen_both |= read["active"] and write["active"]
        self.assertTrue(seen_outstanding)
        self.assertTrue(seen_both)
        final = snapshots[-1]["memory"]
        for direction in ("read", "write"):
            for key, value in batch["memory"][direction].items():
                self.assertEqual(final[direction][key], value, f"{direction}.{key}")
            self.assertFalse(final[direction]["active"])
            self.assertEqual(final[direction]["outstanding"], 0)
        self.assertEqual(final["dram"], batch["memory"]["dram"])

    def test_paused_live_session_exits_cleanly(self):
        # A transfer in flight when stdin closes belongs to the accelerator,
        # not to a coroutine stack, so the process exits 0 (and leak-free
        # under LeakSanitizer).
        for model in ("serial", "axi"):
            process = subprocess.run(
                [str(BINARY), "--m", "8", "--n", "8", "--k", "32", "--tile-k", "8",
                 "--memory-model", model, "--live-stdio"],
                input="advance 9\n", capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            lines = [json.loads(line) for line in process.stdout.splitlines()]
            self.assertEqual(lines[-1]["cycles"], 9)
            self.assertTrue(lines[-1]["dma_busy"])

    def test_config_echo_reloads_the_axi_port(self):
        c = dict(BASE, axi_data_bytes=32, axi_burst_beats=7, axi_read_outstanding=5,
                 axi_write_outstanding=6, axi_read_latency=3, axi_write_latency=4,
                 dram_bytes_per_cycle=9)
        result, _ = run(c, trace=False)
        axi = result["config"]["arch"]["memory"]
        self.assertEqual(axi["model"], "axi")
        self.assertEqual(axi["dram_bytes_per_cycle"], 9)
        self.assertEqual(axi["axi"], {"data_bytes": 32, "max_burst_beats": 7,
                                      "read_outstanding": 5, "write_outstanding": 6,
                                      "read_latency": 3, "write_latency": 4})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "echo.json"
            path.write_text(json.dumps(result["config"]))
            process = subprocess.run([str(BINARY), "--config", str(path)],
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            again = json.loads(process.stdout)
        for key in result:
            if key != "simulation_wall_seconds":
                self.assertEqual(again[key], result[key], key)

    def assert_rejected(self, arguments: list[str], message: str) -> None:
        process = subprocess.run([str(BINARY), *arguments], capture_output=True, text=True,
                                 timeout=30)
        self.assertEqual(process.returncode, 2, f"{arguments}: {process.stderr}")
        lines = process.stderr.strip().splitlines()
        self.assertTrue(lines and lines[-1].startswith("npu-sim: "), process.stderr)
        self.assertIn(message, lines[-1])

    def test_invalid_memory_options_are_rejected(self):
        axi = ["--memory-model", "axi"]
        cases = [
            (["--memory-model", "dram"], "--memory-model: expected serial|axi, got 'dram'"),
            ([*axi, "--axi-data-bytes", "12"], "AXI data bytes must be 8, 16, 32 or 64"),
            ([*axi, "--axi-data-bytes", "128"], "AXI data bytes must be 8, 16, 32 or 64"),
            ([*axi, "--axi-burst-beats", "0"], "AXI burst beats must be in 1..256"),
            ([*axi, "--axi-burst-beats", "257"], "AXI burst beats must be in 1..256"),
            ([*axi, "--axi-read-outstanding", "0"], "AXI outstanding bursts must be in 1..64"),
            ([*axi, "--axi-write-outstanding", "65"], "AXI outstanding bursts must be in 1..64"),
            ([*axi, "--axi-read-latency", "1000000001"],
             "AXI latencies must be at most 1000000000"),
            ([*axi, "--dram-bytes-per-cycle", "1000000001"],
             "DRAM bytes per cycle must be at most 1000000000"),
            ([*axi, "--axi-burst-beats", "-1"], "--axi-burst-beats: expected an unsigned"),
            ([*axi, "--memory-latency", "5"],
             "--memory-latency applies to --memory-model serial only"),
            ([*axi, "--dma-burst-bytes", "64"],
             "--dma-burst-bytes applies to --memory-model serial only"),
            (["--axi-read-latency", "5"], "--axi-read-latency requires --memory-model axi"),
            (["--dram-bytes-per-cycle", "8"],
             "--dram-bytes-per-cycle requires --memory-model axi"),
            ([*axi, "--timing", "cycle", "--m", "1024", "--n", "1024", "--k", "1024"],
             "cycle mode serial work must be at most 10000000 cycles"),
        ]
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                self.assert_rejected(arguments, message)
        # A config file may carry the other model's keys, as every echo does.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "both.json"
            path.write_text(json.dumps({"arch": {"memory": {
                "model": "axi", "latency": 40, "axi": {"read_latency": 7}}}}))
            process = subprocess.run([str(BINARY), "--config", str(path)],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)["memory"]["model"], "axi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
