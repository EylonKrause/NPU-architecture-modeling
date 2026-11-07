#!/usr/bin/env python3
"""Black-box checks of the multi-core cluster (--cores, --partition).

The oracle is written from the cluster description (SPEC section 5), not from
the C++: it splits the output tiles into contiguous per-core bands of units,
replays every core's load, compute and store pipeline with its own LRU weight
cache and AXI port, and shares one DRAM reservation timeline granted in core
order, read before write. Bursts, channels and the DRAM gap search are the
single-core oracle's in memory_model_tests.py. Closed forms check speedup,
the DRAM floor and the m-versus-n weight re-reads independently of it.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import io
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integration  # noqa: E402
import memory_model_tests as mm  # noqa: E402

BINARY: Path

BASE = dict(mm.BASE, weight_cache_bytes=0, cores=2, partition="m")


def ceildiv(value: int, divisor: int) -> int:
    return -(-value // divisor)


# ------------------------------------------------------------------ split --

def units_of(c: dict) -> tuple[str, int]:
    """The unit kind and count: whole groups when G >= P, else row or column tiles."""
    groups, cores = c["groups"], c["cores"]
    if groups >= cores:
        return "groups", groups
    if c["partition"] == "m":
        return "rows", groups * ceildiv(c["m"], c["tile_m"])
    return "columns", groups * ceildiv(c["n"], c["tile_n"])


def owners(c: dict) -> list[int]:
    """Core of every unit: contiguous runs, the first U % P one unit longer."""
    _, count = units_of(c)
    result = []
    for core in range(c["cores"]):
        result += [core] * (count // c["cores"] + (core < count % c["cores"]))
    return result


def unit(c: dict, tile) -> int:
    g, row, col, _, _ = tile
    kind, _ = units_of(c)
    if kind == "groups":
        return g
    if kind == "rows":
        return g * ceildiv(c["m"], c["tile_m"]) + row // c["tile_m"]
    return g * ceildiv(c["n"], c["tile_n"]) + col // c["tile_n"]


def assignment(c: dict) -> list[list[tuple[int, tuple]]]:
    """(global tile id, tile) per core, each in single-core order."""
    owner = owners(c)
    cores: list[list[tuple[int, tuple]]] = [[] for _ in range(c["cores"])]
    for tile_id, tile in enumerate(mm.tile_list(c)):
        cores[owner[unit(c, tile)]].append((tile_id, tile))
    return cores


# ----------------------------------------------------------------- oracle --

class CoreSim:
    """One core's load, compute and store threads, slots and weight cache."""

    def __init__(self, c: dict, index: int, tiles, offsets):
        self.c, self.index, self.tiles, self.offsets = c, index, tiles, offsets
        self.blocks = len(mm.depths(c))
        self.buffers = min(c["panel_buffers"], self.blocks)
        self.read = mm.Channel(c["axi_read_outstanding"], c["axi_read_latency"], 0)
        self.write = mm.Channel(c["axi_write_outstanding"], 0, c["axi_write_latency"])
        self.free_slots, self.computed_panels, self.stored = c["buffer_slots"], 0, 0
        self.load = dict(tile=0, block=0, state="slot")
        self.loaded: list[tuple[int, int]] = []
        self.finished: list[int] = []
        self.compute = None
        self.spans: list[tuple] = []
        self.cache: OrderedDict = OrderedDict()
        self.cached = self.hits = self.misses = self.weight_read_bytes = 0
        self.finish = 0

    def done(self) -> bool:
        return self.stored == len(self.tiles)

    def cached_panel(self, key, size: int) -> bool:
        """LRU of whole panels; a panel larger than the cache bypasses it."""
        capacity = self.c["weight_cache_bytes"]
        if not capacity:
            return False
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        if size > capacity:
            return False
        while self.cached + size > capacity:
            _, evicted = self.cache.popitem(last=False)
            self.cached -= evicted
        self.cache[key] = size
        self.cached += size
        return False

    def panel_runs(self, tile, block: int) -> list[tuple[int, int]]:
        c = self.c
        g, row, col, rows, cols = tile
        depth = mm.depths(c)[block]
        start = block * (c["tile_k"] or c["k"])
        runs = [(mm.ACTIVATIONS + (row + r) * c["groups"] * c["k"] + g * c["k"] + start, depth)
                for r in range(rows)]
        size = mm.weight_bytes(c, depth, cols)
        if not self.cached_panel((g, col, block), size):
            runs.append((mm.WEIGHTS + self.offsets[g, col, block], size))
            self.weight_read_bytes += size
        if block == 0:
            runs.append((mm.PARAMS + (g * c["n"] + col) * 4, cols * 4))
        return runs

    def step(self, now: int) -> bool:
        """One pass over everything that can happen at `now` in zero time."""
        c, load, progressed = self.c, self.load, False
        if load["state"] == "slot" and load["tile"] < len(self.tiles) and self.free_slots:
            self.free_slots -= 1
            load["state"] = "buffer"
            progressed = True
        if load["state"] == "buffer":
            released = load["tile"] * self.blocks + load["block"] - self.buffers + 1
            if load["block"] < self.buffers or self.computed_panels >= released:
                _, tile = self.tiles[load["tile"]]
                self.read.submit(now, self.panel_runs(tile, load["block"]),
                                 (load["tile"], load["block"]), c)
                load["state"] = "reading"
                progressed = True
        if load["state"] == "reading" and self.read.finished(now):
            request = self.read.retire()
            local, block = request["tag"]
            self.spans.append(("load", self.tiles[local][0], request["arrival"], now,
                               request["bytes"], block, self.index))
            self.loaded.append(request["tag"])
            load["block"] += 1
            if load["block"] == self.blocks:
                load.update(tile=load["tile"] + 1, block=0, state="slot")
            else:
                load["state"] = "buffer"
            progressed = True
        if self.compute is not None and self.compute[2] == now:
            (local, block), start, _ = self.compute
            self.spans.append(("compute", self.tiles[local][0], start, now, 0, block,
                               self.index))
            self.computed_panels += 1
            if block + 1 == self.blocks:
                self.finished.append(local)
            self.compute = None
            progressed = True
        if self.compute is None and self.loaded:
            local, block = self.loaded.pop(0)
            self.compute = ((local, block), now,
                            now + mm.compute_cycles(c, self.tiles[local][1], block))
            progressed = True
        if self.write.request is not None and self.write.finished(now):
            request = self.write.retire()
            self.spans.append(("store", self.tiles[request["tag"]][0], request["arrival"], now,
                               request["bytes"], self.blocks - 1, self.index))
            self.stored += 1
            self.free_slots += 1
            if self.done():
                self.finish = now
            progressed = True
        if self.write.request is None and self.finished:
            local = self.finished.pop(0)
            self.write.submit(now, mm.store_runs(c, self.tiles[local][1]), local, c)
            progressed = True
        return progressed

    def upcoming(self) -> list[int]:
        times = [ch.next_issue() for ch in (self.read, self.write) if ch.pending()]
        times += [ch.request["completion"] for ch in (self.read, self.write)
                  if ch.request is not None and not ch.pending()]
        if self.compute is not None:
            times.append(self.compute[2])
        return times

    def busy_cycles(self) -> int:
        """Union of this core's request intervals over both channels."""
        busy = reach = 0
        for start, end in sorted((s[2], s[3]) for s in self.spans if s[0] != "compute"):
            busy += max(0, end - max(start, reach))
            reach = max(reach, end)
        return busy


def simulate(c: dict) -> dict:
    offsets = mm.weight_offsets(c)
    cores = [CoreSim(c, index, tiles, offsets) for index, tiles in enumerate(assignment(c))]
    dram = mm.Dram(c["dram_bytes_per_cycle"])
    now = 0
    while True:
        for core in cores:
            while core.step(now):
                pass
        # Grants after every same-cycle arrival: core order, read before write.
        for core in cores:
            for channel in (core.read, core.write):
                if channel.pending() and channel.next_issue() == now:
                    channel.issue(now, dram, c["axi_data_bytes"])
        if all(core.done() for core in cores):
            break
        upcoming = [t for core in cores for t in core.upcoming() if t > now]
        assert upcoming, "oracle deadlock"
        now = min(upcoming)
    return dict(
        cycles=now, spans=sorted(s for core in cores for s in core.spans),
        dram_bytes=sum(core.read.stats["beats"] + core.write.stats["beats"]
                       for core in cores) * c["axi_data_bytes"],
        cores=[dict(cycles=core.finish, read=core.read.stats, write=core.write.stats,
                    dma_busy_cycles=core.busy_cycles(), weight_cache_hits=core.hits,
                    weight_cache_misses=core.misses, weight_read_bytes=core.weight_read_bytes,
                    tiles=len(core.tiles))
               for core in cores])


# -------------------------------------------------------------- simulator --

def flags(c: dict) -> list[str]:
    return mm.flags(c)


def run(arguments: list, *, trace: bool = False, timeout: float = 120) -> tuple[dict, list]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.csv"
        command = [str(BINARY), *map(str, arguments)] + (["--trace", str(path)] if trace else [])
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if process.returncode:
            raise AssertionError(f"simulator failed ({process.returncode}): {process.stderr}")
        result = json.loads(process.stdout)
        spans = []
        if trace:
            for row in csv.DictReader(io.StringIO(path.read_text())):
                spans.append((row["stage"], int(row["tile_id"]), int(row["start_cycle"]),
                              int(row["end_cycle"]), int(row["bytes"]), int(row["block_id"]),
                              int(row["core"])))
    return result, sorted(spans)


def without_wall_time(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "simulation_wall_seconds"}


CHANNEL_KEYS = ("requests", "bursts", "beats", "payload_bytes", "busy_cycles",
                "max_latency_cycles", "peak_outstanding")


class Cluster(unittest.TestCase):
    def assert_matches_oracle(self, c: dict) -> dict:
        result, spans = run(flags(c), trace=True)
        expected = simulate(c)
        self.assertIs(result["numerical_match"], True)
        self.assertEqual(result["cycles"], expected["cycles"], c)
        self.assertEqual(result["cluster"]["makespan_cycles"], expected["cycles"])
        self.assertEqual(result["memory"]["dram"]["bytes"], expected["dram_bytes"])
        self.assertEqual(spans, expected["spans"], c)
        self.assertEqual(len(result["cores"]), c["cores"])
        for core, (got, want) in enumerate(zip(result["cores"], expected["cores"])):
            self.assertEqual(got["core"], core)
            for key in ("cycles", "dma_busy_cycles", "weight_cache_hits", "weight_cache_misses",
                        "weight_read_bytes", "tiles"):
                self.assertEqual(got[key], want[key], f"core {core} {key} {c}")
            for direction in ("read", "write"):
                for key in CHANNEL_KEYS:
                    self.assertEqual(got["memory"][direction][key], want[direction][key],
                                     f"core {core} {direction}.{key} {c}")
            self.assertEqual(got["read_bytes"], want["read"]["payload_bytes"])
            self.assertEqual(got["write_bytes"], want["write"]["payload_bytes"])
        # The cluster totals add the cores; peaks take the largest.
        for direction in ("read", "write"):
            total = result["memory"][direction]
            for key in ("requests", "bursts", "beats", "payload_bytes", "busy_cycles"):
                self.assertEqual(total[key], sum(core["memory"][direction][key]
                                                 for core in result["cores"]), key)
            for key in ("max_latency_cycles", "peak_outstanding"):
                self.assertEqual(total[key], max(core["memory"][direction][key]
                                                 for core in result["cores"]), key)
        for key in ("read_bytes", "write_bytes", "weight_read_bytes", "weight_cache_hits",
                    "dma_busy_cycles", "compute_busy_cycles", "macs"):
            self.assertEqual(result[key], sum(core[key] for core in result["cores"]), key)
        self.assertEqual(result["peak_inflight_tiles"],
                         max(core["peak_inflight_tiles"] for core in result["cores"]))
        return result

    def test_random_clusters_match_the_oracle(self):
        rng = random.Random(11)
        for case in range(70):
            m, n, k = rng.randint(1, 24), rng.randint(1, 24), rng.randint(1, 60)
            c = dict(BASE, m=m, n=n, k=k, groups=rng.choice([1, 1, 1, 2, 3, 5]),
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
                     dram_bytes_per_cycle=rng.choice([0, 0, 1, 5, 8, 16, 33]),
                     weight_cache_bytes=rng.choice([0, 0, 16, 64, 400, 4096]),
                     cores=rng.choice([1, 2, 3, 4, 5, 8]), partition=rng.choice(["m", "n"]))
            with self.subTest(case=case, config=c):
                result = self.assert_matches_oracle(c)
                self.assertEqual(result["cluster"]["partition"], c["partition"])
                self.assertEqual(result["cluster"]["split"], units_of(c)[0])

    def test_one_core_cluster_is_the_single_core_model(self):
        rng = random.Random(3)
        for case in range(16):
            single = dict(mm.BASE, m=rng.randint(1, 20), n=rng.randint(1, 20),
                          k=rng.randint(1, 50), groups=rng.choice([1, 2]),
                          tile_m=rng.randint(1, 6), tile_n=rng.randint(1, 6),
                          tile_k=rng.choice([0, 3]), axi_read_outstanding=rng.randint(1, 6),
                          dram_bytes_per_cycle=rng.choice([0, 7, 16]))
            plain, plain_spans = run(mm.flags(single), trace=True)
            # The single-core oracle times the run on its own.
            self.assertEqual(plain["cycles"], mm.simulate(single)["cycles"], single)
            for partition in ("m", "n", "auto"):
                with self.subTest(case=case, partition=partition):
                    cluster, spans = run(mm.flags(single) + ["--cores", 1, "--partition",
                                                             partition], trace=True)
                    self.assertEqual(spans, plain_spans)
                    # One core runs every group whatever the partition says;
                    # auto resolves the tie to n.
                    expected = dict(plain, cluster=dict(
                        plain["cluster"], partition="n" if partition == "auto" else partition))
                    for key in plain:
                        if key not in {"config", "simulation_wall_seconds"}:
                            self.assertEqual(cluster[key], expected[key], key)
                    self.assertEqual(plain["cluster"]["partition"], "n")
                    self.assertEqual(cluster["cluster"]["split"], "groups")
                    self.assertEqual(cluster["cluster"]["imbalance"], 1)
        # The serial model takes --cores 1 as well.
        serial = ["--m", 9, "--n", 7, "--k", 13, "--tile-m", 2, "--tile-n", 3]
        plain, _ = run(serial)
        cluster, _ = run(serial + ["--cores", 1, "--partition", "n"])
        for key in plain:
            if key not in {"config", "simulation_wall_seconds"}:
                self.assertEqual(cluster[key], plain[key], key)

    def test_numerics_do_not_depend_on_the_split(self):
        shapes = [
            dict(m=13, n=11, k=19, tile_m=3, tile_n=4),
            dict(m=6, n=9, k=24, tile_m=2, tile_n=2, tile_k=7, groups=2),
            dict(m=5, n=1, k=9, tile_m=2, tile_n=1, groups=12),
        ]
        extras = [[], ["--weight-encoding", "palette", "--palette-bits", "8"],
                  ["--weight-encoding", "sparse", "--sparse-block", "2"],
                  ["--weight-bits", "4", "--array-rows", "3", "--array-cols", "2",
                   "--dataflow", "ws"],
                  ["--rescale", "tosa-double", "--multiplier", "1518500250", "--shift", "38",
                   "--relu"]]
        for shape in shapes:
            c = dict(integration.DEFAULTS, **shape)
            accumulators, outputs = integration.expected_outputs(c)
            base = ["--memory-model", "axi", "--sram-bytes", 1 << 24, "--dump-output"]
            for key in ("m", "n", "k", "tile_m", "tile_n", "tile_k", "groups"):
                if key in c:
                    base += ["--" + key.replace("_", "-"), c[key]]
            for extra in extras:
                reference = None
                for cores in (1, 2, 3, 8):
                    for partition in ("m", "n", "auto"):
                        with self.subTest(shape=shape, extra=extra, cores=cores,
                                          partition=partition):
                            result, _ = run(base + extra + ["--cores", cores,
                                                            "--partition", partition])
                            self.assertIs(result["numerical_match"], True)
                            if not extra:
                                self.assertEqual(result["accumulators"], accumulators)
                                self.assertEqual(result["outputs"], outputs)
                            values = (result["accumulators"], result["outputs"],
                                      result["output_hash"], result["macs"],
                                      result["executed_macs"])
                            reference = reference or values
                            self.assertEqual(values, reference)
                            self.assertEqual(sum(core["tiles"] for core in result["cores"]),
                                             integration.ceildiv(c["m"], c["tile_m"]) *
                                             integration.ceildiv(c["n"], c["tile_n"]) *
                                             c.get("groups", 1))

    def test_compute_bound_speedup_follows_the_unit_split(self):
        # 8 x 8 tiles of 8 x 8 outputs over K = 256 at 16 MACs per cycle: each
        # tile computes for 1024 cycles while its 4 KiB panel loads in under
        # 100 at 64 bytes per beat, so memory never binds.
        c = dict(BASE, m=64, n=64, k=256, tile_m=8, tile_n=8, macs_per_cycle=16,
                 pipeline_cycles=0, axi_data_bytes=64, axi_burst_beats=64,
                 axi_read_outstanding=16, axi_write_outstanding=16, buffer_slots=2)
        tile_cycles = 8 * 8 * 256 // 16
        bands = 8  # row bands under m, column bands under n, 8 tiles each
        baseline = self.assert_matches_oracle(dict(c, cores=1))["cycles"]
        for partition in ("m", "n"):
            for cores in (2, 3, 4, 5, 8, 16):
                with self.subTest(partition=partition, cores=cores):
                    result = self.assert_matches_oracle(dict(c, cores=cores,
                                                              partition=partition))
                    longest = ceildiv(bands, cores)
                    for core in result["cores"]:
                        self.assertEqual(core["compute_busy_cycles"],
                                         core["tiles"] * tile_cycles)
                    # P cores finish ceil(U / P) bands each at best, and only
                    # the pipeline fill and drain separate them from it.
                    ideal = bands / longest
                    speedup = baseline / result["cycles"]
                    self.assertLessEqual(speedup, ideal)
                    self.assertGreater(speedup, 0.97 * ideal)
                    self.assertGreaterEqual(result["cycles"], longest * 8 * tile_cycles)
                    self.assertAlmostEqual(result["cluster"]["imbalance"],
                                           longest * cores / bands)
                    self.assertEqual(sum(core["tiles"] > 0 for core in result["cores"]),
                                     min(cores, bands))

    def test_dram_rate_floors_the_makespan(self):
        # A GEMV streams 256 KiB of weights through 32 column bands; DRAM moves
        # 32 bytes per cycle while each core's read channel moves 16.
        c = dict(BASE, m=1, n=512, k=512, tile_m=1, tile_n=16, macs_per_cycle=4096,
                 pipeline_cycles=0, axi_data_bytes=16, axi_burst_beats=16,
                 axi_read_outstanding=8, axi_read_latency=20, dram_bytes_per_cycle=32,
                 partition="n")
        for cores in (1, 2, 4, 8, 16):
            with self.subTest(cores=cores):
                result = self.assert_matches_oracle(dict(c, cores=cores))
                dram = result["memory"]["dram"]["bytes"]
                floor = ceildiv(dram, 32)
                payload = result["read_bytes"] + result["write_bytes"]
                self.assertGreaterEqual(dram, payload)
                self.assertGreaterEqual(result["cycles"], floor)
                self.assertEqual(result["cluster"]["dram_bound_cycles"], floor)
                self.assertGreaterEqual(result["resource_lower_bound_cycles"], floor)
                self.assertLessEqual(result["resource_lower_bound_cycles"], result["cycles"])
                self.assertLessEqual(result["memory"]["dram"]["utilization"], 1.0)
                if cores == 1:
                    # One read channel moves 16 bytes per cycle: port-bound.
                    self.assertGreater(result["cycles"], 1.8 * floor)
                else:
                    # Two read channels already match the DRAM rate, so more
                    # cores only queue for it.
                    self.assertLess(result["cycles"], 1.05 * floor)

    def test_m_and_n_traffic_differ_by_the_weight_rereads(self):
        # With a weight cache holding a whole group's weights, a core reads each
        # panel it uses once. Under m every core with a row band of a group
        # reads all of that group's weights; under n each panel has one core.
        # Activations, parameters and outputs are the same tile by tile.
        for groups, cores in ((1, 3), (2, 3), (2, 5), (3, 8), (1, 7)):
            c = dict(BASE, m=20, n=12, k=40, tile_m=3, tile_n=4, tile_k=16, groups=groups,
                     weight_cache_bytes=1 << 16, cores=cores)
            weights = sum(mm.weight_bytes(c, depth, min(4, 12 - col))
                          for col in range(0, 12, 4) for depth in mm.depths(c))
            rows = dict(c, partition="m")
            reread = 0
            for tiles in assignment(rows):
                touched = {tile[0] for _, tile in tiles}
                reread += len(touched) * weights
            reread -= groups * weights
            with self.subTest(groups=groups, cores=cores):
                by_rows = self.assert_matches_oracle(rows)
                by_columns = self.assert_matches_oracle(dict(c, partition="n"))
                self.assertEqual(by_columns["weight_read_bytes"], groups * weights)
                self.assertEqual(by_rows["weight_read_bytes"] - by_columns["weight_read_bytes"],
                                 reread)
                self.assertEqual(by_rows["read_bytes"] - by_columns["read_bytes"], reread)
                self.assertEqual(by_rows["write_bytes"], by_columns["write_bytes"])
                for result, partition in ((by_rows, "m"), (by_columns, "n")):
                    predicted = result["cluster"]["predicted"]
                    self.assertEqual(predicted[partition]["read_bytes"], result["read_bytes"])
                    self.assertEqual(predicted[partition]["write_bytes"], result["write_bytes"])
                    self.assertEqual(predicted["m"]["read_bytes"] - predicted["n"]["read_bytes"],
                                     reread)
        # Without a cache every tile reads its own panels: the splits move the
        # same bytes.
        c = dict(BASE, m=20, n=12, k=40, tile_m=3, tile_n=4, cores=4)
        by_rows, by_columns = run(flags(dict(c, partition="m")))[0], \
            run(flags(dict(c, partition="n")))[0]
        self.assertEqual(by_rows["read_bytes"], by_columns["read_bytes"])

    def test_auto_takes_the_smaller_predicted_bound(self):
        common = ["--memory-model", "axi", "--sram-bytes", 1 << 24, "--cores", 4]
        cases = [
            # A GEMV has one row band: m would leave three cores idle.
            (["--m", 1, "--n", 256, "--k", 64, "--tile-m", 1, "--tile-n", 16], "n", "columns"),
            # One column band: n would leave three cores idle.
            (["--m", 96, "--n", 8, "--k", 64, "--tile-m", 8, "--tile-n", 8], "m", "rows"),
            # Whole groups: the splits coincide and the tie goes to n.
            (["--m", 16, "--n", 16, "--k", 16, "--groups", 4], "n", "groups"),
            # Eight bands either way and no cache: identical bounds and bytes.
            (["--m", 64, "--n", 64, "--k", 32], "n", "columns"),
        ]
        for arguments, partition, split in cases:
            with self.subTest(arguments=arguments):
                auto, _ = run(common + arguments)
                self.assertEqual(auto["config"]["mapping"]["partition"], "auto")
                self.assertEqual(auto["cluster"]["partition"], partition)
                self.assertEqual(auto["cluster"]["split"], split)
                predicted = auto["cluster"]["predicted"]
                other = "m" if partition == "n" else "n"
                self.assertLessEqual(predicted[partition]["bound_cycles"],
                                     predicted[other]["bound_cycles"])
                chosen, _ = run(common + arguments + ["--partition", partition])
                for key in auto:
                    if key not in {"config", "simulation_wall_seconds"}:
                        self.assertEqual(auto[key], chosen[key], key)

    def test_waveforms_carry_one_scope_per_core(self):
        c = dict(BASE, m=12, n=10, k=20, tile_m=3, tile_n=4, cores=3, partition="m")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster.vcd"
            result, _ = run(flags(c) + ["--vcd", path])
            changes = read_scoped_vcd(path)
        for core in range(3):
            scope = f"SystemC.npu.core{core}."
            stats = result["cores"][core]
            final = {name[len(scope):]: values[-1][1] for name, values in changes.items()
                     if name.startswith(scope)}
            self.assertEqual(final["done"], 1)
            self.assertEqual(final["inflight_tiles"], 0)
            for key in ("cycles", "read_bytes", "write_bytes", "macs", "compute_busy_cycles",
                        "dma_busy_cycles"):
                self.assertEqual(final[key], stats[key], f"core {core} {key}")
            self.assertTrue(any(value for _, value in changes[scope + "compute_busy"]))
        self.assertFalse(any(name.startswith("SystemC.npu.done") for name in changes))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single.vcd"
            run(flags(dict(c, cores=1)) + ["--vcd", path])
            single = read_scoped_vcd(path)
        self.assertIn("SystemC.npu.done", single)
        self.assertFalse(any(".core0." in name for name in single))

    def test_snapshots_add_up_the_cores(self):
        c = dict(BASE, m=16, n=16, k=48, tile_m=4, tile_n=4, tile_k=16, cores=3,
                 partition="n", dram_bytes_per_cycle=12)
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            batch, _ = run(flags(c) + ["--progress-jsonl", progress, "--slice-cycles", 7])
            snapshots = [json.loads(line) for line in progress.read_text().splitlines()]
        self.assertGreater(len(snapshots), 10)
        seen_parallel = False
        for snap in snapshots:
            cores = snap["cores"]
            self.assertEqual(len(cores), 3)
            self.assertEqual(snap["cluster"], {key: batch["cluster"][key]
                                               for key in ("cores", "partition", "split",
                                                           "units")})
            for key in ("macs", "read_bytes", "write_bytes", "compute_busy_cycles",
                        "dma_busy_cycles", "inflight_tiles", "tiles_completed",
                        "panels_completed", "loaded_panels"):
                self.assertEqual(snap[key], sum(core[key] for core in cores), key)
            self.assertEqual(snap["tiles_total"], sum(core["tiles_total"] for core in cores))
            self.assertEqual(snap["dma_busy"], any(core["dma_busy"] for core in cores))
            self.assertEqual(snap["compute_busy"], any(core["compute_busy"] for core in cores))
            self.assertEqual(snap["completed"], all(core["completed"] for core in cores))
            self.assertEqual(snap["cycles"], max(core["cycles"] for core in cores))
            for core in cores:
                self.assertLessEqual(core["inflight_tiles"], c["buffer_slots"])
                self.assertEqual(core["dma_busy"],
                                 core["memory"]["read"]["active"] or
                                 core["memory"]["write"]["active"])
            busy = [core for core in cores if core["compute_busy"]]
            if busy:
                self.assertEqual(snap["compute_tile_id"], busy[0]["compute_tile_id"])
            seen_parallel |= len(busy) > 1
            self.assertAlmostEqual(snap["compute_utilization"],
                                   snap["compute_busy_cycles"] / (3 * snap["cycles"])
                                   if snap["cycles"] else 0)
        self.assertTrue(seen_parallel)
        final = snapshots[-1]
        self.assertTrue(final["completed"])
        for got, want in zip(final["cores"], batch["cores"]):
            for key in ("cycles", "macs", "read_bytes", "write_bytes", "compute_busy_cycles",
                        "dma_busy_cycles"):
                self.assertEqual(got[key], want[key], key)
            for direction in ("read", "write"):
                for key in CHANNEL_KEYS:
                    self.assertEqual(got["memory"][direction][key],
                                     want["memory"][direction][key])

    def test_live_session_steps_a_cluster(self):
        arguments = [str(BINARY), *flags(dict(BASE, m=12, n=12, k=24, tile_m=4, tile_n=4,
                                                 cores=2)), "--live-stdio"]
        process = subprocess.run(arguments, input="advance 20\nadvance 100000\nstop\n",
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)
        lines = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual([line["type"] for line in lines],
                         ["snapshot", "snapshot", "snapshot", "stopped"])
        self.assertEqual(lines[1]["cycles"], 20)
        self.assertFalse(lines[1]["completed"])
        self.assertTrue(lines[-1]["completed"])
        self.assertIs(lines[-1]["numerical_match"], True)
        self.assertEqual(len(lines[-1]["cores"]), 2)

    def test_timing_only_cluster_keeps_the_schedule(self):
        c = dict(BASE, m=24, n=20, k=36, tile_m=5, tile_n=4, tile_k=12, cores=3,
                 dram_bytes_per_cycle=24)
        exact, _ = run(flags(c))
        timed, _ = run(flags(c) + ["--timing-only"])
        self.assertEqual(timed["numerics"], "skipped")
        for key in ("cycles", "read_bytes", "write_bytes", "dma_busy_cycles",
                    "compute_busy_cycles", "cluster", "cores", "memory"):
            self.assertEqual(timed[key], exact[key], key)

    def test_event_and_cycle_timing_agree(self):
        c = dict(BASE, m=10, n=9, k=14, tile_m=3, tile_n=2, tile_k=5, cores=3,
                 dram_bytes_per_cycle=6, axi_read_latency=3)
        event, event_spans = run(flags(c) + ["--timing", "event"], trace=True)
        cycle, cycle_spans = run(flags(c) + ["--timing", "cycle"], trace=True)
        self.assertEqual(event_spans, cycle_spans)
        for key in event:
            if key not in {"config", "simulation_wall_seconds"}:
                self.assertEqual(event[key], cycle[key], key)

    def test_config_echo_reproduces_the_cluster(self):
        c = dict(BASE, m=12, n=10, k=20, tile_m=3, tile_n=4, cores=5, partition="n",
                 dram_bytes_per_cycle=20)
        result, _ = run(flags(c))
        self.assertEqual(result["config"]["arch"]["cores"], 5)
        self.assertEqual(result["config"]["mapping"]["partition"], "n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "echo.json"
            path.write_text(json.dumps(result["config"]))
            again, _ = run(["--config", path])
        self.assertEqual(without_wall_time(again), without_wall_time(result))

    def assert_rejected(self, arguments: list, message: str) -> None:
        process = subprocess.run([str(BINARY), *map(str, arguments)], capture_output=True,
                                 text=True, timeout=30)
        self.assertEqual(process.returncode, 2, f"{arguments}: {process.stderr}")
        self.assertEqual(process.stdout, "")
        lines = process.stderr.strip().splitlines()
        self.assertTrue(lines and lines[-1].startswith("npu-sim: "), process.stderr)
        self.assertIn(message, lines[-1])

    def test_invalid_cluster_options_are_rejected(self):
        axi = ["--memory-model", "axi"]
        cases = [
            ([*axi, "--cores", 0], "cores must be in 1..64"),
            ([*axi, "--cores", 65], "cores must be in 1..64"),
            (["--cores", 2], "more than one core requires --memory-model axi"),
            ([*axi, "--cores", "two"], "--cores: expected an unsigned 64-bit integer, got 'two'"),
            ([*axi, "--cores", -1], "--cores: expected an unsigned 64-bit integer"),
            ([*axi, "--partition", "k"], "--partition: expected m|n|auto, got 'k'"),
        ]
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                self.assert_rejected(arguments, message)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster.json"
            path.write_text(json.dumps({"arch": {"cores": 4}}))
            self.assert_rejected(["--config", path],
                                 "more than one core requires --memory-model axi")
            path.write_text(json.dumps({"mapping": {"partition": "rows"}}))
            self.assert_rejected(["--config", path], "mapping.partition expects one of m|n|auto")


def read_scoped_vcd(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Value changes keyed by the full scope path of each signal."""
    head, body = path.read_text().split("$enddefinitions", 1)
    names, scope = {}, []
    for line in head.splitlines():
        parts = line.split()
        if parts[:1] == ["$scope"]:
            scope.append(parts[2])
        elif parts[:1] == ["$upscope"]:
            scope.pop()
        elif parts[:1] == ["$var"]:
            names[parts[3]] = ".".join(scope + [parts[4]])
    changes: dict[str, list[tuple[int, int]]] = {name: [] for name in names.values()}
    time = 0
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("#"):
            time = int(line[1:])
        elif line.startswith("b"):
            bits, identifier = line[1:].split()
            if identifier in names and re.fullmatch("[01]+", bits):
                changes[names[identifier]].append((time, int(bits, 2)))
        elif len(line) > 1 and line[0] in "01" and line[1:] in names:
            changes[names[line[1:]]].append((time, int(line[0])))
    return changes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
