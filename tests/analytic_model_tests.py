#!/usr/bin/env python3
"""npuperf.analytic against npu-sim: the request-level schedule, the closed form and the
result parser, on seeded random configurations and on known simulator facts.

schedule() replays every request, so it must equal npu-sim cycle for cycle; analyze() must
be exact in traffic, request counts, compute busy cycles and both caches' hits, its lower
bound must hold, and its estimate must stay within the stated error. The convolution
configurations add implicit convolution, the depthwise datapath and the activation cache;
their footprints are also checked against a brute-force enumeration of kernel taps.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf import analytic, npufile, sim  # noqa: E402

BINARY: Path


def run(flags: dict) -> sim.SimResult:
    return sim.run(flags, BINARY)


def random_config(rng: random.Random, model: str, max_cores: int = 1) -> dict:
    flags = {"m": rng.choice([1, 3, 8, 17, 33, 64, 100]),
             "n": rng.choice([1, 5, 8, 16, 40, 64]),
             "k": rng.choice([7, 16, 31, 64, 100, 200]),
             "tile_m": rng.choice([1, 4, 8, 16, 32]), "tile_n": rng.choice([1, 4, 8, 16, 32]),
             "tile_k": rng.choice([0, 0, 8, 16, 32, 48]),
             "buffer_slots": rng.choice([1, 2, 3]), "panel_buffers": rng.choice([1, 2]),
             "weight_bits": rng.choice([4, 8]), "groups": rng.choice([1, 1, 1, 2, 3]),
             "pipeline_cycles": rng.choice([0, 4, 9]), "timing_only": True,
             "memory_model": model}
    if rng.random() < 0.5:
        flags["macs_per_cycle"] = rng.choice([16, 64, 256])
    else:
        flags["array_rows"], flags["array_cols"] = rng.choice([(4, 4), (8, 4), (3, 5), (16, 16)])
        flags["dataflow"] = rng.choice(["os", "ws"])
    if model == "serial":
        flags["memory_bytes_per_cycle"] = rng.choice([4, 16, 64])
        flags["memory_latency"] = rng.choice([0, 12, 50])
        flags["dma_burst_bytes"] = rng.choice([0, 0, 16, 50, 256])
    else:
        flags["axi_data_bytes"] = rng.choice([8, 16, 32, 64])
        flags["axi_burst_beats"] = rng.choice([1, 4, 16, 256])
        flags["axi_read_outstanding"] = rng.choice([1, 2, 4, 16])
        flags["axi_write_outstanding"] = rng.choice([1, 2, 4])
        flags["axi_read_latency"] = rng.choice([0, 12, 100])
        flags["axi_write_latency"] = rng.choice([0, 12, 40])
        flags["dram_bytes_per_cycle"] = rng.choice([0, 0, 3, 8, 16, 37])
        flags["cores"] = rng.choice([1, 1, 1] + list(range(2, max_cores + 1)))
        flags["partition"] = rng.choice(["m", "n", "auto"])
    if rng.random() < 0.3:
        flags["weight_cache_bytes"] = rng.choice([64, 256, 1024, 4096])
    return flags


def configs(model: str, count: int, seed: int, max_cores: int = 1):
    rng = random.Random(seed)
    for _ in range(count):
        flags = random_config(rng, model, max_cores)
        yield flags, run(flags)


def counters(result) -> dict:
    """The schedule's view of a result: makespan, traffic, compute and request counters."""
    return {"cycles": result["cycles"], "read_bytes": result["read_bytes"],
            "write_bytes": result["write_bytes"], "compute": result["compute_busy_cycles"],
            "requests": result["dma_requests"], "bursts": result["dma_bursts"],
            "hits": result["weight_cache_hits"],
            "core_cycles": [core["cycles"] for core in result["cores"]],
            "core_dma_busy": [core["dma_busy_cycles"] for core in result["cores"]]}


def random_conv_config(rng: random.Random) -> dict:
    """A convolution on its input footprint, a GEMM or a depthwise problem: half on the
    depthwise datapath, half with an activation cache, some clusters."""
    depthwise = rng.random() < 0.5
    if rng.random() < 0.6:
        while True:
            h, w = rng.randint(1, 12), rng.randint(1, 12)
            kh, kw = rng.randint(1, 3), rng.randint(1, 3)
            sh, sw = rng.randint(1, 3), rng.randint(1, 3)
            dh, dw = rng.choice([1, 1, 2]), rng.choice([1, 1, 2])
            pt, pl, pb, pr = (rng.randint(0, 2) for _ in range(4))
            if dh * (kh - 1) + 1 <= h + pt + pb and dw * (kw - 1) + 1 <= w + pl + pr:
                break
        group_channels = rng.randint(1, 4)
        groups = rng.randint(2, 7) if depthwise else rng.choice([1, 1, 2, 3])
        oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
        ow = (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
        row_k = kw * group_channels
        flags = {"m": oh * ow, "n": 1 if depthwise else rng.randint(1, 6), "k": kh * row_k,
                 "groups": groups, "conv_input_h": h, "conv_input_w": w,
                 "conv_input_c": groups * group_channels, "conv_kernel_h": kh,
                 "conv_kernel_w": kw, "conv_stride_h": sh, "conv_stride_w": sw,
                 "conv_dilation_h": dh, "conv_dilation_w": dw, "conv_pad_top": pt,
                 "conv_pad_left": pl, "conv_pad_bottom": pb, "conv_pad_right": pr,
                 "tile_k": rng.randint(0, kh) * row_k}
    else:
        flags = {"m": rng.choice([1, 3, 8, 17, 33, 64]),
                 "n": 1 if depthwise else rng.randint(1, 20), "k": rng.choice([3, 9, 16, 31]),
                 "groups": rng.randint(2, 7) if depthwise else rng.choice([1, 1, 2, 3]),
                 "tile_k": rng.choice([0, 0, 4, 8, 16])}
    flags.update(tile_m=rng.randint(1, flags["m"]), tile_n=rng.randint(1, 6),
                 buffer_slots=rng.choice([1, 2, 3]), panel_buffers=rng.choice([1, 2]),
                 pipeline_cycles=rng.choice([0, 3]), timing_only=True, sram_bytes=1 << 24,
                 memory_model=rng.choice(["serial", "axi"]))
    if depthwise:
        flags["depthwise_macs_per_cycle"] = rng.choice([1, 4, 16, 64])
    if rng.random() < 0.5:
        flags["macs_per_cycle"] = rng.choice([16, 64])
    else:
        flags["array_rows"], flags["array_cols"] = rng.choice([(4, 4), (3, 5)])
        flags["dataflow"] = rng.choice(["os", "ws"])
    if rng.random() < 0.5:
        flags["activation_cache_bytes"] = rng.choice([1, 16, 64, 256, 4096])
    if rng.random() < 0.3:
        flags["weight_cache_bytes"] = rng.choice([16, 256, 4096])
    if rng.random() < 0.3:
        flags["weight_bits"] = 4
    if flags["memory_model"] == "serial":
        flags.update(memory_bytes_per_cycle=rng.choice([4, 16]),
                     memory_latency=rng.choice([0, 12]), dma_burst_bytes=rng.choice([0, 16, 50]))
    else:
        flags.update(axi_data_bytes=rng.choice([8, 16, 32]), axi_burst_beats=rng.choice([1, 4, 16]),
                     axi_read_outstanding=rng.choice([1, 2, 8]),
                     axi_write_outstanding=rng.choice([1, 2]),
                     axi_read_latency=rng.choice([0, 12, 100]),
                     axi_write_latency=rng.choice([0, 40]),
                     dram_bytes_per_cycle=rng.choice([0, 3, 16]),
                     cores=rng.choice([1, 1, 2, 3, 5]), partition=rng.choice(["m", "n", "auto"]))
    return flags


def conv_configs(count: int, seed: int):
    rng = random.Random(seed)
    for _ in range(count):
        flags = random_conv_config(rng)
        yield flags, run(flags)


def replayed(schedule: analytic.Schedule) -> dict:
    return {"cycles": schedule.cycles, "read_bytes": schedule.read_bytes,
            "write_bytes": schedule.write_bytes, "compute": schedule.compute_busy_cycles,
            "requests": schedule.total("dma_requests"), "bursts": schedule.total("dma_bursts"),
            "hits": schedule.total("weight_cache_hits"),
            "core_cycles": [core.cycles for core in schedule.cores],
            "core_dma_busy": [core.dma_busy_cycles for core in schedule.cores]}


class ScheduleTests(unittest.TestCase):
    """schedule() equals npu-sim on every counter it models."""

    def check(self, model: str, count: int, seed: int, max_cores: int = 1) -> None:
        for flags, result in configs(model, count, seed, max_cores):
            schedule = analytic.schedule(analytic.Design.from_result(result))
            with self.subTest(flags=flags):
                self.assertEqual(replayed(schedule), counters(result))
                self.assertEqual(schedule.plan.partition, result["cluster"]["partition"])
                self.assertEqual(schedule.predicted, result["cluster"]["predicted"])

    def test_serial_dma(self):
        self.check("serial", 150, 2)

    def test_axi_single_core(self):
        self.check("axi", 150, 3)

    def test_axi_clusters_share_the_dram_timeline(self):
        self.check("axi", 120, 5, max_cores=8)

    def test_conv_depthwise_and_activation_cache(self):
        for flags, result in conv_configs(400, 21):
            design = analytic.Design.from_result(result)
            schedule = analytic.schedule(design)
            with self.subTest(flags=flags):
                self.assertEqual(replayed(schedule), counters(result))
                self.assertEqual(schedule.predicted, result["cluster"]["predicted"])
                self.assertEqual(
                    tuple(schedule.total(name) for name in (
                        "activation_read_bytes", "activation_cache_hits",
                        "activation_cache_misses", "activation_bytes_saved", "depthwise_macs")),
                    tuple(result[name] for name in (
                        "activation_read_bytes", "activation_cache_hits",
                        "activation_cache_misses", "activation_bytes_saved", "depthwise_macs")))
                self.assertEqual(max(core.peak_cached_activation_bytes for core in schedule.cores),
                                 result["peak_cached_activation_bytes"])
                self.assertEqual(analytic.scratchpad_bytes(design), result["scratchpad_bytes"])
                self.assertEqual(design.peak_macs_per_cycle, result["effective_macs_per_cycle"])

    def test_axi_channel_statistics(self):
        for flags, result in configs("axi", 40, 7):
            schedule = analytic.schedule(analytic.Design.from_result(result))
            for name in ("read", "write"):
                got, want = getattr(schedule.cores[0], name), result["memory"][name]
                with self.subTest(flags=flags, channel=name):
                    self.assertEqual((got.requests, got.bursts, got.beats, got.payload_bytes,
                                      got.busy_cycles, got.max_latency_cycles,
                                      got.peak_outstanding),
                                     (want["requests"], want["bursts"], want["beats"],
                                      want["payload_bytes"], want["busy_cycles"],
                                      want["max_latency_cycles"], want["peak_outstanding"]))
                    self.assertAlmostEqual(got.latency_cycles / max(1, got.requests),
                                           want["mean_latency_cycles"])
            self.assertEqual(schedule.dram_bytes, result["memory"]["dram"]["bytes"])

    def test_palette_decoder_runs_ahead_of_the_array(self):
        """Compressed panels take max(MAC, decode) + pipeline; with every LUT entry present
        the encoded sizes, and so the whole schedule, match."""
        rng = random.Random(11)
        with tempfile.TemporaryDirectory() as directory:
            for trial in range(24):
                m, n, k = rng.choice([1, 3, 8, 20]), rng.choice([16, 33]), rng.choice([16, 64])
                groups, bits = rng.choice([1, 2]), rng.choice([2, 4])
                values = list(range(-(1 << (bits - 1)), 1 << (bits - 1)))
                weights = [values[(row + col) % len(values)] for _ in range(groups)
                           for row in range(k) for col in range(n)]
                activations = [rng.randint(-128, 127) for _ in range(m * groups * k)]
                path = Path(directory) / f"p{trial}.npu"
                npufile.write_problem(path, m, n, k, activations, weights,
                                      [0] * (groups * n), groups)
                flags = {"input": str(path), "groups": groups, "weight_encoding": "palette",
                         "palette_bits": bits, "tile_m": 4, "tile_n": 16,
                         "tile_k": rng.choice([0, 16]),
                         "weight_decode_values_per_cycle": rng.choice([1, 4, 16, 64]),
                         "memory_model": rng.choice(["serial", "axi"]),
                         "macs_per_cycle": rng.choice([16, 64])}
                result = run(flags)
                design = analytic.Design.from_result(result)
                schedule = analytic.schedule(design)
                with self.subTest(flags=flags):
                    self.assertEqual(result["numerics"], "bit-exact")
                    self.assertEqual((schedule.cycles, schedule.read_bytes,
                                      schedule.compute_busy_cycles),
                                     (result["cycles"], result["read_bytes"],
                                      result["compute_busy_cycles"]))
                    self.assertEqual(analytic.analyze(design).compute_busy_cycles,
                                     result["compute_busy_cycles"])


class ClosedFormTests(unittest.TestCase):
    def check(self, model: str, count: int, seed: int, max_cores: int, p95: float) -> None:
        errors, agree = [], 0
        for flags, result in configs(model, count, seed, max_cores):
            model_ = analytic.analyze(analytic.Design.from_result(result))
            with self.subTest(flags=flags):
                self.assertEqual((model_.read_bytes, model_.write_bytes,
                                  model_.compute_busy_cycles, model_.dma_requests,
                                  model_.weight_cache_hits),
                                 (result["read_bytes"], result["write_bytes"],
                                  result["compute_busy_cycles"], result["dma_requests"],
                                  result["weight_cache_hits"]))
                self.assertLessEqual(model_.lower_bound_cycles, result["cycles"])
            errors.append(abs(model_.estimate_cycles - result["cycles"]) / result["cycles"])
            agree += model_.bound == analytic.observed_bound(result)
        errors.sort()
        self.assertLess(statistics.median(errors), 0.01)
        self.assertLess(errors[int(0.95 * len(errors))], p95)
        self.assertGreater(agree / len(errors), 0.95)

    def test_serial(self):
        self.check("serial", 150, 12, 1, 0.15)

    def test_axi(self):
        self.check("axi", 150, 13, 1, 0.25)

    def test_axi_clusters(self):
        self.check("axi", 100, 14, 8, 0.30)

    def test_conv_traffic_and_caches_are_exact(self):
        for flags, result in conv_configs(300, 22):
            model = analytic.analyze(analytic.Design.from_result(result))
            with self.subTest(flags=flags):
                self.assertEqual((model.read_bytes, model.write_bytes, model.compute_busy_cycles,
                                  model.dma_requests, model.weight_cache_hits,
                                  model.activation_read_bytes, model.activation_cache_hits),
                                 (result["read_bytes"], result["write_bytes"],
                                  result["compute_busy_cycles"], result["dma_requests"],
                                  result["weight_cache_hits"], result["activation_read_bytes"],
                                  result["activation_cache_hits"]))
                self.assertLessEqual(model.lower_bound_cycles, result["cycles"])

    def test_exact_where_nothing_overlaps(self):
        exact = 0
        for flags, result in configs("serial", 80, 15):
            model = analytic.analyze(analytic.Design.from_result(result))
            if model.exact_cycles is not None:
                exact += 1
                self.assertEqual(model.exact_cycles, result["cycles"], flags)
                self.assertEqual(model.estimate_cycles, result["cycles"], flags)
        self.assertGreater(exact, 5)

    def test_cluster_predictions_match_npu_sim(self):
        for flags, result in configs("axi", 60, 16, 8):
            design = analytic.Design.from_result(result)
            plan, predicted = analytic.choose_plan(design)
            with self.subTest(flags=flags):
                self.assertEqual(predicted, result["cluster"]["predicted"])
                self.assertEqual((plan.partition, plan.split, plan.units),
                                 (result["cluster"]["partition"], result["cluster"]["split"],
                                  result["cluster"]["units"]))


class KnownFactTests(unittest.TestCase):
    """Facts the core engineers measured on npu-sim."""

    GEMV = {"m": 1, "n": 4096, "k": 4096, "weight_bits": 4, "memory_model": "axi",
            "macs_per_cycle": 512, "axi_data_bytes": 16, "axi_burst_beats": 16,
            "axi_read_latency": 100, "axi_write_latency": 100, "timing_only": True}

    def test_int4_gemv_outstanding(self):
        """One 16-beat burst in flight costs latency + beats per burst; eight cover the
        latency, so each tile's load costs latency + its beats."""
        for outstanding, cycles in ((1, 4_803_753), (8, 707_753)):
            result = run({**self.GEMV, "axi_read_outstanding": outstanding})
            design = analytic.Design.from_result(result)
            self.assertEqual(result["cycles"], cycles)
            self.assertEqual(analytic.schedule(design).cycles, cycles)
            self.assertEqual(analytic.analyze(design).estimate_cycles, cycles)

    def test_int4_gemv_scaling_saturates_at_the_dram_rate(self):
        """Each core's read channel moves at most one 16 B beat per cycle, so the n-split
        GEMV stops scaling at P = dram_bytes_per_cycle / 16."""
        base = {**self.GEMV, "axi_read_outstanding": 16, "dram_bytes_per_cycle": 64,
                "partition": "n", "tile_n": 64, "tile_k": 512}
        cycles = {}
        for cores in (1, 2, 4, 8):
            result = run({**base, "cores": cores})
            cycles[cores] = result["cycles"]
            self.assertEqual(analytic.schedule(analytic.Design.from_result(result)).cycles,
                             result["cycles"])
        self.assertGreater(cycles[1] / cycles[4], 3.5)
        self.assertLess(cycles[4] / cycles[8], 1.1)


class BurstAndTimelineTests(unittest.TestCase):
    def test_bursts_split_at_4_kib_and_count_partial_beats(self):
        # 100 bytes from 3 bytes below a 16 B slot boundary: a partial first beat.
        self.assertEqual(analytic.axi_bursts(13, 100, 16, 16), [(100, 8)])
        # A run across a 4 KiB boundary splits there.
        self.assertEqual(analytic.axi_bursts(4096 - 32, 64, 16, 16), [(32, 2), (32, 2)])
        # The beat limit counts from the aligned slot of a misaligned start.
        self.assertEqual(analytic.axi_bursts(8, 64, 16, 2), [(24, 2), (32, 2), (8, 1)])

    def test_dram_timeline_fills_gaps_and_merges(self):
        timeline = analytic.DramTimeline()
        self.assertEqual(timeline.reserve(0, 10), 0)
        self.assertEqual(timeline.reserve(20, 10), 20)
        self.assertEqual(timeline.reserve(0, 5), 10)      # the gap 10..20
        self.assertEqual(timeline.reserve(0, 6), 30)      # the remaining gap holds 5
        self.assertEqual((timeline.begins, timeline.ends), ([0, 20], [15, 36]))
        timeline.release(15)
        self.assertEqual((timeline.begins, timeline.ends), ([20], [36]))


class FootprintTests(unittest.TestCase):
    """input_spans and activation_runs against every kernel tap of every output pixel."""

    @staticmethod
    def taps(design: analytic.Design, row: int, rows: int, block: int) -> dict[int, set[int]]:
        g = design
        per_block = (g.tile_k or g.k) // (g.conv_kernel_w * g.conv_input_c // g.groups)
        out_w = (g.conv_input_w + g.conv_pad_left + g.conv_pad_right
                 - g.conv_dilation_w * (g.conv_kernel_w - 1) - 1) // g.conv_stride_w + 1
        needed: dict[int, set[int]] = {}
        for m in range(row, row + rows):
            oy, ox = divmod(m, out_w)
            for ky in range(block * per_block, min(g.conv_kernel_h, (block + 1) * per_block)):
                for kx in range(g.conv_kernel_w):
                    y = oy * g.conv_stride_h + ky * g.conv_dilation_h - g.conv_pad_top
                    x = ox * g.conv_stride_w + kx * g.conv_dilation_w - g.conv_pad_left
                    if 0 <= y < g.conv_input_h and 0 <= x < g.conv_input_w:
                        needed.setdefault(y, set()).add(x)
        return needed

    def test_spans_are_the_hull_of_every_tap(self):
        rng = random.Random(31)
        checked = 0
        while checked < 400:
            flags = random_conv_config(rng)
            if "conv_input_h" not in flags:
                continue
            design = analytic.Design.from_flags(flags)
            for row in range(0, design.m, design.tile_m):
                rows = min(design.tile_m, design.m - row)
                for block in range(design.reduction_blocks):
                    needed = self.taps(design, row, rows, block)
                    spans = analytic.input_spans(design, row, rows, block)
                    with self.subTest(flags=flags, row=row, block=block):
                        self.assertEqual(spans, tuple((y, min(xs), max(xs))
                                                      for y, xs in sorted(needed.items())))
            checked += 1

    def test_runs_follow_the_nhwc_layout(self):
        # 6x5x4 input, 2 groups of 2 channels, 3x3 kernel, padding 1, depthwise datapath
        # off: per pixel one 2-byte run of the group's channels.
        design = analytic.Design(m=30, n=3, k=18, groups=2, tile_m=7, tile_n=3, conv_input_h=6,
                                 conv_input_w=5, conv_input_c=4, conv_kernel_h=3,
                                 conv_kernel_w=3, conv_pad_top=1, conv_pad_left=1,
                                 conv_pad_bottom=1, conv_pad_right=1)
        tile = analytic.Tile(1, 7, 0, 7, 3)
        runs = analytic.activation_runs(design, tile, 0)
        # Output pixels 7..13 are output row 1 (columns 2..4) and row 2 (columns 0..3). Input
        # row 0 is read by output row 1 only, columns 1..4; input rows 1..3 by output row 2
        # too, columns 0..4. Each pixel is a run of group 1's channels 2..3.
        hull = [(0, 1)] + [(y, 0) for y in (1, 2, 3)]
        self.assertEqual(runs, [(analytic.ACTIVATION_BASE + (y * 5 + x) * 4 + 2, 2)
                                for y, low in hull for x in range(low, 5)])
        # All channels in the panel: one run per input row over its hull.
        whole = design.replace(groups=1, k=36)
        runs = analytic.activation_runs(whole, analytic.Tile(0, 7, 0, 7, 3), 0)
        self.assertEqual(runs, [(analytic.ACTIVATION_BASE + (y * 5 + low) * 4, (5 - low) * 4)
                                for y, low in hull])
        self.assertEqual(analytic.activation_bytes(whole, 7, 7, 3, 0), (4 + 3 * 5) * 4)


class ObservedTests(unittest.TestCase):
    def test_reads_schema_v2_and_matches_the_design_roofs(self):
        for flags in ({"m": 64, "n": 64, "k": 256, "tile_m": 32, "tile_n": 32, "tile_k": 64,
                       "array_rows": 16, "array_cols": 16, "memory_model": "axi",
                       "dram_bytes_per_cycle": 16, "cores": 2, "timing_only": True},
                      {"m": 16, "n": 16, "k": 32, "timing_only": True}):
            result = run(flags)
            design = analytic.Design.from_result(result)
            observed = analytic.observed(result)
            self.assertEqual(observed.effective_macs_per_cycle, design.peak_macs_per_cycle)
            self.assertEqual(observed.peak_memory_bytes_per_cycle, design.bandwidth)
            self.assertEqual(observed.cores, design.cores)
            self.assertLessEqual(observed.mac_utilization, 1)
            self.assertEqual(observed.compute_cycles_per_core,
                             analytic.analyze(design).compute_cycles_per_core)
        with self.assertRaisesRegex(ValueError, "per-core"):
            analytic.observed({"config": result["config"], "cores": []})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve(strict=True)
    unittest.main(argv=[sys.argv[0], *remaining])
