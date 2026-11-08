#!/usr/bin/env python3
"""npuperf.mapper on the design points: aligned tiles that fit SRAM and run on npu-sim, the
lowest analytic estimate among its candidates, the free SRAM split between the caches, the
lowerings of convolutions, and npu-sim cycles close to the best of an exhaustive mapping
search.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf import analytic, autotune, designs, mapper, sim, workloads  # noqa: E402

BINARY: Path


def point(name: str) -> tuple[analytic.Design, int, str]:
    """A design point's Design, minimum panel depth and config path."""
    return (
        designs.design(designs.load(name)),
        designs.min_panel_depth(name),
        str(designs.config_path(name)),
    )


def map_on(item: tuple[str, workloads.Gemm]) -> mapper.Mapping:
    name, gemm = item
    design, min_depth, _ = point(name)
    return mapper.map_call(design, gemm, min_depth)


def map_all(items: list[tuple[str, workloads.Gemm]]) -> dict:
    """map_on of every distinct (point, GEMM), four processes at a time: the mapper replays
    most candidates of a layer."""
    distinct = list(dict.fromkeys(items))
    workers = max(1, min(4, os.cpu_count() or 1))
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        return dict(zip(distinct, pool.map(map_on, distinct), strict=True))


class MapperTests(unittest.TestCase):
    def test_tiles_align_fit_and_run(self):
        """Every ResNet-50 and MobileNetV2 layer on two points: aligned tiles (multiples of
        the array, of 8 on the depthwise datapath, or whole output rows of a convolution),
        panels at least the minimum depth or whole kernel rows, slots within SRAM with the
        caches taking the rest, and npu-sim accepts the mapping with the scratchpad,
        traffic and estimate the model predicts."""
        names = ("edge-256", "client-16c")
        layers = [
            gemm
            for workload in ("resnet50_v1.5_224", "mobilenet_v2_1.0_224")
            for _, gemm in workloads.iter_calls(workloads.load(workload))
        ]
        res3 = next(
            gemm
            for layer, gemm in workloads.iter_calls(workloads.load("resnet50_v1.5_224"))
            if layer["name"] == "layer2.1-3.conv2"
        )
        depthwise = next(
            gemm
            for layer, gemm in workloads.iter_calls(workloads.load("mobilenet_v2_1.0_224"))
            if layer["name"] == "block_3_depthwise"
        )
        runs = (res3, depthwise, res3.plain(), workloads.Gemm("dw", 196, 1, 9, 384))
        mapped = map_all([(name, gemm) for name in names for gemm in (*layers, *runs)])
        for name in names:
            design, min_depth, config = point(name)
            for gemm in layers:
                mapping = mapped[(name, gemm)]
                trial = design.replace(**gemm.flags(conv=mapping.conv))
                rows, cols = (
                    (8, 8) if trial.channel_wise else (design.array_rows, design.array_cols)
                )
                width = analytic.conv_geometry(trial).output_w if mapping.conv else 0
                with self.subTest(point=name, layer=gemm.name):
                    self.assertTrue(
                        mapping.tile_m % rows == 0
                        or mapping.tile_m == gemm.m
                        or (width and mapping.tile_m % width == 0)
                    )
                    self.assertTrue(
                        mapping.tile_n % cols == 0 or mapping.tile_n == trial.grid_columns
                    )
                    if mapping.conv:
                        row_k = analytic.conv_geometry(trial).kernel_row_k
                        self.assertEqual(mapping.tile_k % row_k, 0)
                    else:
                        self.assertTrue(mapping.tile_k == 0 or mapping.tile_k >= min_depth)
                    # 1x1 convolutions may run on their plain operand; the rest on their
                    # input footprint.
                    self.assertTrue(mapping.conv or gemm.pointwise or not gemm.conv)
                    self.assertEqual(mapping.scratchpad_bytes, design.sram_bytes)
                    self.assertLessEqual(mapping.buffer_slots, design.buffer_slots)
                    if mapping.activation_cache_bytes:
                        self.assertFalse(trial.channel_wise)
                        self.assertGreater(trial.grid_columns, mapping.tile_n)
            for gemm in runs:
                mapping = mapped[(name, gemm)]
                result = sim.run(
                    {
                        "config": config,
                        **gemm.flags(conv=mapping.conv),
                        **mapping.flags(),
                        "timing_only": True,
                    },
                    BINARY,
                )
                with self.subTest(point=name, layer=gemm.name):
                    self.assertEqual(result["scratchpad_bytes"], mapping.scratchpad_bytes)
                    model = analytic.analyze(analytic.Design.from_result(result))
                    self.assertEqual(model.estimate_cycles, mapping.estimate_cycles)
                    self.assertEqual(model.dram_bytes, mapping.dram_bytes)
                    self.assertEqual(
                        (model.read_bytes, model.activation_read_bytes),
                        (result["read_bytes"], result["activation_read_bytes"]),
                    )
                    channel_wise = design.replace(**gemm.flags(conv=False)).channel_wise
                    self.assertEqual(result["depthwise_macs"] > 0, channel_wise)
                    self.assertEqual(
                        channel_wise, gemm.groups > 1 and design.depthwise_macs_per_cycle > 0
                    )

    def test_activation_cache_takes_the_largest_row_tile(self):
        """With several column tiles the mapper also tries the activation cache sized to
        the largest row tile's panels; that split helps only when those panels hit."""
        design, min_depth, _ = point("mobile-2k")
        trial = design.replace(
            m=196, n=1024, k=512, tile_m=128, tile_n=64, tile_k=128, buffer_slots=1
        )
        need = mapper.activation_need(trial)
        self.assertEqual(need, 128 * 512)
        self.assertEqual(mapper.cache_splits(trial, need + 10), [(need + 10, 0), (10, need)])
        self.assertEqual(mapper.cache_splits(trial, need - 1), [(need - 1, 0)])
        self.assertEqual(mapper.activation_need(trial.replace(n=64)), 0)
        hit = analytic.analyze(trial.replace(activation_cache_bytes=need))
        self.assertEqual(hit.activation_read_bytes, 196 * 512)
        # One byte short, the full row tile's panels evict each other across its 16 column
        # tiles; the 68-row edge tile still fits.
        miss = analytic.analyze(trial.replace(activation_cache_bytes=need - 1))
        self.assertEqual(miss.activation_read_bytes, 16 * 128 * 512 + 68 * 512)

    def test_pointwise_layers_try_both_lowerings(self):
        design, _, _ = point("edge-256")
        pointwise = next(
            gemm
            for layer, gemm in workloads.iter_calls(workloads.load("resnet50_v1.5_224"))
            if layer["name"] == "layer4.1-2.conv1"
        )
        lowered = mapper.lowerings(design.replace(**pointwise.flags()))
        self.assertEqual([d.conv for d in lowered], [True, False])
        self.assertEqual(lowered[1].replace(**pointwise.conv_flags()), lowered[0])
        res3 = next(
            gemm
            for layer, gemm in workloads.iter_calls(workloads.load("resnet50_v1.5_224"))
            if layer["name"] == "layer2.1-3.conv2"
        )
        self.assertEqual(len(mapper.lowerings(design.replace(**res3.flags()))), 1)
        # K = 2048 whole does not fit 50 KiB with the array's 16-row tiles: the plain
        # operand's panels do.
        mapping = mapper.map_call(design, pointwise, 128)
        self.assertFalse(mapping.conv)
        self.assertGreater(mapping.tile_k, 0)

    def ranked(self, design: analytic.Design, min_depth: int) -> list:
        """The mapper's candidates of ``design`` with their analyses, in rank order."""
        scored = []
        for trial in mapper.candidates(design, min_depth):
            model = analytic.analyze(trial)
            scored.append((mapper._rank(trial, model), trial, model))
        return sorted(scored, key=lambda entry: entry[0])

    def test_choice_is_the_fastest_replay_of_every_candidate(self):
        """The pick has the fewest replayed cycles of the whole candidate set, earliest rank
        on a tie, which npu-sim reproduces. Every candidate's lower bound is at most its
        replay, so the candidates the mapper skips cannot beat the pick. On the Llama decode
        gate_proj GEMV most candidates share the DRAM-bound estimate and the fastest replay
        ranks fifth or lower, so replaying only the top four candidates would miss it."""
        design, min_depth, config = point("client-16c")
        layers = {
            "conv": design.replace(m=196, n=256, k=2304),
            "gemv": design.replace(m=1, n=8192, k=2048, depthwise_macs_per_cycle=0),
        }
        replayed = {}
        for name, layer in layers.items():
            mapping = mapper.map_layer(layer, min_depth)
            ranked = self.ranked(layer, min_depth)
            replays = [analytic.schedule(trial).cycles for _, trial, _ in ranked]
            replayed[name] = (ranked, replays)
            with self.subTest(layer=name):
                for (_, _, model), cycles in zip(ranked, replays, strict=True):
                    self.assertLessEqual(model.lower_bound_cycles, cycles)
                best = min(replays)
                chosen = ranked[replays.index(best)][1]
                self.assertEqual(mapping.replayed_cycles, best)
                self.assertEqual(
                    (
                        mapping.tile_m,
                        mapping.tile_n,
                        mapping.tile_k,
                        mapping.buffer_slots,
                        mapping.weight_cache_bytes,
                    ),
                    (
                        chosen.tile_m,
                        chosen.tile_n,
                        chosen.tile_k,
                        chosen.buffer_slots,
                        chosen.weight_cache_bytes,
                    ),
                )
                result = sim.run(
                    {
                        "config": config,
                        "m": layer.m,
                        "n": layer.n,
                        "k": layer.k,
                        **mapping.flags(),
                        "timing_only": True,
                    },
                    BINARY,
                )
                self.assertEqual(result.cycles, mapping.replayed_cycles)
        gemv, replays = replayed["gemv"]
        estimates = [model.estimate_cycles for _, _, model in gemv]
        self.assertGreater(estimates.count(estimates[0]), 4)
        self.assertGreaterEqual(replays.index(min(replays)), 4)
        plain = mapper.map_layer(layers["conv"], min_depth, cache=False)
        self.assertEqual((plain.weight_cache_bytes, plain.activation_cache_bytes), (0, 0))
        self.assertLessEqual(plain.scratchpad_bytes, design.sram_bytes)

    def test_close_to_the_exhaustive_search(self):
        """ResNet-50 res3 3x3 on mobile-2k on its input footprint, caches off: the mapper's
        npu-sim cycles are within 10% of the fastest mapping of a 48-point search that
        fits."""
        design, min_depth, config = point("mobile-2k")
        gemm = next(
            gemm
            for layer, gemm in workloads.iter_calls(workloads.load("resnet50_v1.5_224"))
            if layer["name"] == "layer2.1-3.conv2"
        )
        plain = mapper.map_call(design, gemm, min_depth, cache=False)
        self.assertTrue(plain.conv)
        grid = [
            (m, n, k, s)
            for m in (32, 64, 128, 256)
            for n in (64, 128)
            for k in (0, 384, 768)
            for s in (1, 2)
        ]
        base = {
            "config": config,
            **gemm.flags(),
            "weight_cache_bytes": 0,
            "activation_cache_bytes": 0,
            "timing_only": True,
        }
        _, rows = autotune.search(BINARY, [gemm.m, gemm.n, gemm.k], grid, base)
        best = min(row["cycles"] for row in rows if row["feasible"])
        chosen = sim.run({**base, **plain.flags()}, BINARY)
        self.assertLessEqual(chosen.cycles, 1.10 * best)

    def test_autotune_reads_the_config_file(self):
        flags = {"config": str(designs.config_path("edge-256")), "m": 8}
        design = autotune.design_of(flags)
        self.assertEqual((design.array_rows, design.sram_bytes, design.m), (16, 51200, 8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve(strict=True)
    unittest.main(argv=[sys.argv[0], *remaining])
