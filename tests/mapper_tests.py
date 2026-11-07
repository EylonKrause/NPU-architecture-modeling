#!/usr/bin/env python3
"""npuperf.mapper on the design points: aligned tiles that fit SRAM and run on npu-sim, the
lowest analytic estimate among its candidates, and npu-sim cycles close to the best of an
exhaustive mapping search.
"""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf import analytic, autotune, designs, mapper, sim, workloads  # noqa: E402

BINARY: Path


def point(name: str) -> tuple[analytic.Design, int, str]:
    """A design point's Design, minimum panel depth and config path."""
    return (designs.design(designs.load(name)), designs.min_panel_depth(name),
            str(designs.config_path(name)))


class MapperTests(unittest.TestCase):
    def test_tiles_align_fit_and_run(self):
        """Every ResNet-50 and MobileNetV2 layer on two points: aligned tiles, panels at
        least the minimum depth, slots within SRAM with the cache taking the rest, and
        npu-sim accepts the mapping with the scratchpad the model predicts."""
        for name in ("edge-256", "client-16c"):
            design, min_depth, config = point(name)
            for workload in ("resnet50_v1.5_224", "mobilenet_v2_1.0_224"):
                for _, gemm in workloads.iter_calls(workloads.load(workload)):
                    mapping = mapper.map_gemm(design, gemm.m, gemm.n, gemm.k, gemm.groups,
                                              min_depth)
                    with self.subTest(point=name, layer=gemm.name):
                        self.assertTrue(mapping.tile_m % design.array_rows == 0
                                        or mapping.tile_m == gemm.m)
                        self.assertTrue(mapping.tile_n % design.array_cols == 0
                                        or mapping.tile_n == gemm.n)
                        self.assertTrue(mapping.tile_k == 0 or
                                        mapping.tile_k >= min_depth)
                        self.assertEqual(mapping.scratchpad_bytes, design.sram_bytes)
                        self.assertLessEqual(mapping.buffer_slots, design.buffer_slots)
            for gemm in (workloads.Gemm("conv", 784, 128, 1152),
                         workloads.Gemm("dw", 196, 1, 9, 384)):
                mapping = mapper.map_gemm(design, gemm.m, gemm.n, gemm.k, gemm.groups,
                                          min_depth)
                result = sim.run({"config": config, **gemm.flags(), **mapping.flags(),
                                  "timing_only": True}, BINARY)
                self.assertEqual(result["scratchpad_bytes"], mapping.scratchpad_bytes)
                model = analytic.analyze(analytic.Design.from_result(result))
                self.assertEqual(model.estimate_cycles, mapping.estimate_cycles)

    def test_choice_is_the_lowest_estimate(self):
        design, min_depth, _ = point("mobile-2k")
        design = design.replace(m=196, n=256, k=2304)
        mapping = mapper.map_layer(design, min_depth)
        estimates = [analytic.analyze(trial).estimate_cycles
                     for trial in mapper.candidates(design, min_depth)]
        self.assertEqual(mapping.estimate_cycles, min(estimates))
        plain = mapper.map_layer(design, min_depth, cache=False)
        self.assertEqual(plain.weight_cache_bytes, 0)
        self.assertLessEqual(plain.scratchpad_bytes, design.sram_bytes)

    def test_close_to_the_exhaustive_search(self):
        """ResNet-50 res3 3x3 on mobile-2k without the weight cache: the mapper's npu-sim
        cycles are within 10% of the fastest mapping of a 64-point search that fits."""
        design, min_depth, config = point("mobile-2k")
        gemm = workloads.Gemm("res3", 784, 128, 1152)
        plain = mapper.map_gemm(design, gemm.m, gemm.n, gemm.k, gemm.groups,
                                min_depth, cache=False)
        grid = [(m, n, k, s) for m in (32, 64, 128, 256) for n in (64, 128)
                for k in (0, 128, 256, 512) for s in (1, 2)]
        base = {"config": config, **gemm.flags(), "weight_cache_bytes": 0, "timing_only": True}
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
