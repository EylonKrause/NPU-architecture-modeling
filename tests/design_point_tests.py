#!/usr/bin/env python3
"""The design points in configs/: derivation, sizing rules, simulator agreement, numerics.

The derivation arithmetic of docs/design-points.md is written out here from the public
figures, not read back from the config files: peak MAC/cycle, TOP/s at two operations per
MAC, DRAM bytes per cycle and GB/s, SRAM. The sizing rules in npuperf.designs must reproduce
every tile, burst and weight-cache field; npu-sim must report the same peaks, fill the SRAM
exactly with full tiles, reject the next larger tile, and accept every layer of every bundled
workload. One small GEMM per point is checked bit-exact against the integration oracle.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import integration  # noqa: E402
from npuperf import designs, sim, workloads  # noqa: E402

BINARY: Path
KIB = 1024
GIGA = 1e9

# Public figures; docs/design-points.md cites each source.
U55_TOP = {"macs": 256, "sram_kb": 50, "axi_bytes": 8}          # Ethos-U product brief
U85_TOP = {"macs": 2048, "sram_kb": 267, "tops": 4}             # Ethos-U product brief
M4 = {"ane_cores": 16, "tops": 38, "gbps": 120}                 # apple.com newsroom
M3_ULTRA = {"ane_cores": 32, "gbps": 800}                       # apple.com newsroom
# Vela 5.2.0 ethosu/config_files/Arm/vela.ini.
U55_HIGH_END_SRAM = {"latency": 32, "max_reads": 4, "max_writes": 4, "clock_scale": 1.0,
                     "burst_length": 32}
U85_DRAM_HIGH = {"read_latency": 500, "write_latency": 250, "max_reads": 64, "max_writes": 32,
                 "clock_scale": 0.75, "ports": 2, "port_bytes": 16, "burst_length": 128}
CLIENT_MHZ = 1200


def derived() -> dict[str, dict]:
    """Each point's figures by the derivation in docs/design-points.md."""
    client_dram = M4["gbps"] * GIGA / (CLIENT_MHZ * 1e6)
    workstation_dram = math.ceil(M3_ULTRA["gbps"] * GIGA / (CLIENT_MHZ * 1e6))
    dram = U85_DRAM_HIGH
    dram_ports = {"read_latency": dram["read_latency"], "write_latency": dram["write_latency"],
                  "read_outstanding": dram["max_reads"],
                  "write_outstanding": dram["max_writes"]}
    # U85's SRAM per MAC, for a 32x32 core.
    cluster_sram = U85_TOP["sram_kb"] * KIB * (32 * 32) // U85_TOP["macs"]
    return {
        "edge-256": {
            "array": (16, 16), "cores": 1, "mhz": 1000, "sram": U55_TOP["sram_kb"] * KIB,
            "dram": int(U55_TOP["axi_bytes"] * U55_HIGH_END_SRAM["clock_scale"]),
            "axi_bytes": U55_TOP["axi_bytes"], "min_depth": U55_HIGH_END_SRAM["burst_length"],
            "ports": {"read_latency": U55_HIGH_END_SRAM["latency"],
                      "write_latency": U55_HIGH_END_SRAM["latency"],
                      "read_outstanding": U55_HIGH_END_SRAM["max_reads"],
                      "write_outstanding": U55_HIGH_END_SRAM["max_writes"]},
            "tops": 0.512, "gbps": 8.0},
        "mobile-2k": {
            "array": (32, 64), "cores": 1, "mhz": 1000, "sram": U85_TOP["sram_kb"] * KIB,
            "dram": int(dram["port_bytes"] * dram["ports"] * dram["clock_scale"]),
            "axi_bytes": dram["port_bytes"] * dram["ports"], "min_depth": dram["burst_length"],
            "ports": dram_ports, "tops": 4.096, "gbps": 24.0},
        "client-16c": {
            "array": (32, 32), "cores": M4["ane_cores"], "mhz": CLIENT_MHZ,
            "sram": cluster_sram, "dram": int(client_dram), "axi_bytes": 32,
            "min_depth": dram["burst_length"], "ports": dram_ports, "tops": 39.3216,
            "gbps": 120.0},
        "workstation-32c": {
            "array": (32, 32), "cores": M3_ULTRA["ane_cores"], "mhz": CLIENT_MHZ,
            "sram": cluster_sram, "dram": workstation_dram, "axi_bytes": 32,
            "min_depth": dram["burst_length"], "ports": dram_ports, "tops": 78.6432,
            "gbps": 800.4},
    }


EXPECTED = derived()


def lookup(node: dict, path: str):
    for key in path.split("."):
        node = node[key]
    return node


def simulate(name: str, *flags, timeout: float = 120) -> subprocess.CompletedProcess:
    command = [str(BINARY), "--config", str(designs.config_path(name)), *map(str, flags)]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def result_of(name: str, *flags) -> dict:
    proc = simulate(name, *flags)
    if proc.returncode:
        raise AssertionError(f"{name} {flags}: exit {proc.returncode}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


class DesignPoints(unittest.TestCase):
    def test_every_point_loads_and_is_echoed(self):
        self.assertEqual(sorted(EXPECTED), sorted(designs.DESIGN_POINTS))
        for name in designs.DESIGN_POINTS:
            with self.subTest(name=name):
                document = designs.load(name)
                self.assertEqual(document["name"], name)
                described = result_of(name, "--describe-config")
                self.assertEqual(described["name"], name)
                self.assertEqual(described["description"], document["description"])
                for path, value in designs.leaves(document).items():
                    self.assertEqual(lookup(described, path), value, path)
                self.assertEqual(described["arch"]["memory"]["model"], "axi")

    def test_headline_figures_follow_the_derivation(self):
        # Rounding the published figures: the brief's 512 GOP/s and "4 TOP/s", M4's 38,
        # 120 GB/s exactly, and the smallest integer rate reaching "over 800GB/s".
        self.assertEqual(U55_TOP["macs"] * 2 * 1e9 / 1e12, EXPECTED["edge-256"]["tops"])
        self.assertEqual(round(EXPECTED["mobile-2k"]["tops"]), U85_TOP["tops"])
        self.assertLess(abs(EXPECTED["client-16c"]["tops"] / M4["tops"] - 1), 0.05)
        self.assertEqual(EXPECTED["client-16c"]["dram"] * CLIENT_MHZ / 1e3, M4["gbps"])
        workstation = EXPECTED["workstation-32c"]["dram"]
        self.assertGreaterEqual(workstation * CLIENT_MHZ / 1e3, M3_ULTRA["gbps"])
        self.assertLess((workstation - 1) * CLIENT_MHZ / 1e3, M3_ULTRA["gbps"])
        for name, want in EXPECTED.items():
            with self.subTest(name=name):
                document = designs.load(name)
                arch = document["arch"]
                rows, cols = want["array"]
                macs = rows * cols * want["cores"]
                self.assertEqual((arch["compute"]["array_rows"], arch["compute"]["array_cols"]),
                                 want["array"])
                self.assertEqual(arch["cores"], want["cores"])
                self.assertEqual(arch["frequency_mhz"], want["mhz"])
                self.assertEqual(arch["sram_bytes"], want["sram"])
                self.assertEqual(arch["memory"]["dram_bytes_per_cycle"], want["dram"])
                axi = arch["memory"]["axi"]
                self.assertEqual(axi["data_bytes"], want["axi_bytes"])
                for key, value in want["ports"].items():
                    self.assertEqual(axi[key], value, key)
                self.assertAlmostEqual(2 * macs * want["mhz"] / 1e6, want["tops"], places=9)
                self.assertAlmostEqual(want["dram"] * want["mhz"] / 1e3, want["gbps"], places=9)
                row = designs.summary(document)
                self.assertEqual(row.macs_per_cycle, macs)
                self.assertAlmostEqual(row.peak_tops, want["tops"], places=9)
                self.assertAlmostEqual(row.dram_gbps, want["gbps"], places=9)
                self.assertEqual(row.sram_bytes, want["sram"])
                self.assertEqual(row.axi_bits, 8 * want["axi_bytes"])
                self.assertAlmostEqual(row.ridge_macs_per_byte, macs / want["dram"], places=9)

    def test_cluster_ports_reach_the_dram_rate(self):
        # One core design for both clusters: the narrowest AXI width whose read channels,
        # one per core, carry the larger cluster's DRAM rate.
        width = EXPECTED["client-16c"]["axi_bytes"]
        self.assertEqual(width, EXPECTED["workstation-32c"]["axi_bytes"])
        widest = max(EXPECTED[name]["dram"] / EXPECTED[name]["cores"]
                     for name in ("client-16c", "workstation-32c"))
        self.assertGreaterEqual(width, widest)
        self.assertLess(width // 2, widest)

    def test_sizing_rules_reproduce_the_mapping(self):
        for name, want in EXPECTED.items():
            with self.subTest(name=name):
                document = designs.load(name)
                d = designs.design(document)
                self.assertEqual((d.buffer_slots, d.panel_buffers), (2, 2))
                self.assertEqual(d.tile_m, d.tile_n)
                self.assertEqual(designs.tile_rule(d, want["min_depth"]), (d.tile_m, d.tile_k))
                self.assertEqual(designs.cache_rule(d), d.weight_cache_bytes)
                beats = d.axi_max_burst_beats
                self.assertEqual(designs.burst_rule(d.axi_read_outstanding, d.axi_read_latency),
                                 beats)
                # The burst inequality directly: B covers the latency, B/2 does not.
                outstanding, latency = d.axi_read_outstanding, d.axi_read_latency
                self.assertGreaterEqual(outstanding * beats, latency + beats)
                self.assertLess(outstanding * (beats // 2), latency + beats // 2)
                for size in (d.tile_m, d.tile_k):
                    self.assertEqual(size & (size - 1), 0, size)
                self.assertEqual(d.tile_m % math.lcm(d.array_rows, d.array_cols), 0)
                self.assertGreaterEqual(d.tile_k, want["min_depth"])

    def test_full_tiles_fill_the_sram_and_larger_tiles_do_not_fit(self):
        for name in designs.DESIGN_POINTS:
            with self.subTest(name=name):
                d = designs.design(designs.load(name))
                tile, depth = d.tile_m, d.tile_k
                full = ("--m", tile, "--n", tile, "--k", 2 * depth, "--timing-only")
                self.assertEqual(result_of(name, *full)["scratchpad_bytes"], d.sram_bytes)
                min_depth = EXPECTED[name]["min_depth"]
                for flags in (("--tile-k", 2 * depth),
                              ("--tile-m", 2 * tile, "--tile-n", 2 * tile, "--tile-k",
                               min_depth)):
                    proc = simulate(name, "--m", 2 * tile, "--n", 2 * tile, "--k", 4 * depth,
                                    "--weight-cache-bytes", 0, "--timing-only", *flags)
                    self.assertEqual(proc.returncode, 2, (flags, proc.stdout[:200]))
                    self.assertIn("tile buffers exceed configured SRAM capacity",
                                  proc.stderr.strip().splitlines()[-1])

    def test_small_gemm_is_bit_exact(self):
        rng = random.Random(2026)
        for name, want in EXPECTED.items():
            with self.subTest(name=name):
                d = designs.design(designs.load(name))
                # Ragged edges on every axis, three panels with a short last one, and at
                # least one column tile for every core.
                m = 2 * d.tile_m - 1
                n = max(2, d.cores) * d.tile_n - 8
                k = 2 * d.tile_k + 8
                result = result_of(name, "--m", m, "--n", n, "--k", k, "--dump-output")
                self.assertEqual(result["numerics"], "bit-exact")
                self.assertIs(result["numerical_match"], True)
                self.assertEqual(result["macs"], m * n * k)
                self.assertEqual(result["reduction_blocks"], 3)
                rows, cols = want["array"]
                self.assertEqual(result["effective_macs_per_cycle"], rows * cols * want["cores"])
                self.assertEqual(result["peak_memory_bytes_per_cycle"], want["dram"])
                self.assertEqual(result["scratchpad_bytes"], want["sram"])
                self.assertAlmostEqual(result["modeled_latency_ns"],
                                       result["cycles"] * 1000 / want["mhz"])
                tiles = math.ceil(m / d.tile_m) * math.ceil(n / d.tile_n)
                per_core = [core["tiles"] for core in result["cores"]]
                self.assertEqual(len(per_core), want["cores"])
                self.assertEqual(sum(per_core), tiles)
                self.assertGreater(min(per_core), 0)

                quant = result["config"]["quant"]
                self.assertEqual((quant["rescale"], quant["rounding"]), ("native", "away"))
                c = {key: quant[key] for key in ("input_zero", "weight_zero", "output_zero",
                                                 "multiplier", "shift", "relu")}
                c.update(m=m, n=n, k=k, groups=1, seed=result["config"]["shape"]["seed"])
                accumulators, outputs = result["accumulators"], result["outputs"]
                self.assertEqual(len(outputs), m * n)
                self.assertEqual(result["output_hash"],
                                 integration.expected_hash(accumulators, outputs))
                problem = integration.fixtures(c)
                samples = [(0, 0), (m - 1, n - 1)] + [(rng.randrange(m), rng.randrange(n))
                                                      for _ in range(254)]
                for row, channel in samples:
                    index = row * n + channel
                    self.assertEqual((accumulators[index], outputs[index]),
                                     integration.output_value(c, problem, row, channel),
                                     (row, channel))

    def test_every_workload_layer_fits(self):
        for name in designs.DESIGN_POINTS:
            sram = EXPECTED[name]["sram"]
            for workload in workloads.names():
                for layer, gemm in workloads.iter_calls(workloads.load(workload)):
                    with self.subTest(name=name, workload=workload, layer=layer["name"]):
                        # The config's tiles size the im2col operand; a convolution on
                        # its input footprint runs with the mapper's tiles
                        # (tests/mapper_tests.py).
                        result = result_of(name, *sim.to_flags(gemm.flags(conv=False)),
                                           "--timing-only")
                        self.assertEqual(result["numerics"], "skipped")
                        self.assertLessEqual(result["scratchpad_bytes"], sram)
                        self.assertEqual(result["macs"], gemm.macs)

    def test_docs_table_is_the_generated_one(self):
        text = (ROOT / "docs" / "design-points.md").read_text(encoding="utf-8")
        begin, end = "<!-- table: python -m npuperf.designs -->\n", "<!-- end table -->"
        self.assertIn(begin, text)
        table = text.split(begin, 1)[1].split(end, 1)[0]
        rows = [designs.summary(designs.load(name)) for name in designs.DESIGN_POINTS]
        self.assertEqual(table, designs.markdown(rows))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(designs.main([]), 0)
        self.assertEqual(stdout.getvalue(), table)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(designs.main(["no-such-point"]), 2)
        self.assertIn("no design point 'no-such-point'", stderr.getvalue())


def main() -> int:
    global BINARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    BINARY = args.binary.resolve()
    if not BINARY.is_file():
        parser.error(f"simulator binary not found: {BINARY}")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DesignPoints)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
