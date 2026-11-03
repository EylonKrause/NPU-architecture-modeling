#!/usr/bin/env python3
import argparse
import itertools
from pathlib import Path
import sys
import subprocess
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from autotune import search

BINARY = None


class MappingTests(unittest.TestCase):
    def test_latency_choice_and_resource_constraint(self):
        candidates = list(itertools.product([1, 3], [1, 4], [0, 2], [1, 2]))
        flags = ["--m", 4, "--n", 5, "--k", 6, "--sram-bytes", 90]
        report, rows = search(BINARY, [4, 5, 6], candidates, flags)
        feasible = [x for x in rows if x["feasible"]]
        self.assertGreater(len(feasible), 0)
        self.assertLess(len(feasible), len(candidates))
        self.assertEqual(report["best"]["cycles"], min(x["cycles"] for x in feasible))
        self.assertTrue(all(x["scratchpad_bytes"] <= 90 for x in feasible))
        self.assertEqual(len({x["output_hash"] for x in feasible}), 1)
        self.assertTrue(report["pareto"])
        for entry in report["pareto"]:
            self.assertFalse(any(x["cycles"] <= entry["cycles"] and
                                 x["scratchpad_bytes"] <= entry["scratchpad_bytes"] and
                                 (x["cycles"], x["scratchpad_bytes"]) !=
                                 (entry["cycles"], entry["scratchpad_bytes"])
                                 for x in feasible))

    def test_sram_objective_and_no_feasible_mapping(self):
        candidates = [(1, 1, 0, 1), (4, 4, 0, 1)]
        report, rows = search(BINARY, [4, 4, 4], candidates,
                              ["--m", 4, "--n", 4, "--k", 4], "sram")
        self.assertEqual(report["best"]["scratchpad_bytes"], min(x["scratchpad_bytes"] for x in rows))
        with self.assertRaisesRegex(ValueError, "no feasible"):
            search(BINARY, [4, 4, 4], candidates,
                   ["--m", 4, "--n", 4, "--k", 4, "--sram-bytes", 1])

    def test_search_limit_and_unexpected_error_not_hidden(self):
        with self.assertRaisesRegex(ValueError, "256"):
            search(BINARY, [1, 1, 1], [(1, 1, 0, 1)] * 257, [])
        with self.assertRaisesRegex(RuntimeError, "unknown option"):
            search(BINARY, [1, 1, 1], [(1, 1, 0, 1)], ["--bad-architecture", 1])

    def test_cli_rejects_oversized_grid_before_simulation(self):
        grid = ",".join(map(str, range(1, 8)))
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "autotune.py"),
                   "--binary", "/does-not-exist", "--tile-m-options", grid,
                   "--tile-n-options", grid, "--tile-k-options", grid, "--slot-options", grid]
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("256 candidates", result.stderr)
        self.assertNotIn("binary does not exist", result.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True)
    args, rest = p.parse_known_args()
    BINARY = Path(args.binary).resolve()
    unittest.main(argv=[sys.argv[0]] + rest)
