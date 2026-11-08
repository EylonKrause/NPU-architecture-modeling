#!/usr/bin/env python3
"""npuperf.autotune: candidate search under an SRAM limit, the latency and SRAM objectives, the
Pareto front, and the scripts/autotune.py CLI."""

import argparse
import itertools
import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf import analytic
from npuperf.autotune import search

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
            self.assertFalse(
                any(
                    x["cycles"] <= entry["cycles"]
                    and x["scratchpad_bytes"] <= entry["scratchpad_bytes"]
                    and (x["cycles"], x["scratchpad_bytes"])
                    != (entry["cycles"], entry["scratchpad_bytes"])
                    for x in feasible
                )
            )

    def test_sram_objective_and_no_feasible_mapping(self):
        candidates = [(1, 1, 0, 1), (4, 4, 0, 1)]
        report, rows = search(BINARY, [4, 4, 4], candidates, ["--m", 4, "--n", 4, "--k", 4], "sram")
        self.assertEqual(
            report["best"]["scratchpad_bytes"], min(x["scratchpad_bytes"] for x in rows)
        )
        with self.assertRaisesRegex(ValueError, "no feasible"):
            search(BINARY, [4, 4, 4], candidates, ["--m", 4, "--n", 4, "--k", 4, "--sram-bytes", 1])

    def test_search_limit_and_unexpected_error_not_hidden(self):
        with self.assertRaisesRegex(ValueError, "256"):
            search(BINARY, [1, 1, 1], [(1, 1, 0, 1)] * 257, [])
        with self.assertRaisesRegex(RuntimeError, "unknown option"):
            search(BINARY, [1, 1, 1], [(1, 1, 0, 1)], ["--bad-architecture", 1])

    def test_cli_rejects_oversized_grid_before_simulation(self):
        grid = ",".join(map(str, range(1, 8)))
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "autotune.py"),
            "--binary",
            "/does-not-exist",
            "--tile-m-options",
            grid,
            "--tile-n-options",
            grid,
            "--tile-k-options",
            grid,
            "--slot-options",
            grid,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.startswith("autotune: "), result.stderr)
        self.assertIn("256 candidates", result.stderr)
        self.assertNotIn("binary does not exist", result.stderr)

    def test_infeasible_candidates_are_skipped_without_a_launch(self):
        flags = ["--m", 4, "--n", 5, "--k", 6, "--sram-bytes", 90]
        candidates = [(3, 4, 2, 2), (1, 1, 0, 1)]
        report, rows = search(BINARY, [4, 5, 6], candidates, flags)
        skipped = [run for run in report["runs"] if not run["feasible"]]
        self.assertEqual(len(skipped), 1)
        design = analytic.Design.from_flags(
            {"m": 4, "n": 5, "k": 6, "tile_m": 3, "tile_n": 4, "tile_k": 2, "buffer_slots": 2}
        )
        self.assertEqual(
            skipped[0]["predicted_scratchpad_bytes"],
            analytic.scratchpad_bytes(design, panel_buffers=1),
        )
        self.assertNotIn("command", skipped[0])

    def test_cli_table_and_json(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "autotune.py"
        base = [
            sys.executable,
            str(script),
            "--binary",
            str(BINARY),
            "--m",
            "8",
            "--n",
            "8",
            "--k",
            "8",
            "--tile-m-options",
            "4,8",
            "--tile-n-options",
            "8",
            "--tile-k-options",
            "0",
            "--slot-options",
            "1,2",
        ]
        table = subprocess.run(base, capture_output=True, text=True, timeout=60)
        self.assertEqual(table.returncode, 0, table.stderr)
        self.assertIn("Pareto front", table.stdout)
        data = subprocess.run(
            base + ["--format", "json"], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(data.returncode, 0, data.stderr)
        self.assertEqual(json.loads(data.stdout)["candidate_count"], 4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True)
    args, rest = p.parse_known_args()
    BINARY = Path(args.binary).resolve()
    unittest.main(argv=[sys.argv[0]] + rest)
