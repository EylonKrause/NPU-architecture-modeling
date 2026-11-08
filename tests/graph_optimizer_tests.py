#!/usr/bin/env python3
"""npuperf.graph_optimize ReLU fusion: fused graphs reproduce the unfused npu-sim outputs with
no host traffic, and rewrites that would change an observable tensor are refused."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf.graph_optimize import optimize_graph, verify_graphs

BINARY = ROOT / "build/npu-sim"


def fixture(op="Dense", zero=-3, rounding="away"):
    a = {"dtype": "int8", "shape": [2, 2], "data": [-128, 127, 7, -9], "scale": 1, "zero_point": -2}
    w = {"dtype": "int8", "shape": [2, 2], "data": [1, -2, 3, -4], "scale": 1, "zero_point": 0}
    if op in ("Conv2D", "DepthwiseConv2D"):
        a["shape"] = [1, 1, 2, 2]
        w["shape"] = [1, 1, 2, 2]
    return {"schema_version": 1, "tensors": {"x": a, "w": w},
            "nodes": [{"name": "project", "op": op, "inputs": ["x", "w"], "output": "pre",
                       "quant": {"multiplier": 1, "shift": 1, "output_scale": 2, "output_zero": zero,
                                 "rounding": rounding}},
                      {"name": "clamp", "op": "ReLU", "inputs": ["pre"], "output": "y"}], "outputs": ["y"]}


class OptimizerTests(unittest.TestCase):
    def test_dense_matmul_conv_depthwise_fusion_matches_and_removes_host_traffic(self):
        for op in ("Dense", "MatMul", "Conv2D", "DepthwiseConv2D"):
            for rounding in ("away", "even"):
                with self.subTest(op=op, rounding=rounding):
                    original = fixture(op, rounding=rounding)
                    saved = deepcopy(original)
                    compiled, report = optimize_graph(original)
                    self.assertEqual(original, saved)
                    self.assertEqual(report["fused_nodes"], 1)
                    self.assertEqual(len(compiled["nodes"]), 1)
                    result = verify_graphs(original, compiled, BINARY)
                    self.assertTrue(result["outputs_match"])
                    self.assertGreater(result["original"]["host_read_bytes"], 0)
                    self.assertEqual(result["compiled"]["host_read_bytes"], 0)
                    self.assertEqual(result["compiled"]["host_write_bytes"], 0)
                    self.assertEqual(result["compiled"]["cpu_fallback_nodes"], [])
                    # The output stage applies ReLU as a clamp at no cycle cost
                    # (src/core.cpp), so fusion leaves NPU cycles and writes unchanged.
                    self.assertEqual(result["original"]["npu_cycles"], result["compiled"]["npu_cycles"])
                    self.assertEqual(result["original"]["npu_write_bytes"], result["compiled"]["npu_write_bytes"])

    def test_shared_intermediate_is_preserved(self):
        graph = fixture()
        graph["nodes"].append({"name": "skip", "op": "Add", "inputs": ["pre", "pre"], "output": "z"})
        compiled, report = optimize_graph(graph)
        self.assertEqual(report["fused_nodes"], 0)
        self.assertEqual(compiled, graph)

    def test_observed_intermediate_cannot_be_changed(self):
        graph = fixture()
        graph["outputs"].append("pre")
        compiled, report = optimize_graph(graph)
        self.assertEqual(report["fused_nodes"], 0)
        self.assertTrue(verify_graphs(graph, compiled, BINARY)["outputs_match"])

    def test_redundant_chain_unsorted_input_and_idempotence(self):
        graph = fixture()
        graph["nodes"].append({"name": "again", "op": "ReLU", "inputs": ["y"], "output": "z"})
        graph["nodes"].reverse()
        graph["outputs"] = ["z"]
        compiled, report = optimize_graph(graph)
        self.assertEqual(report["fused_nodes"], 2)
        self.assertEqual(len(compiled["nodes"]), 1)
        self.assertTrue(verify_graphs(graph, compiled, BINARY)["outputs_match"])
        twice, second = optimize_graph(compiled)
        self.assertEqual(twice, compiled)
        self.assertEqual(second["fused_nodes"], 0)

    def test_invalid_attributes_are_rejected(self):
        for invalid in (0, 1, "false", None):
            graph = fixture()
            graph["nodes"][0]["quant"]["relu"] = invalid
            with self.assertRaisesRegex(ValueError, "boolean"):
                optimize_graph(graph)
        graph = fixture()
        graph["nodes"][1]["quant"] = {}
        with self.assertRaises(ValueError):
            optimize_graph(graph)

    def test_cli_verify_and_create_only(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source.json", Path(temp) / "compiled.json"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/graph_optimize.py"), str(source),
                       "--binary", str(BINARY), "--verify", "--output", str(output)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(json.loads(run.stdout)["verification"]["outputs_match"])
            previous = output.read_bytes()
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 2)
            self.assertTrue(run.stderr.startswith("graph_optimize: "), run.stderr)
            self.assertIn("exists", run.stderr)
            self.assertEqual(output.read_bytes(), previous)
            output.write_text("{}", encoding="utf-8")
            run = subprocess.run(command + ["--force"], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(output.read_bytes(), previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
