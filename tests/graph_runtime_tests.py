#!/usr/bin/env python3
"""npuperf.graph: schema and attribute validation, scheduling and tensor liveness, and every
operator end to end against hand-computed outputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf import graph, sim

BINARY = ROOT / "build" / "npu-sim"


def tensor(shape, data, dtype="int8", scale=1.0, zero=0):
    return {"dtype": dtype, "shape": shape, "data": data,
            "scale": scale, "zero_point": zero}


def quant(multiplier=1, shift=0, output_scale=1.0, zero=0, relu=False):
    return {"multiplier": multiplier, "shift": shift, "output_scale": output_scale,
            "output_zero": zero, "relu": relu}


def single(op, tensors, inputs, expected, **attributes):
    return {"schema_version": 1, "name": "test", "tensors": tensors,
            "nodes": [{"name": "operator", "op": op, "inputs": inputs,
                       "output": "result", **attributes}],
            "outputs": ["result"], "expected_outputs": {"result": expected}}


class GraphRuntimeTests(unittest.TestCase):
    def execute(self, document, **kwargs):
        return graph.run_graph(document, BINARY, **kwargs)

    def example(self, name):
        return json.loads((ROOT / "examples" / name).read_text())

    def test_cnn_hand_calculated_output_and_mac_counts(self):
        report = self.execute(self.example("cnn.json"))
        self.assertEqual(report["outputs"]["logits"]["data"], [21, -17])
        self.assertEqual(report["macs"], 44)
        # The 2-channel depthwise layer is one grouped call when npu-sim has --groups.
        depthwise_calls = 1 if graph.grouped_calls(BINARY) else 2
        self.assertEqual([x["npu_invocations"] for x in report["layers"]],
                         [1, depthwise_calls, 0, 0, 1])
        self.assertEqual(report["cpu_fallback_nodes"], ["average_pool", "flatten"])
        self.assertTrue(all(x["cycles"] == 0 for x in report["layers"] if x["execution"] == "cpu_fallback"))

    def test_residual_liveness_retains_skip_input_until_add(self):
        report = self.execute(self.example("residual_mlp.json"))
        self.assertEqual(report["outputs"]["output"]["data"], [8, 1, 5])
        self.assertNotIn("input", report["layers"][0]["released_tensors"])
        self.assertIn("input", report["layers"][2]["released_tensors"])
        self.assertEqual(report["initial_live_tensor_bytes"], 45)
        self.assertEqual(report["peak_live_tensor_bytes"], 48)
        self.assertEqual(report["peak_live_activation_bytes"], 9)
        self.assertEqual(report["final_live_tensor_bytes"], 3)
        self.assertEqual(report["macs"], 18)

    def test_unsorted_nodes_are_topologically_scheduled(self):
        document = self.example("residual_mlp.json")
        document["nodes"].reverse()
        report = self.execute(document)
        self.assertEqual([x["name"] for x in report["layers"]], ["dense1", "dense2", "residual", "activation"])

    def test_event_and_cycle_match_graph_outputs_and_target_cycles(self):
        document = self.example("cnn.json")
        event = self.execute(document, architecture={"timing": "event"})
        cycle = self.execute(document, architecture={"timing": "cycle"})
        self.assertEqual(event["outputs"], cycle["outputs"])
        self.assertEqual(event["sum_npu_cycles"], cycle["sum_npu_cycles"])
        self.assertEqual(event["npu_read_bytes"], cycle["npu_read_bytes"])

    def test_dense_nonzero_zero_points_rounding_and_saturation(self):
        # centered activation=[1,-1], weights=[1,-1;1,1], dot=[0,-2].
        document = single("Dense", {"a": tensor([1, 2], [4, 2], scale=0.5, zero=3),
                                    "w": tensor([2, 2], [-1, -3, -1, -1], scale=0.5, zero=-2),
                                    "b": tensor([2], [1, 1], "int32", scale=0.25)},
                          ["a", "w", "b"], [6, 4], quant=quant(1, 1, 0.5, zero=5))
        report = self.execute(document)
        self.assertEqual(report["outputs"]["result"]["data"], [6, 4])

    def test_convolution_same_padding_uses_quantized_input_zero(self):
        document = single("Conv2D", {"a": tensor([1, 2, 2, 1], [4, 5, 6, 7], zero=3),
                                     "w": tensor([2, 2, 1, 1], [1, 1, 1, 1])},
                          ["a", "w"], [10, 6, 7, 4], padding="SAME", quant=quant())
        report = self.execute(document)
        self.assertEqual(report["outputs"]["result"]["shape"], [1, 2, 2, 1])
        self.assertEqual(report["layers"][0]["macs"], 16)

    def test_convolution_stride_and_dilation(self):
        document = single("Conv2D", {"a": tensor([1, 3, 3, 1], list(range(1, 10))),
                                     "w": tensor([2, 2, 1, 1], [1, 1, 1, 1])},
                          ["a", "w"], [20], strides=[2, 2], dilations=[2, 2], quant=quant())
        self.execute(document)

    def test_convolution_explicit_asymmetric_padding(self):
        document = single("Conv2D", {"a": tensor([1, 1, 2, 1], [2, 3]),
                                     "w": tensor([1, 2, 1, 1], [1, -1])},
                          ["a", "w"], [-2, -1], padding=[0, 0, 1, 0], quant=quant())
        self.execute(document)

    def test_depthwise_channel_multiplier_and_reassembly(self):
        document = single("DepthwiseConv2D", {"a": tensor([1, 1, 2, 2], [2, 3, 4, 5]),
                                             "w": tensor([1, 1, 2, 2], [1, 2, -1, 3])},
                          ["a", "w"], [2, 4, -3, 9, 4, 8, -5, 15], quant=quant())
        report = self.execute(document)
        self.assertEqual(report["outputs"]["result"]["shape"], [1, 1, 2, 4])
        # One grouped call when npu-sim has --groups, else one call per channel.
        self.assertEqual(report["layers"][0]["npu_invocations"],
                         1 if graph.grouped_calls(BINARY) else 2)
        self.assertEqual(report["macs"], 8)

    def test_max_pool_negative_values_do_not_use_zero_padding(self):
        document = single("MaxPool2D", {"a": tensor([1, 2, 2, 1], [-8, -7, -6, -5])},
                          ["a"], [-5, -5, -5, -5], kernel=[2, 2], padding="SAME")
        self.execute(document)

    def test_average_pool_rounding_and_pad_exclusion(self):
        base = single("AveragePool2D", {"a": tensor([1, 1, 2, 1], [1, -2], zero=-1)},
                      ["a"], [-0, -2], kernel=[1, 2], padding="SAME")
        # First centered average=(2-1)/2 rounds +1, then zero=-1 gives 0.
        self.execute(base)
        base["nodes"][0]["count_include_pad"] = True
        base["expected_outputs"]["result"] = [0, -2]
        self.execute(base)

    def test_average_pool_include_pad_changes_result(self):
        document = single("AveragePool2D", {"a": tensor([1, 1, 1, 1], [8], zero=2)},
                          ["a"], [8], kernel=[2, 2], padding="SAME")
        self.execute(document)
        document["nodes"][0]["count_include_pad"] = True
        document["expected_outputs"]["result"] = [4]
        self.execute(document)

    def test_add_quantized_zero_and_saturation(self):
        document = single("Add", {"a": tensor([1, 3], [127, -128, 6], zero=5),
                                   "b": tensor([1, 3], [127, -128, 3], zero=5)},
                          ["a", "b"], [127, -128, 4])
        self.execute(document)

    def test_fused_and_separate_relu_are_equivalent(self):
        document = single("Dense", {"a": tensor([1, 2], [2, -3]),
                                     "w": tensor([2, 2], [1, -1, 1, 1])},
                          ["a", "w"], [0, 0], quant=quant(relu=True))
        fused = self.execute(document)
        document["nodes"][0]["quant"]["relu"] = False
        document["nodes"][0]["output"] = "before_relu"
        document["nodes"].append({"name": "relu", "op": "ReLU", "inputs": ["before_relu"], "output": "result"})
        separate = self.execute(document)
        self.assertEqual(fused["outputs"], separate["outputs"])
        self.assertEqual(fused["sum_npu_cycles"], separate["sum_npu_cycles"])

    def test_int4_reduction_tiling_and_array_geometry_preserve_graph_outputs(self):
        document = self.example("cnn.json")
        baseline = self.execute(document)
        candidate = self.execute(document, architecture={"weight_bits": 4, "tile_k": 2,
                                                        "array_rows": 4, "array_cols": 4})
        self.assertEqual(baseline["outputs"], candidate["outputs"])
        self.assertEqual(baseline["macs"], candidate["macs"])

    def test_host_relu_uses_nonzero_quantized_zero(self):
        self.execute(single("ReLU", {"a": tensor([1, 3], [4, 5, 6], zero=5)}, ["a"], [5, 5, 6]))

    def test_flatten_preserves_batch_axis_and_values(self):
        report = self.execute(single("Flatten", {"a": tensor([2, 2, 2, 1], list(range(8)))},
                                     ["a"], list(range(8))))
        self.assertEqual(report["outputs"]["result"]["shape"], [2, 4])

    def test_per_channel_dense_zero_points_and_scales(self):
        document = single("Dense", {"a": tensor([1, 2], [2, 3]),
                                     "w": tensor([2, 2], [2, 3, 4, 5])},
                          ["a", "w"], [11, 6], quant={**quant(),
                          "weight_zero_points": [1, 2], "multipliers": [1, 1],
                          "shifts": [0, 1], "weight_scales": [1, 0.5]})
        self.execute(document)

    def test_per_channel_depthwise_quantization_selects_global_output_channels(self):
        document = single("DepthwiseConv2D", {"a": tensor([1, 1, 1, 2], [2, 3]),
                                             "w": tensor([1, 1, 2, 2], [2, 3, 4, 5])},
                          ["a", "w"], [2, 1, 1, 0], quant={**quant(),
                          "weight_zero_points": [1, 2, 3, 4], "multipliers": [1, 1, 1, 1],
                          "shifts": [0, 1, 2, 3], "weight_scales": [1, 0.5, 0.25, 0.125]})
        self.execute(document)

    def test_per_channel_length_and_scale_mismatch_rejected(self):
        document = single("Dense", {"a": tensor([1, 1], [1]), "w": tensor([1, 2], [1, 1])},
                          ["a", "w"], [1, 1], quant={**quant(), "multipliers": [1]})
        with self.assertRaisesRegex(ValueError, "per output channel"):
            self.execute(document)
        document["nodes"][0]["quant"] = {**quant(), "shifts": [0, 2]}
        with self.assertRaisesRegex(ValueError, "declared tensor scales"):
            self.execute(document)

    def test_cycle_and_missing_dependency_rejected_before_model(self):
        document = single("ReLU", {"a": tensor([1], [1])}, ["missing"], [1])
        with self.assertRaisesRegex(ValueError, "missing tensor"):
            self.execute(document)
        document["nodes"][0]["inputs"] = ["result"]
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            self.execute(document)

    def test_unsupported_operator_and_attributes_rejected(self):
        document = single("Softmax", {"a": tensor([1], [1])}, ["a"], [1])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.execute(document)
        document["nodes"][0]["op"] = "ReLU"
        document["nodes"][0]["silent_broadcast"] = True
        with self.assertRaisesRegex(ValueError, "unsupported node attributes"):
            self.execute(document)

    def test_bad_tensor_types_shapes_and_scales_rejected(self):
        valid = tensor([1], [1])
        invalid = [dict(valid, dtype="float32"), dict(valid, data=[True]),
                   dict(valid, data=[128]), dict(valid, shape=[0]),
                   dict(valid, data=[]), dict(valid, scale=0),
                   dict(valid, scale=float("nan")), dict(valid, zero_point=-129)]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                self.execute(single("ReLU", {"a": spec}, ["a"], [1]))

    def test_bad_quantization_and_bias_scale_rejected(self):
        document = single("Dense", {"a": tensor([1, 1], [1]), "w": tensor([1, 1], [1]),
                                     "b": tensor([1], [1], "int32", scale=2)},
                          ["a", "w", "b"], [2], quant=quant())
        with self.assertRaisesRegex(ValueError, "bias scale"):
            self.execute(document)
        document["tensors"]["b"]["scale"] = 1
        document["nodes"][0]["quant"]["output_scale"] = 4
        with self.assertRaisesRegex(ValueError, "declared tensor scales"):
            self.execute(document)

    def test_add_cannot_silently_broadcast_or_rescale(self):
        document = single("Add", {"a": tensor([1], [1]), "b": tensor([1, 1], [1])}, ["a", "b"], [2])
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            self.execute(document)
        document["tensors"]["b"]["shape"] = [1]
        document["tensors"]["b"]["scale"] = 0.5
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            self.execute(document)

    def test_int32_accumulator_overflow_rejected_before_invocation(self):
        document = single("Dense", {"a": tensor([1, 1], [1]), "w": tensor([1, 1], [1]),
                                     "b": tensor([1], [(1 << 31) - 1], "int32")},
                          ["a", "w", "b"], [127], quant=quant())
        with patch.object(graph.sim, "run") as simulator:
            with self.assertRaisesRegex(ValueError, "accumulator"):
                self.execute(document)
            simulator.assert_not_called()

    def test_python_reference_checks_every_output_element(self):
        document = single("MatMul", {"a": tensor([1, 1], [2]), "w": tensor([1, 1], [3])},
                          ["a", "w"], [6], quant=quant())
        original = sim.run
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            result.raw["outputs"][0] = 5
            return result
        with patch.object(graph.sim, "run", side_effect=corrupt):
            with self.assertRaisesRegex(sim.CheckError, "independent scalar reference"):
                self.execute(document)

    def test_wrong_golden_is_a_failure(self):
        document = self.example("residual_mlp.json")
        document["expected_outputs"]["output"] = [0, 0, 0]
        with self.assertRaisesRegex(sim.CheckError, "independent golden"):
            self.execute(document)

    def test_declared_even_rounding_changes_halfway_cases(self):
        document = single("Dense", {"a": tensor([4, 1], [1, 3, -1, -3]),
                                     "w": tensor([1, 1], [1], scale=0.5)},
                          ["a", "w"], [0, 2, 0, -2],
                          quant={**quant(1, 1), "rounding": "even"})
        self.execute(document)
        document["nodes"][0]["quant"]["rounding"] = "away"
        document["expected_outputs"]["result"] = [1, 2, -1, -2]
        self.execute(document)

    def test_malformed_depthwise_rejects_before_channel_quantization_allocation(self):
        document = single("DepthwiseConv2D", {
            "a": tensor([1, 1, 1, 4096], [0] * 4096),
            "w": tensor([1, 1, 1, 4096], [0] * 4096)},
            ["a", "w"], [], quant=quant())
        with patch.object(graph, "quantization") as quantization:
            with self.assertRaisesRegex(ValueError, "convolution requires"):
                self.execute(document)
            quantization.assert_not_called()

    def test_depthwise_output_channel_bound_precedes_quantization_allocation(self):
        document = single("DepthwiseConv2D", {
            "a": tensor([1, 1, 1, 64], [0] * 64),
            "w": tensor([1, 1, 64, 128], [0] * (64 * 128))},
            ["a", "w"], [], quant=quant())
        with patch.object(graph, "quantization") as quantization:
            with self.assertRaisesRegex(ValueError, "dimension"):
                self.execute(document)
            quantization.assert_not_called()

    def test_lowered_reduction_bound_precedes_quantization_and_reference(self):
        document = single("Conv2D", {
            "a": tensor([1, 65, 65, 1], [0] * (65 * 65)),
            "w": tensor([65, 65, 1, 1], [0] * (65 * 65))},
            ["a", "w"], [], quant=quant())
        with patch.object(graph, "quantization") as quantization:
            with self.assertRaisesRegex(ValueError, "lowered convolution"):
                self.execute(document)
            quantization.assert_not_called()

    def test_duplicate_producers_rejected(self):
        document = self.example("residual_mlp.json")
        document["nodes"][1]["output"] = "hidden"
        with self.assertRaisesRegex(ValueError, "unique producer"):
            self.execute(document)

    def test_architecture_passes_any_simulator_flag_but_not_runtime_owned_ones(self):
        document = self.example("residual_mlp.json")
        report = self.execute(document, architecture={"tile_m": 1, "memory_latency": 3})
        self.assertEqual(report["outputs"]["output"]["data"], [8, 1, 5])
        for key in ("shift", "input", "groups", "bad key"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "architecture field"):
                self.execute(document, architecture={key: 1})
        with self.assertRaisesRegex(sim.SimError, "unknown option"):
            self.execute(document, architecture={"not_a_simulator_flag": 1})

    def test_cli_table_and_rejected_input(self):
        command = [sys.executable, str(ROOT / "scripts" / "graph_runtime.py"),
                   str(ROOT / "examples" / "cnn.json"), "--binary", str(BINARY),
                   "--format", "table"]
        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("NPU cycles", run.stdout)
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "bad.json"
            bad.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
            run = subprocess.run(command[:2] + [str(bad), "--binary", str(BINARY)],
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 2)
        self.assertTrue(run.stderr.startswith("graph_runtime: duplicate JSON object key"),
                        run.stderr)
        self.assertNotIn("Traceback", run.stderr)

    def test_duplicate_operand_freed_once(self):
        document = single("Add", {"a": tensor([1], [2])}, ["a", "a"], [4])
        report = self.execute(document)
        self.assertEqual(report["released_tensors"], ["a"])
        self.assertEqual(report["final_live_tensor_bytes"], 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--binary", type=Path)
    arguments, rest = parser.parse_known_args()
    if arguments.binary is not None:
        BINARY = arguments.binary.resolve()
    unittest.main(argv=[sys.argv[0]] + rest)
