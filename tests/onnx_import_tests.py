#!/usr/bin/env python3
"""Optional ONNX Runtime full-output differential checks and rejection tests."""

import argparse
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf import graph as graph_runtime
from npuperf import onnx_import

try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper
except ImportError as error:
    raise SystemExit("install optional requirements-onnx.txt before running ONNX tests") from error

BINARY = ROOT / "build" / "npu-sim"


def make_model(nodes, shapes, initializers, outputs, input_type=onnx.TensorProto.INT8):
    graph = helper.make_graph(nodes, "quantized_fixture",
        [helper.make_tensor_value_info(name, input_type, shape) for name, shape in shapes.items()],
        [helper.make_tensor_value_info(name, onnx.TensorProto.INT8, shape) for name, shape in outputs.items()],
        [numpy_helper.from_array(value, name) for name, value in initializers.items()])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)


def constants(weights, xs=1.0, ws=1.0, ys=1.0, xz=0, wz=0, yz=0):
    return {"xs": np.asarray(xs, dtype=np.float32), "xz": np.asarray(xz, dtype=np.int8),
            "w": np.asarray(weights, dtype=np.int8), "ws": np.asarray(ws, dtype=np.float32),
            "wz": np.asarray(wz, dtype=np.int8), "ys": np.asarray(ys, dtype=np.float32),
            "yz": np.asarray(yz, dtype=np.int8)}


def node(op, **attributes):
    operands = ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"]
    return helper.make_node(op, operands, ["y"], name="quantized", **attributes)


def input_specs(arrays):
    return {name: {"dtype": "int8", "shape": list(array.shape), "data": array.reshape(-1).tolist()}
            for name, array in arrays.items()}


class OnnxImportTests(unittest.TestCase):
    def compare(self, model, arrays):
        imported = onnx_import.import_model(model, input_specs(arrays))
        report = graph_runtime.run_graph(imported, BINARY)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        expected = session.run(None, arrays)
        for descriptor, array in zip(session.get_outputs(), expected):
            actual = report["external_outputs"][descriptor.name]
            self.assertEqual(actual["shape"], list(array.shape))
            self.assertEqual(actual["data"], array.reshape(-1).tolist())
        return imported, report, expected

    def test_qlinear_matmul_ties_to_even_positive_and_negative(self):
        x = np.asarray([[1], [3], [-1], [-3]], dtype=np.int8)
        values = constants([[1]], ws=0.5)
        model = make_model([node("QLinearMatMul")], {"x": [4, 1]}, values, {"y": [4, 1]})
        imported, report, expected = self.compare(model, {"x": x})
        self.assertEqual(expected[0].reshape(-1).tolist(), [0, 2, 0, -2])
        self.assertEqual(imported["nodes"][0]["quant"]["rounding"], "even")
        self.assertEqual(report["layers"][0]["npu_invocations"], 1)

    def test_qlinear_matmul_nonzero_zero_points(self):
        values = constants([[-1], [-3]], xs=0.5, ws=0.5, xz=3, wz=-2, yz=-5)
        model = make_model([node("QLinearMatMul")], {"x": [1, 2]}, values, {"y": [1, 1]})
        _, _, expected = self.compare(model, {"x": np.asarray([[5, 1]], np.int8)})
        self.assertEqual(expected[0].item(), -4)

    def test_qlinear_matmul_saturation(self):
        values = constants([[127]])
        model = make_model([node("QLinearMatMul")], {"x": [1, 1]}, values, {"y": [1, 1]})
        _, _, expected = self.compare(model, {"x": np.asarray([[127]], np.int8)})
        self.assertEqual(expected[0].item(), 127)

    def test_qlinear_conv_nchw_layout_padding_bias_and_per_channel_scales(self):
        weights = [[[[1, 0], [0, 1]]], [[[0, 1], [1, 0]]]]
        values = constants(weights, ws=[1, 0.5], wz=[0, 0])
        values["bias"] = np.asarray([1, -1], dtype=np.int32)
        operation = node("QLinearConv", pads=[0, 1, 1, 0])
        operation.input.append("bias")
        model = make_model([operation], {"x": [1, 1, 2, 2]}, values, {"y": [1, 2, 2, 2]})
        imported, report, _ = self.compare(model, {"x": np.asarray([[[[1, 2], [3, 4]]]], np.int8)})
        self.assertEqual(imported["tensors"]["x"]["shape"], [1, 2, 2, 1])
        self.assertEqual(report["outputs"]["y"]["shape"], [1, 2, 2, 2])
        self.assertEqual(imported["io_layouts"]["y"], {"internal": "NHWC", "external": "NCHW"})

    def test_qlinear_conv_same_upper_and_lower(self):
        for auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            with self.subTest(auto_pad=auto_pad):
                values = constants([[[[1, 1], [1, 1]]]], xs=1, ws=0.5, xz=2)
                model = make_model([node("QLinearConv", auto_pad=auto_pad, strides=[2, 2])],
                                   {"x": [1, 1, 3, 3]}, values, {"y": [1, 1, 2, 2]})
                self.compare(model, {"x": np.arange(1, 10, dtype=np.int8).reshape(1, 1, 3, 3)})

    def test_qlinear_conv_dilation(self):
        values = constants([[[[1, 1], [1, 1]]]])
        model = make_model([node("QLinearConv", dilations=[2, 2])],
                           {"x": [1, 1, 3, 3]}, values, {"y": [1, 1, 1, 1]})
        _, _, expected = self.compare(model, {"x": np.arange(1, 10, dtype=np.int8).reshape(1, 1, 3, 3)})
        self.assertEqual(expected[0].item(), 20)

    def test_depthwise_channel_multiplier_and_per_channel_scales(self):
        values = constants(np.asarray([1, 2, -1, 1], np.int8).reshape(4, 1, 1, 1), ws=[1, 0.5, 1, 0.5], wz=[0, 0, 0, 0])
        model = make_model([node("QLinearConv", group=2)], {"x": [1, 2, 2, 2]}, values, {"y": [1, 4, 2, 2]})
        imported, report, _ = self.compare(model, {"x": np.arange(1, 9, dtype=np.int8).reshape(1, 2, 2, 2)})
        self.assertEqual(imported["nodes"][0]["op"], "DepthwiseConv2D")
        self.assertEqual(report["layers"][0]["npu_invocations"], 1)

    def test_two_conv_layers_preserve_internal_nhwc_and_restore_external_output(self):
        values = constants(np.asarray([1, -1], np.int8).reshape(2, 1, 1, 1))
        values.update({"w2": np.asarray([1, 2], np.int8).reshape(1, 2, 1, 1),
                       "w2s": np.asarray(0.5, np.float32), "w2z": np.asarray(0, np.int8)})
        first = node("QLinearConv")
        first.output[0] = "hidden"
        second = helper.make_node("QLinearConv", ["hidden", "ys", "yz", "w2", "w2s", "w2z", "ys", "yz"], ["y"])
        model = make_model([first, second], {"x": [1, 1, 2, 2]}, values, {"y": [1, 1, 2, 2]})
        imported, report, _ = self.compare(model, {"x": np.arange(1, 5, dtype=np.int8).reshape(1, 1, 2, 2)})
        self.assertEqual(len(imported["nodes"]), 2)
        self.assertEqual(report["tensor_metadata"]["hidden"]["shape"], [1, 2, 2, 2])

    def test_exact_rescale_accepts_only_representable_ratios(self):
        self.assertEqual(onnx_import.exact_rescale(0.5, 0.25, 1), (1, 3))
        for scales in ((1, 1, 3), (2 ** -16, 2 ** -16, 1), (65536, 32768, 1), (0.1, 0.2, 0.3)):
            with self.subTest(scales=scales), self.assertRaisesRegex(ValueError, "exactly representable"):
                onnx_import.exact_rescale(*scales)

    def test_nonrepresentable_onnx_scale_is_rejected(self):
        model = make_model([node("QLinearMatMul")], {"x": [1, 1]}, constants([[1]], ys=3), {"y": [1, 1]})
        with self.assertRaisesRegex(ValueError, "exactly representable"):
            onnx_import.import_model(model, input_specs({"x": np.asarray([[1]], np.int8)}))

    def test_unsupported_operator_and_dynamic_shape_are_rejected(self):
        operation = helper.make_node("Relu", ["x"], ["y"])
        model = make_model([operation], {"x": [1, 1]}, {}, {"y": [1, 1]})
        with self.assertRaisesRegex(ValueError, "unsupported ONNX operator"):
            onnx_import.import_model(model, input_specs({"x": np.asarray([[1]], np.int8)}))
        model = make_model([node("QLinearMatMul")], {"x": ["batch", 1]}, constants([[1]]), {"y": ["batch", 1]})
        with self.assertRaisesRegex(ValueError, "dynamic ONNX"):
            onnx_import.import_model(model, input_specs({"x": np.asarray([[1]], np.int8)}))

    def test_unsigned_activations_are_rejected(self):
        model = make_model([node("QLinearMatMul")], {"x": [1, 1]}, constants([[1]]), {"y": [1, 1]}, input_type=onnx.TensorProto.UINT8)
        with self.assertRaises(ValueError):
            onnx_import.import_model(model, input_specs({"x": np.asarray([[1]], np.int8)}))

    def test_external_data_is_never_read_and_is_rejected(self):
        model = make_model([node("QLinearMatMul")], {"x": [1, 1]}, constants([[1]]), {"y": [1, 1]})
        weight = next(value for value in model.graph.initializer if value.name == "w")
        weight.ClearField("raw_data")
        weight.data_location = onnx.TensorProto.EXTERNAL
        weight.external_data.add(key="location", value="/does-not-exist/never-read.weights")
        with tempfile.TemporaryDirectory(prefix="onnx-external-reject-") as directory:
            path = Path(directory) / "external.onnx"
            path.write_bytes(model.SerializeToString())
            loaded = onnx_import.load_model(path)  # succeeds without looking for external weights
            with patch.object(onnx.numpy_helper, "to_array") as decoder:
                with self.assertRaisesRegex(ValueError, "external ONNX tensor data"):
                    onnx_import.import_model(loaded, input_specs({"x": np.asarray([[1]], np.int8)}))
                decoder.assert_not_called()

    def test_initializer_shape_limit_precedes_decode(self):
        model = make_model([node("QLinearMatMul")], {"x": [1, 1]}, constants([[1]]), {"y": [1, 1]})
        weight = next(value for value in model.graph.initializer if value.name == "w")
        del weight.dims[:]
        weight.dims.extend([2048, 2048])
        with patch.object(onnx.numpy_helper, "to_array") as decoder:
            with self.assertRaisesRegex(ValueError, "element limit"):
                onnx_import.import_model(model, input_specs({"x": np.asarray([[1]], np.int8)}))
            decoder.assert_not_called()

    def test_grouped_conv_except_depthwise_is_rejected(self):
        values = constants(np.ones((4, 2, 1, 1), np.int8))
        model = make_model([node("QLinearConv", group=2)], {"x": [1, 4, 1, 1]}, values, {"y": [1, 4, 1, 1]})
        with self.assertRaisesRegex(ValueError, "grouped Conv"):
            onnx_import.import_model(model, input_specs({"x": np.ones((1, 4, 1, 1), np.int8)}))

    def test_matmul_vector_scales_are_rejected(self):
        model = make_model([node("QLinearMatMul")], {"x": [1, 1]}, constants([[1, 2]], ws=[1, 1], wz=[0, 0]), {"y": [1, 2]})
        with self.assertRaisesRegex(ValueError, "scalar"):
            onnx_import.import_model(model, input_specs({"x": np.asarray([[1]], np.int8)}))

    def test_command_line_exports_runnable_graph_and_restored_outputs(self):
        model = make_model([node("QLinearMatMul")], {"x": [4, 1]}, constants([[1]], ws=0.5), {"y": [4, 1]})
        with tempfile.TemporaryDirectory(prefix="onnx-cli-") as directory:
            directory = Path(directory)
            model_path = directory / "model.onnx"
            input_path = directory / "inputs.json"
            graph_path = directory / "graph.json"
            model_path.write_bytes(model.SerializeToString())
            input_path.write_text(json.dumps(input_specs({"x": np.asarray([[1], [3], [-1], [-3]], np.int8)})))
            command = [sys.executable, str(ROOT / "scripts" / "onnx_import.py"), str(model_path),
                       "--inputs", str(input_path), "--output", str(graph_path), "--run", "--binary", str(BINARY)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual(report["external_outputs"]["y"]["data"], [0, 2, 0, -2])
            document = graph_runtime.load_graph(graph_path)
            graph_runtime.validate_graph(document)
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            self.assertEqual(repeated.returncode, 2)
            self.assertTrue(repeated.stderr.startswith("onnx_import: "), repeated.stderr)
            self.assertIn("File exists", repeated.stderr)

    def test_input_shape_data_scale_and_names_are_validated(self):
        model = make_model([node("QLinearMatMul")], {"x": [1, 1]}, constants([[1]]), {"y": [1, 1]})
        valid = input_specs({"x": np.asarray([[1]], np.int8)})
        bad_inputs = [{}, {"other": valid["x"]}, {"x": {**valid["x"], "shape": [2, 1]}},
                      {"x": {**valid["x"], "data": [True]}}, {"x": {**valid["x"], "scale": 0.5}},
                      {"x": {**valid["x"], "zero_point": 1}}]
        for supplied in bad_inputs:
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                onnx_import.import_model(model, supplied)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--binary", type=Path)
    arguments, rest = parser.parse_known_args()
    if arguments.binary is not None:
        BINARY = arguments.binary.resolve()
    unittest.main(argv=[sys.argv[0]] + rest)
