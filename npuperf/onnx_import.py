"""Convert a signed-INT8 QLinearMatMul/QLinearConv ONNX model to graph-runtime JSON.

Needs the optional numpy and onnx packages (requirements-onnx.txt). Static shapes only;
NCHW inputs and outputs convert to and from the runtime's NHWC on the host.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from . import cli, jsonio
from . import graph as runtime

MAX_MODEL_BYTES = 32 * 1024 * 1024


def dependencies():
    try:
        import numpy as np
        import onnx
    except ImportError as error:
        raise ValueError(
            "the numpy and onnx packages are required: pip install -r requirements-onnx.txt"
        ) from error
    return onnx, np


def exact_rescale(input_scale, weight_scale, output_scale):
    """(multiplier, shift) with multiplier / 2^shift equal to the exact ratio of the three
    float32 scales; a ratio that needs rounding is a ValueError."""
    for value in (input_scale, weight_scale, output_scale):
        runtime.scale_value(value)
    ratio = (
        Fraction.from_float(float(input_scale))
        * Fraction.from_float(float(weight_scale))
        / Fraction.from_float(float(output_scale))
    )
    denominator = ratio.denominator
    if (
        denominator & (denominator - 1)
        or denominator > (1 << 31)
        or ratio.numerator > runtime.INT32_MAX
    ):
        raise ValueError(
            "ONNX scale ratio is not exactly representable by multiplier/2^shift "
            "(INT32 multiplier, shift 0..31)"
        )
    return ratio.numerator, denominator.bit_length() - 1


def load_model(path):
    onnx, _ = dependencies()
    if Path(path).stat().st_size > MAX_MODEL_BYTES:
        raise ValueError("ONNX model exceeds 32 MiB limit")
    # External-data paths come from the untrusted model file, so they are never followed.
    return onnx.load_model(str(path), load_external_data=False)


def static_shape(value_info, onnx):
    if not value_info.type.HasField("tensor_type"):
        raise ValueError("only tensor-valued ONNX graph values are supported")
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type != onnx.TensorProto.INT8:
        raise ValueError("ONNX activation and output values must be signed INT8")
    if not tensor_type.HasField("shape"):
        raise ValueError("ONNX value has no static shape")
    dimensions = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.HasField("dim_param"):
            raise ValueError("dynamic ONNX shapes are unsupported")
        dimensions.append(int(dimension.dim_value))
    return runtime.shape_value(dimensions)


def _input_spec(spec, shape, scale, zero):
    if not isinstance(spec, dict) or set(spec) - {"dtype", "shape", "data", "scale", "zero_point"}:
        raise ValueError("ONNX inputs require typed tensor objects")
    if not {"dtype", "shape", "data"} <= set(spec):
        raise ValueError("ONNX inputs require dtype, shape and flat data")
    if spec["dtype"] != "int8" or runtime.shape_value(spec["shape"]) != shape:
        raise ValueError("ONNX supplied input dtype/shape does not match model")
    if "scale" in spec and runtime.scale_value(spec["scale"]) != scale:
        raise ValueError("supplied input scale conflicts with ONNX initializer")
    if (
        "zero_point" in spec
        and runtime.integer(spec["zero_point"], "input zero", -128, 127) != zero
    ):
        raise ValueError("supplied input zero point conflicts with ONNX initializer")
    return runtime.parse_tensor({**spec, "scale": scale, "zero_point": zero})


def import_model(model, supplied_inputs):
    """Graph-runtime document of ``model`` with ``supplied_inputs`` bound. The model is
    parsed and checked only."""
    onnx, np = dependencies()
    if model.ByteSize() > MAX_MODEL_BYTES:
        raise ValueError("ONNX model exceeds 32 MiB limit")
    if model.functions or model.graph.sparse_initializer:
        raise ValueError("ONNX local functions and sparse initializers are unsupported")
    opsets = {entry.domain: entry.version for entry in model.opset_import}
    if len(opsets) != len(model.opset_import) or set(opsets) != {""} or not 10 <= opsets[""] <= 21:
        raise ValueError("only standard-domain ONNX opsets 10 through 21 are supported")
    if not 1 <= len(model.graph.node) <= runtime.MAX_NODES:
        raise ValueError("ONNX graph must contain one through 256 nodes")
    initializers = {}
    for initializer in model.graph.initializer:
        if not initializer.name or initializer.name in initializers:
            raise ValueError("ONNX initializers require unique names")
        if initializer.external_data or initializer.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError("external ONNX tensor data is forbidden")
        dimensions = list(initializer.dims)
        if dimensions:
            runtime.shape_value(dimensions)
        if initializer.data_type not in (
            onnx.TensorProto.INT8,
            onnx.TensorProto.INT32,
            onnx.TensorProto.FLOAT,
        ):
            raise ValueError(
                "unsupported ONNX initializer type; only INT8, INT32 and FLOAT scales are supported"
            )
        initializers[initializer.name] = initializer
    for node in model.graph.node:
        if node.domain or node.op_type not in ("QLinearMatMul", "QLinearConv"):
            raise ValueError(
                "unsupported ONNX operator: "
                + (node.domain + ":" if node.domain else "")
                + node.op_type
            )
        allowed = (
            set()
            if node.op_type == "QLinearMatMul"
            else {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"}
        )
        if any(attribute.name not in allowed for attribute in node.attribute):
            raise ValueError("unsupported ONNX node attribute")
        expected_types = {"auto_pad": onnx.AttributeProto.STRING, "group": onnx.AttributeProto.INT}
        for attribute in node.attribute:
            if attribute.type != expected_types.get(attribute.name, onnx.AttributeProto.INTS):
                raise ValueError("incorrect ONNX attribute type")
        if len({attribute.name for attribute in node.attribute}) != len(node.attribute):
            raise ValueError("duplicate ONNX node attribute")
    try:
        onnx.checker.check_model(model)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as error:
        raise ValueError("ONNX checker/shape inference rejected model: " + str(error)) from error
    shapes = {}
    for info in (
        list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    ):
        if info.name not in initializers:
            shapes[info.name] = static_shape(info, onnx)
    external_names = {value.name for value in model.graph.input if value.name not in initializers}
    if not isinstance(supplied_inputs, dict) or set(supplied_inputs) != external_names:
        raise ValueError(
            "supplied input names must exactly match non-initializer ONNX graph inputs"
        )
    decoded = {}

    def constant(name, dtype):
        if name not in initializers:
            raise ValueError("weights, scales, zero points and bias must be ONNX initializers")
        if initializers[name].data_type != dtype:
            raise ValueError("ONNX initializer has unsupported dtype for its role")
        if name not in decoded:
            try:
                decoded[name] = onnx.numpy_helper.to_array(initializers[name])
            except (ValueError, TypeError) as error:
                raise ValueError("malformed ONNX initializer: " + name) from error
        return decoded[name]

    def scalar(name, dtype):
        value = constant(name, dtype)
        if value.ndim != 0:
            raise ValueError(
                "input/output quantization and MatMul weights require scalar scales and zero points"
            )
        result = value.item()
        return (
            runtime.scale_value(result)
            if dtype == onnx.TensorProto.FLOAT
            else runtime.integer(int(result), "zero point", -128, 127)
        )

    def parameter(name, shape, values, scale, zero, dtype="int8"):
        if name in tensors:
            raise ValueError("internal ONNX parameter name collision")
        spec = {
            "dtype": dtype,
            "shape": list(shape),
            "data": values,
            "scale": scale,
            "zero_point": zero,
        }
        runtime.parse_tensor(spec)
        tensors[name] = spec

    tensors, nodes, states = {}, [], {}
    # Choose generated parameter names that cannot alias any model value.
    occupied = set(shapes) | set(initializers)
    for source_node in model.graph.node:
        occupied.update(source_node.output)

    def fresh_name(index, suffix):
        candidate = f"__onnx_{index}_{suffix}"
        while candidate in occupied:
            candidate += "_"
        occupied.add(candidate)
        return candidate

    def bind_activation(name, scale, zero, layout):
        if name not in shapes:
            raise ValueError("missing static shape for ONNX activation: " + name)
        shape = shapes[name]
        if name in states:
            if states[name] != (scale, zero, layout):
                raise ValueError("inconsistent quantization/layout for shared ONNX activation")
            return
        if name not in supplied_inputs:
            raise ValueError(
                "ONNX activation must be an external input or earlier supported node output"
            )
        tensor = _input_spec(supplied_inputs[name], shape, scale, zero)
        values = tensor.data
        runtime_shape = shape
        if layout == "NHWC":
            if len(shape) != 4:
                raise ValueError("QLinearConv requires rank-four NCHW input")
            batches, channels, height, width = shape
            runtime_shape = (batches, height, width, channels)
            values = (
                np.asarray(values, dtype=np.int8)
                .reshape(shape)
                .transpose(0, 2, 3, 1)
                .reshape(-1)
                .tolist()
            )
        parameter(name, runtime_shape, values, scale, zero)
        states[name] = (scale, zero, layout)

    for index, source in enumerate(model.graph.node):
        matmul = source.op_type == "QLinearMatMul"
        if (
            len(source.output) != 1
            or len(source.input) not in ((8,) if matmul else (8, 9))
            or any(not name for name in source.input[:8])
        ):
            raise ValueError("unsupported ONNX operator input/output arity")
        x, xs, xz, w, ws, wz, ys, yz = source.input[:8]
        input_scale, input_zero = (
            scalar(xs, onnx.TensorProto.FLOAT),
            scalar(xz, onnx.TensorProto.INT8),
        )
        output_scale, output_zero = (
            scalar(ys, onnx.TensorProto.FLOAT),
            scalar(yz, onnx.TensorProto.INT8),
        )
        layout = "NC" if matmul else "NHWC"
        bind_activation(x, input_scale, input_zero, layout)
        weight = constant(w, onnx.TensorProto.INT8)
        output_name = source.output[0]
        if output_name in tensors or output_name in states or output_name not in shapes:
            raise ValueError("duplicate or untyped ONNX output")
        if matmul:
            if len(shapes[x]) != 2 or weight.ndim != 2 or shapes[x][1] != weight.shape[0]:
                raise ValueError("only rank-two QLinearMatMul without broadcasting is supported")
            channels = int(weight.shape[1])
            weight_scales = [scalar(ws, onnx.TensorProto.FLOAT)] * channels
            weight_zeros = [scalar(wz, onnx.TensorProto.INT8)] * channels
            weight_shape, weight_values = weight.shape, weight.reshape(-1).tolist()
            expected_shape = (shapes[x][0], channels)
            node = {"name": (source.name or "onnx_node") + "_" + str(index), "op": "MatMul"}
        else:
            if weight.ndim != 4 or len(shapes[x]) != 4:
                raise ValueError("only two-dimensional NCHW QLinearConv is supported")
            channels = int(weight.shape[0])
            scales_array = constant(ws, onnx.TensorProto.FLOAT)
            zeros_array = constant(wz, onnx.TensorProto.INT8)
            if scales_array.shape != zeros_array.shape or scales_array.shape not in (
                (),
                (channels,),
            ):
                raise ValueError(
                    "Conv weight scale/zero shapes must match and be scalar or per-output-channel"
                )
            weight_scales = [
                runtime.scale_value(float(v))
                for v in (
                    scales_array.reshape(-1).tolist()
                    if scales_array.ndim
                    else [scales_array.item()] * channels
                )
            ]
            weight_zeros = [
                int(v)
                for v in (
                    zeros_array.reshape(-1).tolist()
                    if zeros_array.ndim
                    else [zeros_array.item()] * channels
                )
            ]
            attrs = {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in source.attribute
            }
            group = runtime.integer(attrs.get("group", 1), "Conv group", 1, 4096)
            input_channels = shapes[x][1]
            if input_channels != int(weight.shape[1]) * group:
                raise ValueError("Conv input channels do not match weights and group")
            if group == 1:
                weight_shape = (weight.shape[2], weight.shape[3], weight.shape[1], weight.shape[0])
                weight_values = weight.transpose(2, 3, 1, 0).reshape(-1).tolist()
                op = "Conv2D"
            elif (
                group == input_channels
                and int(weight.shape[1]) == 1
                and channels % input_channels == 0
            ):
                multiplier = channels // input_channels
                weight_shape = (weight.shape[2], weight.shape[3], input_channels, multiplier)
                weight_values = (
                    weight.reshape(input_channels, multiplier, weight.shape[2], weight.shape[3])
                    .transpose(2, 3, 0, 1)
                    .reshape(-1)
                    .tolist()
                )
                op = "DepthwiseConv2D"
            else:
                raise ValueError("grouped Conv is unsupported except depthwise")
            node = {
                "name": (source.name or "onnx_node") + "_" + str(index),
                "op": op,
                "strides": list(attrs.get("strides", [1, 1])),
                "dilations": list(attrs.get("dilations", [1, 1])),
            }
            if "kernel_shape" in attrs and list(attrs["kernel_shape"]) != list(weight.shape[2:]):
                raise ValueError("Conv kernel_shape disagrees with weight dimensions")
            auto_pad = attrs.get("auto_pad", b"NOTSET").decode("ascii")
            if auto_pad != "NOTSET" and "pads" in attrs:
                raise ValueError("Conv cannot combine auto_pad and explicit pads")
            if auto_pad in ("SAME_UPPER", "VALID"):
                node["padding"] = "SAME" if auto_pad == "SAME_UPPER" else "VALID"
            elif auto_pad == "NOTSET":
                pads = list(attrs.get("pads", [0, 0, 0, 0]))
                if len(pads) != 4:
                    raise ValueError("Conv pads must contain four values")
                node["padding"] = [pads[0], pads[2], pads[1], pads[3]]
            elif auto_pad == "SAME_LOWER":
                sh, sw = runtime.pair(node["strides"], "strides", [1, 1])
                dh, dw = runtime.pair(node["dilations"], "dilations", [1, 1])
                height, width = shapes[x][2:]
                ph = max(
                    0, ((height + sh - 1) // sh - 1) * sh + (weight.shape[2] - 1) * dh + 1 - height
                )
                pw = max(
                    0, ((width + sw - 1) // sw - 1) * sw + (weight.shape[3] - 1) * dw + 1 - width
                )
                node["padding"] = [(ph + 1) // 2, ph // 2, (pw + 1) // 2, pw // 2]
            else:
                raise ValueError("unsupported Conv auto_pad")
            geometry = runtime.spatial(
                node, shapes[x][2], shapes[x][3], int(weight.shape[2]), int(weight.shape[3])
            )
            expected_shape = (shapes[x][0], channels, geometry[0], geometry[1])
        if shapes[output_name] != expected_shape:
            raise ValueError(
                "ONNX inferred/declared output shape disagrees with the supported operator form"
            )
        ratios = [exact_rescale(input_scale, scale, output_scale) for scale in weight_scales]
        weight_name = fresh_name(index, "weight")
        parameter(weight_name, weight_shape, weight_values, weight_scales[0], weight_zeros[0])
        operands = [x, weight_name]
        q = {
            "multiplier": ratios[0][0],
            "shift": ratios[0][1],
            "output_scale": output_scale,
            "output_zero": output_zero,
            "rounding": "even",
            "weight_zero_points": weight_zeros,
            "multipliers": [pair[0] for pair in ratios],
            "shifts": [pair[1] for pair in ratios],
            "weight_scales": weight_scales,
        }
        if not matmul and len(source.input) == 9 and source.input[8]:
            bias = constant(source.input[8], onnx.TensorProto.INT32)
            if bias.shape != (channels,):
                raise ValueError("Conv bias shape must equal output channels")
            bias_name = fresh_name(index, "bias")
            q["bias_scales"] = [input_scale * scale for scale in weight_scales]
            parameter(bias_name, bias.shape, bias.tolist(), q["bias_scales"][0], 0, "int32")
            operands.append(bias_name)
        node.update(inputs=operands, output=output_name, quant=q)
        nodes.append(node)
        states[output_name] = (output_scale, output_zero, layout)
    if external_names - set(tensors):
        raise ValueError("unused ONNX graph inputs are unsupported")
    outputs = [value.name for value in model.graph.output]
    if any(name not in states for name in outputs):
        raise ValueError("ONNX graph outputs must be supported node outputs")
    graph = {
        "schema_version": 1,
        "name": model.graph.name or "onnx_quantized_graph",
        "tensors": tensors,
        "inputs": sorted(external_names),
        "nodes": nodes,
        "outputs": outputs,
        "io_layouts": {
            name: {"internal": "NHWC", "external": "NCHW"}
            for name in outputs
            if states[name][2] == "NHWC"
        },
    }
    runtime.validate_graph(graph)
    return graph


@cli.main_guard("onnx_import")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("onnx_import", __doc__)
    parser.add_argument("model", type=Path, help="ONNX model")
    parser.add_argument(
        "--inputs", type=Path, required=True, help="JSON map of ONNX-layout INT8 inputs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="graph-runtime JSON to write; an existing file needs --force",
    )
    parser.add_argument("--force", action="store_true", help="overwrite --output if it exists")
    parser.add_argument("--run", action="store_true", help="also run the graph on npu-sim")
    cli.add_binary_options(parser)
    args = parser.parse_args(argv)
    inputs = runtime.load_graph(args.inputs)
    document = import_model(load_model(args.model), inputs)
    jsonio.write_text(args.output, jsonio.dumps(document), overwrite=args.force)
    if args.run:
        report = runtime.run_graph(document, cli.simulator(args), timeout=args.timeout)
    else:
        report = {
            "schema_version": 1,
            "graph_path": str(args.output),
            "nodes": len(document["nodes"]),
            "outputs": document["outputs"],
            "rounding": "even",
            "scale_ratios": "exact",
        }
    print(jsonio.dumps(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
