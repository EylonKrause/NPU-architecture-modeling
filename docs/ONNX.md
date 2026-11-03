# Optional strict ONNX import

The importer converts a **small, explicit signed-INT8 subset** of ONNX into the
runnable graph format. The C++ simulator and normal graph examples need none of
these optional packages. Install the pinned importer and independent validation
backend in a virtual environment:

```sh
python3 -m venv .venv-onnx
.venv-onnx/bin/python -m pip install -r requirements-onnx.txt
.venv-onnx/bin/python tests/onnx_import_tests.py --binary build/npu-sim
```

The requirements pin `onnx==1.19.1` and `onnxruntime==1.23.2`. NumPy is their
transitive dependency. ONNX Runtime is used for independent tests; importing or
executing an already converted graph does not use it. The importer never executes
custom model code.

## A complete example

Create a one-column matrix multiply with positive and negative half-integer
results. The model specifies signed INT8 input/output, a scale ratio of one half,
and nearest-even quantization. Its expected output is `[0, 2, 0, -2]`.

```sh
.venv-onnx/bin/python - <<'PY'
import json
import numpy as np
import onnx
from onnx import helper, numpy_helper

values = {
    "xs": np.array(1, np.float32), "xz": np.array(0, np.int8),
    "w": np.array([[1]], np.int8), "ws": np.array(0.5, np.float32),
    "wz": np.array(0, np.int8), "ys": np.array(1, np.float32),
    "yz": np.array(0, np.int8),
}
node = helper.make_node("QLinearMatMul",
    ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"], ["y"])
graph = helper.make_graph([node], "nearest_even_fixture",
    [helper.make_tensor_value_info("x", onnx.TensorProto.INT8, [4, 1])],
    [helper.make_tensor_value_info("y", onnx.TensorProto.INT8, [4, 1])],
    [numpy_helper.from_array(value, name) for name, value in values.items()])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                          ir_version=10)
onnx.save_model(model, "tiny-quantized.onnx")
with open("tiny-inputs.json", "w", encoding="utf-8") as output:
    json.dump({"x": {"dtype": "int8", "shape": [4, 1],
                     "data": [1, 3, -1, -3]}}, output)
PY

.venv-onnx/bin/python scripts/onnx_import.py tiny-quantized.onnx \
  --inputs tiny-inputs.json --output tiny-graph.json \
  --run --binary build/npu-sim

# Once converted, the runtime uses only the standard library:
python3 scripts/graph_runtime.py tiny-graph.json --binary build/npu-sim
```

`--output` creates a new graph file and refuses to overwrite an existing one.
Without `--run`, conversion prints a short JSON conversion report. With `--run`,
it also executes the converted graph and prints the ordinary graph timing,
traffic, reference-check and output report. The model's weights, bias and
quantization are initializers; the separate input JSON supplies real activation
values in the **original ONNX layout**. Input objects require `dtype`, `shape`
and flat `data`; optional `scale` and `zero_point` must agree with the model.

## Supported contract

| ONNX operator | Accepted subset | Lowering |
| --- | --- | --- |
| `QLinearMatMul` | Rank-two signed INT8 matrices, static dimensions, initializer weights, scalar FLOAT32 scales and scalar INT8 zero points; no broadcasting | Graph `MatMul`, one actual SystemC invocation |
| `QLinearConv` | Two spatial dimensions, static NCHW activations, signed INT8 OIHW initializer weights, scalar input/output quantization, scalar or per-output-channel weight quantization, optional INT32 initializer bias | Explicit NCHW-to-NHWC and OIHW-to-HWIO conversion, then graph convolution |
| Depthwise `QLinearConv` | `group` equals input channels, one input channel per filter, output channels an integer multiple of input channels | HWIM weights and graph `DepthwiseConv2D`; channel multipliers retained |

The importer accepts standard-domain opsets 10 through 21. Input/output scales
and zero points must be rank-zero scalar initializers. Conv weight scale and
zero-point shapes must match: both scalars or both one-dimensional with one
entry per output channel. All scales must be finite and positive. Bias is INT32
with the input-scale-times-weight-scale contract. The original ONNX operator
schemas define these quantization and layout relationships. [QLinearMatMul
schema](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html), [QLinearConv
schema](https://onnx.ai/onnx/operators/onnx__QLinearConv.html).

Conv strides, dilation, explicit pads, `VALID`, `SAME_UPPER`, and `SAME_LOWER` are
preserved. ONNX's `[top,left,bottom,right]` explicit padding is reordered into the
graph runtime's `[top,bottom,left,right]` convention. Optional `kernel_shape`
must match the weights. Quantized padding uses the input zero point.

Quantized arithmetic uses nearest-even ties through the simulator's explicit
`--rounding even` contract. **Scale conversion is exact rather than approximate.**
The importer interprets the actual FLOAT32 initializer values as exact rational
numbers, forms the input-scale-times-weight-scale divided by output-scale ratio,
and requires an exact nonnegative INT32 multiplier and a power-of-two denominator
with shift at most 31. Ratios such as one third, larger multipliers, or a required
shift of 32 fail with an explanation. This deliberately rejects many otherwise
valid quantized models; silently approximating their scale would change the
specified arithmetic.

## Layout, scheduling and timing

The converted document is an ordinary graph-runtime workload. Its activation
tensors are NHWC for convolution; MatMul tensors retain their rank-two layout.
Quantization continuity is checked when an activation is shared between nodes.
Weights and bias are materialized as typed graph parameters. Consecutive
convolution nodes retain NHWC internally.

`io_layouts` records each external convolution output's NHWC-to-NCHW conversion.
The result's `outputs` are runtime-layout tensors; **`external_outputs` restores
original ONNX output names, shapes and NCHW ordering**. MatMul outputs have the
same ordering in both sections. Input/output transposition, import work, im2col,
Python references and process launches are host work excluded from accelerator
cycles. Summed NPU cycles represent sequential layer invocations, without
cross-layer SRAM residency or compiler-driven inter-layer overlap.

The ordinary graph element, dimension, MAC and lowering limits still apply;
unsupported large invocations fail explicitly. This front end does not imply
that general ONNX operators can run on this accelerator or that im2col traffic
is a direct-convolution line-buffer model.

## Rejections and validation

Unsupported operators, unsigned quantization, floating-point execution,
dynamic dimensions, vector MatMul quantization, general grouped convolution,
custom domains/functions, sparse initializers, duplicate/invalid attributes,
non-initializer parameters, missing inputs and contradictory quantization are
rejected. A declared but unused external input is rejected in this subset.
ONNX's checker and static shape inference run before graph conversion.

Models are limited to 32 MiB and 256 nodes. Initializer shapes are bounded before
array decoding. External tensor data is forbidden: model loading explicitly uses
`load_external_data=False`, and the importer rejects external-data references
before decoding any initializer. It does not follow paths embedded in an ONNX
model.

The optional suite compares **full output arrays and shapes** against ONNX
Runtime's CPU backend for signed nearest-even MatMul, nonzero zero points,
saturation, convolution padding/dilation, per-channel scales and bias, depthwise
channel multipliers, and multiple convolution layers. Rejection tests cover
unsupported shapes/types/operators, scale representation, malformed inputs,
external-data refusal, and pre-decode size limits. A CLI test converts and runs
a model through the actual binary. The simulator additionally checks full
accumulators and outputs against its independent scalar references. These are
reproducible subset checks, not certification of arbitrary ONNX models or every
backend's numerical implementation.
