# Quantized graph runtime

The standard-library Python front end executes real graph tensors through the
C++/SystemC simulator. It is a deliberately bounded inference and mapping tool,
not an ONNX/TFLite importer or a commercial NPU software stack.

```sh
python3 scripts/graph_runtime.py examples/cnn.json --binary build/npu-sim
python3 scripts/graph_runtime.py examples/residual_mlp.json --binary build/npu-sim
python3 tests/graph_runtime_tests.py --binary build/npu-sim
```

The small CNN has independently calculated logits `[21, -17]`; the residual MLP
has output `[8, 1, 5]`. Every accelerated node also checks **all accumulators and
all output values** against scalar Python arithmetic. Convolution's reference
uses direct nested spatial/channel loops, independently of the im2col lowering
used to invoke the simulator. A C++ `numerical_match` flag alone is insufficient.
The host operators have explicit known-answer tests, including nonzero zero
points, padding, saturation and rounding.

## Graph schema

A graph has `schema_version: 1`, an initial `tensors` map, a `nodes` list and
an `outputs` list. `name`, `inputs`, `architecture` and `expected_outputs` are
optional. `inputs` declares external input names; values are supplied in the
initial tensor map, so the JSON is a self-contained, reproducible workload.
`expected_outputs` contains independently established flat golden arrays keyed
by graph output name. Golden values are checked when supplied.

Each tensor explicitly declares all five fields:

```json
{
  "dtype": "int8",
  "shape": [1, 3],
  "data": [2, -1, 3],
  "scale": 1.0,
  "zero_point": 0
}
```

Data is flat row-major storage. Activations and weights are signed INT8; bias is
INT32 with zero point zero. Tensor scales must be finite and positive. Booleans
are not accepted as integer data. Bias scale must match input scale times weight
scale. Floating-point tensor execution is unsupported.

Nodes name one output and list ordered input tensor names:

```json
{
  "name": "projection",
  "op": "Dense",
  "inputs": ["input", "weights", "bias"],
  "output": "projected",
  "quant": {
    "multiplier": 1,
    "shift": 0,
    "output_scale": 1.0,
    "output_zero": 0,
    "relu": true
  }
}
```

Requantization is integer multiplication, right scaling with nearest rounding,
output-zero addition and INT8 saturation. `quant.rounding` is `away` (default,
ties away from zero) or `even` (ties to an even integer). Fused ReLU
clamps to the quantized output zero. The declared scale ratio must agree with
the integer multiplier to within half a multiplier quantization step; a loose
or contradictory scale annotation is rejected. The integer contract is explicit
and does not claim exact compatibility with a particular framework's rounding.

For per-output-channel quantization, `quant` additionally accepts the arrays
`weight_zero_points`, `multipliers`, `shifts`, `weight_scales` and `bias_scales`.
Each supplied array must contain exactly one entry per output channel. Missing
arrays expand the scalar tensor or quantization defaults. `weight_scales` and
`bias_scales` explicitly override scalar weight/bias scale metadata by output
channel and are checked against each channel's integer rescaling contract.
The runtime emits the documented `NPUQ1` interchange file and passes it to the
simulator; values are not merely annotations. Depthwise channels use the order
input channel first, channel multiplier second.

## Operators and placement

| Operator | Tensor/layout contract | Execution |
| --- | --- | --- |
| `Dense`, `MatMul` | Activations `[M,K]`, weights `[K,N]`, optional bias `[N]`; both names have the same rank-two contract | One real simulator invocation |
| `Conv2D` | Activations NHWC, weights `[KH,KW,IC,OC]`, optional bias `[OC]` | Host im2col then one dense simulator invocation |
| `DepthwiseConv2D` | Activations NHWC, weights `[KH,KW,IC,multiplier]`, optional bias `[IC*multiplier]` | One lowered dense invocation per input channel, output channels reassembled |
| `Add` | Two identical shapes, scales and zero points; centered addition and saturation | Explicit CPU fallback |
| `ReLU` | Quantized clamp at input zero point; preserves quantization | Explicit CPU fallback; accelerated alternative is declared fused `quant.relu` |
| `MaxPool2D` | NHWC, spatial kernel, preserves quantization; padded cells do not win maximum | Explicit CPU fallback |
| `AveragePool2D` | NHWC, centered integer mean with ties away from zero; preserves quantization | Explicit CPU fallback |
| `Flatten` | Rank at least two; preserves first/batch axis and flattens the remaining axes | Explicit CPU fallback/copy |

Convolutions and pooling accept two-element `strides` and `dilations`, both
defaulting to `[1,1]`. Padding is `VALID` (default), `SAME` (ceil output shape;
extra padding on bottom/right), or `[top,bottom,left,right]`. Convolution padding
uses the activation's **quantized zero point**, not the byte value zero.
Pooling `kernel` defaults to `[2,2]`; pooling stride still defaults to `[1,1]`.
Average pooling excludes padded cells from the denominator unless
`count_include_pad: true` is declared, in which case padding contributes the
quantized zero. Completely empty pooling windows are rejected.

There is no implicit layout conversion, broadcasting, residual rescaling,
grouped convolution other than depthwise, softmax, attention, division by a
silently chosen multiplier, or unsupported-op approximation. Unknown operators,
attributes, inconsistent scales, dimension mismatch and cyclic graphs fail
explicitly. Operators must receive the documented number and dtype of inputs.

## Scheduling, buffers and reports

The runtime validates graph dependencies and topologically orders nodes, so the
JSON node order need not already be executable. It retains a residual/skip tensor
until every consumer has finished, releases each tensor after its last use, and
retains declared graph outputs. Duplicate input references are counted
individually. Unknown inputs and multiple producers are rejected.

Reports contain each node's placement, output shape, NPU cycles, MACs, DMA reads
and writes, simulator invocation details, released tensors and remaining live
logical tensor bytes. Totals include summed NPU cycles and traffic, peak live
tensor bytes, peak live activation bytes, final live bytes, and a separate peak
lowering-workspace estimate. Activation accounting includes produced tensors and
external input/activation roles, excluding tensors used only as weights or bias.
All tensors, including weights and INT32 bias, participate in liveness. Peak live
tensor storage accounts for both a node's inputs and newly allocated output
before freeing inputs. This is a logical tensor storage model, **not Python RSS**;
the input JSON object, scalar reference arrays, serialization and subprocess
memory are excluded. Depthwise invocations are sequential. Lowering storage
estimates count im2col INT8 values and materialized sliced weights/bias; they do
not claim to include reference workspace, allocator overhead or complete runtime
peak memory.

CPU fallback nodes report zero NPU cycles, zero NPU MACs and zero NPU DMA bytes,
with separate logical host read/write bytes. Their Python execution is real,
but its wall time is not a target CPU latency model. The whole-run host elapsed
time includes Python orchestration and subprocess launches. It is distinct from
the summed target accelerator cycles.

Each accelerated invocation is a fresh SystemC process. All layer outputs are
materialized off chip and reloaded by the next invocation. There is **no
cross-layer overlap, SRAM residency, compiler liveness-aware SRAM allocation,
command-queue processor, direct-convolution line buffer or on-chip operator
fusion** beyond an explicitly declared dense/convolution ReLU. The dense engine's
traffic model applies to the lowered matrices, so im2col can amplify read
traffic; its host transformation cost and original-image reads are excluded from
NPU timing. Depthwise lowering is a functional mapping, not a claim that a dense
array is the most efficient production depthwise engine.

`architecture` forwards supported simulator fields, including `tile_m`, `tile_n`,
`tile_k`, `macs_per_cycle`, `array_rows`, `array_cols`, `memory_bytes_per_cycle`,
`memory_latency`, `pipeline_cycles`, `dma_burst_bytes`, `weight_cache_bytes`,
`buffer_slots`, `sram_bytes`, `frequency_mhz`, `timing`, and `weight_bits`.
It also forwards `weight_encoding`, `palette_bits`, `sparse_block` and
`weight_decode_values_per_cycle`; see [actual weight compression](WEIGHT-COMPRESSION.md).
An explicit array geometry and explicit `macs_per_cycle` cannot be combined;
the simulator validates architecture values and capacity. INT4 weight execution
is selected by `weight_bits: 4`; signed values must fit the simulator's contract,
and activations remain INT8. Graph-host tensor storage still counts each source
weight value as one byte; packed off-chip weight traffic belongs to the simulator
report. `--timing event|cycle` overrides graph timing.

Limits are explicit: one through 256 nodes, at most six tensor dimensions,
each dimension one through 4096, at most one million values per tensor, a 32 MiB
JSON input limit, and at most 100 million MACs per graph compute node. A lowered
convolution must also fit dense dimensions and the one-million-value lowering
limit; an unsupported large workload fails rather than silently splitting into
a different timing contract. These limits keep the educational simulator and
independent scalar checks bounded; they are not production deployment limits.

Optional `io_layouts` maps declared outputs to `{"internal":"NHWC","external":"NCHW"}`.
The usual `outputs` remain runtime-layout tensors; `external_outputs` restores
original ONNX output layout with a host transpose. Conversion costs are excluded
from NPU cycles. Other layout pairs are rejected. See [ONNX import](ONNX.md) for
the strictly supported optional interchange subset.

## Compile safe activation fusion

```sh
python3 scripts/graph_optimize.py examples/mobile_fusion.json \
  --verify --binary build/npu-sim --output results/fused.json
```

The optional rewrite pass fuses a Dense, MatMul, Conv2D or DepthwiseConv2D
producer followed by a single-consumer ReLU into the engine's existing
requantization clamp. It preserves quantization, output names and complete
numerical results, including saturation, nonzero zero points and both rounding
policies. Shared or declared pre-activation outputs prevent fusion. Repeated
ReLU chains are handled to a fixed point. The original document is preserved;
the output path must be new. Create its parent directory before running.

`--verify` executes both graphs and compares every declared output and layout.
The report retains original/compiled NPU cycles, traffic, host logical access
counts and liveness. Eliminating the host ReLU removes its host reads/writes;
the model does not assign a fabricated NPU speedup to that work. This exact
activation fusion does not supply general retained-SRAM cross-layer execution.
