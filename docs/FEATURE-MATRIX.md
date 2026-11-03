# Supported features and model boundaries

The table distinguishes runnable support from future scope. A research source identifies a useful capability, not an implementation claim. See [production NPU research](PRODUCTION-NPU-RESEARCH.md) for primary-source links and design rationale.

Status labels: **implemented** means executable code exists; **bounded** means useful support has an explicit narrower contract; **not implemented** means the simulator must not be described as supporting it. Functional support does not imply independently calibrated hardware timing.

| Capability | Status | Contract or remaining work |
| --- | --- | --- |
| Dense signed INT8 inference | Implemented | INT8 activations, INT8 or INT4 weights, INT32 bias, checked final accumulation, integer requantization, saturation and optional ReLU. Selectable nearest ties-away or ties-even rounding. |
| Independent arithmetic oracle | Implemented | Scalar C++ and independent Python checks compare complete accumulators and outputs. |
| Deterministic workload generation | Implemented | Specified fixture sequence; user tensor input also supported. |
| Output-row/column tiling | Implemented | Partial edge tiles included. |
| Reduction/K blocking | Implemented | Reduction panels retain an output tile's partial-sum allocation, load bias/quantization metadata once, and requantize only the final result. Panel transfer and compute timing are explicit; the numerical kernel evaluates the full dot product at the final panel. |
| Per-output-channel quantization | Implemented | Optional output-channel weight zero points, multipliers and shifts; scalar fallback. See [quantization](QUANTIZATION.md). |
| Packed INT4 weight traffic | Bounded | Signed INT4 numerical range; host values remain sign-extended INT8. Traffic packs two weights per byte, with each reduction panel padded independently. No sparse encoding or decompressor timing. |
| Conv2D and depthwise convolution | Bounded | [Graph frontend](GRAPH-RUNTIME.md) lowers NHWC tensors to dense simulator invocations; stride, dilation, padding and depthwise channel multipliers. Materialized lowering is not a direct-convolution line-buffer model. General grouped Conv2D is unsupported. |
| Pooling and residual elementwise operations | Bounded | Add, ReLU, MaxPool2D, AveragePool2D and Flatten execute as explicit host fallbacks. Their CPU time is excluded from NPU architecture cycles. |
| Network execution and tensor lifetimes | Bounded | Validated custom graph, topological execution, use-count release and logical live tensor bytes. CNN and residual-MLP fixtures with known outputs. Sequential invocations; no cross-layer overlap or retained SRAM. |
| ONNX quantized-model import | Bounded | Optional [strict importer](../scripts/onnx_import.py): signed INT8, static shapes, standard-domain opsets ten through twenty-one, rank-two QLinearMatMul and two-dimensional QLinearConv, including depthwise. Requires exactly representable integer rescaling; uses ties-even and explicit NCHW/NHWC conversion. Unsupported operators, layouts, general grouped convolution, external data and dynamic shapes are rejected. Compared with actual ONNX Runtime in the optional test suite. |
| Full compiler / TFLite or TOSA import | Not implemented | The strict ONNX subset and custom graph runtime do not provide a general inference compiler or arbitrary graph coverage. |
| Complete attention / transformer inference | Not implemented | No softmax, normalization, masking, dynamic sequence or key/value-cache model. |
| Floating-point and mixed-precision compute | Not implemented | Integer numerical contract only; no FP16/BF16/FP8 datapath. |
| Shared DMA with read/write arbitration | Implemented | One non-preemptive server, request-boundary round-robin and explicit latency/bandwidth. |
| DMA burst limits | Implemented | A configurable burst limit charges startup/service cost per burst; arbitration remains at whole-request boundaries. |
| Multiple outstanding memory transactions | Not implemented | The shared DMA server serializes service. |
| Bounded tile buffering and backpressure | Implemented | Slots are acquired before load and released after store. |
| SRAM capacity bound | Implemented | Conservative per-slot reservation; invalid configurations rejected. |
| Weight-cache reuse | Bounded | Optional capacity-limited LRU of weight panels across output-row tiles, with hit/miss/saved-byte statistics. No cache lookup latency, coherence or physical banking. |
| SRAM banks, ports, DRAM or network-on-chip | Not implemented | A capacity bound does not model bank conflicts or an external memory controller. |
| Systolic-array geometry | Bounded | Optional array dimensions charge per-wave fill/drain and ragged-tile inefficiency. This is an analytical timing contract, without PE/register-level simulation or hardware calibration. |
| Alternative stationary dataflows | Not implemented | No interchangeable full output-stationary/weight-stationary mapping compiler or interconnect model. |
| Event and cycle-step SystemC modes | Implemented | Same stage timing contract and outputs; cycle-step mode is not validated RTL cycle accuracy. |
| Traces and utilization | Implemented | Stage spans, architecture cycles, traffic, occupancy and useful MAC/DMA payload utilization. |
| Architecture sweeps | Implemented | Reproducible resource comparisons; paired arithmetic validation. |
| Mapping search | Bounded | [Autotuning](../scripts/autotune.py) exhaustively evaluates at most two hundred fifty-six declared tile/slot candidates, rejects SRAM-infeasible mappings and preserves output hashes. Latency/SRAM objectives and a nondominated frontier apply only to the enumerated set and the simulator contract. |
| Sparse compression and sparse execution | Not implemented | No claimed speedup from zeros alone. |
| Coefficient-based energy and average power | Bounded | [Energy analysis](ENERGY.md) requires user-supplied MAC/external-byte costs and accepts optional constant static power. Reports components, units and assumptions; excludes SRAM/clock/interconnect/host effects and has no silicon or thermal calibration. |
| Hardware timing calibration | Not implemented | No RTL or silicon reference validation. |
| Commercial command-stream / ISA compatibility | Not implemented | Generic architecture, without a proprietary ISA. |

The supported implementation is deliberately smaller than production inference stacks. Changes to this table must follow the executable feature and its tests; do not promote a planned feature solely because it appears in the research document.
