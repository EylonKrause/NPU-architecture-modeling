# Production NPU research and modelling priorities

Research checked on 3 November 2026. The sources below are maintained by the architecture, framework, or tool owners. They motivate modelling requirements; they do not establish that different NPUs share an instruction set, datapath, numerical contract, or performance model.

This repository is an executable architectural simulator. Its numerical outputs are testable independently of its timing assumptions. Production deployment additionally needs a selected hardware target, compiler/runtime integration, and calibration against that implementation. Passing simulator tests alone does not establish silicon timing or energy accuracy.

## What representative systems expose

| Primary source | Relevant production capability | Consequence for this simulator |
| --- | --- | --- |
| [Arm Vela documentation](https://arm-software.github.io/CMSIS-Ethos-U/main/vela/index.html) | Target-specific operator constraints, graph transformations, memory scheduling, striping and cascading; performance figures are estimates. | Make supported operators and memory policy explicit. Separate arithmetic validation from target timing calibration. |
| [NVDLA hardware architecture](https://nvdla.org/hw/v1/hwarch.html) | Separate convolution, activation and pooling engines; convolution-buffer reuse, weight compression, configurable memory interfaces. | Model more than multiply throughput: operator coverage, data movement, buffering and engine limits determine useful performance. |
| [Gemmini architecture and components](https://github.com/ucb-bar/gemmini#architecture) | Output/weight-stationary choices, banked scratchpad and accumulator, DMA and independent load/store/execute controllers with dependency management. | Define dataflow, partial-sum residency, transfer costs and hazards rather than assigning a constant cycle cost to a whole neural layer. |
| [ONNX QLinearConv](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) | Per-output-channel weight quantization, INT32 bias and explicit convolution geometry. | Carry channel-specific parameters through arithmetic and test grouped/depthwise, strided, dilated and padded cases. |
| [MLIR TOSA dialect](https://mlir.llvm.org/docs/Dialects/TOSA/) | Tensor operations have explicit types, shapes and numerical semantics, including convolution, depthwise convolution, pooling and rescaling. | Publish a bounded operator contract and reject unsupported graphs or invalid shapes before execution. |
| [Arm Ethos-U85](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85) and [ONNX Attention](https://onnx.ai/onnx/operators/onnx__Attention.html) | Contemporary NPU workloads include transformers; attention has explicit scaling, masking, softmax and optional key/value-cache semantics. | Dense matrix multiplication alone does not provide a complete transformer model. Treat these numerical and stateful operations as separate supported features. |
| [Timeloop/Accelergy energy and area tables](https://timeloop.csail.mit.edu/v4/parsing_and_intermediate_files/energy-and-area-reference-tables) | Energy estimation uses component/action-specific reference costs. | Count actions and accept declared coefficients; label resulting energy as an estimate with provenance rather than measured hardware power. |

## Implementation priorities

The [feature matrix](FEATURE-MATRIX.md) records actual support and remaining limits. These priorities describe how features should be implemented and verified, including where a simple extension would otherwise be misleading.

### Numerical and operator coverage

Per-channel quantization must be indexed by the output channel, including across partial column tiles. Bias enters once, before reduction; requantization and saturation happen once, after the complete reduction. A reduction tile must never round or saturate a partial sum. Nonzero input zero points mean a padded activation must represent real zero, rather than blindly storing integer zero.

Convolution lowering is a useful functional frontend: map output positions to rows, output channels to columns and kernel/input-channel products to the reduction dimension. Grouped convolution requires separate channel groups, and depthwise is a special grouping case. Materialized lowering increases host storage and can increase modelled traffic; it is not equivalent to a line-buffered direct-convolution datapath. Report this distinction explicitly.

Residual addition and pooling extend workload realism without pretending that every operator belongs on the same MAC array. Their rounding, quantization-domain rules and memory traffic need independent tests. A graph frontend should reject unsupported operators instead of silently computing them without timing or charging them to the accelerator.

Transformer support would additionally need batched matrix multiplication, normalization, nonlinear operations, masks, sequence-dependent shapes and cache lifetime/traffic. Precision and approximation error need a defined contract. A matrix-multiplication experiment may study one transformer component, but it should not be advertised as complete transformer inference.

### Memory, mapping and timing

Reduction blocking is necessary when full-reduction operand tiles exceed SRAM. Preserve accumulator state across blocks, expose extra transfer/issue costs and test incomplete last blocks. Distinguish operand buffers from accumulator storage. A capacity bound is useful but does not simulate SRAM banks or physical ports.

Systolic-array dimensions can expose underutilization from small or ragged tiles and explicit fill/drain costs. Such timing remains a stated analytical contract until checked against a particular array. Output-stationary and weight-stationary dataflows require different residency and traffic rules; naming the modes alone is insufficient.

DMA burst limits must charge transaction overhead for every burst, with request arbitration and bandwidth treatment stated. Supporting multiple outstanding requests, caches, DRAM banks or a network-on-chip requires additional state and contention modelling. One shared serial DMA server is a legitimate bounded architecture, but it must remain identified as such.

### Workloads, tooling and validation

Useful architecture experiments hold the tensors and numerical contract fixed while varying tile dimensions, compute width, SRAM, buffering and bandwidth. Report output agreement, latency, traffic, resource occupancy and individual cases before aggregate statistics. Keep simulator host runtime separate from simulated accelerator time.

A multi-layer driver can demonstrate end-to-end tensor flow. Unless intermediate residency and scheduling are implemented, layer boundaries must retain their full write/read costs and no cross-layer overlap or fusion should be implied. Standard-format import additionally requires versioned shape, layout and quantization handling; an import filename does not establish framework conformance.

For future hardware calibration, retain the exact model revision and configuration, numerical fixtures, compiler settings, trace definitions and observed hardware measurements. Calibrate transfer startup, issue costs and pipeline behaviour separately before fitting aggregate latency.

### Sparsity and energy

Zero-valued weights do not automatically imply faster execution. Sparse timing needs a specified format, metadata bytes, decoding cost, packing efficiency and execution/imbalance policy. A dense baseline remains essential. Compression-only savings and skipped-MAC savings should be separately selectable and reported.

An energy estimate can use declared action counts and coefficients:

$$
E_{\mathrm{estimate}}=N_{\mathrm{MAC}}e_{\mathrm{MAC}}+B_{\mathrm{read}}e_{\mathrm{read/byte}}+B_{\mathrm{write}}e_{\mathrm{write/byte}}+tP_{\mathrm{leak}}.
$$

This limited model omits uncounted SRAM/clock/interconnect actions. Default coefficients, when provided, are illustrative assumptions. Neither a lower estimated energy nor a higher simulated MAC rate establishes measured power, thermal behaviour or production suitability.

## Explicit scope boundary

No finite generic simulator contains everything required by every NPU. This project prioritizes inspectable integer inference, explicit resource scheduling and reproducible experiments. Commercial ISA emulation, RTL verification, physical design, coherence/security, device drivers, complete compiler import and target-specific performance or energy accuracy remain separate deliverables unless individually implemented and validated.
