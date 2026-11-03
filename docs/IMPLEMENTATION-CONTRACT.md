# Implementation contract

This repository models a configurable generic NPU in C++17 and SystemC 3.0.2. Its numerical outputs, traffic rules and scheduling semantics are executable and independently checked. Its timing is a documented architectural contract, without supplied RTL or silicon calibration. [Architecture](ARCHITECTURE.md) derives the equations; the [feature matrix](FEATURE-MATRIX.md) identifies narrower features and omissions.

## Numerical behavior

The engine consumes row-major activations and weights, subtracts integer zero points, adds INT32 bias once, and accumulates the complete reduction in INT64. Final accumulators outside INT32 are rejected; intermediate cancellation is permitted. Signed INT8 activations combine with either signed INT8 or signed INT4 weights. INT4 host values remain sign-extended INT8, while simulated weight traffic is packed.

Requantization uses a nonnegative multiplier at most `INT32_MAX` and a shift from zero through 31. It rounds to nearest, with configurable ties away from zero (`away`, default) or ties to even (`even`), adds the output zero point, applies optional ReLU at that zero point, and saturates to INT8. Per-output-channel weight zero points, multipliers and shifts are independently optional vectors; every supplied vector has exactly the output-column count. The global column indexes these values across tiles. The `NPUQ1` CLI file supplies all three vectors. See [quantization](QUANTIZATION.md).

The scalar reference computes independent complete dot products. The tiled implementation uses reduction-major outer products. The executable compares all accumulators and quantized outputs before reporting success. The output hash serializes little-endian INT32 accumulators followed by INT8 output bytes using documented FNV-1a constants; it does not substitute for full comparison.

Without an input file, deterministic fixture state begins with the seed masked to 32 bits:

$$s_{i+1}=(1664525s_i+1013904223)\bmod 2^{32}.$$

Generate all activations, all weights, then all biases. Activations and INT8 weights use $(s\bmod31)-15$, INT4 weights use $(s\bmod16)-8$, and biases use $(s\bmod127)-63$. The `NPU1` tensor file instead supplies shapes, activations, weights and bias explicitly; malformed storage and out-of-range values are rejected.

## Mapping and storage

Output tiles are issued in row-major tile order and use actual edge extents. `tile_k` splits reduction into panels; zero means full reduction. A panel reads its activation bytes and packed weights. Bias and all supplied channel-parameter vectors are read only on the first panel. Each INT4 panel independently pads an odd final nibble. The final output is written once as INT8.

The SRAM bound reserves per slot: a maximum-sized activation panel, packed weight panel, bias/channel metadata, INT32 accumulator tile and INT8 output tile. Partial-sum allocation persists across panels. The functional kernel evaluates the full reduction once at the final panel rather than simulating intermediate register writes. It never requantizes a partial reduction.

Optional `weight_cache_bytes` adds a separately reserved fully associative LRU weight-panel cache. Keys identify output-column tile and reduction block. Hits remove only packed weight bytes from DMA. Misses insert when the panel fits, evicting least-recently-used lines; oversized panels bypass insertion. A disabled cache reports zero cache accesses. Lookup and replacement add no modeled cycle cost. Reservation is the configured capacity, while `peak_cached_weight_bytes` reports actual peak occupancy.

No implicit spilling, physical SRAM banks, coherence, compression, or sparse-MAC skipping is implemented. Accelerator SRAM is distinct from host fixture/reference/trace storage and graph logical tensor storage.

## Compute and scheduling

Vector compute duration is the ceiling of panel MACs over configured throughput, plus panel pipeline overhead. Alternatively, explicit array rows and columns select analytical output-stationary timing: fold the output tile over array dimensions, charge reduction length plus full-array fill/drain per wave, and add panel pipeline overhead. Both array dimensions must be supplied; explicit array geometry and an explicit aggregate MAC budget cannot be combined at the CLI.

One SystemC tick represents one architecture cycle. Load, compute and store processes communicate through bounded FIFO state. Acquire a slot before a tile's first load; release it only after output store completion. Every nonfinal panel waits for compute acknowledgment before its operand buffer is reused. A final panel allows the next output tile's load to overlap compute when another slot exists.

A single non-preemptive DMA server serializes read and write requests. Each burst pays startup latency plus ceiling byte count over bandwidth. Zero burst limit means one unlimited burst. A multi-burst request remains atomic; arbitration cannot interleave its bursts. Read/write service alternates round-robin when both classes are ready, initially preferring reads; a sole ready class proceeds. Simultaneous pending requests are coalesced before selection, and completion notifications wake the requester in a later SystemC delta cycle.

Event mode jumps by whole stage durations. Cycle mode repeats single-cycle waits inside the same scheduling processes. They must produce identical architecture cycles, traffic, traces and outputs. Cycle mode rejects a conservative serial work bound over ten million cycles; both modes enforce exact cycle-representation limits. Neither mode is a PE-level or independently established silicon cycle-accurate model.

## Graphs and analysis tools

The Python graph frontend validates dependencies, shapes, integer domains, declared scales and operator attributes, then topologically executes the graph. Dense/MatMul and materialized Conv2D/depthwise lowering invoke the accelerator. Add, ReLU, MaxPool2D, AveragePool2D and Flatten are explicit host fallbacks. Unsupported operators and incompatible quantization fail; there is no silent approximation. Graph outputs may be checked against independent golden values.

Every accelerator invocation starts fresh. There is no cross-layer overlap, retained cache/SRAM, direct-convolution line buffer, or general compiler/fusion claim. Host fallback work, references, lowering and process launches are excluded from NPU cycle totals. Graph liveness and lowering workspace are logical byte counts, not process RSS. [Graph documentation](GRAPH-RUNTIME.md) specifies layouts and limits. The optional [ONNX importer](ONNX.md) supports a strict signed-INT8 subset: rank-two QLinearMatMul and static QLinearConv with ordinary/depthwise grouping, exact representable scale ratios and nearest-even rounding. It explicitly converts NCHW convolution layouts to the frontend layout. Unsupported models are rejected; import is not a general ONNX runtime.

The mapping search enumerates bounded tile/slot choices, records SRAM-infeasible candidates and verifies numerical identity. Its best result is restricted to that candidate set and objective. The energy tool requires explicit MAC and external-byte coefficients, with optional constant static power; it reports assumptions and omitted effects rather than default silicon accuracy.

## Reporting and bounds

JSON stdout contains configuration, elapsed cycles, MACs, actual external bytes, resource service cycles/utilization, slot pressure, SRAM requirement, cache/DMA counters, arithmetic agreement, output hash, modeled target latency and simulator wall time. Full result arrays are optional. CSV trace fields are `stage,tile_id,start_cycle,end_cycle,bytes,block_id`; output stores use the final reduction block.

Dimensions, MAC count, host working-storage estimate, numeric ranges and serial cycle work are bounded before expensive allocation. Host storage has a conservative 256 MiB limit including reduction-panel trace budgets. Frequency converts cycles to reported time; it does not change scheduling or infer different physical memory behavior. Diagnostics go to stderr, and successful stdout remains one machine-readable JSON object.

## Verification and publication

Use `main`. Before publication, build the executable, run CTest and independent numerical/scheduling checks, compare timing modes, and verify new boundaries and malformed inputs. Preserve default-fixture behavior unless an explicit contract change is intended. Keep generated builds, local reports and credentials out of source control. Do not describe unimplemented, untested or uncalibrated behavior as production support.
