# Methodology

This document defines how the repository models an NPU, how the model is checked, what it
covers, which hardware measurements would calibrate its timing parameters, and how it relates
to other accelerator models. Section 3 defines the scope of the model.

<a id="modeling-approach"></a>

## 1 Modeling approach

### 1.1 Abstraction level

`npu-sim` is a transaction-level, cycle-approximate SystemC model of a tiled INT8 GEMM
accelerator. Each core is a SystemC module with three threads, load, compute and store
(`src/model.cpp`), connected by `sc_fifo` token channels that stand for tile slots and
reduction-panel buffers. A thread blocks when its channel is empty or full, so backpressure,
overlap and contention follow from the event schedule. The unit of work is the reduction
panel: an $M_t \times N_t$ output tile over $K_p$ reduction elements.

| Transaction | Issued by | Content | Timed by |
|---|---|---|---|
| Panel load | load thread | $M_t$ activation runs of $K_p$ bytes (with `shape.conv`, the runs of the tile's input footprint), the weight panel as one run, and on the first panel of a tile the bias and per-channel parameter run; a cache hit omits the activation or weight runs | memory model, `src/memory.cpp` |
| Panel compute | compute thread | $M_t N_t K_p$ MACs ($M_t C_t K_p$ on the depthwise datapath), and the decode of a compressed weight panel | closed form per panel, `src/core.cpp` |
| Tile store | store thread | $M_t$ output runs of $N_t$ bytes | memory model, `src/memory.cpp` |

A load or store is a `MemRequest`, a list of address runs over the fixed address map in
`include/npu/core.hpp` (activations at `0x8000_0000`, weights pre-packed per panel at
`0x9000_0000`, bias and parameters at `0xA000_0000`, outputs at `0xB000_0000`). The issuing
thread blocks in `MemoryPort::transport` (`include/npu/memory.hpp`) until the memory model
completes the request. The serial DMA times a request as one unit. The AXI model splits it
into INCR bursts and times each burst on its channel and on the shared DRAM timeline.

Inside a transaction the model is a closed form; across transactions it is an event
schedule. Arithmetic runs alongside the timing over the same tiles: the compute thread
evaluates a tile's integer result once, at its final reduction panel (`compute_tile`,
`src/core.cpp`), while the timing charges every panel and holds the tile's partial sums in
its slot until the store.

### 1.2 Time base

One cycle is one period of the NPU clock. Every rate in the configuration is per NPU cycle:
MACs (`arch.compute`), AXI beats (one per channel per cycle), DRAM bytes
(`arch.memory.dram_bytes_per_cycle`, which folds the DRAM clock into a byte rate) and
decoded weight values. `src/main.cpp` sets the SystemC time resolution to 1 ns and every
wait advances an integer number of resolution units, so the SystemC timestamp is the cycle
count. `arch.frequency_mhz` enters one conversion only,

$$t_{\mathrm{ns}} = \frac{1000 \cdot \mathrm{cycles}}{f_{\mathrm{MHz}}},$$

reported as `modeled_latency_ns`; `tests/integration.py` checks that a frequency change
leaves cycles unchanged. Delta cycles carry no architecture time. VCD output uses a 1 ps
time unit to order delta-cycle changes inside one cycle; the `cycles` signal carries
architecture time. All cycle and byte arithmetic is checked unsigned 64-bit integer
arithmetic (`checked_add`, `checked_mul`), and `validate()` rejects a configuration whose
serial cycle bound reaches $2^{53}$, so every reported count is exact in a JSON double.

### 1.3 Cost model

Each transaction is timed by a closed form of [architecture](architecture.md), evaluated once
per transaction; the event schedule of section 1.1 composes them.

| Mechanism | Closed form | Source |
|---|---|---|
| Panel compute: vector datapath, output- and weight-stationary array, depthwise datapath | Equations 3.1 to 3.5 | `mac_cycles`, `compute_panel_cycles`, `src/core.cpp` |
| Compressed weight panel | Equation 3.1 with the decoder cycles of architecture section 3.7 | `encoded_panel_compute_cycles`, `src/core.cpp` |
| Serial DMA request | Equation 5.2 | `dma_transfer_cycles`, `src/core.cpp` |
| AXI burst formation | Equation 5.3: INCR bursts split at the burst limit and at 4 KiB, partial first and last beats | `next_burst`, `src/core.cpp` |
| AXI issue, data phase and completion per channel | Equation 5.4 | `AxiMemory::issue`, `src/memory.cpp` |
| DRAM | Equation 5.6: one byte-granular reservation timeline shared by every channel and core | `DramTimeline`, `src/core.cpp` |

### 1.4 Event and cycle modes

`--timing event|cycle` (`sim.timing`, [architecture](architecture.md) section 7.4) selects
whether a thread advances through a duration of $d$ cycles in one `wait(d)` or in $d$ waits of
one cycle, with the AXI arbiter waking every cycle in cycle mode. Both modes run the same
processes and make the same scheduling decisions; only the number of kernel activations
differs. Cycle mode removes every multi-cycle wait, so agreement between the modes shows that
no decision depends on wait granularity. `tests/integration.py`, `tests/memory_model_tests.py`,
`tests/cluster_tests.py` and `tests/systemc_observability_tests.py` require identical results,
stage traces and VCD architecture values in both modes.

### 1.5 Determinism

Contention at one timestamp resolves by fixed rules in the model, independent of SystemC process
evaluation order: the serial DMA's two-delta request collection and read-write alternation and the
AXI arbiter's grant order ([architecture](architecture.md) sections 5.3, 5.4.4 and 7.5), the `auto`
partition's tie-breaks (section 6.2 there), and the mobile platform's round-robin bus grant ([SoC
integration](soc-integration.md) section 2.4). With the seeded tensor generator of architecture
section 7.5, a run is a pure function of its configuration, and every result echoes the complete
effective configuration (`config`, schema version 2), which `npu-sim --config` reloads to reproduce
the run (`tests/config_file_tests.py`).

<a id="validation"></a>

## 2 Validation

### 2.1 Evidence map

Sections 2.2 to 2.8 give the evidence for numerical and schedule correctness; accuracy against
hardware depends on the calibration of section 4.

| Claim | Evidence | Tests (CTest name) |
|---|---|---|
| Outputs are bit-exact | every functional run compares the tiled datapath with a scalar reference; independent Python oracles, including a direct convolution over the NHWC input; ONNX Runtime | `core.arithmetic`, `sim.gemm-oracle`, `sim.quantization`, `sim.weight-compression`, `sim.conv-traffic`, `onnx.ort-differential` |
| Requantization matches published definitions | TOSA 1.0.2, TFLite, gemmlowp and CMSIS-NN formulations, published golden vectors | `core.arithmetic`, `sim.quantization`, `tools.npuperf` |
| The schedule follows the stated rules | independent timestamp schedulers in Python predict cycles, wait counters and complete stage traces; footprint, run and cache oracles predict convolution traffic | `sim.gemm-oracle`, `sim.memory-hierarchy`, `sim.memory-model`, `sim.cluster`, `sim.conv-traffic` |
| Two implementations of the rules agree | `npuperf.analytic.schedule()` against `npu-sim`, cycle for cycle | `tools.analytic-model`, study S6 |
| Instrumentation is passive | traced, untraced, stepped and batch runs give identical results | `sim.observability`, `live.protocol` |
| The C++ runs without memory errors, leaks or undefined behavior on the test inputs | ASan, UBSan with the vptr check, LeakSanitizer | CI job `linux-x64-asan-ubsan` |
| Published tables are reproducible | regenerated quick studies equal the committed tables | CI job `results-drift` |

CTest labels group the suite by area ([usage](usage.md) section 1.5); `ctest -L sim` runs
one area.

### 2.2 Numerical oracles

Every functional `npu-sim` run checks itself before it reports. `reference()` in
`src/core.cpp` evaluates each output as an independent dot product in reduction order; the
pipeline's `compute_tile` accumulates K-major outer products per tile. `check_numerics`
(`src/main.cpp`) requires equal INT32 accumulators and INT8 outputs and exits with status 2
otherwise. The result carries `"numerics": "bit-exact"` and a 64-bit FNV-1a `output_hash`
over the accumulators and outputs. A `--timing-only` run generates no tensors and reports
`"numerics": "skipped"`.

The Python oracles share no code with the C++. They derive expected values with unbounded
integers from the published formulations; [numerics](numerics.md) defines the arithmetic
they check, and its section 6.2 lists each oracle, its source and its coverage.

[Numerics](numerics.md) section 2.4 lists the verified equivalences of `tosa-single` and
`tosa-double` with the TFLite and CMSIS-NN builds, and how `native` differs from both.

The weight codecs are checked the same way: `tests/mobile_compression_tests.py` derives
encoded sizes, decode cycles and skipped MACs in Python and compares complete outputs with
the integer oracle, and `tests/mobile_compression_core_tests.cpp` shows that outputs come
from the stored encoding (mutating it changes the result) and that corrupted headers are
rejected. The other executables have their own oracles: `tests/transformer_tests.py`
recomputes the decoder block in FP32, FP16 and BF16 with independent rounding,
`tests/mobile_platform_tests.py` and `tests/tlm_platform_tests.py` recompute every job's
integer output, and `npuperf.graph` checks each accelerator node against a scalar Python
reference before the next node runs.

### 2.3 ONNX Runtime differential test

`tests/onnx_import_tests.py` builds ONNX models in the test, imports them with
`npuperf.onnx_import`, runs them through `npu-sim` and compares the complete output tensor
with ONNX Runtime. The cases cover `QLinearMatMul` with ties to even, nonzero zero points and
saturation; `QLinearConv` in NCHW with padding (including `SAME_UPPER` and `SAME_LOWER`),
dilation, bias and per-channel scales; depthwise convolution with a channel multiplier; and a
two-layer graph that keeps NHWC internally and restores NCHW at its output. Further cases
check that unsupported operators, dynamic shapes, unsigned activations, external data,
non-depthwise grouping and scale ratios that $\text{multiplier}/2^{\text{shift}}$ cannot
represent are rejected. Versions are pinned in `requirements-onnx.txt` (ONNX 1.19.1, ONNX
Runtime 1.23.2); CI runs the suite as `onnx-runtime-differential`.

### 2.4 Timing oracles

The schedule oracles are timestamp simulations written from the architecture rules,
independent of SystemC threads and delta cycles.

| Oracle | Mechanism | Checked against `npu-sim` |
|---|---|---|
| `schedule()`, `tests/integration.py` | slots, panel buffers, the serial DMA's alternation | `cycles`, `peak_inflight_tiles`, `slot_wait_cycles`, `panel_wait_cycles` and the complete stage trace CSV |
| `tests/architecture_tests.py` | serial sum of panel transfer and compute durations with one slot and one panel buffer; multi-slot runs through the scheduler above | exact cycles, INT4 padding, parameter bytes, LRU cache hits, SRAM reservation at its boundary |
| `tests/memory_model_tests.py` | AXI bursts formed by walking beats, DRAM gaps found by fixpoint iteration over every reservation, the load-compute-store pipeline replayed | cycles and traces of random mappings; closed forms for single-burst latency, steady-state bandwidth, 4 KiB splits, misaligned beats, read-write overlap and DRAM caps |
| `tests/cluster_tests.py` | per-core pipelines on one DRAM timeline, m and n splits | cycles of random clusters, traffic difference between splits, the DRAM floor on the makespan |
| `tests/conv_traffic_tests.py` | input footprints enumerated tap by tap, NHWC runs, AXI beats walked slot by slot, LRU replay of both caches in the traced load order, a direct convolution | footprint bytes and runs with halo and padding, depthwise-datapath timing, activation-cache hits, timing-only traffic equal to bit-exact traffic, rejected geometries |

Closed forms check limiting cases. Study S3 compares the read bandwidth of a weight-streaming
GEMV with Little's law for $N_r$ outstanding bursts of $b$ beats of $D$ bytes at read latency
$L_r$, equation 5.5 of [architecture](architecture.md) in bytes per cycle,

$$\mathrm{BW} = \min\left(D, \frac{N_r b D}{L_r + b}\right),$$

and reports both columns ([results](results.md#s3-axi-outstanding-transactions)).

### 2.5 Analytic model cross-check

`npuperf/analytic.py` restates the timing rules in Python, written from the architecture
description and the interfaces in `include/npu`, without reference to the C++ scheduler. It
has two models.

| Model | Method | Requirement |
|---|---|---|
| `schedule()` | replays every request: address map, burst splitting, each channel's outstanding window, the DRAM byte timeline and its grant order, the serial alternation, slots, panel buffers, per-core LRU caches | equals `npu-sim` cycle for cycle |
| `analyze()` | closed form over tile and panel classes: exact traffic, request counts, compute busy cycles and cache hits; lower bounds; a three-stage pipeline estimate that times every request alone on the memory system | traffic, request counts, compute busy cycles and cache hits exact; lower bound at or below `npu-sim`; estimate exact without overlap (one slot, one panel buffer); median error below 1%; p95 error below 15% (serial), 25% (AXI) and 30% (clusters of up to 8 cores); bound class agreeing on more than 95% of configurations; the cluster plan `npu-sim` picks |

`tests/analytic_model_tests.py` enforces these requirements on seeded random serial, AXI and
cluster configurations. Study S6 ([results](results.md#s6-analytic-model-against-npu-sim))
measures the estimate on a second seeded set, seed 20261107, and on the S1 layers of each design
point, and reports its median, p95 and maximum error on every set. The estimate charges the
shared DRAM as an aggregate byte bound, so its error grows with the queueing of channels and
cores on one timeline; `schedule()`, which replays that queueing, stays exact. `npuperf.mapper`
uses both ([usage](usage.md) section 9.7); `tests/mapper_tests.py` checks on a GEMV and a
convolution that it keeps the fastest replay of its candidate set, and holds its pick for
ResNet-50 res3 3x3 on `mobile-2k`, caches off, within 10% of the fastest mapping of a 48-point
`npu-sim` search.

### 2.6 Serial-model regression baseline

`--memory-model serial`, the default, is the regression baseline. Its cycles, wait counters
and stage traces are predicted exactly by the timestamp scheduler of `tests/integration.py`
and the serial sums of `tests/architecture_tests.py`, which cover burst limits, INT4
weights, weight caches and multi-slot overlap. A change to the server's request collection,
read-write alternation or burst charging, or to the order in which the threads compete for
it, changes a predicted trace and fails `sim.gemm-oracle` or `sim.memory-hierarchy`. The AXI
model is checked against the serial model where the two must agree:
`tests/memory_model_tests.py` requires equal traffic for the same mapping, with palette
weights and weight-cache hits, and `tests/cluster_tests.py` requires a one-core cluster to
equal the single-core model.

### 2.7 Sanitizers and static checks

| Check | Configuration | Source |
|---|---|---|
| AddressSanitizer, UndefinedBehaviorSanitizer, LeakSanitizer | preset `asan-ubsan`: Debug, `-fsanitize=address,undefined -fno-sanitize-recover=all`; SystemC is fetched and built with the same instrumentation; `detect_leaks=1`, `abort_on_error=1`, `halt_on_error=1` | `CMakePresets.json`, `cmake/NpuCompileOptions.cmake` |
| UBSan vptr check | off for SystemC's own objects, whose process dispatch through member-function pointers trips it; on for project code, which `build.ubsan-vptr` verifies with a probe that makes a member call on an object of the wrong dynamic type | `tests/ubsan_vptr_tests.py` |
| Warnings as errors | GCC and Clang: `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wold-style-cast -Wnon-virtual-dtor -Woverloaded-virtual -Wimplicit-fallthrough`; MSVC: `/W4 /permissive-` | `cmake/NpuCompileOptions.cmake` |
| Formatting and lint | `clang-format --dry-run --Werror` on C++; `ruff check` and `ruff format --check` on Python | CI job `lint` |
| Line coverage | gcov over `src/` and `include/`, reported by gcovr | CI job `coverage` |

### 2.8 CI matrix and determinism gate

`.github/workflows/ci.yml` runs on pushes to `main`, version tags, pull requests and weekly.
SystemC 3.0.2 (`cmake/SystemCSource.cmake`) is built once per runner image and compiler and
cached.

| Job | Runner | Toolchain and preset | Scope |
|---|---|---|---|
| `linux-x64-gcc` | `ubuntu-24.04` | GCC, `ci-gcc` | full CTest; uploads VCD waveforms and result files |
| `linux-x64-clang` | `ubuntu-24.04` | Clang, `ci-clang` | full CTest |
| `linux-arm64-gcc` | `ubuntu-24.04-arm` | GCC, `ci-gcc`, native AArch64 | full CTest |
| `macos-arm64-clang` | `macos-15` | Clang, `ci-clang` | full CTest; `run.py` from-scratch build and reuse |
| `windows-x64-msvc` | `windows-2025` | MSVC, `ci-msvc` | full CTest; `run.py` from-scratch build and reuse |
| `linux-x64-asan-ubsan` | `ubuntu-24.04` | preset workflow `asan-ubsan` | full CTest under sanitizers |
| `coverage` | `ubuntu-24.04` | GCC, preset `coverage` | full CTest; line coverage |
| `onnx-runtime-differential` | `ubuntu-24.04` | GCC, `ci-gcc`, `NPU_ENABLE_ONNX_TESTS=ON` | `ctest -L onnx` |
| `results-drift` | `ubuntu-24.04` | GCC, `release` | `python -m npuperf.reproduce --quick --check` |
| `lint` | `ubuntu-24.04` | clang-format 21.1.8, ruff 0.16.10 | formatting and lint |

The `ci-*` presets build Release with warnings as errors.

`python3 -m npuperf.reproduce` regenerates every table between `generated` markers in
[results](results.md) and every figure in [figures/](figures) from `npu-sim` runs. The quick
subset (S3, S4, S5, S6 and S8) runs the same configurations as the full run, so its tables
are byte-comparable with the committed ones. `--check` regenerates them, writes no
documents, prints a unified diff per differing table and exits nonzero on any difference.
The gate holds because the model is deterministic (section 1.5), the configuration sets are
drawn from fixed seeds and the SVG writer emits fixed number formats with no timestamps
(`tests/studies_tests.py`, `tests/npuperf_tests.py`). A change to timing semantics therefore
fails `results-drift` until the regenerated tables are committed with it.

<a id="model-fidelity"></a>

## 3 Model fidelity

Status: Modeled means the mechanism is simulated at the granularity stated; Approximated
means it is represented by a closed form or a simplified rule; Not modeled means it is
absent and contributes no cycles, bytes or energy. Timing parameters are not calibrated
against RTL or silicon (section 4).

| Area | Mechanism | Status | Treatment | Defined in |
|---|---|---|---|---|
| Numerics | INT8 activations, INT8 or INT4 weights, INT32 bias, zero points | Modeled | bit-exact; the final accumulator must lie in INT32 | `src/core.cpp` |
| Numerics | Requantization `native`, `tosa-single`, `tosa-double`; per-channel multipliers, shifts and weight zero points; ReLU at the output zero point | Modeled | bit-exact to TOSA 1.0.2 `apply_scale_32` and its TFLite and CMSIS-NN equivalents | `src/core.cpp` |
| Numerics | Grouped GEMM and depthwise convolution as $G$ groups | Modeled | GEMM-engine tiles never straddle a group; depthwise-datapath tiles span channels, each channel reducing its own group | `src/core.cpp` |
| Numerics | Convolution on its NHWC input (`shape.conv`) | Modeled | the im2col view of the input, padding at the activation zero point; bit-exact against the scalar reference on the im2col matrix | `src/conv.cpp` |
| Numerics | Palette (1, 2, 4, 8-bit indices) and block-sparse weight codecs | Modeled | lossless; outputs computed from the decoded encoding | `src/core.cpp` |
| Numerics | Partial sums across reduction panels | Modeled | held in the slot's INT32 buffer for timing and SRAM; the integer result is evaluated at the final panel, which gives the same value | `src/model.cpp` |
| Numerics | Floating point (FP32, FP16, BF16 storage with FP32 accumulation) | Modeled | `npu-transformer-demo` only | `src/transformer.cpp` |
| Compute | Vector datapath | Approximated | $\lceil M_t N_t K_p / P \rceil$ per panel | `src/core.cpp` |
| Compute | Systolic array, output- and weight-stationary | Approximated | closed-form waves or folds per panel; fill and drain exposed once per panel; no processing-element state | `src/core.cpp` |
| Compute | Depthwise datapath | Approximated | one MAC rate, `arch.compute.depthwise_macs_per_cycle`, for every kernel size when $G > 1$ and $N = 1$; tiles spanning channels; dense weights only; no microblock or kernel-step term | `src/core.cpp` |
| Compute | Per-panel issue overhead | Approximated | constant `pipeline_cycles` per reduction panel | `src/core.cpp` |
| Compute | Weight decoder | Approximated | `weight_decode_values_per_cycle`, running ahead of the array: $\max(\text{MAC}, \text{decode})$ | `src/core.cpp` |
| Compute | Input-stationary dataflow, register-level array timing, operand gating | Not modeled | | |
| Compute | Elementwise add, pooling, activation functions other than the ReLU clamp, softmax and normalization in `npu-sim` | Not modeled | the graph runtime runs them on the host and charges no NPU cycles; `npu-transformer-demo` times its vector operations | `npuperf/graph.py` |
| On-chip memory | Tile slots, panel double buffering, backpressure | Modeled | `sc_fifo` tokens; a load blocks only when all of its slot's panel buffers are full | `src/model.cpp` |
| On-chip memory | SRAM capacity | Modeled | per-slot reservation (bias, parameters, panel buffers, INT32 partial sums, INT8 outputs) plus the caches; an oversubscribed mapping is rejected | `scratchpad_bytes`, `src/core.cpp` |
| On-chip memory | Weight-panel cache | Approximated | fully associative LRU over (group, column panel, reduction panel) lines, capacity in bytes, zero lookup latency | `src/model.cpp` |
| On-chip memory | Activation-panel cache | Approximated | fully associative LRU over (group or channel tile, row tile, reduction panel) lines, capacity in bytes, zero lookup latency | `src/model.cpp` |
| On-chip memory | SRAM banks, ports, bank conflicts, access energy | Not modeled | | |
| External memory | Serial DMA | Modeled | one non-preemptive server for reads and writes; per-burst latency plus $\lceil \text{bytes}/R_{\mathrm{dma}} \rceil$ | `src/memory.cpp` |
| External memory | AXI4 read and write channels | Modeled | INCR bursts split at `max_burst_beats` and 4 KiB; partial beats; per-channel outstanding limits; read latency; write response latency; one beat per channel per cycle; in-order data per channel | `src/memory.cpp` |
| External memory | Request issue | Approximated | a core's load process holds one panel request at a time and its store process one tile store; a request's bursts pipeline through the outstanding window, so a panel's read latency is exposed once per request (S3 in [results](results.md)) | `src/model.cpp`, `src/memory.cpp` |
| External memory | AXI IDs, out-of-order completion, WRAP and FIXED bursts, QoS, address-channel backpressure from the interconnect | Not modeled | | |
| External memory | DRAM | Approximated | one byte-rate timeline shared by all channels and cores, exact integer reservation, fixed grant order; every beat is charged at full width | `src/memory.cpp`, `src/core.cpp` |
| External memory | DRAM banks, row-buffer hits and misses, refresh, read-write turnaround, controller scheduling, LPDDR command timing | Not modeled | | |
| External memory | Convolution activation traffic | Approximated | without `shape.conv`, the im2col matrix; with `shape.conv`, per input row the span of columns the tile's output pixels need, halo included, once per reduction panel; skipped columns inside a span are read, padding is not; `tile_k` 0 or a multiple of $k_w C_g$ | `src/conv.cpp`, `src/core.cpp` |
| System | Multi-core cluster | Modeled | 1 to 64 cores, each with its own slots, SRAM, caches and AXI port, on one DRAM timeline; m, n and `auto` partitions | `src/model.cpp` |
| System | Network on chip, shared last-level cache, inter-core or die-to-die links, coherence, IOMMU | Not modeled | every core reaches DRAM at the same cost | |
| System | Layer sequencing | Approximated | each GEMM is an independent `npu-sim` run; outputs go to external memory between layers; the graph runtime orders the calls on the host | `npuperf/graph.py` |
| System | Cross-layer fusion, overlap or on-chip residency | Not modeled | the graph optimizer folds a single-consumer ReLU into requantization only | `npuperf/graph_optimize.py` |
| Control | Command streams, descriptors, ISA, firmware, driver | Not modeled | `npu-sim` takes one configuration per GEMM | |
| Control | Host interface: MMIO registers, start and status, interrupt, reset | Modeled | `npu-tlm-demo`: loosely timed TLM-2.0 coprocessor with a fixed 3 x 5 x 7 GEMM and serial stages | `src/tlm_platform.cpp`, `include/npu/tlm_platform.hpp` |
| Control | Concurrent jobs, priorities, deadlines, operating points, idle gating | Approximated | `npu-mobile-demo`: synthetic OPP table, one shared bus with round-robin grants, non-preemptive scheduling | `src/mobile_platform.cpp` |
| Control | Transformer decode with a DRAM-resident KV cache | Approximated | `npu-transformer-demo`: DMA, tensor and vector stages back to back; KV reads and writes DMA-charged; prefill as one batch; TTFT and TPOT reported | `src/transformer.cpp` |
| Clocking | One NPU clock | Modeled | cycles; `frequency_mhz` converts to time only | `src/main.cpp` |
| Clocking | Clock-domain crossings, DVFS in `npu-sim`, reset during a job | Not modeled | | |
| Power | Energy | Approximated | coefficients times executed MACs and DRAM bytes, plus static power times modeled latency; SRAM, register files, clock tree, interconnect, DMA control, weight decode and host excluded | `npuperf/energy.py` |
| Power | Thermal behavior, measured power | Not modeled | | |
| Frontends | ONNX import | Modeled | signed INT8, static shapes, standard-domain opsets 10 to 21, rank-2 `QLinearMatMul`, 2-D `QLinearConv` including depthwise, exactly representable scale ratios; anything else is rejected | `npuperf/onnx_import.py` |
| Frontends | TOSA and TFLite graph import | Not modeled | | |
| Mapping | Tile, slot, cache-split and lowering search | Approximated | `npuperf.mapper`: heuristic candidate set ranked by `analyze()` and replayed with `schedule()`; no loop-order, layout or fusion choice | `npuperf/mapper.py` |

`npu-sim` accepts each dimension up to 1,048,576. A functional run is limited to
$4 \times 10^{9}$ MACs and 256 MiB of host storage; a `--timing-only` run, dense weights only,
to $10^{13}$ MACs (`include/npu/core.hpp`).

<a id="calibration-plan"></a>

## 4 Calibration plan

Each timing parameter maps to a microbenchmark that isolates it. The fit order runs from the
compute core outward, so each step holds the previously fitted terms fixed.

| Parameter | Config path | Measurement on the target | Fit |
|---|---|---|---|
| NPU clock | `arch.frequency_mhz` | clock at the operating point | read directly |
| MAC throughput | `arch.compute.macs_per_cycle`, `array_rows`, `array_cols` | cycles of a compute-bound GEMM with operands resident, swept over $M$, $N$, $K$ well beyond the array | slope of cycles against MACs |
| Array quantization, fill and drain | `arch.compute.dataflow` | single-panel tiles with $M_t$ at 1, $A_r$ and $A_r + 1$ and $N_t$ at 1, $A_c$ and $A_c + 1$ | step positions and the intercept $A_r + A_c - 2$ |
| Weight-stationary preload exposure | `arch.compute.dataflow` = `ws` | $K_p$ swept across multiples of $A_r$, with $M_t$ below and above $A_r$ | the $\max(M_t, A_r)$ term |
| Per-panel overhead | `arch.compute.pipeline_cycles` | fixed MAC count, `tile_k` swept so the panel count varies | intercept per panel |
| Decoder rate | `arch.compute.weight_decode_values_per_cycle` | palette panels whose MAC work is below their decode work | slope against decoded values |
| Depthwise rate | `arch.compute.depthwise_macs_per_cycle` | depthwise layers with the channel tile swept | slope against MACs |
| Read latency | `arch.memory.axi.read_latency` | one read burst with one outstanding: AR to first R beat on a bus monitor, or the cycles of a one-burst load | direct |
| Write response latency | `arch.memory.axi.write_latency` | last W beat to B response | direct |
| Outstanding limits, burst length | `arch.memory.axi.read_outstanding`, `write_outstanding`, `max_burst_beats` | DMA configuration registers; confirmed by a read-bandwidth sweep over request depth (the S3 curve) | knee at $N_r b \ge L_r + b$ |
| DRAM rate | `arch.memory.dram_bytes_per_cycle` | sustained read-only, write-only and mixed streaming bandwidth with outstanding requests saturated | plateau of the read-only stream |
| DRAM sharing | `arch.cores` with `dram_bytes_per_cycle` | $N_c$ cores streaming concurrently | aggregate plateau and per-core spread against the fixed-order arbiter |
| Serial DMA | `arch.memory.latency`, `bytes_per_cycle`, `burst_bytes` | transfer time against size | intercept per burst and slope |
| Cache capacity and policy | `arch.weight_cache_bytes`, `arch.activation_cache_bytes` | hit counters for a layer whose panels repeat across row tiles | hit count against the LRU prediction |
| Energy coefficients | `npuperf.energy` inputs | power during compute-bound, memory-bound and idle microbenchmarks | linear regression on executed MACs, DRAM bytes and time |

Validation follows calibration on held-out layers of the bundled workloads
(`workloads/*.json`), reported with the statistics of S6: median, p95 and maximum cycle
error and bound-class agreement per configuration set. Three measurements expose mechanisms
that section 3 lists as not modeled: the mixed read-write stream measures DRAM turnaround,
the multi-core sweep measures interconnect and controller scheduling, and layers with
SRAM-heavy access patterns measure bank conflicts. A mechanism enters the model when its
residual exceeds the measurement's run-to-run spread. Open RTL with cycle-level simulation, such as
Gemmini or NVDLA, provides reference measurements before silicon counters are available.
Every calibrated run keeps the `config` echo of its result, the model revision
(`npu-sim --version`) and the workload file, so a fitted parameter set reloads with
`--config`.

<a id="related-work"></a>

## 5 Related work

| Model | What it is | Timing basis | Memory system | Numerics | Relation to this repository |
|---|---|---|---|---|---|
| [Arm Vela performance estimator][vela-perf] | part of the Ethos-U compiler; estimates each operation of the compiled schedule | per operation, the maximum of the cycles required for computing (microblock counts, pipeline stalls) and the cycles required for bandwidth | a fixed latency per block plus bytes over each memory area's bandwidth, scaled by a burst-efficiency factor; no outstanding-transaction or queueing model | none in the estimator | The design points take Vela's reference system configurations (latencies, outstanding limits, burst lengths, clock scales, [design points](design-points.md)). This model replaces the per-operation maximum with an event schedule over AXI channels with outstanding limits and a shared DRAM timeline, and has no Ethos-U operator coverage or command stream. |
| [SCALE-Sim v2][scalesim2] and [v3][scalesim3] | systolic-array simulator for GEMM and convolution as im2col | analytical compute cycles, with cycle-level SRAM and DRAM traces; output-, weight- and input-stationary dataflows | double-buffered SRAM per operand; DRAM bandwidth computed for stall-free operation or given; v3 adds a Ramulator DRAM model, multiple cores, sparsity and Accelergy energy | none | The array closed forms of [architecture](architecture.md) sections 3.3 and 3.4 play the role of SCALE-Sim's compute model for output- and weight-stationary dataflows. This model times DRAM traffic at burst granularity through AXI outstanding windows and checks numerics bit-exact; it has no input-stationary dataflow and no DRAM bank model. |
| [Timeloop][timeloop] with [Accelergy][accelergy] | mapping-space exploration for tensor algebra on hierarchical buffer architectures; Accelergy estimates energy and area from component action counts | analytical per-level data movement and throughput over loop-nest mappings; a mapper searches the mapping space | buffer hierarchy with per-level capacity and bandwidth | none | `npuperf.analytic` and `npuperf.mapper` cover a narrower mapping space (one loop order, tile and slot choice) and are checked against the transaction model. `npuperf.energy` uses action counts times coefficients without component plug-ins. |
| [Gemmini][gemmini] | Chisel generator of systolic-array accelerator RTL, integrated with RISC-V cores in Chipyard | cycle-exact RTL in software simulation or FireSim FPGA-accelerated simulation; output- and weight-stationary | scratchpad and accumulator SRAM, DMA through the SoC memory system | bit-exact RTL | One microarchitecture per generated instance, simulated at RTL cost. This model is parametric at transaction level and configured from one file. A Gemmini instance is a calibration reference for section 4. |
| [NVDLA][nvdla] | open accelerator RTL with separate convolution, activation and pooling engines, a C model and a register-accurate SystemC/TLM [virtual platform][nvdla-vp] with QEMU | RTL; the virtual platform targets software bring-up | convolution buffer, configurable memory interfaces, weight compression | bit-exact RTL | `npu-tlm-demo` models a register-level host interface (MMIO, interrupt, reset) for a fixed GEMM; `npu-sim` models the performance parameters the NVDLA virtual platform does not time. NVDLA RTL is a calibration reference for section 4. |
| [gem5][gem5], with [gem5-Aladdin][gem5-aladdin] and [gem5-SALAM][gem5-salam] for accelerators | full-system simulator | cycle-level CPU models; accelerators from datapath models built from LLVM IR | coherent caches, interconnect and DRAM controller models | functional execution of the host program | gem5 models the host, coherence, interconnect and DRAM controller that section 3 lists as not modeled. This model has no CPU, operating system or coherent memory system and simulates accelerator-internal scheduling at transaction level. |

[vela-perf]: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/ethosu/vela/npu_performance.py
[scalesim2]: https://github.com/scalesim-project/scale-sim-v2
[scalesim3]: https://github.com/scalesim-project/scale-sim-v3
[timeloop]: https://timeloop.csail.mit.edu/
[accelergy]: https://accelergy.mit.edu/
[gemmini]: https://github.com/ucb-bar/gemmini
[nvdla]: https://nvdla.org/hw/v1/hwarch.html
[nvdla-vp]: https://nvdla.org/vp.html
[gem5]: https://www.gem5.org/
[gem5-aladdin]: https://github.com/harvard-acc/gem5-aladdin
[gem5-salam]: https://github.com/TeCSAR-UNCC/gem5-SALAM
