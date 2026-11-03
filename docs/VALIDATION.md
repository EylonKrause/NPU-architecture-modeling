# Validation and experiment method

Numerical correctness, agreement with a scheduling contract, simulator runtime and hardware accuracy are separate claims. A matching output does not validate timing, and agreement between execution modes does not calibrate a device.

## Build and regression suite

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The standard suite covers the following evidence boundaries:

| Area | Checks |
| --- | --- |
| C++ arithmetic and validation | Hand-calculated dense/per-channel results, INT4 fixtures, rounding modes, extreme products, saturation, ReLU, global tile indexing, overflow/cancellation, exact resource limits and deterministic randomized tiled workloads. |
| Independent integration | Python integer arithmetic/hash oracle, timestamp-based DMA/compute scheduler, real tensors, default timing traces, malformed input, resource scaling and mode agreement. |
| Quantization and INT4 | External tensor files, manual nearest-even examples, asymmetric channel parameters, independent unbounded-integer reference, signed INT4 endpoints and invalid channel/storage values. |
| Extended architecture contract | Exact one-slot panel/array/burst timing oracle, panel acknowledgment ordering, shared-resource exclusion, INT4 padding, channel metadata, cache LRU/capacity behavior, SRAM boundaries and event/cycle trace identity. |
| Graph inference | CNN/residual golden outputs, direct convolution reference versus lowered execution, padding/stride/dilation, depthwise channel mapping, host placement, scale validation, dependency scheduling and tensor liveness. |
| Energy analysis | Hand-calculated units/components, required coefficients, latency consistency, strict result schema, overflow/nonfinite rejection and create-only output. |
| Mapping search | Resource-constrained selection, latency/SRAM objectives, infeasible candidates, bounded search and unexpected-error propagation. |
| Native SystemC observability | Parse actual VCD declarations and changes, check final byte/MAC/burst totals and completion, occupancy bounds, real busy activity, traced/untraced output identity and event/cycle agreement. |
| Live stdio protocol | Drive the actual child process with bounded response waits; verify frozen paused state, explicit advancement, complete output/hash identity with batch, recoverable bounded command errors, EOF, progress and waveform output. |
| Local live dashboard | Actual SystemC time progression, pause/step/resume, repeated verified jobs with changing inputs, all workload profiles, owned-child teardown, rejected arbitrary commands and HTTP browser boundaries. |
| TLM coprocessor platform | Persistent repeated jobs, real tensor memory transactions and complete output readback, independent Python integer results, startup reset, clock/IRQ behavior and masking, malformed protocol rejection, DRAM bounds and host/engine arbitration. |
| Compressed weights | Complete Python integer outputs for palettes and asymmetric sparse zero points; actual codec-storage mutation proof, malformed header rejection, exact encoded bytes/decode cost, cache/SRAM accounting, INT4 boundaries and event/cycle/live identity. |
| Stateful floating-point decoder | Independent complete attention/RoPE/RMSNorm/SwiGLU outputs for FP32/FP16/BF16, causal-prefix and prefill partition agreement, cache overflow/reset/replay, SRAM/traffic/resource formulas, VCD and live progress. Exhaustive finite FP16 encoding roundtrips plus independent rounding edge cases. |
| Mobile multicore platform | Independent integer outputs, serialized bus service and actual concurrent core times, priorities/arrivals/deadlines, LRU eviction, OPP scaling, wakeup/idle states, coefficient energy units, native VCD and invalid-resource rejection. |
| Graph compiler | Actual original/compiled outputs for all four accelerated operators and both rounding modes, observed/shared intermediates, redundant ReLU chains, idempotence, malformed inputs and create-only output. |

The executable compares full accumulators and outputs; hashes are additional identity evidence. Python tests derive expected values independently rather than calling the C++ reference. The extended architecture test uses exact serial sums plus multi-slot causal/resource invariants; the original default-model scheduler additionally checks concurrent timestamps exactly. Event/cycle agreement shares implementation and therefore supplements, rather than replaces, independent checks.

The default configuration contains sixteen CTest suites: two C++ executables and 136 Python test methods. Enabling optional ONNX checks adds one suite and eighteen Python methods, for seventeen suites and 154 Python methods. Methods with parameterized cases validate additional fixtures. Counts describe the checked-in tests; successful runs must still be recorded for the exact revision and environment. A configured GitHub Actions workflow is not evidence that an unpublished revision has passed remote CI.

Optional ONNX import has a separate pinned dependency/test boundary. Its tests construct actual ONNX models and compare outputs with ONNX Runtime for the supported signed-INT8 subset, including nearest-even ties and convolution layout conversion. Unsupported operators, types, shapes, grouping and scale ratios are rejected. Do not infer general framework conformance from a supported fixture; consult [the import contract](ONNX.md).

```sh
python3 -m venv .venv-onnx
.venv-onnx/bin/python -m pip install -r requirements-onnx.txt
cmake -S . -B build-onnx -DCMAKE_BUILD_TYPE=Release \
  -DNPU_ENABLE_ONNX_TESTS=ON -DPython3_EXECUTABLE="$PWD/.venv-onnx/bin/python"
cmake --build build-onnx -j2
ctest --test-dir build-onnx --output-on-failure
```

## Sanitizers and compilers

```sh
cmake -S . -B build-ubsan -DCMAKE_BUILD_TYPE=Debug -DNPU_ENABLE_SANITIZERS=ON
cmake --build build-ubsan -j2
ctest --test-dir build-ubsan --output-on-failure
```

The sanitizer option instruments project arithmetic, scheduling, CLI and C++ test code. SystemC is a separate dependency. GCC/Clang builds and sanitizer checks catch important regressions but do not certify every dependency or eliminate all lifetime, threading or portability bugs. CI configuration and the latest run determine which environments have actually passed.

## Reproducible experiments

```sh
python3 scripts/sweep.py --binary build/npu-sim --output-dir results/sweep
python3 scripts/compare_modes.py --binary build/npu-sim --output-dir results/modes
python3 scripts/autotune.py --binary build/npu-sim --output-dir results/search
python3 scripts/graph_runtime.py examples/cnn.json --binary build/npu-sim
python3 scripts/graph_runtime.py examples/residual_mlp.json --binary build/npu-sim
```

Hold tensor values, shapes and quantization fixed when changing architecture. Inspect SRAM feasibility, external bytes, busy cycles, slot pressure, cache statistics and output identity. One slot supplies the serial timing baseline. Reduction tiling should preserve outputs while changing SRAM and panel overhead. Cache capacity should alter traffic only when the specified access pattern reuses resident panels. Array geometry should expose ragged-wave inefficiency rather than create useful throughput above its area budget.

Mapping search reports every enumerated candidate, skips only recognized SRAM infeasibility, and propagates other failures. It is not a global optimum across unsearched dataflows or devices. Report the objective, candidate set and constraints with the winning result.

For paired cycle ratios,

$$
r_i=\frac{C_{\mathrm{baseline},i}}{C_{\mathrm{candidate},i}},\qquad
G=\exp\left(\frac{1}{N}\sum_{i=1}^{N}\log r_i\right).
$$

Keep individual cases alongside the geometric mean. Do not aggregate unrelated comparisons into a hardware-speedup claim. A resource increase can legitimately leave elapsed cycles unchanged when another resource dominates.

## Timing-mode and energy boundaries

`compare_modes.py` verifies identical architecture results and repeats host measurements. Median subprocess wall time includes startup, scheduling and library effects. The executable's simulation wall time has a narrower boundary. Small cases can be dominated by startup noise. Neither host metric changes modeled accelerator cycles.

Live snapshots run on the kernel's host thread while execution is paused. Their busy counters include elapsed active operations, while bytes and MACs commit at operation completion. Completion snapshots retain the actual model completion cycle if the last host slice runs to a later deadline. Tracing and slicing must preserve batch outputs and architecture cycles; the regression suite tests this explicitly. Native VCD uses picosecond resolution for delta-cycle ordering, not a claim of picosecond circuit accuracy. Read the cycle signal or CSV cycle columns for architecture timing.

The dashboard intentionally paces host commands for inspection. Its browser refresh rate and inferences per host second are not NPU throughput. Its continuous mode launches a new process per inference. The separate TLM platform validates repeated submissions in one kernel, with its own fixed workload, memory transactions and loosely timed serial schedule. It does not inherit the overlapped pipeline's elapsed-cycle predictions. Startup reset and the specified MMIO protocol are tested; arbitrary mid-job resets or undocumented software-driver behavior are outside the supplied evidence.

Energy analysis requires declared coefficients and a valid simulator result. Keep coefficient provenance, units and omitted components with each report. Synthetic test coefficients verify arithmetic; they are not technology measurements. Graph host fallbacks and lowering are excluded from NPU cycles and from the single-invocation energy tool's action counts.

## Evidence and limits

Retain commands, model revision, compiler/build settings, workload identity, configuration and JSON/CSV when publishing results. Record failures and regressions. Generated reports remain outside tracked source by default.

No supplied RTL/silicon timing reference validates transfer startup, array fill/drain, physical memory behavior or energy. A capacity bound does not model SRAM bank conflicts, DRAM refresh, NoC routing or coherence. Standard-model import, when supported for a strict subset, does not establish complete ONNX/TFLite runtime conformance. Calibrate and independently validate against a selected implementation before treating architectural predictions as hardware results.
