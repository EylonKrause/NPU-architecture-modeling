# Verified results

The original default-contract baseline was validated on November 3, 2026 with GCC 13.3.0, CMake 3.28.3, Python 3.12.3, and hash-verified SystemC 3.0.2 on Linux x86-64. A clean default build downloaded its dependency using the checked-in CMake configuration. Both Release and undefined-behavior-sanitized Debug builds passed CTest.

The core suite covers hand-computed outputs, integer boundary cases, exact resource limits, and 64 deterministic randomized tiled workloads. The twelve independent Python test groups include 20 timestamp-based concurrent scheduling cases, exact trace comparisons, numerical and hash comparisons, real tensor inputs, malformed inputs, runtime bounds, resource scaling, and event/cycle agreement. The scheduler oracle uses no SystemC code.

The two-layer INT8 inference example produced the independently expected output:

```text
[4, 3, -1, 3]
```

Each layer took 32 modeled cycles in that example, for a sum of 64. These are separate invocations, with host orchestration between layers.

## Architecture comparisons

These results use the default transaction latencies, pipeline overhead, seed, and quantization. Output tiles have eight rows and eight columns. The serial baseline has one tile slot; other configurations have two. MAC capacity and DMA bandwidth are per architecture cycle.

| Matrix shape: rows, columns, reduction | Serial: 64 MACs, 16 bytes | Overlap: 64 MACs, 16 bytes | Overlap: 128 MACs, 16 bytes | Overlap: 64 MACs, 32 bytes | Overlap: 128 MACs, 32 bytes |
|---|---:|---:|---:|---:|---:|
| 16, 16, 32 | 392 | 268 | 252 | 201 | 178 |
| 17, 13, 31 | 446 | 313 | 313 | 242 | 233 |
| 64, 64, 128 | 18560 | 10228 | 10164 | 8539 | 5878 |

All entries are modeled total cycles, and all 15 runs preserved the full arithmetic result. The middle workload illustrates a memory bottleneck: doubling compute leaves elapsed cycles unchanged, while doubling bandwidth reduces them. This follows the model's timing contract; it is not a hardware measurement.

Reproduce the table with:

```sh
python3 scripts/sweep.py --binary build/npu-sim
```

The event/cycle comparison also passed five repetitions per mode for each of three workloads. It checks identical architecture outcomes while measuring host execution separately. Host timing ratios vary with the machine and load; run `scripts/compare_modes.py` to obtain the local measurements rather than treating one host's ratio as a portable guarantee.

## Expanded-feature checks

Additional checks on November 3 exercised the extended arithmetic and architecture contract. The C++ numerical suite passed after adding per-channel parameters, signed INT4 and nearest-even rounding. Six independent quantization groups and six extended architecture groups passed against the updated Release executable. These include manual positive/negative rounding ties, external-tensor integer oracles, exact single-slot panel timing, multi-slot resource/acknowledgment invariants, cache LRU behavior, burst costs and timing-mode trace agreement. Thirty-two additional irregular systolic cases kept useful MAC utilization within its modeled array-area budget, including very small workloads and oversized arrays. These are model-consistency checks, not silicon measurements.

### Weight reuse

The following cases use a generated workload with sixteen rows, eight output columns and sixteen reduction elements; output tiles have four rows and four columns, reduction panels have four elements, and there are two slots. Other parameters retain defaults. All three produced output hash `1589211671334775273`.

| Weight-cache reservation, bytes | Elapsed cycles | External read bytes | Hits | Misses | Saved weight bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0, disabled | 650 | 1152 | 0 | 0 | 0 |
| 16 | 650 | 1152 | 0 | 32 | 0 |
| 128 | 626 | 768 | 24 | 8 | 384 |

The small cache cannot retain this access pattern's working set. The larger cache saves external traffic, while elapsed cycles improve by less because the remaining traffic, compute and scheduling still contribute.

```sh
build/npu-sim --m 16 --n 8 --k 16 --tile-m 4 --tile-n 4 --tile-k 4 \
  --buffer-slots 2 --weight-cache-bytes 128
```

### Reduction tiling and SRAM

A workload with four rows, four columns and a reduction length of 128, mapped as one output tile with two slots and a 500-byte SRAM limit, rejects a full-reduction mapping. Reduction panels of eight elements require 320 bytes and complete in 366 modeled cycles with numerical agreement.

```sh
build/npu-sim --m 4 --n 4 --k 128 --tile-m 4 --tile-n 4 \
  --buffer-slots 2 --sram-bytes 500 --tile-k 8
```

This demonstrates a bounded-memory mapping that preserves the numerical result. The extra panel loads and compute overhead are charged explicitly; there is no implicit spill.

### Mapping search

The default exhaustive search evaluated 54 tile/slot candidates successfully. For its default workload and objective, the selected mapping used eight-row/eight-column tiles, full reduction and two slots: 268 modeled cycles and 1728 reserved SRAM bytes. The result is optimal within that enumerated candidate set, not across all architectures or mappings.

```sh
python3 scripts/autotune.py --binary build/npu-sim
```

The expanded Release and undefined-behavior-sanitized Debug builds both passed all seven standard CTest suites. Those suites include the C++ core and 74 Python tests: 12 legacy integration, six quantization, six extended architecture, 34 graph inference, 12 energy and four mapping-search tests. CNN/residual fixtures, malformed graphs, liveness and preallocation bounds are included. Eighteen additional optional ONNX tests passed, including full-output comparisons against ONNX Runtime 1.23.2 with ONNX 1.19.1; the optional configuration runs all eight CTest suites. See [validation](VALIDATION.md) for reproducible commands. These results apply to the tested environments, without an all-platform certification claim.

## Live SystemC and platform checks

The version 0.3 SystemC expansion added three native-waveform tests, six live-protocol tests, five dashboard tests and four TLM-platform tests. At that revision, CTest contained the C++ core plus 92 standard Python tests; the optional ONNX configuration contained twelve suites and 110 Python tests. These historical counts describe executable checks at that revision, not unobserved remote CI results.

Local native-waveform tests verified all 21 signal declarations, real resource transitions, final DMA and MAC counters, completion, occupancy limits and equality of traced/untraced outputs and modeled cycles. Picosecond VCD resolution records delta ordering; it does not add physical circuit timing. Live child-process checks verified paused time, explicit stepping, clean EOF, recoverable invalid commands, automatic progress and exact agreement of completed output previews and hashes with batch inference. Completed snapshots retain the actual completion cycle rather than a later host slice deadline.

The local dashboard tests drove the compiled SystemC executable through all three profiles, validated repeated jobs with changing inputs, and checked pause/resume/step and child teardown. Browser checks covered desktop and mobile layouts against the actual running service. The display reports copied model state; it does not synthesize NPU activity. Continuous dashboard operation starts a fresh child process for each inference.

The separate persistent TLM platform exercised two jobs in one kernel with changed tensor bytes, host MMIO, DMA reads/writes, startup reset, clock and masked/unmasked completion interrupts. Output validation covers every accumulator and INT8 result read back from modeled DRAM. Its independent Python arithmetic checks and explicit protocol-error tests provide evidence for the documented fixed workload and register/memory contract. This platform is a loosely timed serial coprocessor model; it does not claim the configurable pipeline's overlap timing or commercial driver compatibility.

Reproduce the added checks with:

```sh
python3 tests/systemc_observability_tests.py --binary build/npu-sim
python3 tests/live_protocol_tests.py --binary build/npu-sim
python3 tests/live_dashboard_tests.py --binary build/npu-sim
python3 tests/tlm_platform_tests.py --binary build/npu-tlm-demo
```

The original default-contract timing comparisons above remain the baseline. Live observation is required to preserve those numerical and cycle results. Full regression and compiler/sanitizer outcomes must be associated with the tested revision; no pending remote workflow is treated as already passed.

## Mobile NPU extension: version 0.4

On 3 November 2026, Release and undefined-behavior-sanitized Debug builds passed
all seventeen configured suites, including the optional ONNX comparisons. The
current source contains two C++ test executables and 154 Python test methods;
without optional ONNX, it contains sixteen suites and 136 Python methods. No
compiler warnings or sanitizer findings were observed in project code during
these builds. The GitHub workflow also runs GCC and Clang; its per-commit status
is the evidence for remote compiler checks.

```sh
python3 scripts/mobile_suite.py --build-dir build --output-dir results/mobile
python3 scripts/graph_optimize.py examples/mobile_fusion.json --verify --binary build/npu-sim
```

The mobile suite uses identical tensors within each compression comparison and
checks complete outputs with a separate Python integer oracle. Its fixed
seventeen-row, thirteen-column, thirty-one-reduction fixture produced:

| Case | Architecture cycles | External read bytes | Executed MACs | Decoder cycles |
| --- | ---: | ---: | ---: | ---: |
| Dense baseline | 654 | 2419 | 6851 | 0 |
| Lossless two-bit palette | 664 | 1705 | 6851 | 84 |
| Block-sparse fixture | 615 | 1798 | 1530 | 63 |

Palette storage shrank from 403 to 165 bytes, but decoder cost made this small
case slightly slower. Sparse storage was 196 bytes with 5321 skipped MACs. These
are two different deterministic weight fixtures, each compared against its own
identical dense baseline, not a claim that their output tensors are identical
to each other. Metadata, decoder time and ragged panels are included.

The seven-token transformer workload completed in 2753 modeled cycles with FP32
storage and 1717 with FP16/BF16 storage. All modes performed 17024 tensor MACs;
live KV storage was 448 bytes in FP32 and 224 in either sixteen-bit format.
Prefill/incremental partitions and lifecycle checks passed. The precision modes
are compared with their own independently rounded numerical references, not
asserted to produce bit-identical cross-precision outputs. Normal workload
cycles exclude the deliberately rejected requests and replay verification.

The same six integer jobs took 1282 modeled nanoseconds on one core and 682 on
two cores with the declared synthetic OPP and shared bus. Every per-job output
matched. Two cores incurred two cold weight-cache loads instead of one, with
all shared bus transfers serialized. This result is a scheduling experiment,
not a phone benchmark or an unqualified core-count scaling promise.

Activation fusion reduced the example from three nodes to two and removed four
host read bytes plus four host write bytes. Outputs remained `[0, 9]`; NPU
cycles remained 62, as expected for the existing fused-clamp contract. Shared
and externally observed intermediate tensors were preserved by negative tests.

Review regressions verify that aliased VCD/progress/trace paths fail before
truncating either an artifact or a tensor input. A partially executed rejected
transformer request retains resource/elapsed-work accounting while publishing no
committed token, output or KV state. Waveform tests now validate all 24 integer
pipeline signals, including actual decoder and skipped-MAC counters.

Live browser checks exercised the palettized INT8 profile, repeated complete
validation, paused time, explicit stepping and resumption. The profile used
2304 encoded weight bytes versus 8192 packed-dense bytes, with 4096 decode cycles
per completed inference. Desktop and phone-size layouts had no JavaScript
errors or horizontal overflow. See [mobile capabilities and remaining gaps](MOBILE-NPU-COMPARISON.md).

## Verification boundary

The model now includes convolution/depthwise graph lowering, explicit host fallbacks and the extended memory/compute mechanisms documented in the [feature matrix](FEATURE-MATRIX.md). It still has no supplied silicon or RTL timing reference, proprietary NPU specification, physical memory-system calibration or cross-layer on-chip scheduling. The optional [ONNX importer](ONNX.md) has a strict supported-subset and independent backend-validation boundary; it does not imply general framework conformance.
