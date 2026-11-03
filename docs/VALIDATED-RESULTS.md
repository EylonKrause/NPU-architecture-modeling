# Verified results

Validated on November 3, 2026 with GCC 13.3.0, CMake 3.28.3, Python 3.12.3, and hash-verified SystemC 3.0.2 on Linux x86-64. A clean default build downloaded its dependency using the checked-in CMake configuration. Both Release and undefined-behavior-sanitized Debug builds passed CTest.

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

## Verification boundary

This validates the documented numerical and scheduling model. There is no supplied silicon or RTL timing reference, proprietary NPU specification, ONNX importer, convolution frontend, or cross-layer on-chip scheduling. Those are separate extensions with their own validation requirements.
