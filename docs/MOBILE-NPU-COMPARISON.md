# Mobile NPU comparison and executable additions

Public-source review: 3 November 2026. Apple Neural Engine and Qualcomm Hexagon
are useful reference points, not interchangeable architectures. Public SDK
behavior is not a complete disclosure of either microarchitecture. This project
models the capabilities below with its own explicit contracts; it does not
emulate Apple's firmware, Core ML compiler or instruction set.

## What was missing, and what now runs

| Mobile inference requirement | Public evidence | Repository implementation |
| --- | --- | --- |
| Floating-point transformer operators | Apple's [transformer deployment research](https://machinelearning.apple.com/research/neural-engine-transformers) discusses FP16, multihead attention, layout and locality. | [Decoder-block executable](TRANSFORMER.md): causal multihead/grouped-query attention, stable softmax, RoPE, RMSNorm, residuals and SwiGLU. FP16/BF16 storage with FP32 accumulation, plus FP32 baseline. BF16 is a generic experimental choice, not an assertion about a particular Apple chip. |
| Persistent inference state | Core ML [stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html) retain state across predictions, including key/value caches. This software API alone does not specify physical cache residency. | Bounded persistent KV state inside the SystemC transformer engine, prefill plus incremental decode, capacity rejection and reset/lifecycle checks. Numerical results are compared with full-sequence execution and an independent Python oracle. |
| Weight compression and just-in-time decoding | Apple's [palettization overview](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html) describes lookup tables and compact indices; its [performance guidance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html) explains why decode placement matters. | [Lossless per-panel palettes](WEIGHT-COMPRESSION.md), real packed indices and LUTs decoded by the C++ numerical path. DMA, cache capacity, SRAM reservation and decode cycles include representation costs. No training or lossy clustering is implied. |
| Sparse-weight execution | [NVDLA's open architecture](https://nvdla.org/hw/v1/hwarch.html) specifies compression and metadata; zero weights alone are not a speedup. This source is an open architecture reference, not proof of a particular Apple sparse datapath. | Zero-point-aware block encoding, bitmap/retained-value bytes, explicit decode work and skipped MAC accounting. Unsupported sparse/systolic combinations are rejected. |
| Several inference engines sharing memory | Qualcomm [describes scalar/vector/tensor engines, shared memory and heterogeneous execution](https://www.qualcomm.com/processors/hexagon). | The transformer has distinct timed tensor/vector/DMA stages. The separate [mobile platform](MOBILE-PLATFORM.md) has actual concurrent SystemC core threads, a serialized shared bus, bounded private SRAM/cache and independent jobs. This is not full CPU/GPU/NPU coherence. |
| Priority, latency and operating-point tradeoffs | Qualcomm identifies mobile power and thermal constraints; public product descriptions do not supply this project's OPP table or a calibrated governor. | Nonpreemptive priority/deadline scheduling, synthetic voltage/frequency operating points, fixed/deadline policies, wakeup/idle gating and optional coefficient-based energy in the mobile platform. Deadlines are measured, not guaranteed. |
| Compiler elimination of avoidable host work | Apple's [Core ML deployment session](https://developer.apple.com/videos/play/wwdc2024/10161/) covers execution planning, state, model stitching and performance tools. | A bounded [graph rewrite pass](../scripts/graph_optimize.py) fuses single-consumer Dense/Conv → ReLU into output requantization. Shared or declared intermediate outputs are preserved. Verification executes original and compiled graphs. This is one exact pass, not a general mobile compiler. |
| Visibility into actual execution | Apple provides profiling tools; model placement and useful throughput require observation rather than peak-operation claims. | Native SystemC VCD, stage traces, numerical checks and a live browser. The palettized profile shows actual encoded bytes, decode cycles and verified outputs. |

The three SystemC executables have different contracts: `npu-sim` is an
overlapped quantized tiled pipeline, `npu-transformer-demo` is a serial persistent
decoder block, and `npu-mobile-demo` schedules independent dense jobs over
multiple engines and a shared bus. Their counts must not be added together as
though they were one measured phone SoC.

## Run the additions

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/mobile_suite.py --build-dir build --output-dir results/mobile
build/npu-transformer-demo --precision fp16 --prefill 4 --decode 3 --verify-lifecycle --dump-output
build/npu-mobile-demo --cores 2 --jobs 6 --governor deadline --deadline-ns 2000
python3 scripts/graph_optimize.py examples/mobile_fusion.json --verify --binary build/npu-sim
python3 scripts/live_dashboard.py --binary build/npu-sim
```

The suite creates a new results directory and refuses to replace previous
experiments. It pairs identical dense/compressed tensors, checks full integer
outputs with a separate Python oracle, runs all three transformer precisions,
and compares identical jobs on one and two cores. It retains individual cycle,
traffic and numerical results. A smaller representation can still run slower
after metadata and decoder costs; the report preserves that result.

In the browser select **Palettized weights**. This profile generates reproducible
INT8 tensors whose weights take four distinct values, packs two-bit indices,
and varies activations between inferences. It is an exact synthetic compression
fixture, not a compressed trained model or an accuracy benchmark.

## Important capabilities still outside this model

* Proprietary Apple/Qualcomm instructions, firmware, drivers and compiler partitioning.
* Real unified-memory coherence, IOMMU/page faults, physical SRAM banking, NoC or LPDDR timing.
* Measured OPP, thermal RC/feedback throttling, battery drain or silicon-calibrated latency/energy.
* Arbitrary Core ML/TFLite/ONNX operator coverage, trained LLM checkpoints, tokenizer/sampling or model-quality evaluation.
* General cross-layer microtile streaming/fusion, actual heterogeneous GPU/CPU execution and active-job preemption.
* Hardware security domains, physical design, fabrication or RTL equivalence.

Those require their own implementations and reference measurements. Public
marketing throughput or a passing numerical fixture cannot substitute for them.
