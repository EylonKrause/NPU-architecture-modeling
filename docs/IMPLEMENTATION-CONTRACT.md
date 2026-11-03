# Implementation contract

This is a configurable INT8 neural processing unit architectural simulator in C++17 and SystemC 3.0.2. It supports quantization, executable architecture modelling, memory/compute trade-offs, pipeline timing, arbitration, backpressure, and explicit validation. It models a generic architecture, without a proprietary instruction set or datapath specification. Timing is defined by the contract below and has not been calibrated against RTL or silicon.

The initial executable workload is a dense neural layer: signed INT8 matrix multiplication, INT32 bias/accumulation, fixed-point integer requantization, and optional fused ReLU. A dense layer is the NPU's supported operator; no claim of a full ONNX/TFLite runtime is made. Python tools can invoke successive layers and sweep mappings.

## Arithmetic

For row-major activation and weight matrices, subtract signed INT8 zero points before multiplying, add INT32 bias, accumulate safely in INT64, and reject a final value outside INT32. Independently compare both full accumulators and requantized outputs with a scalar reference. Requantization uses $\mathrm{multiplier}/2^{\mathrm{shift}}$, rounds to nearest with ties away from zero, adds the output zero point, and saturates to INT8; ReLU clamps below `output_zero`. The multiplier is nonnegative and at most `INT32_MAX`; shift is from zero through 31. This is a documented arithmetic contract, without a claim of bit equivalence with a particular framework.

Deterministic input fixture: state starts with the seed masked to 32 bits. For each value,

$$s_{i+1}=(1664525s_i+1013904223)\bmod 2^{32}.$$

Activation and weight values are $(s\bmod31)-15$; bias values are $(s\bmod127)-63$. Generate all activations, all weights, then all biases. Never depend on implementation-specific random distributions. The `--input` option instead accepts user tensors; its header defines the matrix shapes.

## Timing

One SystemC tick represents one architecture cycle, independent of the reported target frequency. The three processes are load, compute, store. A tile spans its entire reduction dimension, with edge tiles supported. For a tile with $M_t$ rows and $N_t$ columns,

$$B_r=M_tK+KN_t+4N_t,\quad C_c=\left\lceil\frac{M_tN_tK}{P}\right\rceil+C_{\mathrm{pipeline}},\quad B_w=M_tN_t.$$

The compute budget $P$ is MACs per cycle. Weights and activations are reloaded per output tile; there is no inter-tile cache or compression. Accumulators are internal SRAM; off-chip outputs are INT8.

A single non-preemptive shared DMA server services load and store queues. Each request costs $L+\lceil B/W\rceil$ cycles, with transaction latency $L$, bytes $B$, and bandwidth $W$. Arbitration is round-robin at request boundaries, initially preferring loads; if only one queue is ready it runs. A request's completion wakes the requester in a later SystemC delta cycle. Simultaneous requests already pending are arbitrated together. The independent Python timing checks implement this ordering.

buffer_slots bounds in-flight tiles end-to-end: acquire before loading; release only after output write completion. One slot serializes tiles; more slots permit overlap subject to DMA arbitration and compute availability. FIFO ordering preserves deterministic tile issue. The conservative SRAM bound per slot is maximum activation tile + weight tile + bias + INT32 accumulator tile + INT8 output tile. Reject configurations over sram_bytes. Host fixtures/reference output are not accelerator SRAM.

event mode advances by whole stage durations. cycle mode advances one cycle at a time inside the same SystemC stage processes. They implement the SAME timing contract and must produce identical architecture cycles/traces/results; host runtime measures simulator speed, not target hardware speed. Cycle mode rejects configurations whose conservative serial stage sum exceeds ten million cycles, bounding the number of single-cycle waits; event mode permits larger durations within the precision guard. This is transaction/stage timing, not per-MAC RTL or independently established silicon cycle accuracy.

Report total cycles, MACs, read/write bytes, compute busy cycles, DMA busy cycles, utilization, load/compute/store trace spans, peak in-flight tiles, slot wait cycles, arithmetic agreement, output hash, simulated latency at configured frequency, and host simulation runtime. Reject zero/overflow/unsafe or impractically huge inputs before allocation. Frequency ranges from one hertz through one billion megahertz; modeled time is a reporting conversion. Conservative simulator host storage, including trace records, is limited to 256 MiB. The CLI JSON stdout must be machine readable; diagnostics go to stderr.

## Validation and publication

Use the main branch. Validate changes with a local build, CTest, independent Python arithmetic/timing checks, timing-mode agreement, bandwidth/compute scaling comparisons, and meaningful input-validation tests. Keep generated build artifacts, machine-local data, and credentials out of the repository.
