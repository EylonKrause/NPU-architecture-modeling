# Architecture and arithmetic

This model answers a bounded question: for a dense INT8 workload and the stated resource and scheduling assumptions, what arithmetic result, cycle count, traffic, and utilization should we expect? It does not infer undocumented hardware behaviour from a cycle counter.

## Dense layer

Activations, weights, and outputs use row-major storage. Each output element is defined by

$$
a_{m,n}=b_n+\sum_{k=0}^{K-1}(x_{m,k}-z_x)(w_{k,n}-z_w).
$$

The input and weight zero points describe the integer representation of real zero. A final accumulator must fit INT32. The implementation uses an INT64 intermediate and rejects an overflowing final accumulator, avoiding C++ signed-overflow undefined behaviour.

For a nonnegative integer multiplier and a power-of-two denominator, output requantization is

$$
q_{m,n}=\operatorname{sat}_{[-128,127]}\left(z_y+\operatorname{round}_{\mathrm{ties\ away\ from\ zero}}\left(\frac{a_{m,n}\,\mathrm{multiplier}}{2^{\mathrm{shift}}}\right)\right).
$$

In a real quantized layer, the multiplier approximates the ratio of activation and weight scales to output scale:

$$
\frac{\mathrm{multiplier}}{2^{\mathrm{shift}}}\approx\frac{s_xs_w}{s_y}.
$$

This executable accepts the integer multiplier and shift directly; it does not estimate scales, calibrate a model, or promise framework-specific rounding. Optional ReLU clamps output values below the output zero point. Saturation and rounding are tested separately from the matrix product.

The input fixture uses a specified 32-bit linear congruential sequence rather than an implementation-dependent random distribution. Shape, seed, and quantization must match when comparing numerical outputs. Host allocation and scalar-reference execution do not consume accelerator SRAM or modeled accelerator cycles.

## Mapping and traffic

An output tile has a row extent, a column extent, and the entire reduction dimension. Partial edge tiles use their actual dimensions. For tile dimensions and reduction length, the memory and compute requirements are

$$
\begin{aligned}
B_\mathrm{read}&=M_tK+KN_t+4N_t,\\
B_\mathrm{write}&=M_tN_t,\\
C_\mathrm{compute}&=\left\lceil\frac{M_tN_tK}{P}\right\rceil+C_\mathrm{pipeline},\\
C_\mathrm{DMA}(B)&=L+\left\lceil\frac{B}{W}\right\rceil.
\end{aligned}
$$

Here the compute budget is MACs per cycle, DMA bandwidth is bytes per cycle, and the memory latency applies to each tile read or write request. Bias is INT32; outputs written off-chip are INT8. Every tile reloads its activations, weights, and bias. There is no inter-tile cache, burst-size limit, or compression model.

The conservative scratchpad reservation is

$$
B_\mathrm{SRAM}=S\left(M_\max K+KN_\max+4N_\max+4M_\max N_\max+M_\max N_\max\right),
$$

where the maximum tile dimensions are clipped by the matrix dimensions and the slot count bounds in-flight tiles. The bound reserves space for input, weight, bias, accumulator, and output storage per slot. It can overestimate storage if a more sophisticated lifetime-aware allocator could share buffers. Configurations exceeding capacity are rejected; there is no implicit spill implementation.

## Timing, arbitration, and backpressure

SystemC has a load process, a compute process, and a store process. A tile acquires a buffer slot before loading. It releases that slot only after its output write completes. One slot therefore serializes tile lifetimes. More slots permit load, compute, and store overlap within the limits of the shared DMA and compute engine.

The single DMA server handles read and write queues. Requests run to completion. Arbitration alternates between ready request classes at request boundaries, initially preferring loads. If only one class is pending, it proceeds. Simultaneously pending requests are considered together; completion notifications wake requesters in a later delta cycle. These ordering details affect pipeline timing and are part of the contract.

FIFO tile issue and bounded slots make stalls and overlap observable. A tile waiting for a slot is backpressure, not free execution. Pipeline fill, drain, DMA queuing, transaction latency, compute overhead, and tile edges contribute beyond an ideal throughput bound.

Each SystemC tick denotes one architecture cycle. Event mode advances by an entire stage duration; cycle mode executes a one-cycle wait repeatedly inside the same stage. Both modes must produce the same architecture cycles, stage traces, and outputs. Neither mode models individual multiplier registers, physical SRAM banks, cache coherence, instruction scheduling, or RTL signal propagation. Calling cycle mode “cycle accurate” would require additional validation against a specified implementation.

## Bounds and baselines

An optimistic compute and memory lower bound is

$$
C_\mathrm{ideal}\geq\max\left(\left\lceil\frac{MNK}{P}\right\rceil,\left\lceil\frac{B_\mathrm{read,total}+B_\mathrm{write,total}}{W}\right\rceil\right).
$$

This omits transaction latency, per-tile rounding, pipeline overhead, arbitration, fill/drain, and limited buffering. The sum of actual compute service times and the sum of actual DMA service times provide tighter resource bounds; total elapsed cycles must cover both. They are bounds, not an independently validated timing prediction.

For reporting target latency at a configured clock,

$$
t_\mathrm{target,\mu s}=\frac{C_\mathrm{elapsed}}{f_\mathrm{MHz}}.
$$

Changing frequency rescales reported time without changing modeled cycle counts. Actual silicon might also require latency and bandwidth changes at a different clock, which this model does not infer.

Resource occupancy and useful throughput are distinct. `compute_utilization` includes pipeline overhead in compute busy time, and `memory_utilization` includes transaction latency in DMA busy time. The additional `mac_utilization` measures useful MACs against the total MAC budget, and `dma_payload_utilization` measures transferred bytes against the total bandwidth budget. `arithmetic_intensity_macs_per_byte` includes all modeled reads and writes, including repeated loads; `resource_lower_bound_cycles` is the larger of the actual compute and DMA service sums.

Use one buffer slot as a serial architecture baseline. Compare extra slots, MAC budgets, memory bandwidth, and tile shapes independently with the same arithmetic fixture. Compare event and cycle modes only for simulator host runtime. Report individual cases alongside aggregate ratios so a geometric mean cannot hide a regression.

## What the experiments can explain

A useful architecture discussion connects workload assumptions to an objective, constraints, baselines, and measured evidence. For example, a tile-shape change alters SRAM capacity and reload traffic; extra MACs reduce compute service time only until another resource dominates. If arithmetic agrees but timing differs, inspect the timing contract, queue arbitration, pipeline fill/drain, memory latency, backpressure, and overlap before changing the numerical kernel.

The repository offers executable examples of that method. It does not substantiate CV performance claims, recommend a particular commercial NPU, or replace compiler, RTL, or hardware validation.
