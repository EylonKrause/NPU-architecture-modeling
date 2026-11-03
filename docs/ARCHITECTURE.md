# Architecture and arithmetic

The simulator answers a defined question: under the selected workload, numerical contract, mapping and resource assumptions, what result, cycles, traffic and utilization should occur? Arithmetic correctness and hardware timing accuracy require different evidence. The [production research](PRODUCTION-NPU-RESEARCH.md) and [feature matrix](FEATURE-MATRIX.md) explain the scope.

## Integer datapath

Activations are signed INT8. Weights are signed INT8 or signed INT4, held in sign-extended INT8 host storage. Bias and final accumulators are INT32. With row-major operands, output-channel weight zero points give

$$
a_{r,c}=b_c+\sum_{j=0}^{K-1}(x_{r,j}-z_x)(w_{j,c}-z_{w,c}).
$$

INT64 intermediate storage avoids signed-overflow undefined behavior, permits intermediate cancellation, and rejects a final value outside INT32. Bias enters once. Requantization and saturation occur only after the complete reduction, never at a reduction-panel boundary.

$$
q_{r,c}=\operatorname{sat}_{[-128,127]}\left(z_y+\operatorname{round}_{R}\left(\frac{a_{r,c}M_c}{2^{s_c}}\right)\right).
$$

The tie policy is nearest-away by default, or nearest-even when selected. Optional ReLU clamps below the output zero point. Each channel may use its own weight zero point, multiplier and shift; omitted vectors use scalar parameters. INT4 zero points and weights must fit their selected signed range. See [quantization](QUANTIZATION.md) for boundaries and interchange details.

In a quantized inference system, integer scale ratios approximate tensor scales:

$$
\frac{M_c}{2^{s_c}}\approx\frac{s_xs_{w,c}}{s_y}.
$$

The C++ engine accepts integer parameters directly. The graph frontend additionally validates declared tensor/bias scales and the represented ratio. Neither layer silently calibrates a floating-point model. Host fixture/reference storage does not consume modeled accelerator SRAM.

## Output tiles and reduction panels

An output tile has actual edge-clipped extents. Its reduction dimension is split by `tile_k`, with zero meaning the entire reduction. Let a panel have reduction extent $K_p$, an output tile have extents $M_t,N_t$, weight width be $b_w$, and the number of provided channel parameter vectors be $Q$.

$$
\begin{aligned}
B_{w,p}&=\left\lceil\frac{K_pN_tb_w}{8}\right\rceil,\\
B_{r,p}&=M_tK_p+B_{w,p}+\mathbf{1}_{p=0}\,4N_t(1+Q),\\
B_{\mathrm{output}}&=M_tN_t.
\end{aligned}
$$

Each panel pads an incomplete INT4 byte independently. Bias and provided channel vectors are loaded on the first panel only. Scalar parameters are treated as configuration state, without a separate metadata DMA charge. Output writes are INT8; accumulator storage remains on chip.

The aggregate vector engine charges

$$
C_{p,\mathrm{vector}}=\left\lceil\frac{M_tN_tK_p}{P}\right\rceil+C_{\mathrm{pipeline}}.
$$

An alternative analytical output-stationary array with dimensions $A_r,A_c$ charges

$$
C_{p,\mathrm{array}}=\left\lceil\frac{M_t}{A_r}\right\rceil\left\lceil\frac{N_t}{A_c}\right\rceil(K_p+A_r+A_c-2)+C_{\mathrm{pipeline}}.
$$

This includes full-array fill/drain and output folds, including inefficient ragged waves. The peak useful MAC budget is the array area. It is an analytical service-time model, without PE/register-level simulation or independent RTL calibration. Pipeline overhead is charged per panel in both compute modes.

## SRAM and weight reuse

Each in-flight output tile reserves a maximum-sized activation panel, packed weight panel, first-panel bias/channel metadata, INT32 accumulator tile and INT8 output tile. With maximum clipped panel/tile extents and cache reservation $B_{\mathrm{cache}}$,

$$
B_{\mathrm{SRAM}}=S\left(M_{\max}K_{\max}+\left\lceil\frac{K_{\max}N_{\max}b_w}{8}\right\rceil+4N_{\max}(1+Q)+5M_{\max}N_{\max}\right)+B_{\mathrm{cache}}.
$$

Over-capacity configurations are rejected; there is no implicit spill. Partial-sum allocation persists across reduction panels. The numerical kernel evaluates the full dot product once at the final panel; this is a functional abstraction of resident partial sums, not an implementation of per-panel accumulator register updates.

The optional fully associative LRU cache keys packed weight panels by output-column tile and reduction block, enabling reuse across output-row tiles. A hit removes only the weight bytes from that load. Activations, bias and channel metadata follow their normal rules. Oversized panels bypass insertion; eviction frees least-recently-used panels until the new panel fits. Reports expose hits, misses, saved external bytes and peak stored weight bytes. The entire configured cache capacity is included in SRAM even if it remains partly unused.

The cache adds no lookup delay, physical bank contention or coherence cost. Its host implementation stores architectural cache metadata rather than another numerical copy of the tensors.

## DMA, scheduling and backpressure

One non-preemptive DMA engine services read and write requests. Unlimited requests cost startup latency plus bandwidth service. A configured burst size splits the byte count into burst lengths $B_i$ and charges

$$
C_{\mathrm{DMA}}(B)=\sum_i\left(L+\left\lceil\frac{B_i}{W}\right\rceil\right).
$$

The engine holds the whole request until all its bursts complete. Read/write arbitration occurs at request boundaries, not between bursts. Ready request classes alternate round-robin, initially preferring reads; a sole pending class proceeds. Simultaneous pending work is coalesced before arbitration. There are no multiple outstanding independently serviced memory transactions.

A tile acquires a slot before its first load and releases it after its output store. Each nonfinal panel's next load waits for compute completion acknowledgment, preventing reuse of the panel buffer before consumption. The final panel permits the next output tile's load to overlap computation if another slot is available. Compute and store process FIFO output-tile order. More slots allow overlap, subject to the single compute engine and shared DMA.

One SystemC tick denotes one architecture cycle. Event mode advances across a whole stage; cycle mode performs the same duration as individual cycle waits. Both must agree on cycles, traffic, traces and outputs. Stage traces include `tile_id` and `block_id`; stores carry the final reduction block. Neither execution mode establishes RTL cycle accuracy.

## Graphs and operator placement

The graph frontend topologically executes declared dependencies, validates shapes and quantization, retains skip tensors through their final consumer, and reports logical tensor liveness. Dense/MatMul and materialized Conv2D/depthwise lowering invoke the C++ accelerator. Padding uses the activation's quantized zero. Add, ReLU, pooling and Flatten are explicit host fallbacks.

Invocations are sequential, including separate depthwise channel mappings. Every layer materializes outputs off chip, and no cache/SRAM state persists across invocations. Lowering cost, host fallback latency, independent references and process launches are excluded from NPU cycle totals. Logical tensor storage and lowering workspace are separate from Python RSS and accelerator SRAM. See [graph runtime](GRAPH-RUNTIME.md).

## Metrics, baselines and energy

Elapsed cycles must cover both actual compute service and actual DMA service:

$$
C_{\mathrm{elapsed}}\geq\max(C_{\mathrm{compute,total}},C_{\mathrm{DMA,total}}).
$$

Resource occupancy includes pipeline/startup overhead. Useful `mac_utilization` divides MAC count by peak MAC budget and elapsed cycles; `dma_payload_utilization` similarly uses bytes and bandwidth. Arithmetic intensity includes actual external traffic after cache savings. `resource_lower_bound_cycles` reports the larger measured service sum.

Target time is a reporting conversion:

$$
t_{\mathrm{target},\mu s}=\frac{C_{\mathrm{elapsed}}}{f_{\mathrm{MHz}}}.
$$

Changing frequency does not alter stage cycle counts or infer voltage/bandwidth changes. The [energy tool](ENERGY.md) applies explicit user coefficients to MACs, external bytes and optional constant static power; it does not model uncounted SRAM, clock, interconnect or thermal effects.

Use one slot as a serial timing baseline. Preserve numerical tensors and quantization while comparing mapping, compute, SRAM or bandwidth. Mapping search is exhaustive only over its declared candidates. Event/cycle host-runtime ratios measure simulator speed, not hardware improvement. [Compression](WEIGHT-COMPRESSION.md) extends the integer path; [attention](TRANSFORMER.md) and [mobile multicore scheduling](MOBILE-PLATFORM.md) have separate executable contracts. Physical memory systems, commercial command streams and target calibration remain separate scope.
