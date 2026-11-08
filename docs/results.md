# Results

This document reports the ten studies of `npuperf/studies.py` on the four configurations of
[design points](design-points.md). Every table between `generated` markers and every figure in
[figures/](figures) is written by `python3 -m npuperf.reproduce` from `npu-sim` runs. The
tables were generated at tag `v1.0.0`. The CI job `results-drift` regenerates the quick studies
(S3, S4, S5, S6 and S8) and fails on any difference.

| Convention | Definition |
|---|---|
| Workloads | The GEMM layers of `workloads/*.json`: MobileNetV2 1.0 224, ResNet-50 v1.5 224, one BERT-base encoder layer at sequence length 128, and Llama-3.2-1B prefill ($S = 128$) and decode (KV length 129); batch 1, INT8 unless stated. A layer repeated $r$ times runs once and counts $r$ times |
| Mapping | Tiles, slots, cache split and lowering per layer from `npuperf.mapper` ([usage](usage.md) section 9.7) |
| Convolutions | On the NHWC input footprint ([architecture](architecture.md) section 4.4), or a 1x1 convolution on its plain operand when the mapper finds that faster; depthwise layers on the point's depthwise datapath |
| Run mode | Bit-exact when a call fits the functional limits and 400,000,000 MACs, `--timing-only` otherwise; dense timing does not depend on tensor values |
| DRAM bytes | Whole AXI beats, as the DRAM timeline charges them: a partial beat counts in full |
| MAC utilization | MACs over cycles times the array's peak MAC/cycle, the depthwise layers' MACs included |
| TOP/s | Two operations per MAC |
| Bound class | `analytic.observed_bound`: compute-bound when the busiest core's compute-busy cycles are at least its memory time; otherwise bandwidth-bound when the data cycles of its busier channel are at least the compute-busy cycles, and latency-bound when they are not. S1 counts bandwidth- and latency-bound layers as memory-bound |
| Consistency checks | A result above a roof, a utilization above 1, a speedup above the core count or a bandwidth above the bus stops the study (`SanityError`) |

| Study | Subject | Figure |
|---|---|---|
| S1 | Latency, throughput and traffic of five workloads on four points | |
| S2 | Where each layer sits under its point's roofs | [rooflines](#s2-roofline-per-design-point) |
| S3 | Read bandwidth against AXI outstanding transactions | [s3-outstanding](figures/s3-outstanding.svg) |
| S4 | Cluster speedup for the m and n partitions | [s4-scaling-conv](figures/s4-scaling-conv.svg), [s4-scaling-gemv](figures/s4-scaling-gemv.svg) |
| S5 | Output- against weight-stationary array utilization | [s5-dataflow](figures/s5-dataflow.svg) |
| S6 | Accuracy of the analytic model (`analyze()`) against `npu-sim` | [s6-analytic](figures/s6-analytic.svg) |
| S7 | LLM decode with palette-compressed weights | [s7-compression](figures/s7-compression.svg) |
| S8 | Cycles against SRAM over a tiling search | [s8-pareto](figures/s8-pareto.svg) |
| S9 | Sizing the depthwise datapath | [s9-depthwise](figures/s9-depthwise.svg) |
| S10 | Sizing the per-core SRAM of the cluster core | [s10-sram](figures/s10-sram.svg) |

## S1 Network results per design point

### S1.1 Networks

Each network runs layer by layer on each point, every tensor in DRAM
([design points](design-points.md); section 2 for `edge-256`).

<!-- generated:s1-networks -->
| Design point | Network | Weights (MB) | Latency (ms) | Inferences/s | Achieved TOP/s | MAC utilization | DRAM MB/inference | Compute-bound layers | Memory-bound layers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `edge-256` | MobileNetV2 | 3.47 | 8.09 | 124 | 0.0743 | 14.5% | 25.1 | 0% | 100% |
| `edge-256` | ResNet-50 v1.5 | 25.5 | 46.3 | 21.6 | 0.177 | 34.5% | 186 | 11% | 89% |
| `edge-256` | BERT-base layer | 7.08 | 10.1 | 99.3 | 0.185 | 36.1% | 35.4 | 0% | 100% |
| `mobile-2k` | MobileNetV2 | 3.47 | 1.40 | 714 | 0.430 | 10.5% | 18.8 | 0% | 100% |
| `mobile-2k` | ResNet-50 v1.5 | 25.5 | 4.30 | 233 | 1.90 | 46.5% | 62.7 | 33% | 67% |
| `mobile-2k` | BERT-base layer | 7.08 | 0.766 | 1,305 | 2.43 | 59.3% | 10.5 | 0% | 100% |
| `client-16c` | MobileNetV2 | 3.47 | 0.270 | 3,710 | 2.23 | 5.7% | 22.9 | 0% | 100% |
| `client-16c` | ResNet-50 v1.5 | 25.5 | 0.916 | 1,091 | 8.93 | 22.7% | 76.3 | 19% | 81% |
| `client-16c` | BERT-base layer | 7.08 | 0.170 | 5,886 | 11.0 | 27.9% | 15.1 | 0% | 100% |
| `workstation-32c` | MobileNetV2 | 3.47 | 0.123 | 8,141 | 4.90 | 6.2% | 31.4 | 0% | 100% |
| `workstation-32c` | ResNet-50 v1.5 | 25.5 | 0.406 | 2,464 | 20.1 | 25.6% | 130 | 22% | 78% |
| `workstation-32c` | BERT-base layer | 7.08 | 0.0657 | 15,229 | 28.4 | 36.1% | 30.4 | 0% | 100% |
<!-- end:s1-networks -->

Only ResNet-50 has compute-bound layers, 11% to 33% of them; every MobileNetV2 and BERT layer
is memory-bound on every point. The configured tiles' operand intensity, 16 MAC/B on
`edge-256` and 32 on the other points, lies below every ridge ([design points](design-points.md)
section 6); only cache hits and the input footprints of convolutions raise a layer's intensity
above that of its tiles.

The DRAM rate a network sustains is its DRAM bytes per inference over its latency, from the
table above:

| Design point | DRAM rate | MobileNetV2 | ResNet-50 v1.5 | BERT-base layer |
|---|---:|---:|---:|---:|
| `edge-256` | 8 GB/s | 39% | 50% | 44% |
| `mobile-2k` | 24 GB/s | 56% | 61% | 57% |
| `client-16c` | 120 GB/s | 71% | 69% | 74% |
| `workstation-32c` | 800.4 GB/s | 32% | 40% | 58% |

On the single-core points the memory-bound layers use 39% to 61% of the DRAM rate. A core's
load process holds one panel request at a time ([architecture](architecture.md) section 5.2),
so every panel load pays the 500-cycle read latency before its first beat; S3 isolates the
effect. The 16 cores of `client-16c` hold up to 16 read requests in flight and run the networks
at 69% to 74% of the rate. From `mobile-2k` to `client-16c` latency falls 4.5x to 5.2x, tracking
the 5x DRAM rate where the peak MAC rate grows 9.6x. From `client-16c` to `workstation-32c` the
DRAM rate grows 6.7x and the core count 2x; latency falls 2.2x to 2.6x, and the share of the
DRAM rate used drops to 32% to 58%.

Traffic grows where SRAM shrinks or cores multiply. `edge-256` moves 186 MB per ResNet-50
inference, 7.3x the 25.5 MB of weights, where `mobile-2k`, with 5.3x the SRAM, moves 62.7 MB:
smaller tiles re-read each operand more often (the traffic table of
[architecture](architecture.md) section 4.3). `workstation-32c` moves 130 MB against 76.3 MB
on `client-16c`: a cluster split re-reads a group's weights on every core that holds its row
tiles, or its activations on every core that holds its column tiles (architecture section
6.2), and twice the cores re-read more.

### S1.2 Depthwise layers

MobileNetV2 with and without its depthwise layers. Burst efficiency is payload bytes over
whole-beat bytes of the depthwise layers' AXI traffic.

<!-- generated:s1-depthwise -->
| Design point | Depthwise MAC/cycle per core | MobileNetV2 latency (ms) | Depthwise share of cycles | Depthwise share of DRAM bytes | Depthwise read burst efficiency | Depthwise write burst efficiency | Latency without depthwise (ms) | MAC utilization without depthwise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `edge-256` | 24 | 8.09 | 35.2% | 43.2% | 100.0% | 100.0% | 5.24 | 20.9% |
| `mobile-2k` | 192 | 1.40 | 35.4% | 39.2% | 100.0% | 97.3% | 0.905 | 15.1% |
| `client-16c` | 128 | 0.270 | 32.6% | 32.7% | 100.0% | 97.3% | 0.182 | 7.8% |
| `workstation-32c` | 128 | 0.123 | 27.5% | 35.7% | 100.0% | 97.3% | 0.0890 | 8.0% |
<!-- end:s1-depthwise -->

MobileNetV2's 17 depthwise layers hold 6.9% of its MACs (`workloads/mobilenet_v2_1.0_224.json`)
and take 27.5% to 35.4% of its cycles and 32.7% to 43.2% of its DRAM bytes. A 3x3 depthwise
layer performs 9 MACs per output byte and at most 9 per input byte, at most 4.5 MAC/B
of traffic, far below every ridge, and S1.1 classifies each of these layers as memory-bound.
Without them MobileNetV2 takes 5.24 ms on `edge-256` and 0.182 ms on `client-16c`, and its MAC
utilization rises from 14.5% to 20.9% and from 5.7% to 7.8%. S9 sizes the datapath that runs
them.

### S1.3 Llama-3.2-1B

Decode at KV length 129 and prefill at $S = 128$, batch 1. The study stops on either of two
mapping checks: INT8 decode slower with the full SRAM than with half of it (every half-SRAM
tiling also fits the full SRAM), or an INT4 decode speedup above 2x (INT4 halves only the
weight bytes).

<!-- generated:s1-llm -->
| Design point | Decode tokens/s INT8 | Decode tokens/s INT4 | INT4 speedup | Decode DRAM MB/token INT8 | Decode DRAM rate used INT8 | Prefill TTFT INT8 (ms) | Prefill TTFT INT4 (ms) | Prefill MAC utilization INT8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `edge-256` | 5.05 | 9.85 | 1.95x | 1,241 | 78.3% | 1,453 | 746 | 33.8% |
| `mobile-2k` | 17.1 | 33.5 | 1.96x | 1,241 | 88.4% | 107 | 79.6 | 57.6% |
| `client-16c` | 89.4 | 169 | 1.89x | 1,246 | 92.8% | 21.8 | 13.5 | 29.3% |
| `workstation-32c` | 449 | 755 | 1.68x | 1,250 | 70.2% | 5.02 | 4.34 | 63.7% |
<!-- end:s1-llm -->

Decode streams every weight once per token: 1,241 to 1,250 MB per token against 1,236 MB of
INT8 weights (`workloads/llama3.2_1b_decode_kv129.json`, `lm_head` counted once). Tokens per
second are therefore $uR/B$, with $u$ the share of the DRAM rate $R$ used and $B$ the bytes per
token; on `edge-256`, 78.3% of 8 GB/s over 1,241 MB is 5.05 tokens/s. INT4 halves the weight
bytes and speeds decode up 1.95x and 1.96x on the single-core points. On `workstation-32c` the
speedup is 1.68x: a batch-1 GEMV keeps one of the 32 rows of each output-stationary array busy
(equation 3.3 of [architecture](architecture.md) with $M_t = 1$), 1,024 MAC/cycle over 32
cores, which caps decode at 988 tokens/s for the 1,244,200,960 MACs of a token, and INT4 decode
reaches 755.

Prefill runs each projection as one GEMM of 128 token rows and reaches 29.3% to 63.7% MAC
utilization. INT4 shortens the time to first token 1.95x on `edge-256` (1,453 to 746 ms), where
the weight bytes set the prefill time, and 1.16x on `workstation-32c` (5.02 to 4.34 ms).

## S2 Roofline per design point

Every GEMM layer of the five workloads as achieved MAC/cycle against MAC/B of DRAM traffic,
with the S1 mappings and INT8 weights. Bytes are whole AXI beats. The solid roofs are the
array's peak and the DRAM rate; the dotted line is the depthwise datapath's peak, the compute
roof of the depthwise layers. The study checks each mark against the roof of the datapath it
ran on.

![edge-256 roofline](figures/s2-roofline-edge-256.svg)
![mobile-2k roofline](figures/s2-roofline-mobile-2k.svg)
![client-16c roofline](figures/s2-roofline-client-16c.svg)
![workstation-32c roofline](figures/s2-roofline-workstation-32c.svg)

*Figure S2. Rooflines of `edge-256`, `mobile-2k`, `client-16c` and `workstation-32c`: every
GEMM layer of the five workloads as achieved MAC/cycle against MAC/B of DRAM traffic.*

A mark's vertical gap to the lower roof at its intensity is time the layer spends outside both
rates: the read latency of each panel request (S3), the pipeline charge per panel, array waves
that leave rows or columns idle (S5), and cluster cores without work units (S4). The depthwise
layers, at most 4.5 MAC/B (S1.2), form the low-intensity end of each plot. The ridge moves
from 32 MAC/B on `edge-256` to 163.8 on `client-16c`, whose 16 cores share 100 B/cycle, and
falls to 49.13 on `workstation-32c`, whose 32 cores share 667.

## S3 AXI outstanding transactions

An INT4 GEMV of $1 \times 4096 \times 4096$ streams its weights through one core's read
channel: 16-byte beats, 16-beat bursts, unlimited DRAM, read and write latencies $L_r = L_w$,
at 100 and at 400 cycles, and read outstanding limit $N_r$ from 1 to 32, in the default
$8 \times 8$ tiles. The formula column is the channel's steady state, equation 5.5 of
[architecture](architecture.md) in bytes per cycle, $\min(D, N_r b D/(L_r + b))$ with
$D = 16$ bytes and $b = 16$ beats.

![AXI outstanding](figures/s3-outstanding.svg)

*Figure S3. Read bandwidth of the INT4 GEMV $1 \times 4096 \times 4096$ against outstanding AXI
reads at $L_r$ = 100 and 400 cycles; dashed lines are the steady-state formula.*

<!-- generated:s3-outstanding -->
| $N_r$ | `npu-sim`, $L_r$ = 100 (B/cycle) | Formula, $L_r$ = 100 (B/cycle) | `npu-sim`, $L_r$ = 400 (B/cycle) | Formula, $L_r$ = 400 (B/cycle) |
|---|---:|---:|---:|---:|
| 1 | 2.19 | 2.21 | 0.61 | 0.62 |
| 2 | 4.33 | 4.41 | 1.20 | 1.23 |
| 4 | 8.47 | 8.83 | 2.35 | 2.46 |
| 8 | 14.84 | 16.00 | 4.50 | 4.92 |
| 16 | 14.84 | 16.00 | 8.26 | 9.85 |
| 24 | 14.84 | 16.00 | 11.53 | 14.77 |
| 32 | 14.84 | 16.00 | 12.19 | 16.00 |
<!-- end:s3-outstanding -->

Below saturation `npu-sim` runs 0.9% to 4.5% under the formula for $N_r \le 4$. The formula reaches
the 16 B/cycle bus from $N_r = 8$ at $L_r = 100$ and from $N_r = 26$ at $L_r = 400$; `npu-sim`
reaches 14.84 B/cycle from $N_r = 8$ at $L_r = 100$ and 12.19 B/cycle at $N_r = 32$ for
$L_r = 400$. The plateau is set by the request boundary: the load process issues one panel request
and waits for its completion before the next ([architecture](architecture.md) section 5.2), so the
read latency is exposed once per request. A $1 \times 8$ tile's panel moves 20,512 bytes in 1,282
beats, and `npu-sim` reports a mean request latency of 1,382 cycles at $L_r = 100$ and 1,682 at
$L_r = 400$, the beats plus $L_r$:

$$\frac{20{,}512}{1{,}282 + 100} = 14.84, \qquad \frac{20{,}512}{1{,}282 + 400} = 12.19 \ \text{B/cycle}$$

A panel request of $x$ bytes therefore sustains at most $x / (L_r + x/D)$ bytes per cycle per
core, whatever the outstanding limit; deeper panels and wider tiles raise the plateau.

## S4 Multi-core scaling

Speedup over one core of two layers on 1 to 32 cores of the `client-16c` point, m and n
partitions, at the DRAM rates of `client-16c` (100 B/cycle) and `workstation-32c` (667 B/cycle).
The mapper picks the tiles of every configuration. The DRAM ceiling is the one-core cycles over
the cycles DRAM needs for the compulsory traffic, every input, weight, bias and output byte
once, which bounds the speedup of any schedule.

| Layer | $M \times N \times K$ | Weights | Operand |
|---|---|---|---|
| ResNet-50 res3 3x3 convolution, `layer2.1-3.conv2` | $784 \times 128 \times 1152$ | INT8 | Input footprint |
| Llama-3.2-1B decode `gate_proj` GEMV | $1 \times 8192 \times 2048$ | INT4 | Plain |

![conv scaling](figures/s4-scaling-conv.svg)
![GEMV scaling](figures/s4-scaling-gemv.svg)

*Figure S4. Speedup over one core of the ResNet-50 res3 3x3 convolution and of the
Llama-3.2-1B `gate_proj` GEMV, m and n partitions at 100 and 667 B/cycle.*

<!-- generated:s4-scaling -->
| Layer | Partition | DRAM B/cycle | 2 cores | 4 cores | 8 cores | 16 cores | 32 cores | DRAM ceiling |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 res3 3x3 conv | m | 100 | 1.78 | 3.25 | 5.11 | 5.11 | 5.11 | 34.40 |
| ResNet-50 res3 3x3 conv | n | 100 | 1.95 | 3.73 | 3.73 | 3.73 | 3.73 | 34.40 |
| ResNet-50 res3 3x3 conv | m | 667 | 1.78 | 3.27 | 5.63 | 10.21 | 12.36 | 229.33 |
| ResNet-50 res3 3x3 conv | n | 667 | 1.95 | 3.78 | 3.78 | 3.78 | 3.78 | 229.33 |
| Llama-3.2-1B gate_proj GEMV INT4 | m | 100 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.35 |
| Llama-3.2-1B gate_proj GEMV INT4 | n | 100 | 1.99 | 3.92 | 5.44 | 5.71 | 5.87 | 6.35 |
| Llama-3.2-1B gate_proj GEMV INT4 | m | 667 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 42.37 |
| Llama-3.2-1B gate_proj GEMV INT4 | n | 667 | 1.99 | 3.94 | 7.71 | 15.00 | 27.12 | 42.37 |
<!-- end:s4-scaling -->

The tile grid caps the convolution below both DRAM ceilings. With $N = 128$ on a 32-column
array the n split has at most 4 column tiles, and it holds 3.73 and 3.78 from 4 cores on. At
100 B/cycle the mapper keeps 128-row tiles from 8 cores on, 7 row tiles of the 784 output
pixels, so the m split holds 5.11 from 8 to 32 cores. At 667 B/cycle it picks 56-row and
32-row tiles, 14 and 25 row units, and the m split reaches 10.21 at 16 cores and 12.36 at 32.
The m split re-reads the weights on every core: the 32-core run moves 4.09 MB against 0.81 MB
on one core (`results/studies/s4.json`).

The GEMV has one row tile, so the m split runs it on one core at every core count. The n split
divides up to 128 column tiles of 64. One core is compute-bound: the output-stationary array
keeps one of its 32 rows busy, and `npu-sim` reports 532,736 compute-busy cycles of the 535,608
of the one-core run. Cores divide that compute until the shared DRAM binds: at 100 B/cycle the
speedup is 5.44 at 8 cores and 5.87 at 32 against a ceiling of 6.35, and at 667 B/cycle 27.12
at 32 cores against 42.37.

## S5 Dataflow

Output- and weight-stationary utilization of every MobileNetV2 and ResNet-50 layer that runs on
the array, one core of `client-16c` (32x32), with the S1 lowering and the mapper's tiles for
each dataflow. Depthwise layers run on the depthwise datapath, which has no dataflow, and are
not in the table. The Layers column counts every layer instance, a repeated block once per
repeat; the figure draws one mark per distinct shape. Utilizations are MAC-weighted: array
utilization divides MACs by compute-busy cycles times 1,024, end-to-end utilization by total
cycles times 1,024.

![dataflow](figures/s5-dataflow.svg)

*Figure S5. Output- against weight-stationary array utilization of every distinct
MobileNetV2 and ResNet-50 layer shape on one `client-16c` core.*

<!-- generated:s5-dataflow -->
| Layer type | Layers | Distinct shapes | MACs (M) | OS array utilization | WS array utilization | OS end-to-end | WS end-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| pointwise | 70 | 34 | 2,387 | 86.2% | 92.8% | 76.4% | 80.1% |
| 3x3 | 17 | 5 | 1,861 | 89.1% | 97.7% | 81.7% | 87.5% |
| 7x7 | 1 | 1 | 118 | 97.3% | 88.5% | 88.1% | 84.2% |
| FC | 2 | 2 | 3.33 | 3.0% | 3.0% | 2.4% | 2.4% |
<!-- end:s5-dataflow -->

The two dataflows round different dimensions up to the array (equations 3.3 and 3.4 of
[architecture](architecture.md)). Output-stationary rounds $M_t$ and $N_t$ to multiples of 32;
weight-stationary rounds $K_p$ and $N_t$ and streams the tile's $M_t$ rows unrounded.

| Layer shape | Example | OS array utilization | WS array utilization | Dimension that rounds |
|---|---|---:|---:|---|
| $M = 49$ (7x7 output) | ResNet-50 $49 \times 2048 \times 512$ | 74.2% | 97.0% | OS rounds 49 rows to 64 |
| $M = 196$ (14x14 output) | ResNet-50 $196 \times 256 \times 1024$ | 85.9% | 97.0% | OS rounds 196 rows to 224 |
| $K = 16$ | MobileNetV2 $12544 \times 96 \times 16$ | 85.3% | 44.3% | WS fills 16 of 32 rows per fold |
| $K = 147$, the stem | ResNet-50 $12544 \times 64 \times 147$ | 97.3% | 88.5% | WS takes 5 folds of 32 rows for 147 |
| $M = 1$, FC | ResNet-50 $1 \times 1000 \times 2048$ | 3.0% | 3.0% | Both fill one row of 32 |

The rows come from `results/studies/s5.json`. The layers with $M$ of 49 and 196 carry most of
the MACs, so weight-stationary leads on pointwise (92.8% against 86.2%) and 3x3 layers (97.7%
against 89.1%); output-stationary leads on the 7x7 stem. End to end the gap narrows, to 80.1%
against 76.4% on pointwise layers, as memory time fills part of both schedules.

## S6 Analytic model against npu-sim

`npuperf.analytic` against `npu-sim` on seeded random configurations and on the distinct S1 layer
runs of each design point. The random dense set draws read latencies up to 300 cycles, at most
16 outstanding reads, DRAM up to 64 B/cycle and up to 8 cores; the random conv set draws
convolutions on their input footprint, a third of them on the depthwise datapath and half with
an activation cache, at Vela's SRAM and DRAM latencies. The design points lie outside the dense
set, with 500-cycle reads, 64 outstanding reads, and 100 and 667 B/cycle shared by 16 and 32
cores on the clusters.

![Analytic model against npu-sim](figures/s6-analytic.svg)

*Figure S6. Ratio of the `analyze()` estimate to `npu-sim` cycles on the random sets and
the S1 layer runs.*

<!-- generated:s6-analytic -->
| Configurations | Count | Median error | p95 error | Max error | Estimate below npu-sim | Bound class agrees | schedule() exact | Traffic and compute exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random serial | 120 | 0.0% | 5.9% | 21.7% | 11/120 | 120/120 | 120/120 | 120/120 |
| random AXI | 83 | 0.0% | 14.2% | 26.8% | 40/83 | 82/83 | 83/83 | 83/83 |
| random AXI cluster | 37 | 0.5% | 20.9% | 44.8% | 28/37 | 36/37 | 37/37 | 37/37 |
| random conv | 80 | 0.1% | 18.7% | 29.1% | 49/80 | 80/80 | 80/80 | 80/80 |
| `edge-256` network layers | 82 | 0.0% | 31.9% | 36.5% | 38/82 | 82/82 | 82/82 | 82/82 |
| `mobile-2k` network layers | 82 | 0.1% | 27.7% | 32.7% | 51/82 | 77/82 | 82/82 | 82/82 |
| `client-16c` network layers | 82 | 19.2% | 33.2% | 37.8% | 82/82 | 78/82 | 82/82 | 82/82 |
| `workstation-32c` network layers | 82 | 6.9% | 23.4% | 28.5% | 82/82 | 75/82 | 82/82 | 82/82 |
<!-- end:s6-analytic -->

`schedule()` equals `npu-sim` on all 648 runs, and the closed-form traffic and compute-busy
cycles of `analyze()` are exact on all of them. `analyze()` times every request alone on the
memory system and charges the shared DRAM only as an aggregate byte bound, so it misses the
queueing of channels and cores on one DRAM timeline. Its error grows with that queueing: the
median is 0.0% to 0.5% on the random sets and the single-core points, 6.9% on
`workstation-32c` and 19.2% on `client-16c`, where the estimate falls below `npu-sim` on 82 of 82
layer runs. Its bound class disagrees on 5, 4 and 7 layer runs of `mobile-2k`, `client-16c` and
`workstation-32c`, which it calls latency-bound against an observed bandwidth bound, or
compute-bound against an observed latency bound (`results/studies/s6.json`).

`npuperf.mapper` uses both ([usage](usage.md) section 9.7): it ranks its candidates by the
`analyze()` estimate and keeps the fastest `schedule()` replay of its whole candidate set.

## S7 Weight compression for LLM decode

Llama-3.2-1B decode tokens/s with dense INT8, dense INT4 and 4-bit palette weights over the
decoder throughput $V$ in values per cycle per core. A 4-bit palette panel holds a 4-byte
header, a full 16-entry LUT and 4-bit indices: the bytes of dense INT4 plus 20 per panel
([numerics](numerics.md) section 4.2). A palette run is functional, its weights read from a
generated input file, and the study holds each call to 18 MiB of host storage, so every weight
GEMM runs as pieces. All three formats run the same pieces; the dense columns therefore differ
from the S1 decode rows, which run each layer whole. The figure divides by dense INT4.

![compression](figures/s7-compression.svg)

*Figure S7. Llama-3.2-1B decode rate with 4-bit palette weights relative to dense INT4,
against the decoder throughput $V$.*

<!-- generated:s7-compression -->
| Design point | Dense INT8 | Dense INT4 | Palette, 4/cycle | Palette, 8/cycle | Palette, 16/cycle | Palette, 32/cycle | Palette, 64/cycle | Palette, 128/cycle | Palette, 256/cycle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `edge-256` | 5.05 | 9.84 | 3.23 | 6.43 | 9.83 | 9.83 | 9.83 | 9.83 | 9.83 |
| `mobile-2k` | 17.1 | 33.4 | 3.23 | 6.45 | 12.9 | 25.5 | 33.4 | 33.4 | 33.4 |
| `client-16c` | 89.1 | 168 | 61.0 | 120 | 159 | 167 | 167 | 167 | 167 |
| `workstation-32c` | 443 | 728 | 117 | 230 | 443 | 726 | 726 | 726 | 726 |
<!-- end:s7-compression -->

A palette panel of $E$ values decodes in $\lceil E / V \rceil$ cycles, and the decoder runs
ahead of the array, so a panel takes the longer of its decode and MAC times (equation 3.1 of
[architecture](architecture.md)). Decode hides behind the memory time once $V$ reaches the rate
at which DRAM delivers indices to a core, two per byte, $2R / N_c$ values per cycle: 16 on
`edge-256`, 48 on `mobile-2k`, 12.5 on `client-16c` and 41.7 on `workstation-32c`. The palette
reaches its dense-INT4 rate at the first swept $V$ at or above that threshold on `edge-256`,
`mobile-2k` and `workstation-32c` (16, 64 and 64), and at 32 on `client-16c`, where 16 values
per cycle give 95% of it. Below the threshold tokens/s scale with $V$: `edge-256` and
`mobile-2k` both give 3.23 tokens/s at 4 values per cycle, one core decoding 1,236 M weights per
token at 1 GHz. At full decoder rate the palette runs within 0.6% of dense INT4 on every point.

## S8 Mapping search Pareto front

Cycles against reserved SRAM for 49 tile and slot choices of ResNet-50 res3 3x3
($784 \times 128 \times 1152$) on `mobile-2k`, on its input footprint, caches off, with the SRAM
raised to 1 MiB so that every choice fits: `tile_m` of 32 to 256, `tile_n` of 64 or 128,
`tile_k` 0 or whole kernel rows ($3 \times 128 = 384$ bytes, and 768), 1 or 2 slots, and the
mapper's pick. All 49 produce one output hash.

![Pareto](figures/s8-pareto.svg)

*Figure S8. Cycles against reserved SRAM for the tile and slot choices of ResNet-50 res3
3x3 on `mobile-2k`, with the Pareto front and the mapper's picks.*

<!-- generated:s8-pareto -->
| Tile (m x n x k) | Slots | Weight cache (KiB) | Activation cache (KiB) | SRAM (KiB) | Cycles | Row |
|---|---:|---:|---:|---:|---:|---|
| 32x64x384 | 1 | 0.00 | 0.00 | 66.75 | 301,998 | Pareto front |
| 64x64x384 | 1 | 0.00 | 0.00 | 84.75 | 184,398 | Pareto front |
| 128x64x384 | 1 | 0.00 | 0.00 | 120.75 | 125,598 | Pareto front |
| 128x64x768 | 1 | 0.00 | 0.00 | 175.75 | 122,798 | Pareto front |
| 256x64x1152 | 1 | 0.00 | 0.00 | 191.50 | 111,394 | Pareto front |
| 256x64x384 | 1 | 0.00 | 0.00 | 192.75 | 98,658 | Pareto front |
| 128x128x384 | 1 | 0.00 | 0.00 | 209.00 | 95,330 | Pareto front |
| 128x64x384 | 2 | 0.00 | 0.00 | 241.50 | 93,984 | Pareto front |
| 112x64x1152 | 2 | 0.00 | 0.00 | 256.50 | 81,641 | Pareto front |
| 128x64x1152 | 2 | 0.00 | 0.00 | 271.00 | 77,337 | Pareto front |
| 256x64x1152 | 2 | 0.00 | 0.00 | 383.00 | 68,663 | Pareto front |
| 256x64x384 | 2 | 0.00 | 0.00 | 385.50 | 68,511 | Pareto front |
| 128x128x1152 | 2 | 0.00 | 0.00 | 495.50 | 66,354 | Pareto front |
| 128x128x768 | 2 | 0.00 | 0.00 | 624.00 | 65,194 | Pareto front |
| 256x128x384 | 2 | 0.00 | 0.00 | 642.00 | 65,100 | Pareto front |
| 256x128x768 | 2 | 0.00 | 0.00 | 848.00 | 64,890 | Pareto front |
| 112x64x1152 | 2 | 0.00 | 0.00 | 256.50 | 81,641 | fastest within 267 KiB |
| 112x64x1152 | 2 | 0.00 | 0.00 | 256.50 | 81,641 | npuperf.mapper, caches off |
| 112x64x1152 | 2 | 10.50 | 0.00 | 267.00 | 81,641 | npuperf.mapper, free SRAM as caches |
<!-- end:s8-pareto -->

Within the 267 KiB of `mobile-2k` the fastest choice is $112 \times 64$ over the whole
reduction with 2 slots, 256.5 KiB and 81,641 cycles, and `npuperf.mapper` picks it. Its 112
rows are 4 output rows of 28 pixels, so an interior tile's panel reads 6 whole input rows,
21,504 bytes, where the same rows of the im2col matrix hold 129,024. The mapper gives the
10.5 KiB left over to the weight cache, which leaves the cycles at 81,641. The front continues
to 64,890 cycles at 848 KiB: 3.3x the SRAM reduces cycles by 20.5%. At the array's 2,048
MAC/cycle the layer's MACs take 56,448 cycles, so the pick runs at 69.1% MAC utilization and
the end of the front at 87.0%.

## S9 Depthwise datapath rate

Arm's performance models give the depthwise rate of the parts `edge-256` and `mobile-2k` are
sized from. Vela 5.2.0 runs the Ethos-U55-256's depthwise on its 2x2x8 output microblock, and
its Regor compiler the Ethos-U85-2048's on its 4x4x16 one; both take 4 cycles per microblock
for each step of 4 kernel taps ([`npu_performance.py`][vela-perf],
[`ethos_u85_performance.cpp`][regor-perf]). That is 32 and 256 MAC/cycle, and 24 and 192 for a
3x3 kernel, whose 9 taps take 3 steps. The two points take the 3x3 rates: `npu-sim`'s datapath has
no kernel step, and every depthwise layer of the workloads is 3x3. Apple publishes no Neural
Engine microarchitecture, so this study sizes the clusters' rate by a latency rule and reports
the same rule on the Arm-sized points beside their Arm-model rate.

MobileNetV2 runs on each point with the rate swept over powers of two from 1 MAC/cycle per core
to the core's array MACs, and at the configured rate, with tiles from the mapper at every rate.
The first row runs the depthwise layers as grouped GEMMs on the array on their im2col operand.
The rule picks the smallest swept rate at which MobileNetV2 runs within 5% of its latency at the
array-wide rate. The configs set the Arm-model rate where there is one and the rule's rate on the
clusters, and the study stops when one disagrees. Cells give latency in ms and, in brackets,
the depthwise layers' share of cycles.

![depthwise rate](figures/s9-depthwise.svg)

*Figure S9. MobileNetV2 latency relative to an array-wide depthwise datapath, against the
depthwise rate per core.*

<!-- generated:s9-depthwise -->
| Depthwise MAC/cycle per core | `edge-256` | `mobile-2k` | `client-16c` | `workstation-32c` |
|---|---:|---:|---:|---:|
| none: grouped GEMM on the array, im2col | 107 (95%) | 22.2 (96%) | 1.80 (90%) | 0.721 (88%) |
| 1 | 26.0 (80%) | 21.6 (96%) | 1.33 (86%) | 0.712 (87%) |
| 2 | 15.6 (66%) | 11.3 (92%) | 0.770 (76%) | 0.408 (78%) |
| 4 | 10.5 (50%) | 6.11 (85%) | 0.490 (63%) | 0.256 (65%) |
| 8 | 8.45 (38%) | 3.52 (74%) | 0.357 (49%) | 0.182 (51%) |
| 16 | 8.15 (36%) | 2.22 (59%) | 0.306 (41%) | 0.148 (40%) |
| 24 | 8.09 (35%) | – | – | – |
| 32 | 8.06 (35%) | 1.64 (45%) | 0.284 (36%) | 0.133 (33%) |
| 64 | 7.94 (34%) | 1.46 (38%) | 0.274 (34%) | 0.127 (30%) |
| 128 | 7.88 (33%) | 1.41 (36%) | 0.270 (33%) | 0.123 (28%) |
| 192 | – | 1.40 (35%) | – | – |
| 256 | 7.84 (33%) | 1.39 (35%) | 0.264 (31%) | 0.120 (26%) |
| 512 | – | 1.37 (34%) | 0.261 (30%) | 0.119 (25%) |
| 1,024 | – | 1.35 (33%) | 0.259 (30%) | 0.118 (25%) |
| 2,048 | – | 1.35 (33%) | – | – |
<!-- end:s9-depthwise -->

<!-- generated:s9-choice -->
| Design point | Configured rate (MAC/cycle per core) | Source | Rule's rate | Latency at the configured rate (ms) | Latency at the rule's rate (ms) | Depthwise share of cycles | Array-wide rate | Latency at the array-wide rate (ms) | Latency without the datapath (ms) | Speedup over no datapath |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `edge-256` | 24 | Vela 5.2.0 3x3 rate | 16 | 8.09 | 8.15 | 35.2% | 256 | 7.84 | 107 | 13.2x |
| `mobile-2k` | 192 | Regor (Vela 5.2.0) 3x3 rate | 128 | 1.40 | 1.41 | 35.4% | 2,048 | 1.35 | 22.2 | 15.9x |
| `client-16c` | 128 | S9 rule | 128 | 0.270 | 0.270 | 32.6% | 1,024 | 0.259 | 1.80 | 6.7x |
| `workstation-32c` | 128 | S9 rule | 128 | 0.123 | 0.123 | 27.5% | 1,024 | 0.118 | 0.721 | 5.9x |
<!-- end:s9-choice -->

Without the datapath the depthwise layers take 88% to 96% of MobileNetV2's cycles. As grouped
GEMMs each tile is one output column, which an array runs on one of its columns
([architecture](architecture.md) section 3.5), and the im2col operand holds each input byte up
to 9 times. The datapath at the configured rate cuts MobileNetV2's latency 13.2x on `edge-256`,
15.9x on `mobile-2k`, 6.7x on `client-16c` and 5.9x on `workstation-32c`. Latency falls steeply
at low rates and flattens once the layers' memory time dominates: `edge-256` runs within 8% of
its array-wide latency from 8 MAC/cycle (8.45 against 7.84 ms), `mobile-2k` within 9% from 64
(1.46 against 1.35 ms). The Arm-model rates of 24 and
192 lie above the rule's picks of 16 and 128 and run within 3.2% and 3.7% of the array-wide
latency; on the clusters the rule picks 128 on both points.

## S10 Per-core SRAM of the cluster core

Apple publishes no Neural Engine SRAM size, so this study sizes the SRAM of the core that
`client-16c` and `workstation-32c` share. ResNet-50 and MobileNetV2 run on both points with the
per-core SRAM swept over powers of two from 64 KiB to 4 MiB, tiles and cache split from the
mapper at every size. The rule: on each point, the smallest size at which both networks reach
90% of their highest MAC utilization in the sweep; the points share one core design, so both
take the larger of the two picks. The configs set it, and the study stops when one disagrees.
Cells give MAC utilization and, in brackets, latency.

![SRAM](figures/s10-sram.svg)

*Figure S10. MAC utilization of ResNet-50 v1.5 and MobileNetV2 against per-core SRAM on
`client-16c` and `workstation-32c`.*

<!-- generated:s10-sram -->
| SRAM per core (KiB) | `client-16c` ResNet-50 v1.5 | `client-16c` MobileNetV2 | `workstation-32c` ResNet-50 v1.5 | `workstation-32c` MobileNetV2 |
|---|---:|---:|---:|---:|
| 64 | 9.7% (2.14 ms) | 5.0% (0.307 ms) | 13.7% (0.761 ms) | 5.7% (0.134 ms) |
| 128 | 15.4% (1.35 ms) | 5.5% (0.279 ms) | 20.4% (0.511 ms) | 6.2% (0.124 ms) |
| 256 | 22.7% (0.916 ms) | 5.7% (0.270 ms) | 25.6% (0.406 ms) | 6.2% (0.123 ms) |
| 512 | 23.2% (0.895 ms) | 5.7% (0.269 ms) | 25.8% (0.403 ms) | 6.2% (0.123 ms) |
| 1,024 | 23.3% (0.893 ms) | 5.7% (0.269 ms) | 25.8% (0.403 ms) | 6.2% (0.123 ms) |
| 2,048 | 23.3% (0.893 ms) | 5.7% (0.269 ms) | 25.8% (0.403 ms) | 6.2% (0.123 ms) |
| 4,096 | 23.3% (0.893 ms) | 5.7% (0.269 ms) | 25.8% (0.403 ms) | 6.2% (0.123 ms) |
<!-- end:s10-sram -->

<!-- generated:s10-choice -->
| Design point | ResNet-50 v1.5 pick (KiB) | MobileNetV2 pick (KiB) | Point's pick (KiB) |
|---|---:|---:|---:|
| `client-16c` | 256 | 128 | 256 |
| `workstation-32c` | 256 | 64 | 256 |
<!-- end:s10-choice -->

ResNet-50 sets the pick on both points. Its MAC utilization climbs from 9.7% at 64 KiB to 22.7%
at 256 KiB on `client-16c` and from 13.7% to 25.6% on `workstation-32c`, then gains at most 0.6
points up to 4 MiB: 256 KiB is the first size at or above 90% of the highest, 21.0% on
`client-16c`, where 128 KiB gives 15.4%. Equation 4.3 of [architecture](architecture.md) bounds
the tiles by the SRAM, and a tile's operand intensity $ab/(a + b)$ grows with its edge
([design points](design-points.md) section 6).
MobileNetV2 stays between 5.0% and 5.7% at every size, and its picks are 128 and 64 KiB.

[vela-perf]: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/ethosu/vela/npu_performance.py
[regor-perf]: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/ethosu/regor/architecture/ethosu85/ethos_u85_performance.cpp
