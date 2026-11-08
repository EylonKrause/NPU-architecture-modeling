# Numerics

This document specifies the arithmetic of the integer GEMM engine (`npu-sim`, and the
`npu-tlm-demo` and `npu-mobile-demo` platforms built on the same core) and of the
floating-point transformer engine (`npu-transformer-demo`). It covers operand types,
accumulation, requantization, per-channel parameters and their file formats, the compressed
weight formats, the floating-point storage formats, and the oracles that check each of them.
The scope of the model as a whole is stated once, in [methodology](methodology.md), section
"Model fidelity".

| Mechanism | Defining source |
|---|---|
| Parameter structure, rescale and rounding enumerations | `include/npu/config.hpp` (`Requant`, `RescaleMode`, `Rounding`) |
| Accumulation, rescale, weight codecs, fixture generator, output hash | `src/core.cpp` |
| Numerics commit point in the SystemC pipeline | `src/model.cpp` (`Core::compute`) |
| `NPU1` and `NPUQ1` parsers | `src/main.cpp` (`input_problem`, `input_quantization`) |
| Flags, JSON paths and defaults | `src/options.cpp` |
| Python requantization oracle, `quantize_multiplier` | `npuperf/quant.py` |
| Floating-point storage conversion and operators | `src/transformer.cpp` |

## 1 Integer datapath

### 1.1 Operand types and layout

| Operand | Type | Range | Host representation | Simulated storage |
|---|---|---|---|---|
| Activation $A$ | signed INT8 | -128..127 | `int8_t` | 1 byte |
| Weight $W$, `--weight-bits 8` | signed INT8 | -128..127 | `int8_t` | 1 byte (dense) |
| Weight $W$, `--weight-bits 4` | signed INT4 | -8..7 | `int8_t`, sign-extended | 4 bits (dense) |
| Bias $b$ | signed INT32 | $-2^{31}..2^{31}-1$ | `int32_t` | 4 bytes per output channel |
| Activation zero point $z_a$ | signed INT8 | -128..127 | `int32_t` | run scalar |
| Weight zero point $z_w$ | signed INT8 or INT4 | -128..127, or -8..7 with INT4 weights | `int32_t` | per channel (section 3) |
| Output zero point $z_o$ | signed INT8 | -128..127 | `int32_t` | run scalar |
| Accumulator | signed INT32 | $-2^{31}..2^{31}-1$ | `int64_t` while summing | 4 bytes per output element |
| Output $q$ | signed INT8 | -128..127 | `int8_t` | 1 byte |

A run is $G$ independent GEMMs that share the row count $M$ (`--groups`, default 1). $N$ and $K$
are the output channels and the reduction depth of one group. The tensors are row-major:

| Tensor | Shape | Element and flat index |
|---|---|---|
| Activations | $M \times GK$ | $A_{r,gK+j}$ at $rGK + gK + j$ |
| Weights | $G$ blocks of $K \times N$ | $W^{(g)}_{j,n}$ at $(gK + j)N + n$ |
| Bias, channel parameters | $GN$ | channel $c = gN + n$ |
| Accumulators, outputs | $M \times GN$ | $(r, c)$ at $rGN + c$ |

A depthwise convolution is $G$ channels with $N = 1$ (the channel multiplier) and $K = k_h k_w$.
With `shape.conv` the activation matrix is the im2col view of an $H \times W \times C$ NHWC input
tensor ([architecture](architecture.md) section 4.4, `im2col` in `src/conv.cpp`); a kernel tap
in the padding reads $z_a$, so its term $(z_a - z_a)(W - z_w)$ is zero.

### 1.2 Accumulation

Output $(r, c)$ of group $g$ and column $n$, with $c = gN + n$, is

$$a_{r,c} = b_c + \sum_{j=0}^{K-1} \left(A_{r,gK+j} - z_a\right)\left(W^{(g)}_{j,n} - z_{w,c}\right).$$

The sum is carried in INT64 (`reference`, `compute_tile` and `compute_tile_encoded` in
`src/core.cpp`). Intermediate partial sums are unconstrained. The final value must lie in INT32;
otherwise the run fails with `accumulator is outside INT32` and `npu-sim` exits with status 2.

| Property | Statement |
|---|---|
| Wrapping INT32 accumulator | Bit-identical on every accepted input: two's-complement addition is exact modulo $2^{32}$ and the final value fits INT32. |
| Saturating INT32 accumulator | Differs when a partial sum leaves INT32 and the total returns to it; the model has no intermediate saturation. |
| Term magnitude | $\lvert (A - z_a)(W - z_w) \rvert \le 255 \cdot 255 = 65{,}025$ for INT8 weights and $\le 255 \cdot 15 = 3{,}825$ for INT4 weights. |
| Overflow-free depth | With $b_c = 0$, no operand values overflow INT32 for $K \le 33{,}025$ (INT8 weights) or $K \le 561{,}433$ (INT4 weights). |
| Summation order | The reference sums each dot product in $j$ order, the tiled path in K-major outer products, the encoded path panel by panel. Integer addition is associative, so the orders agree exactly and the comparison checks indexing and decoding. |

Numerics commit once per output tile, at its final reduction panel (`Core::compute` in
`src/model.cpp`). Earlier panels charge their compute time and hold partial sums in the slot's
accumulator buffer, which the SRAM reservation sizes at 4 bytes (INT32) per output element plus
1 byte for the INT8 output. A partial sum is never requantized.

The result does not depend on `tile_m`, `tile_n`, `tile_k`, buffer slots, panel buffers, the vector
datapath or the systolic array and its dataflow, the memory model, the core count and partition, or
the timing mode. The depthwise datapath (`arch.compute.depthwise_macs_per_cycle` above 0, used when
$G > 1$ and $N = 1$) places several channels in one tile; each output still reduces over its own
group's $K$ values with its own channel parameters (`compute_channel_tile`, `src/core.cpp`). A run
with `shape.conv` produces the accumulators and outputs of the GEMM on its im2col matrix.

### 1.3 INT4 weights

INT4 weights are signed values in -8..7, held in host memory and in `NPU1` files as sign-extended
INT8. A weight outside -8..7 is rejected when the tensor file is read (`tensor value outside
configured signed precision`) and again when palette or sparse encoding packs it. Weight zero
points, scalar and per channel, must lie in -8..7.

| Format | INT4 storage |
|---|---|
| Dense | $\lceil 4 K_p N_t / 8 \rceil$ bytes per weight panel; each panel pads its own odd final nibble. Dense panels are not materialized as byte buffers: the byte count drives traffic, cache occupancy and storage statistics. |
| Sparse | Payload values packed two per byte, element $i$ in bits $4(i \bmod 2)$ to $4(i \bmod 2) + 3$, low nibble first (section 4.1). |
| Palette | LUT entries are INT8 bytes; the index width (`--palette-bits`) is independent of the weight width. |

### 1.4 Output stage

With $\rho$ the rescale of section 2,

$$q_{r,c} = \operatorname{clamp}\left(\rho(a_{r,c}) + z_o,\ -128,\ 127\right),$$

and with `--relu`

$$q_{r,c} = \operatorname{clamp}\left(\max\left(\rho(a_{r,c}) + z_o,\ z_o\right),\ -128,\ 127\right) = \operatorname{clamp}\left(\rho(a_{r,c}) + z_o,\ z_o,\ 127\right).$$

The addition, ReLU and clamp evaluate in INT64 (`requantize_unchecked` in `src/core.cpp`), so a
native rescale with shift 0, whose magnitude reaches $2^{31}(2^{31}-1)$, saturates without
overflow. The output zero point and ReLU are run scalars.

### 1.5 Synthetic tensors and defaults

Without `--input`, tensors come from a 32-bit linear congruential generator (`make_problem` in
`src/core.cpp`). The state starts at the low 32 bits of `--seed`; each draw advances the state
and returns it:

$$s_{i+1} = \left(1664525 s_i + 1013904223\right) \bmod 2^{32}.$$

| Order | Tensor | Value from draw $s$ | Range |
|---|---|---|---|
| 1 | All activations, flat order; with `shape.conv`, the NHWC input tensor | $(s \bmod 31) - 15$ | -15..15 |
| 2 | All weights, INT8 | $(s \bmod 31) - 15$ | -15..15 |
| 2 | All weights, INT4 | $(s \bmod 16) - 8$ | -8..7 |
| 3 | All biases | $(s \bmod 127) - 63$ | -63..63 |

Default parameters: $z_a = -3$, $z_w = 2$, $z_o = 0$, multiplier 1, shift 7, `--rescale native`,
`--rounding away`, ReLU off.

## 2 Requantization

### 2.1 Modes

All modes scale the INT32 accumulator by $\sigma = \mu / 2^S$ for a multiplier $\mu$ and right shift
$S$, then apply the output stage of section 1.4. `--rescale` selects the rounding.

| `--rescale` | Multiplier $\mu$ | Shift $S$ | Accumulator domain | Rounding | Definition |
|---|---|---|---|---|---|
| `native` (default) | $0..2^{31}-1$ | 0..31 | INT32 | Magnitude to nearest; ties per `--rounding away\|even` | `native_scale`, `src/core.cpp` |
| `tosa-single` | $0..2^{31}-1$ | 2..62 | $[-2^{S-1}, 2^{S-1})$ | Half toward $+\infty$ | TOSA 1.0.2 `apply_scale_32`, `SINGLE_ROUND` |
| `tosa-double` | $0..2^{31}-1$ | 2..62 | $[-2^{S-1}, 2^{S-1})$ | Half up at bit 31, then half away from zero | TOSA 1.0.2 `apply_scale_32`, `DOUBLE_ROUND` |

`--rounding even` with a TOSA mode is rejected (`rounding even applies to native rescale only`).
Shifts, scalar and per channel, are signed 32-bit fields; a shift outside the mode's range is
rejected with the range in the message (`requantization shift must be in 2..62 for tosa-double
rescale`).

### 2.2 Native

For $S = 0$, $\rho(a) = a\mu$ exactly. For $S \ge 1$,

$$\rho_{\text{native}}(a) = \operatorname{sgn}(a\mu) \cdot \operatorname{round}_t\left(\frac{\lvert a\mu \rvert}{2^S}\right),$$

where $\operatorname{round}_t$ rounds to the nearest integer and breaks ties away from zero
(`away`, $\lfloor (\lvert a\mu \rvert + 2^{S-1}) / 2^S \rfloor$) or to the even quotient (`even`).
The product is formed in INT64; $\lvert a\mu \rvert \le 2^{31}(2^{31}-1) < 2^{62}$.

On the shift range 2..31 that both admit, `native` with `away` and `tosa-single` agree on every
accumulator except negative ties: both round to nearest and round positive ties up, and native
rounds negative ties down where TOSA rounds them up.

Native mode cannot hold a normalized Q31 multiplier for a scale below $2^{-1}$, because
$\mu \ge 2^{30}$ and $S \le 31$ give $\sigma \ge 2^{-1}$. A scale of 0.0019 normalizes to
$(\mu, S) = (2089072093, 40)$ with 31 significant multiplier bits; at $S = 31$ native mode holds it
as $\mu = 4080219$, 22 significant bits. TFLite and TOSA parameter pairs therefore load bit-exactly
into the TOSA modes only. Native mode remains the default for existing configurations and is the
mode the ONNX importer emits: it accepts only float32 scale ratios that are exactly $\mu/2^S$ with
$\mu \le 2^{31}-1$ and $S \le 31$, and selects `even` (`exact_rescale` in `npuperf/onnx_import.py`).

### 2.3 TOSA modes

`tosa-single` and `tosa-double` implement `apply_scale_32` of TOSA 1.0.2
(`pseudocode/library/scale_helpers.tosac`). The parameters must satisfy

$$0 \le \mu \le 2^{31}-1, \qquad 2 \le S \le 62, \qquad -2^{S-1} \le a < 2^{S-1},$$

and the rescale is an arithmetic right shift of a rounded INT64 product:

$$\rho(a) = \left\lfloor \frac{a\mu + \delta}{2^S} \right\rfloor, \qquad \delta_{\text{single}} = 2^{S-1}, \qquad \delta_{\text{double}} = 2^{S-1} + [S > 31] \cdot \operatorname{sgn}^{+}(a) \cdot 2^{30},$$

with $\operatorname{sgn}^{+}(a) = 1$ for $a \ge 0$ and $-1$ for $a < 0$, and $[S > 31]$ equal to 1
when $S > 31$ and 0 otherwise. `tosa-double` equals `tosa-single` for $S \le 31$. The floor is
computed without right-shifting a negative value (`floor_shift`, `src/core.cpp`), since C++17 leaves
that shift implementation-defined.

| Consequence | Value |
|---|---|
| Accumulator outside $[-2^{S-1}, 2^{S-1})$ | Run fails: `accumulator 8 is outside [-8, 8), the apply_scale_32 input range for shift 4` |
| Shifts admitting every INT32 accumulator | $S \ge 32$ |
| Magnitude of $\rho(a)$ | At most $2^{30}$, so adding $z_o$ cannot overflow INT32 |
| Normalized scale range at shift $S$ | $\mu \in [2^{30}, 2^{31})$ gives $\sigma \in [2^{30-S}, 2^{31-S})$; $S = 62$ reaches $2^{-32}$ |
| Scale 1.0 | $\mu = 2^{30}$, $S = 30$ |

The integer pipeline maps onto TOSA as a MAC stage that subtracts $z_a$ and $z_w$, followed by
`RESCALE` with `scale32 = true`, `in_t = i32`, `out_t = i8`, `input_zp = 0`, `output_zp` equal to
$z_o$, and `per_channel = true` when a multiplier or shift vector is present. The model's ReLU
equals a following `CLAMP` with `min_val` equal to $z_o$ and `max_val = 127`, by the identity in
section 1.4.

TOSA 1.0.2 `RESCALE.tosac` contains an erratum: its signed branch reads
`value = sign_extend<int48_t>(value)`. The specification's `main` branch reads
`sign_extend<int48_t>(in_value)`. The model implements the corrected form; with `input_zp = 0`
the accumulator is `in_value`.

### 2.4 Equivalence with TFLite, gemmlowp and CMSIS-NN

TFLite `MultiplyByQuantizedMultiplier` and CMSIS-NN `arm_nn_requantize` take the same Q31
multiplier and a left-shift exponent $L$, negative for a right shift. The parameter mapping is

$$\mu_{\text{TOSA}} = \mu_{\text{TFLite}}, \qquad S_{\text{TOSA}} = 31 - L_{\text{TFLite}}.$$

| Model mode | TFLite (`common.cc`) | CMSIS-NN (`arm_nnsupportfunctions.h`) |
|---|---|---|
| `tosa-single` | Built with `TFLITE_SINGLE_ROUNDING`: $\lfloor (x\mu + 2^{30-L}) / 2^{31-L} \rfloor$ | Built with `CMSIS_NN_USE_SINGLE_ROUNDING`: $((x\mu \gg (30-L)) + 1) \gg 1$ |
| `tosa-double` | Default build: gemmlowp `SaturatingRoundingDoublingHighMul` of $x \cdot 2^{\max(L,0)}$ and $\mu$, then `RoundingDivideByPOT` by $\max(-L, 0)$ | Default C branch: `arm_nn_doubling_high_mult_no_sat`, then `arm_nn_divide_by_power_of_two` |

| Library parameter | Documented domain | TOSA shift |
|---|---|---|
| TFLite single-rounding `shift` | -31..30 (`TFLITE_DCHECK`) | 1..62; the TOSA modes cover $L$ = -31..29 |
| TFLite double-rounding right shift | 0..31 (`RoundingDivideByPOT` assertion) | at most 62 |
| CMSIS-NN `shift` | -31..30 | 1..62 |

Two facts connect the double-rounding forms. `SaturatingRoundingDoublingHighMul` computes
$\lfloor (ab + 2^{30}) / 2^{31} \rfloor$ for both signs, the computation of the NEON `VQRDMULH`
instruction, and so rounds half toward $+\infty$ at bit 31; its saturating case
$a = b = -2^{31}$ is unreachable with $\mu \ge 0$. `RoundingDivideByPOT` rounds half away from zero.
For $L > 0$ the default TFLite path multiplies $x$ by $2^L$ in INT32; the TOSA input range gives
$\lvert x \cdot 2^L \rvert \le 2^{30}$, so the product is exact.

The equivalences were verified by a harness that compiles verbatim transcriptions of the four
published bodies (g++ 13, `-O2 -std=c++17`) over 20,000,000 random cases: $\mu$ uniform in
$[2^{30}, 2^{31})$, TFLite shift in -31..7 (TOSA shift 24..62), accumulators restricted to each
path's documented input range.

| Comparison | Mismatches |
|---|---:|
| TOSA `SINGLE_ROUND` against TFLite `TFLITE_SINGLE_ROUNDING` | 0 |
| TOSA `DOUBLE_ROUND` against TFLite default (gemmlowp), 16,952,803 right-shift cases included | 0 |
| CMSIS-NN `CMSIS_NN_USE_SINGLE_ROUNDING` against TFLite single rounding | 0 |
| CMSIS-NN default against TFLite default | 0 |
| TFLite single rounding against TFLite default | 273,968 (1.37%) |

The repository's tests repeat the comparison against the model (section 6.2).

### 2.5 Tie behavior

Exact half-integer scaled values, computed through `apply_scale_32` with $\mu = 2^{30}$ at $S = 31$
and $S = 32$ (`tosa-*`) and through `requantize` with $\mu = 1$, $S = 1$ (`native`):

| $a\mu / 2^S$ | `native` `away` | `native` `even` | `tosa-single` | `tosa-double`, $S \le 31$ | `tosa-double`, $S > 31$ |
|---:|---:|---:|---:|---:|---:|
| -2.5 | -3 | -2 | -2 | -2 | -3 |
| -1.5 | -2 | -2 | -1 | -1 | -2 |
| -0.5 | -1 | 0 | 0 | 0 | -1 |
| 0.5 | 1 | 0 | 1 | 1 | 1 |
| 1.5 | 2 | 2 | 2 | 2 | 2 |
| 2.5 | 3 | 2 | 3 | 3 | 3 |

Double rounding also departs from the nearest integer away from ties, as rows 1 and 4 of the
golden vectors show.

### 2.6 Golden vectors

These four vectors come from the harness of section 2.4. `tests/core_tests.cpp` checks them
through `apply_scale_32`, and `tests/quantization_tests.py` checks them through the TFLite and
CMSIS-NN formulations and through `npu-sim`, where the INT8 clamp maps 1209998 and 1209999 to 127.

| $x$ | $\mu$ | TFLite shift | TOSA shift | $x\mu / 2^S$ | Single | Double |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1861152495 | -1 | 32 | 1.300000000 | 1 | 2 |
| -6 | 1073741824 | -1 | 32 | -1.500000000 | -1 | -2 |
| -3 | 1073741824 | 0 | 31 | -1.500000000 | -1 | -1 |
| 5038971 | 2062684434 | -2 | 33 | 1209998.392 | 1209998 | 1209999 |

In rows 1 and 4 the single-rounding result is the correctly rounded one. Rows 2 and 3 separate
negative ties: single rounding gives -1, double rounding with a right shift beyond bit 31 gives -2,
and double rounding at $S = 31$ gives -1 because `SaturatingRoundingDoublingHighMul` rounds half up.

### 2.7 Multiplier and shift from a scale

`npuperf.quant.quantize_multiplier(scale)` follows TFLite `QuantizeMultiplier`
(`quantization_util.cc`): $\sigma = f \cdot 2^e$ with $f \in [0.5, 1)$ by `frexp`,
$\mu = \operatorname{round}(f \cdot 2^{31})$ with halves away from zero, $\mu = 2^{31}$ renormalized
to $2^{30}$ with $e + 1$, and $S = 31 - e$. Scale 0 returns $(0, 31)$; a shift outside 2..62 raises
`ValueError`. `tests/npuperf_tests.py` requires $\mu \in [2^{30}, 2^{31})$ and a relative error of
$\mu/2^S$ below $2^{-30}$.

| Scale | $\mu$ | $S$ |
|---:|---:|---:|
| 1.0 | 1073741824 | 30 |
| 0.75 | 1610612736 | 31 |
| 0.5 | 1073741824 | 31 |
| 0.0019 | 2089072093 | 40 |
| $2^{-32}$ | 1073741824 | 62 |

## 3 Per-channel parameters and file formats

### 3.1 Resolution and validation

The weight zero point, multiplier and shift of channel $c$ each come from a scalar or from a
vector of exactly $GN$ entries. The three vectors are independently optional; a channel takes the
scalar for each absent vector. The index is the global channel $c = gN + n$, whatever tile covers
it. $z_a$, $z_o$, ReLU, the rescale mode and the tie rule are run scalars.

| Field | Scalar flag, JSON path | Vector JSON path | Rule |
|---|---|---|---|
| Weight zero point | `--weight-zero`, `quant.weight_zero` | `quant.weight_zero_points` | INT8; -8..7 with `--weight-bits 4` |
| Multiplier | `--multiplier`, `quant.multiplier` | `quant.multipliers` | 0..2147483647 |
| Shift | `--shift`, `quant.shift` | `quant.shifts` | 0..31 `native`, 2..62 `tosa-*` |
| Activation zero point | `--input-zero`, `quant.input_zero` | none | INT8 |
| Output zero point | `--output-zero`, `quant.output_zero` | none | INT8 |
| ReLU | `--relu`, `quant.relu` | none | boolean |
| Rescale mode | `--rescale`, `quant.rescale` | none | `native\|tosa-single\|tosa-double` |
| Native tie rule | `--rounding`, `quant.rounding` | none | `away\|even` |

A vector of the wrong length fails with `per-channel ... count must equal G*N`. A run takes the
vectors from a config file (`--config`) or from `--quantization-file`; supplying both is rejected.
Every result echoes the effective vectors in its `config` object, so the result file reproduces
the run.

### 3.2 `NPU1` tensor file

An `NPU1` file (`--input`) supplies the shape and every tensor. Tokens are separated by
whitespace, line breaks carry no meaning, and each token is at most 32 characters. Integers are
decimal with an optional leading `-` (`std::from_chars`).

```text
NPU1 M N K
<M*G*K activations, row-major M x (G*K)>
<G*K*N weights: G blocks of K x N, each row-major>
<G*N biases>
```

$G$ comes from `--groups`; the header holds $M$, $N$ and $K$. `--m`, `--n` and `--k` are rejected
alongside `--input`, and a config file may restate the shape only when it agrees with the header.
Activations must lie in -128..127, weights in the range of `--weight-bits`, biases in INT32. A
missing value (`truncated INT8 tensor`, `truncated bias tensor`) or a trailing token is an error.

### 3.3 `NPUQ1` parameter file

An `NPUQ1` file (`--quantization-file`) sets all three per-channel vectors:

```text
NPUQ1 C
z_w(0) M(0) S(0)
...
z_w(C-1) M(C-1) S(C-1)
```

$C$ must equal $GN$. Each row holds one channel's weight zero point, multiplier and shift as
signed decimal INT32 tokens, validated against section 3.1 for the selected mode. A TOSA-mode file
with two channels at $\sigma = 0.0019$ and $\sigma = 1$:

```text
NPUQ1 2
0 2089072093 40
0 1073741824 30
```

```sh
npu-sim --input layer.npu --quantization-file layer.npuq --rescale tosa-single --dump-output
```

### 3.4 Parameter records in external memory

The bias and the present per-channel vectors of output channel $c$ travel as one INT32 record
per channel in the parameter region of [architecture](architecture.md) section 5.1, which
gives its layout and address (`param_record_bytes` and `panel_read_runs` in `src/core.cpp`).

### 3.5 Graph frontend scale check

The Python graph frontend (`quantization` in `npuperf/graph.py`) accepts an accelerated node only
when its integer parameters represent its declared tensor scales. For activation scale $s_a$,
weight scale $s_{w,c}$ and output scale $s_o$:

$$\left\lvert \frac{s_a s_{w,c}}{s_o} - \frac{\mu_c}{2^{S_c}} \right\rvert \le 2^{-(S_c+1)} + 10^{-12} \left\lvert \frac{s_a s_{w,c}}{s_o} \right\rvert.$$

A bias must be INT32 with zero point 0 and scale $s_a s_{w,c}$ to a relative tolerance of
$10^{-12}$.

## 4 Weight compression

`--weight-encoding palette|sparse` stores each weight panel in a lossless byte format. The
encoded buffer is the only weight source of the compressed datapath (`compute_tile_encoded`);
the expanded `Problem::weights` feed the reference only.

### 4.1 Panels, element order and bit packing

The encoding unit is one weight panel: group $g$, a column panel starting at column $n_0$ of
width $N_t$ equal to the smaller of `tile_n` and $N - n_0$, and a reduction panel starting at
row $k_0$ of depth $K_p$ (`--tile-k`, or $K$ when it is 0, shorter at the end of $K$). Element $i$
of the panel is weight row $k_0 + \lfloor i / N_t \rfloor$, column $n_0 + (i \bmod N_t)$:
K-major, column-minor. A panel holds $E = K_p N_t$ elements. Panels are laid out back to back from
`0x9000_0000` in encoding order: group, then column panel, then reduction panel.

| Rule | Value |
|---|---|
| Multi-byte header fields | Little-endian |
| Field $i$ of width $w$ | Byte $\lfloor iw / 8 \rfloor$, from bit $iw \bmod 8$ upward; $w \in \lbrace 1, 2, 4, 8 \rbrace$ divides 8, so no field crosses a byte |
| Unused high bits of a final byte | Zero |
| Signed packed values | $w$-bit two's complement |

### 4.2 Palette format

| Offset | Bytes | Field |
|---|---|---|
| 0 | 2 | $n_{\mathrm{LUT}}$, LUT entries (uint16), $1 \le n_{\mathrm{LUT}} \le 2^{b_p}$ |
| 2 | 1 | $b_p$, index bits: 1, 2, 4 or 8 |
| 3 | 1 | Reserved, 0 |
| 4 | $n_{\mathrm{LUT}}$ | LUT: signed INT8 weight values in order of first occurrence |
| $4 + n_{\mathrm{LUT}}$ | $\lceil E b_p / 8 \rceil$ | Indices, $b_p$ bits each, in element order |

$$\mathrm{Bytes}_{\text{palette}} = 4 + n_{\mathrm{LUT}} + \left\lceil \frac{E b_p}{8} \right\rceil$$

LUT entries are stored weight values before zero-point subtraction, so one LUT serves the
$N_t$ channels of the panel, each with its own zero point. A panel with more than $2^{b_p}$
distinct values fails to encode
(`lossless palette capacity exceeded; increase --palette-bits or use dense`). With $b_p = 8$
every panel encodes; a panel holding all 256 INT8 values has $n_{\mathrm{LUT}} = 256$, header
bytes `00 01`.

### 4.3 Sparse format

| Offset | Bytes | Field |
|---|---|---|
| 0 | 2 | $B_s$, block length (uint16), 1..4096 |
| 2 | 1 | $b_w$, source weight bits: 4 or 8 |
| 3 | 1 | Reserved, 0 |
| 4 | $\lceil \beta / 8 \rceil$ | Bitmap of $\beta = \lceil E / B_s \rceil$ bits; bit $j$ set when block $j$ is retained |
| $4 + \lceil \beta / 8 \rceil$ | $\lceil E_r b_w / 8 \rceil$ | Every element of every retained block, in element order, $b_w$ bits each |

$$\mathrm{Bytes}_{\text{sparse}} = 4 + \left\lceil \frac{\beta}{8} \right\rceil + \left\lceil \frac{E_r b_w}{8} \right\rceil$$

Block $j$ covers elements $jB_s$ to $\min(E, (j+1)B_s) - 1$ in element order, so a block can span
columns and reduction rows. A block is retained when any of its elements differs from that
element's own channel weight zero point; $E_r$ is the number of elements in retained blocks. An
omitted block contributes exactly zero after zero-point subtraction, under asymmetric per-channel
zero points as well. Inside retained blocks, an element equal to its zero point skips its MACs;
$\nu$ denotes the panel's count of elements that differ from their zero point.

### 4.4 Worked example

One panel: $K = 4$, $N = 2$, `--tile-n 2`, `--tile-k 0`, weights in row-major $K \times N$ order
`3 -1 / 3 0 / -1 0 / 3 5`, so the element order is `3 -1 3 0 -1 0 3 5`. The buffers come from
`encode_weights`. With $z_w = (3, 0)$ the centered elements are `0 -1 0 0 -4 0 0 5`; block 1
(`3 0`, centered `0 0`) is omitted, so $E_r = 6$ and $\nu = 3$.

| Encoding | Parameters | Buffer (hex) | Bytes | Dense bytes |
|---|---|---|---:|---:|
| Palette | $b_p = 2$; LUT `3 -1 0 5`; indices `0 1 0 2 1 2 0 3` | `04 00 02 00 03 FF 00 05 84 C9` | 10 | 8 |
| Sparse, INT8 | $B_s = 2$, $z_w = (3, 0)$, bitmap `0b1101` | `02 00 08 00 0D 03 FF FF 00 03 05` | 11 | 8 |
| Sparse, INT4 | as above, `--weight-bits 4` | `02 00 04 00 0D F3 0F 53` | 8 | 4 |

The counters below come from `npu-sim` with one activation row `1 2 3 4`, $z_a = 0$, and an
`NPUQ1` file setting $z_w = (3, 0)$, multiplier 1 and shift 0. Every run produces accumulators
`-12 19` with `numerical_match: true`.

| Run | `weight_read_bytes` | `executed_macs` | `skipped_macs` | `weight_decode_cycles` | `decoded_weight_values` | `compute_busy_cycles` |
|---|---:|---:|---:|---:|---:|---:|
| Dense INT8 | 8 | 8 | 0 | 0 | 0 | 5 |
| Dense INT4 | 4 | 8 | 0 | 0 | 0 | 5 |
| Palette, $b_p = 2$ | 10 | 8 | 0 | 1 | 8 | 5 |
| Sparse INT8, $B_s = 2$ | 11 | 3 | 5 | 2 | 6 | 6 |
| Sparse INT4, $B_s = 2$ | 8 | 3 | 5 | 2 | 6 | 6 |

### 4.5 Decode and compute cycles

[Architecture](architecture.md) section 3.7 specifies the decoder timing: with decoder rate
$V$ (`--weight-decode-values-per-cycle`), a palette panel decodes in $\lceil E / V \rceil$
cycles and a sparse panel in $\lceil \beta / V \rceil + \lceil E_r / V \rceil$, and the decoder
runs ahead of the MAC datapath, so a compressed panel takes the longer of its decode and MAC
times (equation 3.1 of architecture). A sparse panel's MAC time is $\lceil M_t \nu / P \rceil$:
sparse encoding requires the vector datapath, and with `--array-rows` and `--array-cols` it is
rejected (`sparse zero-skipping is supported only by the vector datapath`).

The weight cache stores encoded panels at their encoded byte length; a hit removes the panel's
DMA bytes and leaves its decode work in place. Study S7 in [results](results.md) measures LLM
decode throughput against the decoder rate.

### 4.6 Size bounds

Encoded sizes depend on the weight values. The SRAM reservation and the pre-run validation use the
worst case per panel of $E$ elements (`encoded_weight_bound`, `src/core.cpp`):

| Encoding | Upper bound on panel bytes |
|---|---|
| Dense | $\lceil E b_w / 8 \rceil$ |
| Palette | $4 + \min(E, 2^{b_p}) + \lceil E b_p / 8 \rceil$ |
| Sparse | $4 + \lceil \lceil E / B_s \rceil / 8 \rceil + \lceil E b_w / 8 \rceil$ |

Both codecs can exceed dense storage, as the worked example shows. `--timing-only` accepts dense
weights only, because compressed sizes and decode cycles require the weight values.

### 4.7 Statistics

| JSON field | Definition |
|---|---|
| `macs` | Logical workload, $GMNK$ |
| `executed_macs` | MACs performed: `macs` for dense and palette; for sparse, $M_t \nu$ summed over every computed panel |
| `skipped_macs` | `macs` minus `executed_macs` |
| `weight_decode_cycles` | Decoder occupancy $C_{\mathrm{dec}}$ summed over every computed panel |
| `decoded_weight_values` | $E$ (palette) or $E_r$ (sparse) summed over every computed panel |
| `weight_read_bytes` | Encoded weight bytes moved by DMA after weight-cache hits |
| `encoded_weight_storage_bytes` | Bytes of the unique encoded panels, headers, LUTs and bitmaps included |
| `dense_weight_storage_bytes` | The same panels stored dense at `--weight-bits`, per-panel padding included |

### 4.8 Decoder checks

The decoder (`visit_encoded`, `src/core.cpp`) rejects a buffer shorter than its header, a nonzero
reserved byte, a header whose bit width or block length disagrees with the configuration, a
palette with $n_{\mathrm{LUT}} = 0$ or $n_{\mathrm{LUT}} > 2^{b_p}$, a buffer length that
disagrees with its header and contents, a palette index at or above $n_{\mathrm{LUT}}$, and a
sparse bitmap that disagrees with its payload.

## 5 Floating-point path of the transformer engine

### 5.1 Storage formats

`npu-transformer-demo --precision fp32|fp16|bf16` (default `fp16`) selects the storage format of
every materialized tensor. Arithmetic is IEEE 754 binary32 in all three.

| `--precision` | Format | Sign, exponent, fraction bits | Storage bytes |
|---|---|---|---:|
| `fp32` | binary32 | 1, 8, 23 | 4 |
| `fp16` | binary16 | 1, 5, 10 | 2 |
| `bf16` | bfloat16 | 1, 8, 7 | 2 |

### 5.2 Conversion

Rounding binary32 to the storage format operates on bit patterns with integer arithmetic
(`to_half`, `to_bfloat` in `src/transformer.cpp`), independent of the host floating-point
rounding mode. Expansion back to binary32 is exact.

| Case | `fp16` | `bf16` |
|---|---|---|
| Finite values | Round to nearest, ties to even | Round to nearest, ties to even |
| Overflow | Signed infinity at and above 65520 in magnitude | Signed infinity at and above `0x7F7F8000` in magnitude, the tie above the largest finite value |
| Results below the normal range | binary16 subnormals, rounded to nearest even | Same exponent range as binary32; subnormals kept |
| binary32 subnormal input | Signed zero | Rounded to nearest even |
| Signed zero, infinity | Preserved | Preserved |
| NaN | Quiet NaN, sign preserved | Quiet NaN, sign preserved |

`--round-f32-bits WORDS` (comma-separated decimal binary32 words, or `-` for stdin) rounds each
word to `--precision`, prints the results as binary32 words and exits. Output of the probe:

| Input word | Value | `fp16` result | `bf16` result |
|---|---|---|---|
| `0x3F801000` | $1 + 2^{-11}$ | `0x3F800000` (tie to even) | `0x3F800000` |
| `0x3F808000` | $1 + 2^{-8}$ | `0x3F808000` (exact) | `0x3F800000` (tie to even) |
| `0x3F818000` | $1 + 3 \cdot 2^{-8}$ | `0x3F818000` (exact) | `0x3F820000` (tie to even) |
| `0x33000000` | $2^{-25}$ | `0x00000000` (tie to even) | `0x33000000` |
| `0x33800000` | $2^{-24}$ | `0x33800000` (smallest binary16 subnormal) | `0x33800000` |
| `0x477FE000` | 65504 | `0x477FE000` (largest binary16) | `0x47800000` |
| `0x477FF000` | 65520 | `0x7F800000` (infinity) | `0x47800000` |
| `0x007FFFFF` | Largest binary32 subnormal | `0x00000000` | `0x00800000` |
| `0x7F7F7FFF` | Below the bfloat16 tie | `0x7F800000` | `0x7F7F0000` |
| `0x7F7F8000` | bfloat16 tie | `0x7F800000` | `0x7F800000` |
| `0x7FC00001` | NaN with payload | `0x7FC00000` | `0x7FC00000` |

### 5.3 Arithmetic and rounding points

Each multiplication, addition and division is a separate binary32 operation (`add`, `mul`,
`div32` store through `volatile`), so the compiler cannot fuse them into multiply-adds. A dot
product is one sequential chain from $+0$ over the reduction index $0..K-1$; streaming the weights
in column panels does not change the order. Square roots are binary32. Exponentials, cosines,
sines and the RoPE frequency power evaluate in binary64 through the host C library and round once
to binary32.

| Operator | Evaluated in binary32 | Rounded to the storage format |
|---|---|---|
| Projections Q, K, V, O, gate, up, down | Full reduction | Each output element |
| RMSNorm, $\epsilon = 10^{-5}$ | Sum of squares, mean, $\epsilon$ addition, square root, reciprocal, both products | Each output element; $\gamma$ stored |
| RoPE, base 10000, adjacent pairs | Rotation; angle, cosine and sine in binary64, then binary32 | Each rotated element |
| Attention scores | $QK^\top$ dot products, scale $1/\sqrt{d_{\text{head}}}$ | Never: FP32 score buffer |
| Softmax | Maximum subtraction, sum, division; exponential in binary64, then binary32 | Never: FP32 probabilities |
| Attention output | Probability-weighted sum of V | Each output element |
| SwiGLU | $g / (1 + e^{-g})$ and the product with up; $e^{-g}$ in binary64, then binary32 | Each product |
| Residual add | Addition | Each output element |
| KV cache | Storage only | Keys after RoPE and values, held in DRAM at the storage format |

Requests must hold finite inputs; a non-finite RMSNorm reduction or a non-finite output fails the
request. The fixture values are exact in all three formats: inputs $i/16$ with $i \in [-18, 18]$,
weights $i / (8 \cdot 2^e)$ with $i \in [-15, 15]$ and $e \ge 3$ set by the reduction depth, and
$\gamma = 1 + i/32$ with $i \in [-4, 4]$.

## 6 Verification

### 6.1 In-run checks

| Executable | Check | Failure |
|---|---|---|
| `npu-sim` | Every bit-exact run computes the scalar `reference` and compares all accumulators and INT8 outputs with the simulated datapath | `numerical validation failed`, exit status 2 |
| `npu-sim --timing-only` | No tensors or reference; the result reports `"numerics": "skipped"`, and `numerical_match` and `output_hash` are null | none |
| `npu-tlm-demo` | Accumulators and outputs read back from simulated memory over TLM equal `reference` | `DMA result differs from the scalar reference`, exit status 2 |
| `npu-mobile-demo` | Each job's `compute_tile` output equals `reference`, reported per job as `numerical_match` | exit status 2 after the report; `--inject-output-fault` exercises it |
| `npu-transformer-demo` | Batched prefill and incremental decode outputs equal an untimed one-row-at-a-time replay of the same model, bit for bit | exit status 2; `--inject-output-fault ROW` exercises it |

A bit-exact `npu-sim` result carries `"numerics": "bit-exact"` and `"numerical_match": true`. The
transformer replay detects batching and prefill/decode partition errors; the Python oracle of
section 6.2 checks the values.

`output_hash` is FNV-1a 64 (offset basis 14695981039346656037, prime 1099511628211) over every
row-major INT32 accumulator as four little-endian bytes, followed by every row-major INT8 output as
its unsigned byte, with no separators or length prefixes (`output_hash`, `src/core.cpp`).

### 6.2 Oracles

The Python oracles are written from published definitions or from first principles and import no
simulator code. The C++ tests check hand-derived vectors and compare the encoded datapath with the
scalar reference.

| CTest | Source | Oracle | Coverage |
|---|---|---|---|
| `core.arithmetic` | `tests/core_tests.cpp` | Hand-derived vectors | Native ties under both rules; 15 `apply_scale_32` vectors (the golden vectors, ties at $S = 31$ and $S = 32$, INT32 and multiplier extremes, $S = 2$); every `REQUIRE` clause; per-channel shifts; INT4; final-accumulator overflow; the fixture generator and FNV-1a against golden values |
| `core.weight-codec` | `tests/mobile_compression_core_tests.cpp` | Reference on unmodified weights | Encoded results unchanged after the expanded weights are overwritten; corrupted headers, lengths, bit widths, offsets and retained counts rejected; a 256-entry INT8 palette |
| `sim.gemm-oracle` | `tests/integration.py` | Python integers | Fixture generation, accumulation, requantization, saturation and ReLU; external tensor files with extreme operands |
| `sim.quantization` | `tests/quantization_tests.py` | TFLite (`common.cc`), gemmlowp (`fixedpoint.h`) and CMSIS-NN (`arm_nnsupportfunctions.h`), each transcribed from its source | Golden vectors through all four formulations and `npu-sim`; 2,400 random in-range, tie, extreme and small-shift cases through `npu-sim` in both TOSA modes; TFLite against CMSIS-NN in both roundings; per-channel `NPUQ1` files; INT4; grouped and depthwise GEMMs at three tilings |
| `sim.weight-compression` | `tests/mobile_compression_tests.py` | Python model of encoded size, $\nu$, $R$ and decode cycles | Every palette width with INT4 and INT8 sources, asymmetric zero points, odd INT4 panels, empty sparse payloads, decoder overlap, cache accounting, timing-mode equivalence, capacity rejection |
| `tools.npuperf` | `tests/npuperf_tests.py` | Transcription of TOSA `apply_scale_32` | The TFLite formulation in `npuperf.quant` against TOSA at the edges of every shift 2..62 and on 20,000 random cases; `quantize_multiplier` round trip |
| `soc.transformer` | `tests/transformer_tests.py` | Python float model: `struct` binary16, bfloat16 by nearest-neighbor search, binary64 operations rounded to binary32 | Full block in all three precisions and head groupings; every finite binary16 encoding round-trips; 22 edge words and 1,800 random binary32 words through the probe |

The Python transformer oracle evaluates each operation in binary64 and rounds it to binary32. For
one addition, multiplication, division or square root of binary32 operands this equals the
correctly rounded binary32 result, because binary64 carries at least $2 \cdot 24 + 2$ significand
bits. The comparison is therefore exact.

## 7 References

| Source | Version | Location |
|---|---|---|
| TOSA specification | 1.0.2 | https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html |
| TOSA `apply_scale_32` | tag v1.0.2 | https://github.com/arm/tosa-specification/blob/v1.0.2/pseudocode/library/scale_helpers.tosac |
| TOSA `RESCALE` | tag v1.0.2; corrected on `main` | https://github.com/arm/tosa-specification/blob/v1.0.2/pseudocode/operators/RESCALE.tosac |
| TOSA rounding modes and profile extensions | tag v1.0.2 | https://github.com/arm/tosa-specification/blob/v1.0.2/tosa.xml |
| gemmlowp `fixedpoint.h` | commit 13d57703 | https://github.com/google/gemmlowp/blob/13d57703abca3005d97b19df1f2db731607a7dc2/fixedpoint/fixedpoint.h |
| TFLite `common.cc` | commit 9e303fdb | https://github.com/tensorflow/tensorflow/blob/9e303fdb714aa6c1e83637dbc20bfcd1291f768b/tensorflow/lite/kernels/internal/common.cc |
| TFLite `common.h` | commit d46c3747 | https://github.com/tensorflow/tensorflow/blob/d46c3747aeec5b061c6dbc167306f73ae7b23ee3/tensorflow/lite/kernels/internal/common.h |
| TFLite `quantization_util.cc` | commit 9af339bd | https://github.com/tensorflow/tensorflow/blob/9af339bdcf5767dc00b04b292733f30718e43f70/tensorflow/lite/kernels/internal/quantization_util.cc |
| CMSIS-NN `arm_nnsupportfunctions.h` | file revision V.22.14.0, commit 9a88efea | https://github.com/ARM-software/CMSIS-NN/blob/9a88efea746b98f54d7674835b43b6c32e5a6f6b/Include/arm_nnsupportfunctions.h |
