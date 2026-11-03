# Stateful mobile transformer block

`npu-transformer-demo` executes an actual floating-point decoder block inside a
persistent SystemC engine. It adds a workload that the integer dense/convolution
pipeline cannot represent: stateful autoregressive attention with vector
operators and floating-point storage.

This is a small, reproducible architecture experiment. Its parameters are
deterministic test fixtures, not trained weights. It does not tokenize text,
produce useful language, emulate the Apple Neural Engine instruction set, or
predict the latency of any commercial processor. The simulator's abstract
architecture cycle is represented by one nanosecond of SystemC time; that choice
does not establish a physical clock frequency.

## Run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
build/npu-transformer-demo --dump-output --verify-lifecycle
build/npu-transformer-demo --precision bf16 --heads 4 --kv-heads 1 \
  --prefill 8 --decode 4 --context 32 --dump-output \
  --vcd results/transformer --progress-jsonl results/transformer-progress.jsonl
python3 tests/transformer_tests.py --binary build/npu-transformer-demo
```

Create the `results` directory first if it is absent. The executable prints one
JSON object to stdout; SystemC notices and errors go to stderr. The optional VCD
basename receives the `.vcd` suffix; supplying that suffix explicitly is also
accepted. Equivalent VCD/progress paths, including existing symlink/hard-link
aliases, are rejected before either file is opened or truncated. Files are
written only when explicitly requested.

Default configuration: model width 16, four query heads, two key/value heads,
feed-forward width 32, cache capacity 32 tokens, three prefill tokens, and three
incremental decode tokens. The precision defaults to FP16 storage with FP32
accumulation. FP32 and BF16 are also selectable.

## Numerical execution

The complete block performs these operations, in order:

1. Pre-attention RMS normalization with learned-shape fixture scale values.
2. Query, key and value linear projections without bias.
3. Adjacent-pair rotary position encoding on queries and keys, with base 10000.
4. Causal grouped-query scaled dot-product attention and stable softmax.
5. Attention output projection and the first residual addition.
6. A second RMS normalization.
7. Gate/up projections, SiLU gate, elementwise multiplication, and down projection
   implementing a SwiGLU feed-forward network.
8. The second residual addition.

The RMS epsilon is $10^{-5}$. Attention divides logits by
$\sqrt{d_{\mathrm{head}}}$ and subtracts their maximum before exponentiation.
Softmax reductions and probabilities remain FP32. All tensor dot products use
separate binary32 multiplications and additions; there is no fused multiply-add
or hidden FP64 dot-product accumulation. C++ volatile rounding boundaries prevent
contraction. Transcendental functions use the host C++ math library and are cast
to binary32 at the documented boundary; tiny platform-dependent math-library
differences remain possible.

Inputs, weights, RMS scale values, projected vectors, rotated query/key vectors,
cached key/value vectors, attention outputs, activated feed-forward vectors and
residual outputs are materialized in the selected storage precision. The RMS
and softmax internal intermediates stay FP32. There is no implicit conversion
of the integer pipeline into floating-point arithmetic: this executable is a
separate, explicitly described model.

FP16 and BF16 conversions use software IEEE-style round-to-nearest, ties-to-even.
They preserve signed zero, support subnormals, overflow to signed infinity, and
quiet NaNs while preserving the sign. Inference inputs must be finite, and a
non-finite FP32 RMS reduction is rejected even when its inputs were finite. The
fixture domain is bounded. The precision probe accepts arbitrary binary32
bit patterns, including non-finite conversion edge cases:

```sh
build/npu-transformer-demo --precision fp16 \
  --round-f32-bits 0,2147483648,1065353216,1199570944
```

The result contains decimal binary32 bit patterns after storage conversion and
expansion back to binary32. This exposes the rounding contract without relying
on decimal formatting. BF16 is a portable modeling option here, not a claim
about Apple's documented execution precision or implementation.

## Persistent cache and requests

One SystemC `SC_THREAD` remains alive across the prefill request and each
single-token decode request. A separate host `SC_THREAD` submits commands through
a bounded FIFO and waits for completion events. Keys and values remain internal
to the engine between requests, avoiding a fictitious host round trip of the
entire cache.

Each query head maps to a contiguous group sharing one key/value head. Setting
`--kv-heads` equal to `--heads` selects ordinary multihead attention; setting it
to one selects multiquery attention. The key/value head count must divide the
query head count, and model width must divide into even query-head widths for
rotary pairs.

Only the current causal prefix is read. Prefill appends tokens in order and
incremental decode appends one token at a time. The cache capacity is fixed; no
token is silently evicted, wrapped or truncated. The entire request's shape,
capacity and input finiteness are validated before DMA, timed execution or cache
mutation. A candidate cache is used during execution, and output publication and
cache commitment occur only after the whole request succeeds. Numerical failures
can consume modeled work before rejection, but do not publish partial outputs or
commit a partial cache.

`--verify-lifecycle` additionally checks all of the following in the same live
SystemC engine:

- A cache-overflow request fails with no cache mutation or elapsed cycles.
- Empty, wrong-width and non-finite requests are rejected.
- A request with a valid first token followed by finite values that overflow the
  FP32 RMS reduction does real work but commits no tokens, outputs or cache.
- Reset clears both cache tensors.
- Replaying the entire sequence in one request produces exactly the same outputs
  and final cache as the original prefill plus incremental decode partition.

Normal runs also compare the outputs with an untimed execution using the same
scalar primitives. That partition check is not an independent numerical oracle;
the Python suite separately computes full query/key/value tensors and applies an
explicit causal mask before comparing every output value. No simulator numerical
kernel is imported into that oracle.

## Timed resource model

Execution uses real SystemC timed waits, events, FIFO requests and native traced
signals. The model serializes its DMA, tensor and vector stages. It makes no
claim that these resources overlap, that Apple uses this scheduling policy, or
that the resulting cycles are calibrated against hardware.

Tensor stages account for projection multiply-accumulates and both attention
matrix products. Their cost is:

$$
C_{\mathrm{tensor}} = \left\lceil
  \frac{\text{multiply-accumulates}}{\text{tensor MACs per cycle}}
\right\rceil.
$$

Vector stages separately account for RMS normalization, rotary encoding,
softmax, SiLU/gating and residual additions. Vector operations are abstract
work units, not a claim that exponentiation costs one hardware instruction:

$$
C_{\mathrm{vector}} = \left\lceil
  \frac{\text{modeled vector operations}}{\text{vector operations per cycle}}
\right\rceil.
$$

The operation counts are explicit in `Block::rms`, `Block::rope`,
`Block::residual` and `Block::token`. They can be changed or calibrated without
changing the numerical result. Tensor/vector throughput and DMA throughput are
independently configurable. Every DMA transfer costs its startup plus the
ceiling of transferred bytes divided by bytes per cycle.

All projection weights and RMS scale vectors are streamed from external memory
for each token. Host input/output transfers occur per request. Weights are not
assumed to fit in on-chip SRAM. This deliberately conservative model exposes
decode's weight-bandwidth pressure and the benefit of smaller storage precision;
it does not model a weight-resident cache or a fused batch-prefill GEMM.

The bounded SRAM reservation includes the entire logical KV cache and
conservative FP32 vector scratch. Configurations exceeding `--sram-bytes` fail
before simulation. The report distinguishes cache capacity from live bytes:

$$
\begin{aligned}
B_{\mathrm{KV,capacity}} &= 2 L_{\max} H_{\mathrm{KV}} d_{\mathrm{head}} b,\\
B_{\mathrm{KV,live}} &= 2 L_{\mathrm{current}} H_{\mathrm{KV}} d_{\mathrm{head}} b.
\end{aligned}
$$

Here $b$ is storage bytes per element. KV reads count one reuse of each cached
key/value vector per token, shared across the query-head group. The model assumes
that reuse is on chip; it does not charge an external DMA transfer for every
query head. SRAM byte counts represent modeled storage, while the host C++
implementation expands stored values into `float` containers for portability.

Reports include tensor MACs, vector operation counts, resource cycles, DMA byte
counts and transactions, KV reads/writes, SRAM reservation, prefill/decode cycles,
cache resets and rejected requests. Workload counters describe the first normal
run. When lifecycle verification is requested, `simulation_cycles` also includes
the reset/replay test; `workload_cycles` excludes it. With serial resource
scheduling, workload cycles equal the sum of DMA, vector and tensor cycles.
`tokens_completed` counts only atomically committed tokens. `tokens_executed`
counts completed token arithmetic, including tentative work later discarded by
request rejection. Resource and KV traffic counters retain genuinely executed
work on failure; no elapsed time or physical work is pretended to roll back.
Lifecycle reports expose the rejected request's executed-token and cycle counts
separately from the first normal workload.

## Native observability

VCD records `decoder_stage`, `decoder_busy`, `tokens_completed`, `tokens_executed`
and `kv_tokens`. Completed tokens and KV occupancy update only at request commit;
the executed-token trace can advance during a request that is later rejected.
Stage identifiers are idle 0, DMA 1, tensor 2 and vector 3. The VCD timescale is
one picosecond so delta-cycle signal changes remain visible while actual
architecture cycles are represented by one nanosecond.

Optional progress JSONL contains flushed `stage_complete`, `token_executed` and
`request_commit` events with actual simulation cycles, executed/committed tokens,
cache occupancy and
cumulative arithmetic/traffic counters. These events originate in the executing
SystemC thread. They can drive a separate UI; they are not wall-clock speed
claims and do not advance a simulated clock while the process is idle.

## Validation and bounds

`tests/transformer_tests.py` checks all three storage precisions against a separate
Python scalar oracle for ordinary multihead, grouped-query and multiquery
attention. It checks nondefault shapes, large unsigned seeds, multiple prefill
partitions, causal-prefix independence, exact cache capacity, reset/replay,
invalid request atomicity, arithmetic/traffic formulas, SRAM budgeting, relative
throughput changes and native VCD signals. Numerical comparisons allow small
host math-library differences while checking every output element.

The storage tests exhaustively round-trip every finite binary16 encoding and
independently compare random binary32, halfway, subnormal, signed-zero and
overflow conversions. The BF16 oracle selects the nearest representable
neighbour using binary64 distances instead of copying the C++ bit algorithm.

CLI bounds cap model width at 128, query heads at 16, feed-forward width at 256,
cache capacity at 512, prefill at 128, decode at 128 and SRAM budget at 16 MiB.
Unsigned parsing rejects signs, overflow, trailing garbage and unsupported
precision strings. A simulated-time watchdog also bounds a regressed command
handshake. These bounds make the deterministic demonstration reproducible;
they are not product specifications.

## Relation to mobile NPU software

Apple's [Deploying Transformers on the Apple Neural
Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
describes optimizing transformer execution around memory traffic and supported
operator layouts. This motivates representing vector work and memory costs
separately from tensor MAC throughput here. This model does not reproduce the
article's ANE-specific layout/compiler transformations or measured performance.

The Core ML Tools [stateful models
guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
documents persistent model state and incremental transformer KV-cache updates.
The persistent cache/command lifecycle here models that useful execution pattern.
Core ML state is a software facility and does not establish that every stateful
model or operation executes on the Neural Engine. The guide's cited Mistral
demonstration uses a GPU; this simulator makes no stronger placement claim.

Missing commercial features include trained checkpoints/tokenization, complete
multi-layer language models, speculative decoding, paged KV allocation, arbitrary
dynamic shapes, sparse floating-point tensor hardware, ISA/RTL execution,
compiler placement and measured commercial power/latency calibration.
