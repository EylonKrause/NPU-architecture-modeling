# Lossless weight compression and zero skipping

Apple's public Core ML documentation describes palettization as replacing weights
with lookup-table indices. Its performance guidance connects memory-bound gains
to loading compressed data and decompressing it close to computation, while
warning that additional LUTs can reduce performance. This simulator makes those
storage and decoder costs explicit rather than promising that compression is
always faster. See [Apple's palettization overview](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html)
and [performance guidance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html).

This is a documented experimental architecture inspired by public mobile model
optimization techniques. It does not reproduce Apple's Neural Engine ISA,
undocumented compression format, or device latency. Core ML supports lossy
clustering, grouped/vector palettes and additional widths; this implementation
uses **lossless scalar INT8 LUTs per column/K panel**, with packed 1, 2, 4 or 8-bit
indices. It never changes trained weights to make a palette fit.

## Run it

```sh
# Full-width lossless LUT: valid even when a panel contains all 256 INT8 values.
build/npu-sim --weight-encoding palette --palette-bits 8 --dump-output

# Block-sparse storage and zero-point-aware MAC skipping.
build/npu-sim --weight-encoding sparse --sparse-block 8 --dump-output

# Small LUT on an application tensor with at most four unique values per panel.
build/npu-sim --input model-layer.npu --weight-encoding palette --palette-bits 2
```

Source values still use `--weight-bits 4|8`. That flag describes signed numerical
precision and the dense baseline. Palette bits describe index width and are
independent. A two-bit LUT can contain four arbitrary signed INT8 values. Any
panel exceeding its LUT capacity fails with an explicit error; there is no
silent lossy quantization or fallback to dense storage.

`--weight-decode-values-per-cycle` configures decoder throughput, default 16.
`--sparse-block` accepts lengths 1 through 4096. `dense` remains the default;
existing dense numerics and timing are unchanged.

## Actual buffer and arithmetic

Each output-column tile and reduction panel owns a byte buffer, reused across
output-row tiles. The expanded `Problem::weights` vector remains the scalar
reference. The compressed numerical datapath reads and decodes the byte buffer,
then performs checked integer accumulation. It does not read expanded reference
weights. A C++ test mutates those expanded weights after encoding and proves
that the encoded result remains unchanged.

Both formats start with a four-byte header. Multi-byte fields are little endian.
All packed fields start in the low bits of their first byte; unused high bits in
the final byte are zero.

| Format | Header bytes | Remaining bytes |
| --- | --- | --- |
| Palette | uint16 LUT entries, uint8 index bits, uint8 reserved zero | signed INT8 LUT, then packed indices in K-major/local-column order |
| Sparse | uint16 block length, uint8 source weight bits, uint8 reserved zero | one bit per block, then packed signed values for retained blocks |

A sparse block is retained when any value differs from **its output channel's
weight zero point**. Omitted blocks therefore contribute exactly zero after
zero-point subtraction, including asymmetric per-channel quantization. Retained
blocks store every member, including any centered zeros. The sparse datapath
also skips those centered-zero multiplications, and explicitly counts them.
This is software-constructed block sparsity with a configurable zero-skipping
throughput engine, not a claim of a particular vendor's structured-sparsity unit.
No pruning threshold or lossy parameter is applied.

Decoding produces values directly into outer-product arithmetic without a
full expanded weight-panel allocation. Final INT32 overflow and quantization
rules are identical to the independent dense reference. As with the original
pipeline, numerical arithmetic commits at the final K panel; timing and
partial-sum storage are modeled for every panel.

## Timing, storage and cache contract

- DMA reads the **actual header, LUT/bitmap and packed payload bytes** for each
  cache miss. The source INT8/INT4 byte count is not substituted as a fake
  compressed transfer. Existing activation, bias, per-channel metadata and output
  traffic remain charged separately.
- The LRU cache holds compressed panels. Its capacity, saved bytes and high-water
  counter use the same exact encoded byte lengths. A cache hit avoids DMA bytes
  but does not erase decoder work.
- SRAM validation reserves a conservative **worst-case** encoded panel per slot,
  plus activation, output/partial sums and cache. The reported `scratchpad_bytes`
  is this safe capacity bound; the actual byte buffers are reported separately.
  Sparse expansion and LUT overhead can make a codec larger than dense storage.
- Palette decoder cycles are the ceiling of panel elements divided by decoder
  throughput. Sparse cycles add the ceiling of bitmap-block count divided by
  that rate to the ceiling of retained values divided by the same rate. The rate
  models bitmap descriptor issue as well as value decode; it is an explicit
  architecture assumption, not a measured phone parameter.
- Decode runs serially before that panel's MAC work, on the same compute stage.
  Its cost is added even when DMA is cached. Palette mode preserves dense MAC
  scheduling, including the optional systolic geometry and fill/drain cost.
- Sparse compute cycles use actual nonzero-centered weight count times tile
  rows, divided by configured MAC throughput, plus decode and pipeline overhead.
  Sparse mode **rejects systolic-array timing**: the model does not invent a
  sparse systolic compaction network or assume free array speedups.

| JSON field | Meaning |
| --- | --- |
| `macs` | Logical dense workload, retained for backward compatibility |
| `executed_macs` | Operations performed by the configured arithmetic path |
| `skipped_macs` | Centered-zero operations skipped in sparse mode |
| `weight_decode_cycles` | Committed decoder occupancy, included in compute busy cycles |
| `decoded_weight_values` | Packed values decoded, repeated for each output-row tile |
| `weight_read_bytes` | Actual encoded weight bytes transferred after cache reuse |
| `encoded_weight_storage_bytes` | Unique serialized panels, including codec metadata |
| `dense_weight_storage_bytes` | Same panels stored at configured source precision, including tail padding |

`mac_utilization` uses executed MACs; arithmetic intensity continues to expose the
logical workload per external byte. Live JSON snapshots expose the new counters,
and native SystemC VCD signals include executed MACs, skipped MACs and decoder
cycles. Event-driven, per-cycle, progress-sliced and live-stdio execution retain
the same final cycle count and output.

## Limits and validation

Palette and sparse encodings are alternative formats; joint compression and
lossy clustering are not implemented. Input tensors are compressed before the
simulated run, so host encoding time is not charged as NPU latency. Headers are
an explicit simulator format rather than a Core ML model interchange format.
The TLM demo and Python graph runtime continue to default to dense weights unless
an applicable execution path explicitly forwards these options.

Independent Python tests cover all index widths, INT4 nibble tails, asymmetric
zero points, empty sparse payloads, edge panels, decoder and metadata cycles,
cache accounting, capacity rejection and execution-mode equivalence. C++ tests
cover packed-storage independence, malformed headers/payloads and a full
256-entry signed INT8 palette. Compression is allowed to be larger or slower;
there is no fabricated speedup assertion.
