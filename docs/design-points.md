# Design points

Four hypothetical configurations in [`configs/`](../configs), sized from public data. None is a
model of a vendor product: each takes the public figures named below (MAC count, internal
SRAM, interface width, clock, memory rate, latencies) and derives every other field by the
shared rules. Run one with

```sh
npu-sim --config configs/mobile-2k.json --m 196 --n 256 --k 2304
```

Later flags override the file. `python -m npuperf.designs` prints the table below from the
config files; `sim.design-points` (tests/design_point_tests.py) checks the table, the
derivations and the sizing rules against the files and the simulator, accepts every layer of
every bundled workload on every point, and runs one GEMM per point bit-exact.

<!-- table: python -m npuperf.designs -->
| Design point | MAC/cycle | Array | Cores | Clock | Peak TOP/s | SRAM | AXI | DRAM B/cycle | DRAM GB/s | Ridge MAC/B |
|---|---|---|---|---|---|---|---|---|---|---|
| `edge-256` | 256 | 16x16 | 1 | 1000 MHz | 0.512 | 50 KiB | 64-bit | 8 | 8 | 32 |
| `mobile-2k` | 2048 | 32x64 | 1 | 1000 MHz | 4.096 | 267 KiB | 256-bit | 24 | 24 | 85.33 |
| `client-16c` | 16384 | 32x32 | 16 | 1200 MHz | 39.3216 | 16 x 133.5 KiB | 16 x 256-bit | 100 | 120 | 163.8 |
| `workstation-32c` | 32768 | 32x32 | 32 | 1200 MHz | 78.6432 | 32 x 133.5 KiB | 32 x 256-bit | 667 | 800.4 | 49.13 |
<!-- end table -->

- MAC/cycle counts every core. Peak TOP/s = 2 x MAC/cycle x clock: one INT8 MAC is two
  operations, the convention of Arm's figures (2048 MACs, 4 TOP/s at 1 GHz). The clusters
  apply the same convention to Apple's operations-per-second figure, whose precision and
  operation count Apple does not state.
- SRAM and AXI are per core. Each core has its own SRAM, slots, weight cache and AXI port
  (read and write channels); all cores share one DRAM timeline.
- DRAM GB/s = DRAM B/cycle x clock. For `edge-256` the external memory is system SRAM.
- Ridge = MAC/cycle / memory roof, the roof being min(DRAM B/cycle, 2 x cores x AXI bytes),
  which npu-sim reports as `peak_memory_bytes_per_cycle`. The DRAM rate is the roof on all
  four points.
- KB in Arm's documents is 1024 B: the Ethos-U55-256 has 48 SHRAM banks of 1024 B plus a
  2048 B LUT ([Vela `architecture_features.py`][vela-arch]), 51,200 B, which the product brief
  states as 50 KB.

## Shared rules

| Field | Rule |
|---|---|
| `dataflow` | `os` |
| `buffer_slots`, `panel_buffers` | 2 and 2: load, compute and store overlap across tiles and across reduction panels |
| `tile_m` = `tile_n` = T | The largest power of two that is a multiple of both array dimensions and whose two slots fit the SRAM with `tile_k` at least the minimum panel depth |
| `tile_k` | The largest power of two that fits with T |
| Minimum panel depth | The minimum efficient burst of the memory behind the port, Vela's `Sram_burst_length` (32 B) or `Dram_burst_length` (128 B). Each activation-panel row is one `tile_k`-byte run, so a shallower panel issues shorter runs |
| `weight_cache_bytes` | The SRAM the slots leave free |
| `axi.max_burst_beats` | The shortest power of two B with read outstanding x B >= read latency + B. A read burst holds an outstanding slot for latency + B cycles, so N slots stream one beat per cycle once N x B >= latency + B |
| AXI latencies, outstanding limits | Vela reference system configurations, in cycles. Vela uses the same cycle counts at 500 MHz and 1 GHz (DRAM 500 read, 250 write in `Ethos_U65_Mid_End` at 500 MHz and in `Ethos_U85_SYS_DRAM_High` at 1 GHz); the points use them unchanged at their own clock |
| `pipeline_cycles` | 4, the simulator default; no public figure |

A slot reserves 4T B of bias, two panel buffers of T x `tile_k` activation and `tile_k` x T
weight bytes, and 5 B per output (INT32 partial sum and INT8 result). Partial tiles at the
matrix edges and single-panel tiles reserve less, so every layer fits once full tiles do.
`sim.design-points` checks that full tiles fill each point's SRAM exactly and that npu-sim
rejects the next larger T or `tile_k`.

`npuperf.designs` implements the rules (`tile_rule`, `burst_rule`, `cache_rule`).

## edge-256

Microcontroller-class point sized from the Ethos-U55 top configuration.

| Field | Value | Derivation |
|---|---|---|
| Array | 16x16, 256 MAC/cycle | Ethos-U55 MACs "32, 64, 128, 256" ([product brief][brief]) |
| Clock | 1000 MHz | The brief quotes performance "at 1 GHz": 2 x 256 x 1 GHz = 512 GOP/s, the top of its "64 to 512 GOP/s" |
| SRAM | 51,200 B (50 KiB) | Internal SRAM "18 to 50 KB" ([brief]); 48 SHRAM banks x 1024 B + 2048 B LUT for ethos-u55-256 ([vela-arch]) |
| AXI | One 64-bit port, 8 B beats | "Two 64-bit AXI" ([brief]); 8 B/cycle per U55 port ([vela-arch]). The model's one port is AXI0 to system SRAM, which holds every tensor (Vela `Sram_Only` memory mode); the read-only AXI1 flash port is not modeled |
| Memory rate | 8 B/cycle, 8 GB/s | 8 B/cycle x `Sram_clock_scale` 1.0 in `Ethos_U55_High_End_Embedded` ([vela.ini][vela-ini]); Vela states that system as "SRAM (4 GB/s)" at its 500 MHz |
| Latency | 32 read, 32 write | `Sram_read_latency`, `Sram_write_latency` ([vela-ini]) |
| Outstanding | 4 reads, 4 writes | `Sram_max_reads`, `Sram_max_writes` ([vela-ini]) |
| Burst | Up to 16 beats (128 B) | 4 x 16 >= 32 + 16; 4 x 8 < 32 + 8 |
| Tiles | 32x32x128 | Minimum depth 32 B (`Sram_burst_length`). T = 64 needs 57,856 B at depth 32; at T = 32, depth 256 needs 76,032 B |
| Weight cache | 7,936 B | 51,200 - 2 x 21,632 |

The model sets no capacity on the system SRAM behind the port. Arm's Corstone-300 reference
platform for the Ethos-U55 has 2 MiB of SRAM and places the model and tensor arena in DDR by
default ([`platform.ld`][corstone]). Vela's U55 reference systems attach the second AXI port
to off-chip flash, where its `Shared_Sram` memory mode places the constants ([vela-ini]).
[results.md](results.md) therefore states the weight bytes each `edge-256` row assumes the
system SRAM holds, and runs no LLM workload on this point.

## mobile-2k

Mobile point sized from the Ethos-U85 top configuration.

| Field | Value | Derivation |
|---|---|---|
| Array | 32x64, 2048 MAC/cycle | "Support for configurations from 128 to 2048 MACs/cycle" ([Arm newsroom][u85-blog]) |
| Clock | 1000 MHz | "256 GOPS/s to 4 TOP/s at 1GHz" ([u85-blog]): 2 x 2048 x 1 GHz = 4.096 TOP/s |
| SRAM | 273,408 B (267 KiB) | "Internal SRAM of 29 to 267 KB" ([u85-blog]) |
| AXI | One 256-bit port, 32 B beats | "up to six 128-bit AXI5 interfaces" ([u85-blog]): for ethos-u85-2048, 4 system-SRAM and 2 external ports ([Regor `ethos_u85.cpp`][regor]); 16 B/cycle per U85 port ([vela-arch]); `Dram_ports_used=2` in `Ethos_U85_SYS_DRAM_High` ([vela-ini]). The two 128-bit DRAM ports are modeled as one 256-bit port. Every tensor is in DRAM; the system-SRAM level is not modeled |
| DRAM rate | 24 B/cycle, 24 GB/s | 16 B x 2 ports x `Dram_clock_scale` 0.75 ([vela-ini]), stated by Vela as "DRAMx2 (24 GB/s)" |
| Latency | 500 read, 250 write | `Dram_read_latency`, `Dram_write_latency` ([vela-ini]) |
| Outstanding | 64 reads, 32 writes | `Dram_max_reads`, `Dram_max_writes` ([vela-ini]) |
| Burst | Up to 8 beats (256 B) | 64 x 8 >= 500 + 8; 64 x 4 < 500 + 4 |
| Tiles | 64x64x256 | Minimum depth 128 B (`Dram_burst_length`). T = 128 needs 295,936 B at depth 128; at T = 64, depth 512 needs 303,616 B |
| Weight cache | 100,864 B | 273,408 - 2 x 86,272 |

## client-16c

Client-SoC point in the class of the 16-core Neural Engine Apple states for M4.

| Field | Value | Derivation |
|---|---|---|
| Cores, array | 16 x 32x32, 16,384 MAC/cycle | M4's Neural Engine: "16-core design", "capable of 38 trillion operations per second" ([Apple, May 2024][m4]). Apple does not state the precision; this model's MACs are INT8. 38 T ops/s over 16 cores of 1024 MACs at 2 ops/MAC is 1.16 GHz |
| Clock | 1200 MHz | 1.16 GHz rounded up so that 120 GB/s is a whole 100 B/cycle (npu-sim takes integer bytes per cycle). Peak 2 x 16,384 x 1.2 GHz = 39.3216 TOP/s, 3.5% above 38 |
| DRAM rate | 100 B/cycle, 120 GB/s | M4 "has higher memory bandwidth of 120GB/s" ([Apple, Oct 2024][m4-pro]). That is the SoC's rate, shared with the CPU and GPU; Apple publishes no rate attainable by the Neural Engine |
| SRAM | 136,704 B (133.5 KiB) per core | Apple publishes no Neural Engine SRAM size. The Ethos-U85 top configuration's 267 KiB per 2048 MACs, scaled to 1024 MACs |
| AXI | One 256-bit port per core | One core design serves both clusters: the narrowest AXI width whose read channels, one per core, carry `workstation-32c`'s 667 B/cycle. 32 x 16 B = 512 falls short; 32 x 32 B = 1024 does not. Here 16 x 32 B = 512 B/cycle against 100 |
| Latency, outstanding | 500/250, 64/32 | `Ethos_U85_SYS_DRAM_High` ([vela-ini]), unchanged at 1.2 GHz |
| Burst | Up to 8 beats (256 B) | As `mobile-2k` |
| Tiles | 64x64x128 | Minimum depth 128 B. T = 128 needs 295,936 B at depth 128; at T = 64, depth 256 needs 172,544 B |
| Weight cache | 29,696 B per core | 136,704 - 2 x 53,504 |
| Partition | `auto` | npu-sim picks the m or n split per GEMM |

## workstation-32c

Two-die point: two `client-16c` dies on one DRAM, 32 cores.

| Field | Value | Derivation |
|---|---|---|
| Cores | 32 | M3 Ultra: "A powerful 32-core Neural Engine" ([Apple, March 2025][m3-ultra]). Apple states no operations-per-second figure for it; the peak follows from the core count and the `client-16c` clock |
| Core, clock | 32x32 array, 1200 MHz | `client-16c`'s; peak 2 x 32,768 x 1.2 GHz = 78.6432 TOP/s |
| DRAM rate | 667 B/cycle, 800.4 GB/s | "over 800GB/s of memory bandwidth" ([m3-ultra]): the smallest integer rate at 1.2 GHz that reaches 800 GB/s (666 B/cycle gives 799.2 GB/s). Shared with the CPU and GPU |
| Per-core fields | As `client-16c` | SRAM 133.5 KiB, 256-bit port, Vela latencies and limits, 8-beat bursts, 64x64x128 tiles, 29,696 B weight cache, `auto` partition |
| Die-to-die link | Not modeled | All 32 cores reach the one DRAM timeline at the same cost |

The per-core throughput is `client-16c`'s, sized from the M4 generation's figure. Public data
on the M3 generation implies about half: Apple states the M3-family Neural Engine is "up to
60 percent faster than in the M1 family" ([Apple, October 2023][m3]) and M1's at "11 trillion
operations per second" ([Apple, November 2020][m1]), 1.6 x 11 = 17.6 T ops/s per 16 cores and
35.2 per 32, against this point's 78.6432 TOP/s. The point pairs M4-class cores with the
M3 Ultra's core count and memory rate.

## Reading the points

An a x b tile moves (a + b) x `tile_k` bytes per panel for a x b x `tile_k` MACs: ab/(a + b)
MAC per byte before weight-cache hits and the output store. With the config files' 2-slot
T x T tiles that is T/2, 16 on `edge-256` and 32 on the others. All four ridges lie above it,
so under that default mapping a GEMM whose weights the cache does not hold is DRAM-bound on
every point: the SRAM sets T, and no point's SRAM holds the two slots of a tile wide enough
to reach its ridge. `npuperf.mapper`, which the results use, can give up the second slot for
a wider tile; on `edge-256` a one-slot 49 x 128 tile of a 49-row layer reaches
6272/177 = 35.4 MAC per byte, above the 32 MAC/B ridge, so some `edge-256` layers in
[results.md](results.md) are compute-bound.

## Sources

- Arm, *Ethos-U Processor Series Product Brief*: <https://armkeil.blob.core.windows.net/developer/Files/pdf/datasheets/arm-ethos-u-processor-series-product-brief.pdf>
- Arm Newsroom, Ethos-U85 announcement, 2024-04-09: <https://newsroom.arm.com/blog/ethos-u85>
- Vela 5.2.0, `ethosu/config_files/Arm/vela.ini`: <https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/ethosu/config_files/Arm/vela.ini>. Field meanings are in
  [`OPTIONS.md`](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/OPTIONS.md):
  burst length is the "Minimum efficient burst length", latencies are in cycles.
- Vela 5.2.0, `ethosu/vela/architecture_features.py`: <https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/ethosu/vela/architecture_features.py>
- Vela 5.2.0, `ethosu/regor/architecture/ethosu85/ethos_u85.cpp`: <https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/ethosu/regor/architecture/ethosu85/ethos_u85.cpp>
- Apple Newsroom, *Apple introduces M4 chip*, 2024-05: <https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/>
- Apple Newsroom, *Apple introduces M4 Pro and M4 Max*, 2024-10: <https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/>
- Apple Newsroom, *Apple reveals M3 Ultra*, 2025-03: <https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/>
- Apple Newsroom, *Apple unveils M3, M3 Pro, and M3 Max*, 2023-10: <https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/>
- Apple Newsroom, *Apple unleashes M1*, 2020-11: <https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/>
- Arm, Ethos-U core platform, Corstone-300 linker script at tag 26.08: <https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform/-/blob/26.08/targets/corstone-300/platform.ld>

[brief]: https://armkeil.blob.core.windows.net/developer/Files/pdf/datasheets/arm-ethos-u-processor-series-product-brief.pdf
[u85-blog]: https://newsroom.arm.com/blog/ethos-u85
[vela-ini]: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/ethosu/config_files/Arm/vela.ini
[vela-arch]: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/ethosu/vela/architecture_features.py
[regor]: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/5.2.0/ethosu/regor/architecture/ethosu85/ethos_u85.cpp
[m4]: https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/
[m4-pro]: https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
[m3-ultra]: https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/
[m3]: https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/
[m1]: https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/
[corstone]: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform/-/blob/26.08/targets/corstone-300/platform.ld
