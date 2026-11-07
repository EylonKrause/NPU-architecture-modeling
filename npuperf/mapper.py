"""Per-layer tile mapping for a design point: a heuristic search over the analytic model.

    python -m npuperf.mapper --design client-16c --m 784 --n 128 --k 1152

Candidates. tile_m is a power-of-two multiple of the array rows and tile_n of the array
columns (a vector datapath steps by 8), up to the layer's extent rounded up to that step;
a tile larger than the extent is clipped to it, as npu-sim clips edge tiles. tile_k is
the whole reduction (0) or a power of two of at least ``min_depth`` bytes: each
activation-panel row is one tile_k-byte burst run, so a shallower panel issues runs shorter
than the memory's efficient burst. buffer_slots is 1 up to the design's count: one slot
gives up load-compute overlap across tiles for a tile twice as large. A candidate fits
when its slots and panel buffers fit the SRAM; the SRAM they leave free becomes the
weight cache.

Choice. Per tile shape and slot count the full reduction and the three deepest fitting
panels are scored with ``analytic.analyze``; the mapper keeps the lowest estimate, then
fewer DRAM bytes, then less SRAM. Panel buffers, dataflow and partition stay as the design
sets them.
A heuristic over a fixed candidate set, not a compiler: no layer fusion, no layout or
loop-order choice, no cross-layer buffering.
"""
from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from . import analytic, cli, designs, jsonio, table

VECTOR_STEP = 8
DEEPEST_DEPTHS = 3
MAX_TILE = 1 << 16


@dataclass(frozen=True)
class Mapping:
    tile_m: int
    tile_n: int
    tile_k: int
    buffer_slots: int
    weight_cache_bytes: int
    scratchpad_bytes: int
    estimate_cycles: int
    dram_bytes: int
    bound: str

    def flags(self) -> dict[str, int]:
        """npu-sim flags that apply the mapping on top of the design point's config."""
        return {"tile_m": self.tile_m, "tile_n": self.tile_n, "tile_k": self.tile_k,
                "buffer_slots": self.buffer_slots,
                "weight_cache_bytes": self.weight_cache_bytes}


def axis_options(extent: int, step: int) -> list[int]:
    """Tile sizes along one axis: step * 2^i below the extent, then the extent rounded up
    to the step, each clipped to the extent."""
    top = -(-extent // step) * step
    sizes, size = [], step
    while size < top and size <= MAX_TILE:
        sizes.append(min(size, extent))
        size *= 2
    sizes.append(min(top, extent))
    return sorted(set(sizes))


def depth_options(k: int, min_depth: int) -> list[int]:
    """tile_k values: powers of two of at least min_depth below k, then 0 (all of k)."""
    depths, depth = [], 1
    while depth < max(1, min_depth):
        depth *= 2
    while depth < k:
        depths.append(depth)
        depth *= 2
    return depths + [0]


def slot_bytes(design: analytic.Design) -> int:
    return analytic.scratchpad_bytes(design.replace(weight_cache_bytes=0))


def steps(design: analytic.Design) -> tuple[int, int]:
    if design.array:
        return design.array_rows, design.array_cols
    return VECTOR_STEP, VECTOR_STEP


def candidates(design: analytic.Design, min_depth: int,
               cache: bool = True) -> list[analytic.Design]:
    """Every scored candidate, each with its weight cache set to the free SRAM (or off)."""
    row_step, col_step = steps(design)
    out = []
    shapes = [(tile_m, tile_n, slots) for tile_m in axis_options(design.m, row_step)
              for tile_n in axis_options(design.n, col_step)
              for slots in range(1, design.buffer_slots + 1)]
    for tile_m, tile_n, slots in shapes:
        fitting = []
        for tile_k in depth_options(design.k, min_depth):
            trial = design.replace(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                                   buffer_slots=slots)
            used = slot_bytes(trial)
            if used <= design.sram_bytes:
                free = design.sram_bytes - used if cache else 0
                fitting.append(trial.replace(weight_cache_bytes=free))
        full = [d for d in fitting if d.tile_k == 0]
        panels = [d for d in fitting if d.tile_k][-DEEPEST_DEPTHS:]
        out += full + panels
    return out


def _rank(design: analytic.Design, analysis: analytic.Analysis) -> tuple[int, int, int]:
    return analysis.estimate_cycles, analysis.dram_bytes, slot_bytes(design)


@functools.lru_cache(maxsize=4096)
def map_layer(design: analytic.Design, min_depth: int = 32, cache: bool = True) -> Mapping:
    """The mapping for ``design``'s GEMM shape (m, n, k, groups); see the module text.
    ``cache`` False leaves the weight cache off instead of giving it the free SRAM."""
    best = None
    for trial in candidates(design, min_depth, cache):
        analysis = analytic.analyze(trial)
        key = _rank(trial, analysis)
        if best is None or key < best[0]:
            best = (key, trial, analysis)
    if best is None:
        raise ValueError(f"no tile of a {design.m}x{design.n}x{design.k} layer fits "
                         f"{design.sram_bytes} B of SRAM with panels of at least "
                         f"{min_depth} B")
    _, trial, analysis = best
    return Mapping(trial.tile_m, trial.tile_n, trial.tile_k, trial.buffer_slots,
                   trial.weight_cache_bytes,
                   analysis.scratchpad_bytes, analysis.estimate_cycles, analysis.dram_bytes,
                   analysis.bound)


def map_gemm(design: analytic.Design, m: int, n: int, k: int, groups: int = 1,
             min_depth: int = 32, cache: bool = True, **changes: object) -> Mapping:
    return map_layer(design.replace(m=m, n=n, k=k, groups=groups, **changes), min_depth, cache)


@cli.main_guard("mapper")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("mapper", __doc__)
    parser.add_argument("--design", required=True,
                        help="design point (configs/<NAME>.json or a JSON path)")
    parser.add_argument("--m", type=cli.positive_int, required=True, help="output rows")
    parser.add_argument("--n", type=cli.positive_int, required=True,
                        help="output channels per group")
    parser.add_argument("--k", type=cli.positive_int, required=True,
                        help="reduction depth per group")
    parser.add_argument("--groups", type=cli.positive_int, default=1, help="groups")
    parser.add_argument("--min-depth", type=cli.positive_int,
                        help="smallest panel depth in bytes (default: the design point's)")
    cli.add_format_option(parser)
    args = parser.parse_args(argv)
    document = designs.load(args.design)
    design = designs.design(document)
    min_depth = args.min_depth or designs.min_panel_depth(document["name"])
    mapping = map_gemm(design, args.m, args.n, args.k, args.groups, min_depth)
    if args.format == "json":
        print(jsonio.dumps(asdict(mapping)), end="")
    else:
        print(table.format_table([asdict(mapping)]), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
