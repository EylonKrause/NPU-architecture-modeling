"""Per-layer tile mapping for a design point: a heuristic search over the analytic model.

    python -m npuperf.mapper --design client-16c --m 784 --n 128 --k 1152

Candidates. tile_m is a power-of-two multiple of the array rows and tile_n of the array
columns (a vector datapath, and the depthwise datapath, step by 8), up to the layer's
extent rounded up to that step; a tile larger than the extent is clipped to it, as npu-sim
clips edge tiles. A convolution also tries whole output rows, a power of two of them.
tile_k is the whole reduction (0) or a power of two of at least ``min_depth`` bytes: each
activation-panel row is one tile_k-byte burst run, so a shallower panel issues runs shorter
than the memory's efficient burst. Under conv geometry tile_k is 0 or whole kernel rows,
and the runs are input rows or pixels whatever its depth. buffer_slots is 1 up to the
design's count: one slot gives up load-compute overlap across tiles for a tile twice as
large. A candidate fits when its slots and panel buffers fit the SRAM; when no candidate
fits, one panel buffer per slot is tried. The SRAM the slots leave free is the caches': all
of it the weight cache, or, when the layer has more than one column tile to reuse a row
tile's activations, the largest row tile's activation panels the activation cache and the
rest the weight cache.

Lowering. A convolution runs on its input footprint (conv geometry). A 1x1 convolution
also tries the plain operand, whose rows are its input pixels: tile_k is then free, which
deep 1x1 layers need to fit a small SRAM.

Choice. Per tile shape and slot count the full reduction and the three deepest fitting
panels are scored with ``analytic.analyze``, ranked by the lowest estimate, then fewer DRAM
bytes, then less SRAM. The closed form times each request alone and misses queueing on the
shared DRAM: on a weight-streaming GEMV most candidates share the DRAM-bound estimate,
which cannot order them. So the candidates are replayed with ``analytic.schedule``,
which matches npu-sim cycle for cycle, in rank order, skipping a candidate whose
``analyze`` lower bound is no less than the fewest cycles replayed so far; the mapper keeps
the fewest replayed cycles of the whole candidate set (ties by rank). Only a layer whose
lower bounds separate its candidates replays few of them.
Dataflow and partition stay as the design sets them.
A heuristic over a fixed candidate set, not a compiler: no layer fusion, no layout or
loop-order choice, no cross-layer buffering.
"""
from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from . import analytic, cli, designs, jsonio, table, workloads

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
    activation_cache_bytes: int = 0
    replayed_cycles: int = 0
    panel_buffers: int = 2
    # The layer runs on its conv geometry; False for a plain GEMM or the plain operand of
    # a 1x1 convolution.
    conv: bool = False

    def flags(self) -> dict[str, int]:
        """npu-sim flags that apply the mapping on top of the design point's config."""
        return {"tile_m": self.tile_m, "tile_n": self.tile_n, "tile_k": self.tile_k,
                "buffer_slots": self.buffer_slots, "panel_buffers": self.panel_buffers,
                "weight_cache_bytes": self.weight_cache_bytes,
                "activation_cache_bytes": self.activation_cache_bytes}


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
    return analytic.scratchpad_bytes(design.replace(weight_cache_bytes=0,
                                                    activation_cache_bytes=0))


def steps(design: analytic.Design) -> tuple[int, int]:
    if design.array and not design.channel_wise:
        return design.array_rows, design.array_cols
    return VECTOR_STEP, VECTOR_STEP


def row_options(design: analytic.Design, step: int) -> list[int]:
    """tile_m values: axis_options, and under conv geometry whole output rows."""
    sizes = set(axis_options(design.m, step))
    if design.conv:
        width = analytic.conv_geometry(design).output_w
        rows = 1
        while rows * width < design.m:
            sizes.add(rows * width)
            rows *= 2
    return sorted(sizes)


def tile_depths(design: analytic.Design, min_depth: int) -> list[int]:
    """tile_k values: depth_options, or under conv geometry whole kernel rows and 0."""
    if not design.conv:
        return depth_options(design.k, min_depth)
    row_k = analytic.conv_geometry(design).kernel_row_k
    return [rows * row_k for rows in range(1, design.conv_kernel_h)] + [0]


def activation_need(design: analytic.Design) -> int:
    """Activation-cache bytes at which every row tile's panels fit together, or 0 when no
    panel is read twice: one column tile, or the depthwise datapath."""
    if design.channel_wise or design.grid_columns <= design.tile_n:
        return 0
    return max(sum(analytic.activation_bytes(design, row, min(design.tile_m, design.m - row),
                                             1, block)
                   for block in range(design.reduction_blocks))
               for row in range(0, design.m, design.tile_m))


def cache_splits(trial: analytic.Design, free: int) -> list[tuple[int, int]]:
    """(weight cache, activation cache) splits of ``free`` SRAM bytes."""
    need = activation_need(trial)
    splits = [(free, 0)]
    if 0 < need <= free:
        splits.append((free - need, need))
    return splits


def candidates(design: analytic.Design, min_depth: int, cache: bool = True,
               panel_buffers: int | None = None) -> list[analytic.Design]:
    """Every scored candidate, each with the free SRAM split between its caches (or both
    caches off)."""
    row_step, col_step = steps(design)
    buffers = design.panel_buffers if panel_buffers is None else panel_buffers
    out = []
    shapes = [(tile_m, tile_n, slots) for tile_m in row_options(design, row_step)
              for tile_n in axis_options(design.grid_columns, col_step)
              for slots in range(1, design.buffer_slots + 1)]
    for tile_m, tile_n, slots in shapes:
        fitting = []
        for tile_k in tile_depths(design, min_depth):
            trial = design.replace(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                                   buffer_slots=slots, panel_buffers=buffers,
                                   weight_cache_bytes=0, activation_cache_bytes=0)
            if slot_bytes(trial) <= design.sram_bytes:
                fitting.append(trial)
        full = [d for d in fitting if d.tile_k == 0]
        panels = [d for d in fitting if d.tile_k][-DEEPEST_DEPTHS:]
        for trial in full + panels:
            if not cache:
                out.append(trial)
                continue
            free = design.sram_bytes - slot_bytes(trial)
            out += [trial.replace(weight_cache_bytes=weights, activation_cache_bytes=act)
                    for weights, act in cache_splits(trial, free)]
    return out


def lowerings(design: analytic.Design) -> list[analytic.Design]:
    """The design on its conv geometry and, for a 1x1 convolution, on its plain operand."""
    if not design.conv:
        return [design]
    plain = design.replace(**{name: getattr(analytic.Design(), name)
                              for name in analytic.CONV_FIELDS})
    pointwise = design.conv_kernel_h == design.conv_kernel_w == 1
    return [design, plain] if pointwise else [design]


def _rank(design: analytic.Design, analysis: analytic.Analysis) -> tuple[int, int, int]:
    return analysis.estimate_cycles, analysis.dram_bytes, slot_bytes(design)


def replay_best(ranked: Sequence[tuple[object, analytic.Design, analytic.Analysis]]
                ) -> tuple[int, int]:
    """(fewest replayed cycles, index) over ``ranked``, the earliest index on a tie. A
    candidate whose lower bound is no less than the best so far cannot beat it, nor tie it
    from a later index, so it is not replayed."""
    best: tuple[int, int] | None = None
    for order, (_, trial, analysis) in enumerate(ranked):
        if best is not None and analysis.lower_bound_cycles >= best[0]:
            continue
        cycles = analytic.schedule(trial).cycles
        if best is None or cycles < best[0]:
            best = (cycles, order)
    if best is None:
        raise ValueError("no candidate to replay")
    return best


@functools.lru_cache(maxsize=4096)
def map_layer(design: analytic.Design, min_depth: int = 32, cache: bool = True) -> Mapping:
    """The mapping for ``design``'s GEMM shape (m, n, k, groups) and conv geometry; see
    the module text. ``cache`` False leaves both caches off instead of giving them the free
    SRAM."""
    scored = []
    for buffers in (None, 1):
        for lowered in lowerings(design):
            for trial in candidates(lowered, min_depth, cache, buffers):
                analysis = analytic.analyze(trial)
                scored.append((_rank(trial, analysis), trial, analysis))
        if scored or design.panel_buffers == 1:
            break
    if not scored:
        raise ValueError(f"no tile of a {design.m}x{design.n}x{design.k} layer fits "
                         f"{design.sram_bytes} B of SRAM with panels of at least "
                         f"{min_depth} B")
    # A stable sort: equal ranks keep the candidate order.
    ranked = sorted(scored, key=lambda entry: entry[0])
    cycles, order = replay_best(ranked)
    _, trial, analysis = ranked[order]
    return Mapping(trial.tile_m, trial.tile_n, trial.tile_k, trial.buffer_slots,
                   trial.weight_cache_bytes,
                   analysis.scratchpad_bytes, analysis.estimate_cycles, analysis.dram_bytes,
                   analysis.bound, trial.activation_cache_bytes, cycles, trial.panel_buffers,
                   trial.conv)


def map_gemm(design: analytic.Design, m: int, n: int, k: int, groups: int = 1,
             min_depth: int = 32, cache: bool = True, **changes: object) -> Mapping:
    design = design.replace(m=m, n=n, k=k, groups=groups, **changes)
    if not design.channel_wise:
        # The depthwise rate cannot change a GEMM's mapping; one cache entry serves all.
        design = design.replace(depthwise_macs_per_cycle=0)
    return map_layer(design, min_depth, cache)


def map_call(design: analytic.Design, gemm: workloads.Gemm, min_depth: int = 32,
             cache: bool = True, **changes: object) -> Mapping:
    """map_gemm with the call's conv geometry, if it has one."""
    geometry = dict(zip(analytic.CONV_FIELDS, gemm.conv))
    return map_gemm(design, gemm.m, gemm.n, gemm.k, gemm.groups, min_depth, cache,
                    **geometry, **changes)


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
