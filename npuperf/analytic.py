"""Closed-form timing model of npu-sim, written from the architecture rules, not the C++.

Execution model. Each of G groups is an independent M x n x k GEMM. Its output is cut
into tile_m x tile_n tiles (edge tiles are smaller); each tile walks the reduction in
panels of tile_k (0 = all of k). Per panel the core loads

    A panel  a * c bytes            (a tile rows, c reduction columns)
    W panel  ceil(c * b * bits / 8)  (b tile columns)
    params   4 * b * (1 + vectors)   on the first panel only (INT32 bias + per-channel vectors)

computes it, and after the last panel stores a * b output bytes.

Compute cycles per panel (c = panel depth, P = MACs per cycle, A_r x A_c array):
    vector  ceil(a * b * c / P) + pipeline
    os      ceil(a/A_r) * ceil(b/A_c) * c + (A_r + A_c - 2) + pipeline
    ws      ceil(c/A_r) * ceil(b/A_c) * a + A_r + (A_r + A_c - 2) + pipeline

Serial memory model: one DMA server, one request at a time, each request costs
``latency + ceil(bytes / bytes_per_cycle)`` per burst of at most dma_burst_bytes.

Exact for dense weights without a weight cache: traffic, request counts and the
compute/DMA busy totals; the cycle count itself when the schedule cannot overlap
(one buffer slot and one panel per slot buffer). Otherwise the cycle count lies between
``lower_bound_cycles`` and ``serial_cycles``; ``estimate_cycles`` is a software-pipeline
estimate (prologue + max(remaining compute, remaining DMA) + epilogue).

AXI memory model: lower bounds only, from per-channel beats, bursts, outstanding
transactions and the shared DRAM rate (see ``axi_bounds``).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .sim import config_value

KIB4 = 4096


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Design:
    """One simulator configuration. Field names are npu-sim flag names with underscores."""

    m: int = 16
    n: int = 16
    k: int = 32
    groups: int = 1
    tile_m: int = 8
    tile_n: int = 8
    tile_k: int = 0
    buffer_slots: int = 2
    panel_buffers: int = 2
    weight_bits: int = 8
    weight_encoding: str = "dense"
    per_channel_vectors: int = 0
    macs_per_cycle: int = 64
    array_rows: int = 0
    array_cols: int = 0
    dataflow: str = "os"
    pipeline_cycles: int = 4
    memory_model: str = "serial"
    memory_bytes_per_cycle: int = 16
    memory_latency: int = 12
    dma_burst_bytes: int = 0
    weight_cache_bytes: int = 0
    axi_data_bytes: int | None = None
    axi_max_burst_beats: int | None = None
    axi_read_outstanding: int | None = None
    axi_write_outstanding: int | None = None
    axi_read_latency: int | None = None
    axi_write_latency: int | None = None
    dram_bytes_per_cycle: int = 0
    cores: int = 1
    partition: str = "auto"
    sram_bytes: int = 262144
    frequency_mhz: float = 200.0

    # Alternative spellings accepted from flag dicts and nested result configs.
    ALIASES = {
        "axi_data_bytes": ("data_bytes",),
        "axi_max_burst_beats": ("max_burst_beats",),
        "axi_read_outstanding": ("read_outstanding", "axi_outstanding_reads"),
        "axi_write_outstanding": ("write_outstanding", "axi_outstanding_writes"),
        "axi_read_latency": ("read_latency",),
        "axi_write_latency": ("write_latency",),
        "memory_bytes_per_cycle": ("bytes_per_cycle",),
        "memory_latency": ("latency",),
    }

    @classmethod
    def from_flags(cls, flags: Mapping[str, Any] | None = None, **overrides: Any) -> "Design":
        """Design from npu-sim flags ({"tile_m": 8} or {"--tile-m": 8}); unknown keys ignored."""
        values: dict[str, Any] = {}
        fields = {field.name: field for field in dataclasses.fields(cls)}
        aliases = {alias: name for name, names in cls.ALIASES.items() for alias in names}
        for key, value in {**(flags or {}), **overrides}.items():
            name = str(key).lstrip("-").replace("-", "_")
            name = aliases.get(name, name)
            if name in fields and value is not None:
                values[name] = _coerce(fields[name], value)
        return cls(**values)

    @classmethod
    def from_result(cls, result: Mapping[str, Any]) -> "Design":
        """Design echoed in an npu-sim result. Keys an older simulator does not report take
        the behaviour that simulator had (one panel buffer, serial memory, one core)."""
        config = result["config"]
        legacy = {"panel_buffers": 1, "groups": 1, "memory_model": "serial", "cores": 1,
                  "dataflow": "os", "partition": "auto", "dram_bytes_per_cycle": 0}
        values: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            for key in (field.name,) + cls.ALIASES.get(field.name, ()):
                value = config_value(config, key, None)
                if value is not None:
                    values[field.name] = _coerce(field, value)
                    break
            else:
                if field.name in legacy:
                    values[field.name] = legacy[field.name]
        quant = config.get("quant", {}) if isinstance(config.get("quant"), Mapping) else {}
        values["per_channel_vectors"] = sum(bool(quant.get(key)) for key in
                                            ("weight_zero_points", "multipliers", "shifts"))
        return cls(**values)

    def replace(self, **changes: Any) -> "Design":
        return dataclasses.replace(self, **changes)

    @property
    def array(self) -> bool:
        return self.array_rows > 0 and self.array_cols > 0

    @property
    def peak_macs_per_cycle(self) -> int:
        per_core = self.array_rows * self.array_cols if self.array else self.macs_per_cycle
        return per_core * self.cores

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k * self.groups

    @property
    def panel_depth(self) -> int:
        return min(self.k, self.tile_k) if self.tile_k else self.k

    @property
    def reduction_blocks(self) -> int:
        return ceil_div(self.k, self.panel_depth)

    @property
    def bandwidth(self) -> float:
        """Peak DRAM bytes per cycle (the roofline's memory roof)."""
        if self.memory_model == "axi":
            channels = 2 * self.cores * (self.axi_data_bytes or 0)
            if self.dram_bytes_per_cycle:
                return float(min(self.dram_bytes_per_cycle, channels) if channels
                             else self.dram_bytes_per_cycle)
            return float(channels)
        return float(self.memory_bytes_per_cycle)

    @property
    def ridge(self) -> float:
        """Arithmetic intensity (MAC per DRAM byte) where the compute and memory roofs meet."""
        return self.peak_macs_per_cycle / self.bandwidth


def _coerce(field: dataclasses.Field, value: Any) -> Any:
    if field.type in ("float",):
        return float(value)
    if field.type in ("str",):
        return str(value)
    if isinstance(value, bool):
        raise ValueError(f"{field.name} must be a number")
    return int(value)


@dataclass(frozen=True)
class PanelClass:
    """``count`` identical panels: a x b tile, depth c, first panel of its tile or not."""

    rows: int
    cols: int
    depth: int
    first: bool
    last: bool
    count: int


def _extent_classes(total: int, tile: int) -> list[tuple[int, int]]:
    tile = min(tile, total)
    classes = [(tile, total // tile)]
    if total % tile:
        classes.append((total % tile, 1))
    return classes


def _block_classes(design: Design) -> list[tuple[int, int, bool, bool]]:
    """(depth, count, first, last) for the reduction panels of one tile."""
    depth = design.panel_depth
    full, tail = divmod(design.k, depth)
    blocks = full + (1 if tail else 0)
    if blocks == 1:
        return [(design.k, 1, True, True)]
    classes = [(depth, 1, True, False)]
    middle = full - 1 - (0 if tail else 1)
    if middle > 0:
        classes.append((depth, middle, False, False))
    classes.append((tail, 1, False, True) if tail else (depth, 1, False, True))
    return classes


def panel_classes(design: Design) -> Iterator[PanelClass]:
    for rows, row_count in _extent_classes(design.m, design.tile_m):
        for cols, col_count in _extent_classes(design.n, design.tile_n):
            for depth, count, first, last in _block_classes(design):
                yield PanelClass(rows, cols, depth, first, last,
                                 row_count * col_count * count * design.groups)


def tile_classes(design: Design) -> Iterator[tuple[int, int, int]]:
    for rows, row_count in _extent_classes(design.m, design.tile_m):
        for cols, col_count in _extent_classes(design.n, design.tile_n):
            yield rows, cols, row_count * col_count * design.groups


def weight_panel_bytes(design: Design, depth: int, cols: int) -> int:
    if design.weight_encoding != "dense":
        raise ValueError("the analytic model covers dense weights only")
    return ceil_div(depth * cols * design.weight_bits, 8)


def load_bytes(design: Design, rows: int, cols: int, depth: int, first: bool) -> int:
    params = 4 * cols * (1 + design.per_channel_vectors) if first else 0
    return rows * depth + weight_panel_bytes(design, depth, cols) + params


def store_bytes(rows: int, cols: int) -> int:
    return rows * cols


def panel_compute_cycles(design: Design, rows: int, cols: int, depth: int) -> int:
    if not design.array:
        work = ceil_div(rows * cols * depth, design.macs_per_cycle)
    else:
        ar, ac = design.array_rows, design.array_cols
        fill_drain = ar + ac - 2
        if design.dataflow == "os":
            work = ceil_div(rows, ar) * ceil_div(cols, ac) * depth + fill_drain
        elif design.dataflow == "ws":
            work = ceil_div(depth, ar) * ceil_div(cols, ac) * rows + ar + fill_drain
        else:
            raise ValueError("dataflow must be os or ws")
    return work + design.pipeline_cycles


def dma_cycles(design: Design, nbytes: int) -> int:
    """Serial DMA occupancy of one request."""
    if nbytes == 0:
        return 0
    bw, latency, burst = design.memory_bytes_per_cycle, design.memory_latency, design.dma_burst_bytes
    if not burst:
        return latency + ceil_div(nbytes, bw)
    full, tail = divmod(nbytes, burst)
    cycles = full * (latency + ceil_div(burst, bw))
    return cycles + (latency + ceil_div(tail, bw) if tail else 0)


def dma_bursts(design: Design, nbytes: int) -> int:
    return ceil_div(nbytes, design.dma_burst_bytes) if design.dma_burst_bytes else 1


def scratchpad_bytes(design: Design, panel_buffers: int | None = None) -> int:
    """SRAM reserved per core: slots x (panel buffers x first-panel bytes + 5 B per output)
    plus the weight cache. 5 B per output = INT32 partial sum + INT8 result."""
    buffers = design.panel_buffers if panel_buffers is None else panel_buffers
    rows, cols = min(design.tile_m, design.m), min(design.tile_n, design.n)
    panel = load_bytes(design, rows, cols, design.panel_depth, True)
    return design.buffer_slots * (buffers * panel + 5 * rows * cols) + design.weight_cache_bytes


@dataclass(frozen=True)
class Analysis:
    """Model output for one design. Serial-only fields are None under the AXI model."""

    design: Design
    macs: int
    read_bytes: int
    write_bytes: int
    weight_bytes: int
    dma_requests: int
    compute_busy_cycles: int
    memory_cycles: int
    memory_data_cycles: int
    lower_bound_cycles: int
    estimate_cycles: int
    scratchpad_bytes: int
    dma_bursts: int | None = None
    dma_busy_cycles: int | None = None
    serial_cycles: int | None = None
    exact_cycles: int | None = None

    @property
    def dram_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.macs / self.dram_bytes

    @property
    def attainable_macs_per_cycle(self) -> float:
        return attainable(self.design, self.arithmetic_intensity)

    @property
    def compute_cycles_per_core(self) -> int:
        return ceil_div(self.compute_busy_cycles, self.design.cores)

    @property
    def resource_bound_cycles(self) -> int:
        return max(self.compute_cycles_per_core, self.memory_cycles)

    @property
    def bound(self) -> str:
        """What limits the schedule: compute, bandwidth (memory payload) or latency
        (per-request memory latency that the payload alone would not cost)."""
        if self.compute_cycles_per_core >= self.memory_cycles:
            return "compute"
        return "bandwidth" if self.memory_data_cycles >= self.compute_cycles_per_core else "latency"

    def as_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "design"}
        out.update(arithmetic_intensity=self.arithmetic_intensity, bound=self.bound,
                   attainable_macs_per_cycle=self.attainable_macs_per_cycle,
                   resource_bound_cycles=self.resource_bound_cycles)
        return out


def attainable(design: Design, intensity: float) -> float:
    """Roofline: min(peak MACs per cycle, DRAM bytes per cycle x MACs per byte)."""
    return min(float(design.peak_macs_per_cycle), design.bandwidth * intensity)


def analyze(design: Design) -> Analysis:
    if design.weight_encoding != "dense":
        raise ValueError("the analytic model covers dense weights only")
    if design.weight_cache_bytes:
        raise ValueError("the analytic model does not cover the weight cache")
    read = write = weights = requests = bursts = 0
    compute = data = loads = later_loads = stores = 0
    for panel in panel_classes(design):
        nbytes = load_bytes(design, panel.rows, panel.cols, panel.depth, panel.first)
        if not panel.first:
            later_loads += panel.count * dma_cycles(design, nbytes)
        read += panel.count * nbytes
        weights += panel.count * weight_panel_bytes(design, panel.depth, panel.cols)
        requests += panel.count
        bursts += panel.count * dma_bursts(design, nbytes)
        loads += panel.count * dma_cycles(design, nbytes)
        data += panel.count * ceil_div(nbytes, design.memory_bytes_per_cycle)
        compute += panel.count * panel_compute_cycles(design, panel.rows, panel.cols, panel.depth)
    for rows, cols, count in tile_classes(design):
        nbytes = store_bytes(rows, cols)
        write += count * nbytes
        requests += count
        bursts += count * dma_bursts(design, nbytes)
        stores += count * dma_cycles(design, nbytes)
        data += count * ceil_div(nbytes, design.memory_bytes_per_cycle)
    common = dict(design=design, macs=design.macs, read_bytes=read, write_bytes=write,
                  weight_bytes=weights, dma_requests=requests, compute_busy_cycles=compute,
                  scratchpad_bytes=scratchpad_bytes(design))
    if design.memory_model == "axi":
        axi = axi_bounds(design, compute)
        memory = max(ceil_div(axi["read_channel_cycles"], design.cores),
                     ceil_div(axi["write_channel_cycles"], design.cores), axi["dram_cycles"])
        payload = max(ceil_div(axi["read_beats"], design.cores),
                      ceil_div(axi["write_beats"], design.cores), axi["dram_cycles"])
        lower = axi["lower_bound_cycles"]
        return Analysis(**common, memory_cycles=memory, memory_data_cycles=payload,
                        lower_bound_cycles=lower, estimate_cycles=max(lower, memory))
    if design.memory_model != "serial":
        raise ValueError("memory model must be serial or axi")
    if design.cores != 1:
        raise ValueError("more than one core requires the AXI memory model")
    dma = loads + stores
    # Tiles run in order: the first is a full tile, the last is the bottom-right edge tile.
    rows0, cols0 = min(design.tile_m, design.m), min(design.tile_n, design.n)
    first_load = dma_cycles(design, load_bytes(design, rows0, cols0, design.panel_depth, True))
    last_rows = design.m % design.tile_m or rows0
    last_cols = design.n % design.tile_n or cols0
    last_depth = design.k % design.panel_depth or design.panel_depth
    last_compute = panel_compute_cycles(design, last_rows, last_cols, last_depth)
    last_store = dma_cycles(design, store_bytes(last_rows, last_cols))
    serial = compute + dma
    # Compute cannot start before the first load and the last store follows the last
    # compute; every load precedes the last compute; the DMA is one server.
    lower = max(compute + first_load + last_store, loads + last_compute + last_store, dma)
    if design.panel_buffers == 1:
        # One panel buffer per slot: a tile's next panel loads only after its previous panel
        # computed, so compute and every non-first panel load form one chain.
        lower = max(lower, first_load + compute + later_loads + last_store)
    estimate = max(lower, first_load + last_compute + last_store
                   + max(compute - last_compute, dma - first_load - last_store))
    one_tile = design.groups == 1 and rows0 == design.m and cols0 == design.n
    no_overlap = (design.buffer_slots == 1 or one_tile) and         (design.reduction_blocks == 1 or design.panel_buffers == 1)
    return Analysis(**common, memory_cycles=dma, memory_data_cycles=data,
                    lower_bound_cycles=lower, estimate_cycles=estimate, dma_bursts=bursts,
                    dma_busy_cycles=dma, serial_cycles=serial,
                    exact_cycles=serial if no_overlap else None)


def _runs(rows: int, run: int, stride: int) -> list[int]:
    """Lengths of ``rows`` runs of ``run`` bytes at ``stride``, merging contiguous runs."""
    return [rows * run] if run == stride else [run] * rows


def axi_bounds(design: Design, compute_cycles: int | None = None) -> dict[str, int]:
    """Lower bounds for the AXI model, valid for any burst split and address alignment.

    Per channel: every beat takes one data cycle, one burst issues per cycle, and a burst
    holds one of N outstanding slots for at least latency + beats cycles. The DRAM
    timeline moves at most dram_bytes_per_cycle payload bytes per cycle for all channels.
    """
    needed = ("axi_data_bytes", "axi_max_burst_beats", "axi_read_outstanding",
              "axi_write_outstanding", "axi_read_latency", "axi_write_latency")
    missing = [name for name in needed if getattr(design, name) is None]
    if missing:
        raise ValueError("AXI bounds need " + ", ".join(missing))
    db = design.axi_data_bytes or 1
    burst_bytes = min(KIB4, (design.axi_max_burst_beats or 1) * db)
    row_a, row_o = design.groups * design.k, design.groups * design.n

    def channel(runs: list[tuple[int, int]], latency: int, outstanding: int) -> dict[str, int]:
        beats = bursts = slot_time = 0
        for length, count in runs:
            run_beats = ceil_div(length, db)
            run_bursts = ceil_div(length, burst_bytes)
            beats += count * run_beats
            bursts += count * run_bursts
            slot_time += count * (run_bursts * latency + run_beats)
        return {"beats": beats, "bursts": bursts,
                "cycles": max(beats, bursts, ceil_div(slot_time, outstanding))}

    reads: list[tuple[int, int]] = []
    for p in panel_classes(design):
        for length in _runs(p.rows, p.depth, row_a):
            reads.append((length, p.count))
        reads.append((weight_panel_bytes(design, p.depth, p.cols), p.count))
        if p.first:
            reads.append((4 * p.cols * (1 + design.per_channel_vectors), p.count))
    writes: list[tuple[int, int]] = []
    for rows, cols, count in tile_classes(design):
        for length in _runs(rows, cols, row_o):
            writes.append((length, count))
    read = channel(reads, design.axi_read_latency or 0, design.axi_read_outstanding or 1)
    write = channel(writes, design.axi_write_latency or 0, design.axi_write_outstanding or 1)
    payload = sum(length * count for length, count in reads + writes)
    dram = ceil_div(payload, design.dram_bytes_per_cycle) if design.dram_bytes_per_cycle else 0
    if compute_cycles is None:
        compute_cycles = sum(p.count * panel_compute_cycles(design, p.rows, p.cols, p.depth)
                             for p in panel_classes(design))
    rows0, cols0 = min(design.tile_m, design.m), min(design.tile_n, design.n)
    first = load_bytes(design, rows0, cols0, design.panel_depth, True)
    last = store_bytes(design.m % design.tile_m or rows0, design.n % design.tile_n or cols0)
    chain = (design.axi_read_latency or 0) + ceil_div(first, db) + compute_cycles \
        + ceil_div(last, db) + (design.axi_write_latency or 0)
    per_core = max(ceil_div(read["cycles"], design.cores), ceil_div(write["cycles"], design.cores))
    return {"read_beats": read["beats"], "read_bursts": read["bursts"],
            "read_channel_cycles": read["cycles"], "write_beats": write["beats"],
            "write_bursts": write["bursts"], "write_channel_cycles": write["cycles"],
            "dram_cycles": dram, "compute_chain_cycles": chain,
            "lower_bound_cycles": max(per_core, dram,
                                      chain if design.cores == 1 else ceil_div(compute_cycles,
                                                                               design.cores))}
