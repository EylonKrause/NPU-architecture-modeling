"""Timing model of npu-sim, derived from docs/architecture.md (sections 3 to 7) and the
interface comments in include/npu, independently of the C++ implementation.

Execution model. Each of G groups is an independent M x n x k GEMM. Its output is cut
into tile_m x tile_n tiles (edge tiles are smaller); each tile walks the reduction in
panels of tile_k (0 = all of k). Per panel the core loads

    A panel  a * c bytes            (a tile rows, c reduction columns)
    W panel  ceil(c * b * bits / 8)  (b tile columns), or its palette encoding
    params   4 * b * (1 + vectors)   on the first panel only (INT32 bias + per-channel vectors)

computes it, and after the last panel stores a * b output bytes. A weight panel the core's
LRU weight cache holds is not loaded again, nor an activation panel its LRU activation cache
holds (lines: group, row tile, reduction panel, so the column tiles of a row tile share
them).

Depthwise datapath (depthwise_macs_per_cycle > 0, G > 1, N = 1): the grid is one group of
G columns, so a tile spans b consecutive groups (channels), its A panel is a * b * c bytes
and it computes in ceil(a * b * c / depthwise rate) + pipeline_cycles.

Implicit convolution (conv geometry): the activation window holds the NHWC input and an A
panel is the input footprint of its output pixels over its kernel rows (tile_k is whole
kernel rows): per input row the hull of the needed columns, clipped to the image, times the
panel's input channels. Its runs are one per input row when the panel holds all C
channels, else one per pixel of the hull.

Compute cycles per panel (c = panel depth, P = MACs per cycle, A_r x A_c array), plus
pipeline_cycles; a compressed panel takes max(MAC cycles, decoder cycles) because the
weight decoder runs ahead of the array:
    depthwise    ceil(a * b * c / depthwise rate)
    vector       ceil(a * b * c / P)
    os           ceil(a/A_r) * ceil(b/A_c) * c + (A_r + A_c - 2)
    ws           A_r + (F - 1) * max(a, A_r) + a + (A_r + A_c - 2),
                 F = ceil(c/A_r) * ceil(b/A_c) weight folds: the first A_r-cycle weight
                 preload is exposed and each later one overlaps the previous fold's
                 a-row stream, so a fold occupies max(a, A_r) cycles and the last one a
    os-per-wave  ceil(a/A_r) * ceil(b/A_c) * (c + A_r + A_c - 2)
                 (npu-sim builds without --dataflow expose fill/drain on every wave)

Two models share these rules.

``analyze()`` is closed form over panel classes: exact traffic, request counts and
compute busy cycles (cluster and both caches included), lower bounds, and a three-stage
pipeline estimate (first load + the slowest stage's remaining work + last compute and
store) in which every request takes its time alone on the memory system: under the
serial model ``latency + ceil(bytes / bytes_per_cycle)`` per burst, under the AXI model
the channel recurrence below on a private DRAM timeline. ``exact_cycles`` is set where the
schedule cannot overlap (one buffer slot and one panel per slot buffer).

``schedule()`` replays every request in order: slots, panel buffers, the serial server's
alternation, each AXI channel's outstanding window, the shared DRAM byte timeline and its
fixed arbitration order, the cluster split and each core's two LRU caches. It returns the
cycle count and the counters npu-sim reports.

``observed()`` reads the same quantities from an npu-sim result (schema_version 2:
"memory", "cluster" and "cores"), and ``observed_bound()`` classifies it as ``bound``
classifies a model.
"""
from __future__ import annotations

import bisect
import collections
import dataclasses
import difflib
import functools
import heapq
import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .sim import CONFIG_PATHS, FLAT_KEYS, config_value

KIB4 = 4096
# npu-sim flags whose option-table leaf, and so the Design field, has another name.
FLAG_ALIASES = {"axi_burst_beats": "axi_max_burst_beats"}


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
    palette_bits: int = 4
    weight_decode_values_per_cycle: int = 16
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
    depthwise_macs_per_cycle: int = 0
    activation_cache_bytes: int = 0
    conv_input_h: int = 0
    conv_input_w: int = 0
    conv_input_c: int = 0
    conv_kernel_h: int = 1
    conv_kernel_w: int = 1
    conv_stride_h: int = 1
    conv_stride_w: int = 1
    conv_dilation_h: int = 1
    conv_dilation_w: int = 1
    conv_pad_top: int = 0
    conv_pad_left: int = 0
    conv_pad_bottom: int = 0
    conv_pad_right: int = 0

    @classmethod
    def field_name(cls, key: str) -> str | None:
        """Field for a flag ("--tile-m", "tile_m") or an option-table path ("mapping.tile_m")."""
        name = str(key).lstrip("-").replace("-", "_")
        name = FLAT_KEYS.get(name, name)
        name = FLAG_ALIASES.get(name, name)
        return name if name in _FIELDS else None

    @classmethod
    def normalize(cls, flags: Mapping[str, Any]) -> dict[str, Any]:
        """``flags`` keyed by field name; any key that names no field is a ValueError."""
        values: dict[str, Any] = {}
        for key, value in flags.items():
            name = cls.field_name(key)
            if name is None:
                close = difflib.get_close_matches(str(key).lstrip("-").replace("-", "_"),
                                                  _FIELDS, n=1)
                hint = f"; did you mean {close[0]!r}?" if close else ""
                raise ValueError(f"unknown design parameter {key!r}{hint}")
            values[name] = value
        return values

    @classmethod
    def from_flags(cls, flags: Mapping[str, Any] | None = None, strict: bool = False,
                   **overrides: Any) -> "Design":
        """Design from npu-sim flags ({"tile_m": 8} or {"--tile-m": 8}). Keys that name no
        field are skipped (simulator-only flags such as --input), or rejected when
        ``strict``."""
        given = {**(flags or {}), **overrides}
        if strict:
            given = cls.normalize(given)
        values: dict[str, Any] = {}
        for key, value in given.items():
            name = cls.field_name(key)
            if name is not None and value is not None:
                values[name] = _coerce(_FIELDS[name], value)
        return cls(**values)

    @classmethod
    def from_result(cls, result: Mapping[str, Any]) -> "Design":
        """Design echoed in an npu-sim result, flat or option-table config.

        A key the result does not report takes the value of the build that wrote it, which
        lacks that option: one panel buffer per slot, one group, serial memory, one core, and
        os-per-wave for an array without a dataflow key. Every other key the model reads must
        be present."""
        config = result["config"]
        legacy = {"panel_buffers": 1, "groups": 1, "memory_model": "serial", "cores": 1,
                  "partition": "auto", "dram_bytes_per_cycle": 0,
                  "depthwise_macs_per_cycle": 0, "activation_cache_bytes": 0,
                  **{name: _FIELDS[name].default for name in CONV_FIELDS}}
        optional = {"dataflow", "per_channel_vectors"} | {name for name in _FIELDS
                                                          if name.startswith("axi_")}
        values: dict[str, Any] = {}
        missing = []
        for name, field in _FIELDS.items():
            value = config_value(config, name, None)
            if value is not None:
                values[name] = _coerce(field, value)
            elif name in legacy:
                values[name] = legacy[name]
            elif name not in optional:
                missing.append(CONFIG_PATHS.get(name, name))
        if missing:
            raise ValueError("result config lacks " + ", ".join(missing))
        if "dataflow" not in values:
            array = values["array_rows"] > 0 and values["array_cols"] > 0
            values["dataflow"] = "os-per-wave" if array else "os"
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
    def channel_wise(self) -> bool:
        """The depthwise datapath runs the problem: a rate is set, G > 1 and N = 1."""
        return self.depthwise_macs_per_cycle > 0 and self.groups > 1 and self.n == 1

    @property
    def grid_groups(self) -> int:
        """Groups of the output tile grid: one on the depthwise datapath, else G."""
        return 1 if self.channel_wise else self.groups

    @property
    def grid_columns(self) -> int:
        """Columns per grid group: the G channels on the depthwise datapath, else N."""
        return self.groups if self.channel_wise else self.n

    @property
    def conv(self) -> bool:
        """Implicit convolution: the activations are an NHWC input's im2col view."""
        return bool(self.conv_input_h or self.conv_input_w or self.conv_input_c)

    @property
    def peak_macs_per_cycle(self) -> int:
        if self.channel_wise:
            return self.depthwise_macs_per_cycle * self.cores
        per_core = self.array_rows * self.array_cols if self.array else self.macs_per_cycle
        return per_core * self.cores

    @property
    def array_macs_per_cycle(self) -> int:
        """Peak of the array or vector datapath, whichever datapath runs the problem."""
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


_FIELDS = {field.name: field for field in dataclasses.fields(Design)}
CONV_FIELDS = tuple(name for name in _FIELDS if name.startswith("conv_"))


def _coerce(field: dataclasses.Field, value: Any) -> Any:
    if field.type in ("float",):
        return float(value)
    if field.type in ("str",):
        return str(value)
    if isinstance(value, bool):
        raise ValueError(f"{field.name} must be a number")
    return int(value)


# ---------------------------------------------------------------- implicit convolution --
# The activation window holds an NHWC INT8 input of H x W x C. GEMM row m is output pixel
# (m // W_o, m % W_o); a group's K index is (ky * kernel_w + kx) * C_g + ci, so one kernel
# row is kernel_w * C_g consecutive K and tile_k is 0 or whole kernel rows. A panel reads,
# per input row it touches, the hull of the columns its output pixels need over the
# panel's kernel rows, clipped to the image: padding reads nothing.

@dataclass(frozen=True)
class ConvGeometry:
    output_h: int
    output_w: int
    group_channels: int     # C_g = input channels / G
    kernel_row_k: int       # kernel_w * C_g


def _output_extent(size: int, kernel: int, stride: int, dilation: int, low: int,
                   high: int) -> int:
    reach = dilation * (kernel - 1) + 1
    if reach > size + low + high:
        raise ValueError(f"conv kernel reach {reach} exceeds the padded input {size + low + high}")
    return (size + low + high - reach) // stride + 1


def conv_geometry(design: Design) -> ConvGeometry:
    """Output extents and K layout of ``design``'s convolution, checked against M, K, G."""
    if not design.conv:
        raise ValueError("the design has no conv geometry")
    d = design
    if d.conv_input_c % d.groups:
        raise ValueError(f"conv input channels {d.conv_input_c} are not a multiple of "
                         f"groups {d.groups}")
    oh = _output_extent(d.conv_input_h, d.conv_kernel_h, d.conv_stride_h, d.conv_dilation_h,
                        d.conv_pad_top, d.conv_pad_bottom)
    ow = _output_extent(d.conv_input_w, d.conv_kernel_w, d.conv_stride_w, d.conv_dilation_w,
                        d.conv_pad_left, d.conv_pad_right)
    channels = d.conv_input_c // d.groups
    row_k = d.conv_kernel_w * channels
    if d.m != oh * ow or d.k != d.conv_kernel_h * row_k:
        raise ValueError(f"conv geometry gives M = {oh * ow} and K = {d.conv_kernel_h * row_k}, "
                         f"the design has M = {d.m} and K = {d.k}")
    return ConvGeometry(oh, ow, channels, row_k)


def _conv_key(design: Design) -> tuple[int, ...]:
    return tuple(getattr(design, name) for name in CONV_FIELDS)


@functools.lru_cache(maxsize=1 << 16)
def _column_hull(key: tuple[int, ...], first: int, last: int) -> tuple[int, int] | None:
    """Input columns read by output columns first..last over every kernel column."""
    _, width, _, _, kernel_w, _, stride, _, dilation, _, left, _, _ = key
    low = high = None
    for kx in range(kernel_w):
        offset = kx * dilation - left
        # The smallest output column whose input column is >= 0, and the largest whose
        # input column is <= width - 1.
        lo = max(first, ceil_div(-offset, stride))
        if lo <= last and lo * stride + offset < width:
            value = lo * stride + offset
            low = value if low is None else min(low, value)
        hi = min(last, (width - 1 - offset) // stride)
        if hi >= first and hi * stride + offset >= 0:
            value = hi * stride + offset
            high = value if high is None else max(high, value)
    if low is None or high is None or low > high:
        return None
    return low, high


def _spans(key: tuple[int, ...], output_w: int, first_output: int, outputs: int,
           kernel_first: int, kernel_rows: int) -> tuple[tuple[int, int, int], ...]:
    height = key[0]
    stride_h, dilation_h, top = key[5], key[7], key[9]
    last_output = first_output + outputs - 1
    hulls: dict[int, list[int]] = {}
    for oy in range(first_output // output_w, last_output // output_w + 1):
        first = first_output % output_w if oy == first_output // output_w else 0
        last = last_output % output_w if oy == last_output // output_w else output_w - 1
        columns = _column_hull(key, first, last)
        if columns is None:
            continue
        for ky in range(kernel_first, kernel_first + kernel_rows):
            iy = oy * stride_h + ky * dilation_h - top
            if 0 <= iy < height:
                hull = hulls.setdefault(iy, [columns[0], columns[1]])
                hull[0], hull[1] = min(hull[0], columns[0]), max(hull[1], columns[1])
    return tuple((row, hulls[row][0], hulls[row][1]) for row in sorted(hulls))


def kernel_rows(design: Design, block: int) -> tuple[int, int]:
    """(first kernel row, kernel rows) of reduction panel ``block``."""
    geometry = conv_geometry(design)
    if design.tile_k % geometry.kernel_row_k:
        raise ValueError(f"with conv geometry tile_k must be 0 or a multiple of "
                         f"{geometry.kernel_row_k} (whole kernel rows)")
    per_block = design.tile_k // geometry.kernel_row_k if design.tile_k else design.conv_kernel_h
    first = block * per_block
    return first, min(per_block, design.conv_kernel_h - first)


def input_spans(design: Design, first_output: int, outputs: int,
                block: int) -> tuple[tuple[int, int, int], ...]:
    """(input row, first column, last column) of every input row that output pixels
    first_output..first_output + outputs - 1 read over the kernel rows of ``block``."""
    geometry = conv_geometry(design)
    first, count = kernel_rows(design, block)
    return _spans(_conv_key(design), geometry.output_w, first_output, outputs, first, count)


def footprint_pixels(spans: Sequence[tuple[int, int, int]]) -> int:
    return sum(last - first + 1 for _, first, last in spans)


@dataclass(frozen=True)
class Footprint:
    """One activation panel's input footprint: its spans, hull pixels and hull widths."""

    spans: tuple[tuple[int, int, int], ...]
    pixels: int
    widths: tuple[int, ...]


@functools.lru_cache(maxsize=1024)
def _footprint_table(key: tuple[int, ...], m: int, k: int, groups: int, tile_m: int,
                     tile_k: int) -> tuple[ConvGeometry, tuple[tuple[Footprint, ...], ...]]:
    design = Design(m=m, k=k, groups=groups, tile_m=tile_m, tile_k=tile_k,
                    **dict(zip(CONV_FIELDS, key)))
    geometry = conv_geometry(design)
    blocks = [kernel_rows(design, block) for block in range(design.reduction_blocks)]
    table = []
    for row in range(0, m, tile_m):
        entry = []
        for first, count in blocks:
            spans = _spans(key, geometry.output_w, row, min(tile_m, m - row), first, count)
            entry.append(Footprint(spans, footprint_pixels(spans),
                                   tuple(high - low for _, low, high in spans)))
        table.append(tuple(entry))
    return geometry, tuple(table)


def footprints(design: Design) -> tuple[ConvGeometry, tuple[tuple[Footprint, ...], ...]]:
    """The conv geometry and, per row tile and reduction block, the panel footprint."""
    return _footprint_table(_conv_key(design), design.m, design.k, design.groups,
                            design.tile_m, design.tile_k)


def tile_footprint(design: Design, row: int, rows: int, block: int) -> Footprint:
    """Footprint of output rows row..row + rows - 1 over the kernel rows of ``block``."""
    if row % design.tile_m == 0 and rows == min(design.tile_m, design.m - row):
        return footprints(design)[1][row // design.tile_m][block]
    spans = input_spans(design, row, rows, block)
    return Footprint(spans, footprint_pixels(spans),
                     tuple(high - low for _, low, high in spans))


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
        for cols, col_count in _extent_classes(design.grid_columns, design.tile_n):
            for depth, count, first, last in _block_classes(design):
                yield PanelClass(rows, cols, depth, first, last,
                                 row_count * col_count * count * design.grid_groups)


def tile_classes(design: Design) -> Iterator[tuple[int, int, int]]:
    for rows, row_count in _extent_classes(design.m, design.tile_m):
        for cols, col_count in _extent_classes(design.grid_columns, design.tile_n):
            yield rows, cols, row_count * col_count * design.grid_groups


def lut_entries(design: Design, depth: int, cols: int) -> int:
    """Palette LUT entries of one panel, taking every index value as present: the encoded
    size depends on the data, and a calibrated input fills the LUT."""
    return min(1 << design.palette_bits, depth * cols)


def weight_bytes(design: Design, depth: int, cols: int) -> int:
    """Bytes of one stored weight panel: packed dense values, or the palette encoding
    (4-byte header, signed INT8 LUT, packed indices)."""
    if design.weight_encoding == "dense":
        return ceil_div(depth * cols * design.weight_bits, 8)
    if design.weight_encoding == "palette":
        return (4 + lut_entries(design, depth, cols)
                + ceil_div(depth * cols * design.palette_bits, 8))
    raise ValueError("the analytic model covers dense and palette weights; sparse sizes "
                     "depend on the data")


def param_record_bytes(design: Design) -> int:
    """One output channel's parameter record: INT32 bias, then each per-channel vector."""
    return 4 * (1 + design.per_channel_vectors)


def store_bytes(rows: int, cols: int) -> int:
    return rows * cols


def activation_bytes(design: Design, row: int, rows: int, cols: int, block: int) -> int:
    """Activation bytes of a panel load of output rows row..row + rows - 1 and ``cols``
    columns: rows x depth for a GEMM tile, rows x channels x depth on the depthwise
    datapath, or with conv geometry the input footprint times the panel's channels."""
    width = cols if design.channel_wise else 1
    if not design.conv:
        return rows * width * block_extent(design, block)[1]
    pixels = tile_footprint(design, row, rows, block).pixels
    return pixels * footprints(design)[0].group_channels * width


def largest_activation_bytes(design: Design, cols: int) -> int:
    """The largest activation panel of any row tile and reduction block, ``cols`` wide."""
    if not design.conv:
        return activation_bytes(design, 0, min(design.tile_m, design.m), cols, 0)
    geometry, table = footprints(design)
    pixels = max(panel.pixels for entry in table for panel in entry)
    return pixels * geometry.group_channels * (cols if design.channel_wise else 1)


def mac_cycles(design: Design, rows: int, cols: int, depth: int) -> int:
    """MAC-array cycles of one panel, without the per-panel pipeline charge. The depthwise
    datapath takes rows x channels x depth MACs at its own rate."""
    if design.channel_wise:
        return ceil_div(rows * cols * depth, design.depthwise_macs_per_cycle)
    if not design.array:
        return ceil_div(rows * cols * depth, design.macs_per_cycle)
    ar, ac = design.array_rows, design.array_cols
    fill_drain = ar + ac - 2
    if design.dataflow == "os":
        return ceil_div(rows, ar) * ceil_div(cols, ac) * depth + fill_drain
    if design.dataflow == "ws":
        folds = ceil_div(depth, ar) * ceil_div(cols, ac)
        return ar + (folds - 1) * max(rows, ar) + rows + fill_drain
    if design.dataflow == "os-per-wave":
        return ceil_div(rows, ar) * ceil_div(cols, ac) * (depth + fill_drain)
    raise ValueError("dataflow must be os, ws or os-per-wave")


def decode_cycles(design: Design, depth: int, cols: int) -> int:
    """Weight-decoder cycles of one panel; dense weights need none."""
    if design.weight_encoding == "dense" or design.channel_wise:
        return 0
    return ceil_div(depth * cols, design.weight_decode_values_per_cycle)


def panel_compute_cycles(design: Design, rows: int, cols: int, depth: int) -> int:
    """MAC cycles, or the decoder's when it is slower (it runs ahead of the array), plus
    the per-panel pipeline charge."""
    return (max(mac_cycles(design, rows, cols, depth), decode_cycles(design, depth, cols))
            + design.pipeline_cycles)


def dma_cycles(design: Design, nbytes: int) -> int:
    """Serial DMA occupancy of one request."""
    if nbytes == 0:
        return 0
    bw, latency = design.memory_bytes_per_cycle, design.memory_latency
    burst = design.dma_burst_bytes
    if not burst:
        return latency + ceil_div(nbytes, bw)
    full, tail = divmod(nbytes, burst)
    cycles = full * (latency + ceil_div(burst, bw))
    return cycles + (latency + ceil_div(tail, bw) if tail else 0)


def dma_bursts(design: Design, nbytes: int) -> int:
    return ceil_div(nbytes, design.dma_burst_bytes) if design.dma_burst_bytes else 1


def scratchpad_bytes(design: Design, panel_buffers: int | None = None) -> int:
    """SRAM reserved per core: slots x (parameters + panel buffers x panel bytes + 5 B per
    output) plus the weight and activation caches. A slot holds the tile's bias and channel
    vectors once, for its lifetime, and no more panel buffers than the tile has panels; a
    panel buffer holds the largest activation panel (with conv geometry, the largest input
    footprint of any panel) and the deepest weight panel. 5 B per output = INT32 partial
    sum + INT8 result."""
    buffers = design.panel_buffers if panel_buffers is None else panel_buffers
    buffers = min(buffers, design.reduction_blocks)
    rows, cols = min(design.tile_m, design.m), min(design.tile_n, design.grid_columns)
    panel = (largest_activation_bytes(design, cols)
             + weight_bytes(design, design.panel_depth, cols))
    params = cols * param_record_bytes(design)
    return (design.buffer_slots * (params + buffers * panel + 5 * rows * cols)
            + design.weight_cache_bytes + design.activation_cache_bytes)


# ------------------------------------------------------------------- closed-form model --

@dataclass(frozen=True)
class CoreAnalysis:
    """Closed-form totals of one core: exact traffic, requests and compute; memory time as
    the sum of each request's time alone on the memory system."""

    core: int
    tiles: int
    macs: int
    compute_cycles: int
    read_bytes: int
    write_bytes: int
    weight_bytes: int
    weight_cache_hits: int
    requests: int
    read_cycles: int        # serial: every request's DMA time; axi: read requests'
    write_cycles: int       # axi: write requests'; serial: 0
    read_beats: int
    write_beats: int
    bursts: int
    first_load: int
    last_compute: int
    last_store: int
    later_loads: int        # loads of every panel but each tile's first
    data_cycles: int        # serial: the DMA time without per-burst latency
    one_slot_cycles: int    # the tiles back to back, each its own load-compute-store chain
    one_slot: bool
    panel_chain: bool       # one panel buffer: each panel load waits for the previous compute
    activation_read_bytes: int = 0
    activation_cache_hits: int = 0

    @property
    def estimate_cycles(self) -> int:
        """Three-stage pipeline: the first load, then the slowest stage's remaining work,
        then the last compute and store. One slot runs the tiles back to back; one panel
        buffer chains each tile's panel loads behind their computes."""
        if not self.tiles:
            return 0
        if self.one_slot:
            return self.one_slot_cycles
        rest = max(self.compute_cycles - self.last_compute, self.read_cycles - self.first_load,
                   self.write_cycles - self.last_store)
        estimate = self.first_load + rest + self.last_compute + self.last_store
        if self.panel_chain:
            estimate = max(estimate, self.first_load + self.compute_cycles + self.later_loads
                           + self.last_store)
        return estimate


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
    compute_cycles_per_core: int = 0
    weight_cache_hits: int = 0
    activation_read_bytes: int = 0
    activation_cache_hits: int = 0
    partition: str = "n"
    cores: tuple[CoreAnalysis, ...] = ()
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
    def resource_bound_cycles(self) -> int:
        return max(self.compute_cycles_per_core, self.memory_cycles)

    @property
    def bound(self) -> str:
        """What limits the schedule: compute, bandwidth (memory payload) or latency
        (per-request memory latency that the payload alone would not cost)."""
        return classify(self.compute_cycles_per_core, self.memory_cycles,
                        self.memory_data_cycles)

    def as_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
               if f.name not in ("design", "cores")}
        out.update(arithmetic_intensity=self.arithmetic_intensity, bound=self.bound,
                   attainable_macs_per_cycle=self.attainable_macs_per_cycle,
                   resource_bound_cycles=self.resource_bound_cycles)
        return out


def classify(compute: float, memory: float, memory_data: float) -> str:
    """compute when the compute engine is the busiest resource; otherwise bandwidth when
    the payload alone outlasts compute, else latency."""
    if compute >= memory:
        return "compute"
    return "bandwidth" if memory_data >= compute else "latency"


def attainable(design: Design, intensity: float) -> float:
    """Roofline: min(peak MACs per cycle, DRAM bytes per cycle x MACs per byte)."""
    return min(float(design.peak_macs_per_cycle), design.bandwidth * intensity)


def _range_classes(first: int, end: int, count: int, tile: int,
                   extent: int) -> list[tuple[int, int, int]]:
    """(size, number, first index) of the tiles with indexes first..end-1 along an axis of
    ``count`` tiles; only the last tile can be short."""
    if first >= end:
        return []
    last = extent - (count - 1) * tile
    if end == count and last != tile:
        classes = [(tile, end - 1 - first, first)] if end - 1 > first else []
        return classes + [(last, 1, count - 1)]
    return [(min(tile, extent), end - first, first)]


def _core_segments(design: Design, plan: ClusterPlan,
                   core: int) -> list[tuple[int, int, int, int, int, int]]:
    """(groups, first group, row tiles first..end, column tiles first..end) of one core's
    tiles: whole groups, or the per-group pieces of a run of row or column tiles."""
    row_tiles = ceil_div(design.m, design.tile_m)
    col_tiles = ceil_div(design.grid_columns, design.tile_n)
    first = plan.first_unit(core)
    end = first + plan.unit_count(core)
    if first == end:
        return []
    if plan.split == "groups":
        return [(end - first, first, 0, row_tiles, 0, col_tiles)]
    per_group = row_tiles if plan.split == "rows" else col_tiles
    segments = []
    for group in range(first // per_group, (end - 1) // per_group + 1):
        lo = max(first, group * per_group) - group * per_group
        hi = min(end, (group + 1) * per_group) - group * per_group
        span = (lo, hi, 0, col_tiles) if plan.split == "rows" else (0, row_tiles, lo, hi)
        if segments and segments[-1][2:] == span:
            count, start = segments[-1][:2]
            segments[-1] = (count + 1, start, *span)
        else:
            segments.append((1, group, *span))
    return segments


def _block_kind(design: Design, block_first: bool, block_last: bool) -> int:
    """A reduction-panel index with that position, for address-dependent request times."""
    if block_first:
        return 0
    return design.reduction_blocks - 1 if block_last else min(1, design.reduction_blocks - 1)


def _segment_shape(design: Design, plan: ClusterPlan, core: int) -> tuple:
    """What analyze_core's totals depend on: per segment its group count and the tile
    count along each axis, and whether it holds the short edge tile. With conv geometry
    every row tile has its own footprint, so the row range itself."""
    row_tiles = ceil_div(design.m, design.tile_m)
    col_tiles = ceil_div(design.grid_columns, design.tile_n)
    return tuple((groups, (row0, row1) if design.conv else row1 - row0, row1 == row_tiles,
                  col1 - col0, col1 == col_tiles)
                 for groups, _, row0, row1, col0, col1 in _core_segments(design, plan, core))


def analyze_cores(design: Design, plan: ClusterPlan,
                  timing: bool = True) -> list[CoreAnalysis]:
    """analyze_core for every core, once per distinct segment shape. Request times take
    the addresses of the first core with that shape; traffic and compute do not depend
    on addresses."""
    shapes: dict[tuple, CoreAnalysis] = {}
    cores = []
    for core in range(plan.cores):
        shape = _segment_shape(design, plan, core)
        if shape not in shapes:
            shapes[shape] = analyze_core(design, plan, core, timing)
        cores.append(dataclasses.replace(shapes[shape], core=core))
    return cores


def _request_time(design: Design, direction: int,
                  runs: Sequence[tuple[int, int]]) -> tuple[int, int, int]:
    """(cycles, bursts, beats) of one AXI request alone on the memory system: the channel
    recurrence of _AxiChannel with a private DRAM timeline. Alone, each burst reaches the
    DRAM after the previous one has left it, so the timeline only stretches a burst to its
    DRAM transfer time."""
    write = direction == WRITE
    latency = (design.axi_write_latency if write else design.axi_read_latency) or 0
    outstanding = (design.axi_write_outstanding if write else design.axi_read_outstanding) or 1
    db = design.axi_data_bytes or 16
    reach = (design.axi_max_burst_beats or 16) * db
    rate = design.dram_bytes_per_cycle
    lead, tail = (0, latency) if write else (latency, 0)
    key = (lead, tail, outstanding, db, reach, rate, tuple(runs))
    if key in _REQUEST_TIMES:
        return _REQUEST_TIMES[key]
    if len(_REQUEST_TIMES) > 1 << 16:
        _REQUEST_TIMES.clear()
    window: collections.deque[int] = collections.deque()
    last_issue, data_free, completion, bursts, beats = -1, 0, 0, 0, 0
    for base, length in runs:
        address, end = base, base + length
        while address < end:
            slot = address - address % db
            stop = min(end, (address // KIB4 + 1) * KIB4, slot + reach)
            count = ceil_div(stop, db) - slot // db
            address = stop
            issue = last_issue + 1
            if len(window) == outstanding:
                issue = max(issue, window.popleft())
            last_issue = issue
            busy = max(count, ceil_div(count * db, rate)) if rate else count
            data_free = max(issue + lead, data_free) + busy
            completion = data_free + tail
            window.append(completion)
            bursts += 1
            beats += count
    _REQUEST_TIMES[key] = completion, bursts, beats
    return completion, bursts, beats


# Request times by memory parameters and runs, shared by the analyses of one mapper search.
_REQUEST_TIMES: dict[tuple, tuple[int, int, int]] = {}


def _row_classes(design: Design, row0: int, row1: int) -> list[tuple[int, int, int]]:
    """(rows, count, first index) of row tiles row0..row1-1. With conv geometry the classes
    are row tiles of equal footprint hull widths in every block, in order of first
    appearance; a class's first index is its first member."""
    row_tiles = ceil_div(design.m, design.tile_m)
    if not design.conv:
        return _range_classes(row0, row1, row_tiles, design.tile_m, design.m)
    table = footprints(design)[1]
    classes: dict[tuple, list[int]] = {}
    for index in range(row0, row1):
        rows = min(design.tile_m, design.m - index * design.tile_m)
        key = (rows, tuple(panel.widths for panel in table[index]))
        if key in classes:
            classes[key][1] += 1
        else:
            classes[key] = [rows, 1, index]
    return [tuple(entry) for entry in classes.values()]


def analyze_core(design: Design, plan: ClusterPlan, core: int,
                 timing: bool = True) -> CoreAnalysis:
    """Closed form over the tile and panel classes of one core's tiles.

    Both caches are LRUs that see one cycle of panels repeated: the weight cache, per
    group, the core's column panels once per row tile; the activation cache, per row tile,
    that row tile's panels once per column tile the core runs. After the first pass every
    cacheable panel (one no larger than the cache) hits if the cacheable panels of the
    cycle fit the cache together, and every one misses otherwise. Depthwise-datapath tiles
    never share an activation panel. ``timing`` False skips the request times (choose_plan
    needs bytes and compute only)."""
    axi = design.memory_model == "axi"
    col_tiles = ceil_div(design.grid_columns, design.tile_n)
    blocks = _block_classes(design)
    record = param_record_bytes(design)
    capacity = design.weight_cache_bytes
    act_capacity = design.activation_cache_bytes
    shared = act_capacity > 0 and not design.channel_wise
    bandwidth = design.memory_bytes_per_cycle
    overlap_panels = design.panel_buffers > 1 and len(blocks) > 1
    memo: dict[tuple, tuple[int, int, int]] = {}
    totals: collections.Counter[str] = collections.Counter()
    first_load = last_compute = last_store = None

    def time(direction: int, nbytes: int, runs_of: Any, key: tuple) -> tuple[int, int, int]:
        if not timing:
            return 0, 0, 0
        if not axi:
            return dma_cycles(design, nbytes), dma_bursts(design, nbytes), 0
        if key not in memo:
            memo[key] = _request_time(design, direction, runs_of())
        return memo[key]

    def footprint(row: int, rows: int, block: int) -> tuple | None:
        """What a panel's activation runs depend on besides its size: the hull widths."""
        if not design.conv:
            return None
        return tile_footprint(design, row, rows, block).widths

    for groups, group, row0, row1, col0, col1 in _core_segments(design, plan, core):
        rows = _row_classes(design, row0, row1)
        cols = _range_classes(col0, col1, col_tiles, design.tile_n, design.grid_columns)
        sizes = {(c, depth): weight_bytes(design, depth, c)
                 for c, _, _ in cols for depth, _, _, _ in blocks}
        cacheable = {key: capacity > 0 and size <= capacity for key, size in sizes.items()}
        resident = sum(n_c * n_b * sizes[(c, depth)]
                       for c, n_c, _ in cols for depth, n_b, _, _ in blocks
                       if cacheable[(c, depth)])
        fits = capacity > 0 and resident <= capacity
        for a, n_a, row_index in rows:
            row = row_index * design.tile_m
            # One channel's activation panel per block, and whether this row tile's
            # cacheable panels fit the activation cache together.
            act = [activation_bytes(design, row, a, 1, b) for b in range(design.reduction_blocks)]
            act_fits = shared and sum(v for v in act if v <= act_capacity) <= act_capacity
            # The segment's first row tile is a cold weight pass over the columns, and
            # the core's first column tile of a row tile a cold activation pass.
            passes = [(1, False), (n_a - 1, True)] if row_index == row0 else [(n_a, True)]
            for c, n_c, col_index in cols:
                tile = Tile(group, row, col_index * design.tile_n, a, c)
                width = c if design.channel_wise else 1
                act_passes = ([(1, False), (n_c - 1, True)] if col_index == col0
                              else [(n_c, True)])
                for (repeat, warm), (act_repeat, act_warm) in itertools.product(passes,
                                                                                 act_passes):
                    if not repeat or not act_repeat:
                        continue
                    tiles = groups * repeat * act_repeat
                    # One tile of this class: its loads, compute and the last panel's compute.
                    loads = later = compute = 0
                    for depth, n_b, block_first, block_last in blocks:
                        kind = _block_kind(design, block_first, block_last)
                        hit = warm and fits and cacheable[(c, depth)]
                        size = sizes[(c, depth)]
                        panel = panel_compute_cycles(design, a, c, depth)
                        # Blocks of one kind differ only in their footprint: conv geometry
                        # takes each block on its own.
                        members = range(kind, kind + n_b) if design.conv else (kind,)
                        per = 1 if design.conv else n_b
                        for block in members:
                            act_hit = act_warm and act_fits and act[block] <= act_capacity
                            moved = 0 if act_hit else act[block] * width
                            nbytes = moved + (0 if hit else size) + (c * record
                                                                     if block_first else 0)
                            cycles, bursts, beats = time(
                                READ, nbytes,
                                lambda: panel_read_runs(design, tile, block, hit, act_hit),
                                (READ, a, c, depth, block_first, block_last, hit, act_hit,
                                 footprint(row, a, block)))
                            count = tiles * per
                            totals["read_bytes"] += count * nbytes
                            totals["activation_bytes"] += count * moved
                            totals["activation_hits"] += count if act_hit else 0
                            totals["weight_bytes"] += 0 if hit else count * size
                            totals["hits"] += count if hit else 0
                            totals["read_beats"] += count * beats
                            totals["data_cycles"] += count * ceil_div(nbytes, bandwidth)
                            totals["bursts"] += count * bursts
                            totals["requests"] += count
                            loads += per * cycles
                            later += 0 if block_first else per * cycles
                            compute += per * panel
                            if first_load is None:
                                first_load = cycles
                            last_compute = panel
                    cycles, bursts, beats = time(WRITE, a * c,
                                                 lambda: tile_store_runs(design, tile),
                                                 (WRITE, a, c))
                    totals["write_bytes"] += tiles * a * c
                    totals["macs"] += tiles * a * c * design.k
                    totals["tiles"] += tiles
                    totals["read_cycles"] += tiles * loads
                    totals["write_cycles" if axi else "read_cycles"] += tiles * cycles
                    totals["later_loads"] += tiles * later
                    totals["compute"] += tiles * compute
                    totals["write_beats"] += tiles * beats
                    totals["data_cycles"] += tiles * ceil_div(a * c, bandwidth)
                    totals["bursts"] += tiles * bursts
                    totals["requests"] += tiles
                    # One slot: tiles run one after another, each its load-compute chain
                    # (overlapped across panels by a second panel buffer) and its store.
                    first = loads - later
                    chain = (first + max(compute - last_compute, later) + last_compute
                             if overlap_panels else loads + compute)
                    totals["one_slot"] += tiles * (chain + cycles)
                    last_store = cycles
    return CoreAnalysis(
        core=core, tiles=totals["tiles"], macs=totals["macs"], compute_cycles=totals["compute"],
        read_bytes=totals["read_bytes"], write_bytes=totals["write_bytes"],
        weight_bytes=totals["weight_bytes"], weight_cache_hits=totals["hits"],
        requests=totals["requests"], read_cycles=totals["read_cycles"],
        write_cycles=totals["write_cycles"], read_beats=totals["read_beats"],
        write_beats=totals["write_beats"], bursts=totals["bursts"],
        first_load=first_load or 0, last_compute=last_compute or 0,
        last_store=last_store or 0, later_loads=totals["later_loads"],
        data_cycles=totals["data_cycles"], one_slot_cycles=totals["one_slot"],
        one_slot=design.buffer_slots == 1,
        panel_chain=design.panel_buffers == 1 and len(blocks) > 1,
        activation_read_bytes=totals["activation_bytes"],
        activation_cache_hits=totals["activation_hits"])


def closed_form_plan(design: Design) -> ClusterPlan:
    """The plan of a single core, or the cluster plan npu-sim picks (choose_plan)."""
    if design.cores == 1:
        return cluster_plan(design, "n")
    return choose_plan(design)[0]


def analyze(design: Design) -> Analysis:
    """Closed-form model: exact traffic, requests and compute busy cycles; bounds and a
    pipelined estimate of the cycle count."""
    if design.weight_encoding not in ("dense", "palette"):
        raise ValueError("the analytic model covers dense and palette weights; sparse sizes "
                         "depend on the data")
    if design.memory_model not in ("serial", "axi"):
        raise ValueError("memory model must be serial or axi")
    if design.cores != 1 and design.memory_model != "axi":
        raise ValueError("more than one core requires the AXI memory model")
    plan = closed_form_plan(design)
    cores = tuple(analyze_cores(design, plan))
    busy = [core for core in cores if core.tiles]
    compute = sum(core.compute_cycles for core in cores)
    per_core = max(core.compute_cycles for core in cores)
    common = dict(design=design, macs=design.macs,
                  read_bytes=sum(core.read_bytes for core in cores),
                  write_bytes=sum(core.write_bytes for core in cores),
                  weight_bytes=sum(core.weight_bytes for core in cores),
                  dma_requests=sum(core.requests for core in cores),
                  compute_busy_cycles=compute, scratchpad_bytes=scratchpad_bytes(design),
                  compute_cycles_per_core=per_core,
                  weight_cache_hits=sum(core.weight_cache_hits for core in cores),
                  activation_read_bytes=sum(core.activation_read_bytes for core in cores),
                  activation_cache_hits=sum(core.activation_cache_hits for core in cores),
                  partition=plan.partition, cores=cores)
    if design.memory_model == "axi":
        db = design.axi_data_bytes or 16
        rate = design.dram_bytes_per_cycle
        beats = sum(core.read_beats + core.write_beats for core in cores)
        dram = ceil_div(beats * db, rate) if rate else 0
        memory = max([dram] + [max(c.read_cycles, c.write_cycles) for c in busy])
        payload = max([dram] + [max(c.read_beats, c.write_beats) for c in busy])
        lower = axi_bounds(design, compute, common["weight_bytes"],
                           common["activation_read_bytes"])["lower_bound_cycles"]
        estimate = max([lower, dram] + [core.estimate_cycles for core in busy])
        return Analysis(**common, memory_cycles=memory, memory_data_cycles=payload,
                        lower_bound_cycles=lower, estimate_cycles=estimate,
                        dma_bursts=sum(core.bursts for core in cores))
    core = cores[0]
    dma = core.read_cycles
    data = core.data_cycles
    loads = dma - _store_cycles(design)
    serial = compute + dma
    # Compute cannot start before the first load and the last store follows the last
    # compute; every load precedes the last compute; the DMA is one server.
    lower = max(compute + core.first_load + core.last_store,
                loads + core.last_compute + core.last_store, dma)
    if design.panel_buffers == 1:
        # One panel buffer per slot: a tile's next panel loads only after its previous panel
        # computed, so compute and every non-first panel load form one chain.
        lower = max(lower, core.first_load + compute + core.later_loads + core.last_store)
    estimate = max(lower, core.first_load + core.last_compute + core.last_store
                   + max(compute - core.last_compute, dma - core.first_load - core.last_store))
    if design.buffer_slots == 1:
        estimate = max(lower, core.one_slot_cycles)
    rows0, cols0 = min(design.tile_m, design.m), min(design.tile_n, design.grid_columns)
    one_tile = (design.grid_groups == 1 and rows0 == design.m
                and cols0 == design.grid_columns)
    no_overlap = ((design.buffer_slots == 1 or one_tile)
                  and (design.reduction_blocks == 1 or design.panel_buffers == 1))
    return Analysis(**common, memory_cycles=dma, memory_data_cycles=data,
                    lower_bound_cycles=lower, estimate_cycles=estimate,
                    dma_bursts=core.bursts, dma_busy_cycles=dma, serial_cycles=serial,
                    exact_cycles=serial if no_overlap else None)


def _store_cycles(design: Design) -> int:
    return sum(count * dma_cycles(design, store_bytes(rows, cols))
               for rows, cols, count in tile_classes(design))


def activation_run_lengths(design: Design) -> list[tuple[int, int]]:
    """(length, count) of every activation run of every panel load, no cache hits."""
    runs: list[tuple[int, int]] = []
    if not design.conv:
        for p in panel_classes(design):
            if not design.channel_wise:
                runs.append((p.depth, p.rows * p.count))
            elif p.depth == design.k:
                runs.append((p.cols * design.k, p.rows * p.count))
            else:
                runs.append((p.depth, p.rows * p.cols * p.count))
        return runs
    geometry, table = footprints(design)
    columns = _extent_classes(design.grid_columns, design.tile_n)
    for entry in table:
        for panel in entry:
            for cols, count in columns:
                channels = geometry.group_channels * (cols if design.channel_wise else 1)
                tiles = count * design.grid_groups
                if channels == design.conv_input_c:
                    runs += [((width + 1) * channels, tiles) for width in panel.widths]
                else:
                    runs.append((channels, tiles * panel.pixels))
    return runs


def axi_bounds(design: Design, compute_cycles: int | None = None,
               weight_read_bytes: int | None = None,
               activation_read_bytes: int | None = None) -> dict[str, int]:
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

    # With a cache the panels read are known only in total: one run of that many bytes
    # needs no more beats, bursts or slot time than the panels it stands for.
    lumped = design.weight_cache_bytes > 0
    act_lumped = design.activation_cache_bytes > 0
    if (lumped and weight_read_bytes is None) or (act_lumped and activation_read_bytes is None):
        model = analyze(design)
        weight_read_bytes = model.weight_bytes if weight_read_bytes is None else weight_read_bytes
        if activation_read_bytes is None:
            activation_read_bytes = model.activation_read_bytes
    reads: list[tuple[int, int]] = []
    if lumped and weight_read_bytes:
        reads.append((weight_read_bytes, 1))
    if act_lumped:
        if activation_read_bytes:
            reads.append((activation_read_bytes, 1))
    else:
        # One run per activation row or footprint run, even where runs are contiguous.
        reads += activation_run_lengths(design)
    for p in panel_classes(design):
        if not lumped:
            reads.append((weight_bytes(design, p.depth, p.cols), p.count))
        if p.first:
            reads.append((p.cols * param_record_bytes(design), p.count))
    writes: list[tuple[int, int]] = []
    for rows, cols, count in tile_classes(design):
        writes.append((cols, rows * count))
    read = channel(reads, design.axi_read_latency or 0, design.axi_read_outstanding or 1)
    write = channel(writes, design.axi_write_latency or 0, design.axi_write_outstanding or 1)
    payload = sum(length * count for length, count in reads + writes)
    dram = ceil_div(payload, design.dram_bytes_per_cycle) if design.dram_bytes_per_cycle else 0
    if compute_cycles is None:
        compute_cycles = sum(p.count * panel_compute_cycles(design, p.rows, p.cols, p.depth)
                             for p in panel_classes(design))
    rows0, cols0 = min(design.tile_m, design.m), min(design.tile_n, design.grid_columns)
    first = (activation_bytes(design, 0, rows0, cols0, 0)
             + weight_bytes(design, design.panel_depth, cols0)
             + cols0 * param_record_bytes(design))
    last = store_bytes(design.m % design.tile_m or rows0,
                       design.grid_columns % design.tile_n or cols0)
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


# ================================================================ cycle-level schedule ==
# The rest of the module replays the schedule request by request: the address map, the
# AXI burst split, each channel's outstanding-window recurrence, the shared DRAM byte
# timeline, the serial DMA server, the cluster split and each core's LRU weight cache.
# ``schedule()`` returns the cycle count and the counters npu-sim reports. It follows
# docs/architecture.md sections 4 to 7 and the interface comments in include/npu/memory.hpp
# and include/npu/core.hpp; tests/analytic_model_tests.py checks it against npu-sim.

ACTIVATION_BASE = 0x8000_0000
WEIGHT_BASE = 0x9000_0000
PARAM_BASE = 0xA000_0000
OUTPUT_BASE = 0xB000_0000
READ, WRITE = 0, 1


def axi_bursts(base: int, length: int, data_bytes: int, max_beats: int) -> list[tuple[int, int]]:
    """(payload bytes, beats) of the INCR bursts that one address run splits into.

    A burst ends at the run's end, the next 4 KiB boundary or max_beats beats, whichever
    comes first. Beats count the data_bytes-aligned slots a burst touches, so a misaligned
    first or last beat is a partial beat, and the beat limit counts from the aligned slot."""
    bursts = []
    address, end = base, base + length
    while address < end:
        slot = address - address % data_bytes
        stop = min(end, (address // KIB4 + 1) * KIB4, slot + max_beats * data_bytes)
        bursts.append((stop - address, ceil_div(stop, data_bytes) - slot // data_bytes))
        address = stop
    return bursts


@dataclass(frozen=True)
class Tile:
    """An output tile of one grid group; ``col`` is the column within the group's N. On the
    depthwise datapath the grid is one group of G columns: ``col`` and ``cols`` select
    output channels, each its own group's single column."""

    group: int
    row: int
    col: int
    rows: int
    cols: int


def make_tiles(design: Design) -> list[Tile]:
    """Every output tile in issue order: grid group, then row, then column."""
    tiles = []
    columns = design.grid_columns
    for group in range(design.grid_groups):
        for row in range(0, design.m, design.tile_m):
            for col in range(0, columns, design.tile_n):
                tiles.append(Tile(group, row, col, min(design.tile_m, design.m - row),
                                  min(design.tile_n, columns - col)))
    return tiles


def block_extent(design: Design, block: int) -> tuple[int, int]:
    """(first reduction index, depth) of reduction panel ``block``."""
    start = block * design.panel_depth
    return start, min(design.panel_depth, design.k - start)


def weight_panel_offset(design: Design, tile: Tile, block: int) -> int:
    """Offset of a weight panel from WEIGHT_BASE. Panels are packed in the order the loader
    first reads them: group, column panel, reduction panel."""
    full_cols, edge_cols = divmod(design.grid_columns, design.tile_n)

    def column_bytes(cols: int) -> int:
        return sum(weight_bytes(design, block_extent(design, b)[1], cols)
                   for b in range(design.reduction_blocks))
    group_bytes = (full_cols * column_bytes(design.tile_n)
                   + (column_bytes(edge_cols) if edge_cols else 0))
    return (tile.group * group_bytes + (tile.col // design.tile_n) * column_bytes(design.tile_n)
            + block * weight_bytes(design, design.panel_depth, tile.cols))


def activation_runs(design: Design, tile: Tile, block: int) -> list[tuple[int, int]]:
    """(base, length) activation runs of one panel load.

    GEMM operand (M x G*K, row-major): per tile row its group's depth-byte slice; on the
    depthwise datapath the slices of the tile's channels, one run when the panel covers
    all of K. Conv geometry (NHWC input): per input row of the footprint one run over the
    column hull when the panel's channels are all C, else one run per pixel of the hull."""
    start, depth = block_extent(design, block)
    first, count = (tile.col, tile.cols) if design.channel_wise else (tile.group, 1)
    if not design.conv:
        stride = design.groups * design.k
        runs = []
        for row in range(tile.row, tile.row + tile.rows):
            base = ACTIVATION_BASE + row * stride
            if depth == design.k:
                runs.append((base + first * design.k, count * design.k))
            else:
                runs += [(base + group * design.k + start, depth)
                         for group in range(first, first + count)]
        return runs
    width = footprints(design)[0].group_channels
    channels, channel = count * width, first * width
    c, w = design.conv_input_c, design.conv_input_w
    runs = []
    for row, low, high in tile_footprint(design, tile.row, tile.rows, block).spans:
        base = ACTIVATION_BASE + row * w * c
        if channels == c:
            runs.append((base + low * c, (high - low + 1) * c))
        else:
            runs += [(base + x * c + channel, channels) for x in range(low, high + 1)]
    return runs


def activation_key(design: Design, tile: Tile, block: int) -> tuple[int, int, int]:
    """Activation-cache line: (group, or channel tile on the depthwise datapath; row tile;
    reduction block). Column tiles of one row tile share their group's lines."""
    unit = tile.col // design.tile_n if design.channel_wise else tile.group
    return unit, tile.row // design.tile_m, block


def panel_read_runs(design: Design, tile: Tile, block: int, weight_cached: bool = False,
                    activations_cached: bool = False) -> list[tuple[int, int]]:
    """(base, length) runs of one panel load: the tile's activation runs unless the
    activation cache holds them, its weight panel unless the weight cache holds it, and on
    the first panel its parameter records."""
    depth = block_extent(design, block)[1]
    runs = [] if activations_cached else activation_runs(design, tile, block)
    if not weight_cached:
        runs.append((WEIGHT_BASE + weight_panel_offset(design, tile, block),
                     weight_bytes(design, depth, tile.cols)))
    if block == 0:
        record = param_record_bytes(design)
        runs.append((PARAM_BASE + (tile.group * design.n + tile.col) * record,
                     tile.cols * record))
    return runs


def tile_store_runs(design: Design, tile: Tile) -> list[tuple[int, int]]:
    stride = design.groups * design.n
    return [(OUTPUT_BASE + (tile.row + i) * stride + tile.group * design.n + tile.col, tile.cols)
            for i in range(tile.rows)]


class DramTimeline:
    """Reserved DRAM intervals in byte-time units: cycle c begins at byte c * rate, so a
    fractional-cycle transfer stays an exact integer. Touching intervals merge."""

    def __init__(self) -> None:
        self.begins: list[int] = []  # sorted, disjoint, never touching
        self.ends: list[int] = []

    def reserve(self, earliest: int, nbytes: int) -> int:
        """Reserve ``nbytes`` in the earliest gap at or after ``earliest``; return its start."""
        begins, ends = self.begins, self.ends
        # First interval that ends after ``earliest``: everything before it lies behind.
        index = bisect.bisect_right(ends, earliest)
        begin = earliest
        while index < len(begins) and begins[index] < begin + nbytes:
            begin = max(begin, ends[index])
            index += 1
        end = begin + nbytes
        # Insert [begin, end) at ``index`` and merge with touching neighbors.
        if index > 0 and ends[index - 1] == begin:
            ends[index - 1] = end
            if index < len(begins) and begins[index] == end:
                ends[index - 1] = ends[index]
                del begins[index], ends[index]
        elif index < len(begins) and begins[index] == end:
            begins[index] = begin
        else:
            begins.insert(index, begin)
            ends.insert(index, end)
        return begin

    def release(self, before: int) -> None:
        """Forget intervals ending at or before ``before``."""
        keep = bisect.bisect_right(self.ends, before)
        if keep:
            del self.begins[:keep], self.ends[:keep]


@dataclass
class ChannelStats:
    """One AXI channel's counters, as npu-sim reports them under ``memory``."""

    requests: int = 0
    bursts: int = 0
    beats: int = 0
    payload_bytes: int = 0
    busy_cycles: int = 0
    latency_cycles: int = 0
    max_latency_cycles: int = 0
    peak_outstanding: int = 0


class _AxiChannel:
    """One read or write channel of a core's port: the burst recurrence of
    docs/architecture.md 5.4.2.

        issue_i      = max(arrival, issue_{i-1} + 1, completion_{i-N})
        start_i      = max(issue_i + read latency (reads), data_free, DRAM grant)
        data_free    = start_i + beats_i, or the DRAM end of the burst if later
        completion_i = data_free (+ write latency, the B response, for writes)
    """

    def __init__(self, design: Design, direction: int):
        self.write = direction == WRITE
        self.latency = (design.axi_write_latency if self.write else design.axi_read_latency) or 0
        self.outstanding = (design.axi_write_outstanding if self.write
                            else design.axi_read_outstanding) or 1
        self.data_bytes = design.axi_data_bytes or 16
        self.max_beats = design.axi_max_burst_beats or 16
        self.rate = design.dram_bytes_per_cycle
        self.stats = ChannelStats()
        self.data_free = 0
        self.last_issue = -1
        self.window: collections.deque[int] = collections.deque()
        self.bursts: list[tuple[int, int]] = []
        self.index = 0
        self.arrival = 0
        self.intervals: list[tuple[int, int]] = []

    def begin(self, arrival: int, runs: Sequence[tuple[int, int]]) -> int:
        """Start a request; return the first burst's issue cycle."""
        self.bursts = [burst for base, length in runs
                       for burst in axi_bursts(base, length, self.data_bytes, self.max_beats)]
        self.index = 0
        self.arrival = arrival
        return self.next_issue()

    def next_issue(self) -> int:
        issue = max(self.arrival, self.last_issue + 1)
        if len(self.window) >= self.outstanding:
            issue = max(issue, self.window[-self.outstanding])
        return issue

    def grant(self, issue: int, dram: DramTimeline | None) -> int:
        """Issue the next burst at ``issue``; return its completion cycle."""
        payload, beats = self.bursts[self.index]
        self.index += 1
        earliest = max(issue + (0 if self.write else self.latency), self.data_free)
        start, end = earliest, earliest + beats
        if dram is not None:
            nbytes = beats * self.data_bytes
            begin = dram.reserve(earliest * self.rate, nbytes)
            start = max(earliest, ceil_div(begin, self.rate))
            end = max(start + beats, ceil_div(begin + nbytes, self.rate))
        self.stats.busy_cycles += end - start
        self.data_free = end
        completion = end + (self.latency if self.write else 0)
        self.window.append(completion)
        while len(self.window) > self.outstanding:
            self.window.popleft()
        self.last_issue = issue
        self.stats.bursts += 1
        self.stats.beats += beats
        self.stats.payload_bytes += payload
        # Completions rise along a channel, so the bursts still in flight are a suffix.
        in_flight = len(self.window) - bisect.bisect_right(self.window, issue)
        self.stats.peak_outstanding = max(self.stats.peak_outstanding, in_flight)
        return completion

    @property
    def done(self) -> bool:
        return self.index == len(self.bursts)

    def finish(self, completion: int) -> None:
        latency = completion - self.arrival
        self.stats.requests += 1
        self.stats.latency_cycles += latency
        self.stats.max_latency_cycles = max(self.stats.max_latency_cycles, latency)
        self.intervals.append((self.arrival, completion))


class _PanelCache:
    """LRU cache of weight or activation panels; a panel larger than the cache is never
    cached, and capacity 0 counts nothing."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries: collections.OrderedDict[tuple[int, int, int], int] = \
            collections.OrderedDict()
        self.used = self.hits = self.misses = self.saved = self.peak = 0

    def access(self, key: tuple[int, int, int], size: int) -> bool:
        if not self.capacity:
            return False
        if key in self.entries:
            self.entries.move_to_end(key)
            self.hits += 1
            self.saved += size
            return True
        self.misses += 1
        if size <= self.capacity:
            while self.used + size > self.capacity:
                self.used -= self.entries.popitem(last=False)[1]
            self.entries[key] = size
            self.used += size
            self.peak = max(self.peak, self.used)
        return False


# ----------------------------------------------------------------------------- cluster --

@dataclass(frozen=True)
class ClusterPlan:
    """The work split of docs/architecture.md 6.2: units are whole groups when G >= P, else
    the row tiles (partition m) or column tiles (partition n) of every group, numbered
    group-major. Core c runs a contiguous run of U // P units, one more for each of the
    first U % P."""

    partition: str
    split: str
    cores: int
    units: int

    def first_unit(self, core: int) -> int:
        base, extra = divmod(self.units, self.cores)
        return core * base + min(core, extra)

    def unit_count(self, core: int) -> int:
        base, extra = divmod(self.units, self.cores)
        return base + (1 if core < extra else 0)


def cluster_plan(design: Design, partition: str) -> ClusterPlan:
    if partition not in ("m", "n"):
        raise ValueError("a cluster plan splits by m or n")
    cores, groups = design.cores, design.grid_groups
    if groups >= cores:
        return ClusterPlan(partition, "groups", cores, groups)
    if partition == "m":
        return ClusterPlan("m", "rows", cores, groups * ceil_div(design.m, design.tile_m))
    return ClusterPlan("n", "columns", cores,
                       groups * ceil_div(design.grid_columns, design.tile_n))


def tile_unit(design: Design, plan: ClusterPlan, tile: Tile) -> int:
    if plan.split == "groups":
        return tile.group
    if plan.split == "rows":
        return tile.group * ceil_div(design.m, design.tile_m) + tile.row // design.tile_m
    return tile.group * ceil_div(design.grid_columns, design.tile_n) + tile.col // design.tile_n


def core_tiles(design: Design, plan: ClusterPlan) -> list[list[Tile]]:
    """Each core's tiles under ``plan``, in single-core order."""
    owner: list[int] = []
    for core in range(plan.cores):
        owner += [core] * plan.unit_count(core)
    cores: list[list[Tile]] = [[] for _ in range(plan.cores)]
    for tile in make_tiles(design):
        cores[owner[tile_unit(design, plan, tile)]].append(tile)
    return cores


def panel_cycles(design: Design, tile: Tile, block: int) -> int:
    return panel_compute_cycles(design, tile.rows, tile.cols, block_extent(design, block)[1])


def plan_bound(design: Design, cores: Sequence[CoreAnalysis]) -> int:
    """Makespan lower bound of a plan: the slowest core's compute or port transfer cycles
    (an AXI channel moves data_bytes per cycle; the serial server bytes_per_cycle for both
    directions), or every byte at the DRAM rate."""
    worst = 0
    for core in cores:
        if design.memory_model == "axi":
            db = design.axi_data_bytes or 16
            port = max(ceil_div(core.read_bytes, db), ceil_div(core.write_bytes, db))
        else:
            port = ceil_div(core.read_bytes + core.write_bytes, design.memory_bytes_per_cycle)
        worst = max(worst, core.compute_cycles, port)
    total = sum(core.read_bytes + core.write_bytes for core in cores)
    rate = design.dram_bytes_per_cycle if design.memory_model == "axi" else 0
    return max(worst, ceil_div(total, rate) if rate else 0)


def choose_plan(design: Design) -> tuple[ClusterPlan, dict[str, dict[str, int]]]:
    """The plan npu-sim runs and its per-partition predictions ("cluster.predicted"): auto
    takes the smaller bound, then fewer read plus write bytes, then n."""
    predicted: dict[str, dict[str, int]] = {}
    plans: dict[str, ClusterPlan] = {}
    for partition in ("m", "n"):
        plan = cluster_plan(design, partition)
        cores = analyze_cores(design, plan, timing=False)
        plans[partition] = plan
        predicted[partition] = {"read_bytes": sum(c.read_bytes for c in cores),
                                "write_bytes": sum(c.write_bytes for c in cores),
                                "bound_cycles": plan_bound(design, cores)}
    if design.partition in ("m", "n"):
        return plans[design.partition], predicted
    m, n = predicted["m"], predicted["n"]
    key_m = (m["bound_cycles"], m["read_bytes"] + m["write_bytes"])
    key_n = (n["bound_cycles"], n["read_bytes"] + n["write_bytes"])
    return plans["m" if key_m < key_n else "n"], predicted


# ---------------------------------------------------------------------------- schedule --

@dataclass
class CoreSchedule:
    core: int
    tiles: int
    cycles: int = 0
    macs: int = 0
    compute_busy_cycles: int = 0
    dma_busy_cycles: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    weight_read_bytes: int = 0
    weight_cache_hits: int = 0
    weight_cache_misses: int = 0
    activation_read_bytes: int = 0
    activation_cache_hits: int = 0
    activation_cache_misses: int = 0
    activation_bytes_saved: int = 0
    peak_cached_activation_bytes: int = 0
    depthwise_macs: int = 0
    dma_requests: int = 0
    dma_bursts: int = 0
    read: ChannelStats = dataclasses.field(default_factory=ChannelStats)
    write: ChannelStats = dataclasses.field(default_factory=ChannelStats)


@dataclass
class Schedule:
    """``schedule()`` output: the makespan and the counters npu-sim reports."""

    design: Design
    plan: ClusterPlan
    predicted: dict[str, dict[str, int]]
    cycles: int
    cores: list[CoreSchedule]

    def total(self, name: str) -> int:
        return sum(getattr(core, name) for core in self.cores)

    @property
    def read_bytes(self) -> int:
        return self.total("read_bytes")

    @property
    def write_bytes(self) -> int:
        return self.total("write_bytes")

    @property
    def compute_busy_cycles(self) -> int:
        return self.total("compute_busy_cycles")

    @property
    def dram_bytes(self) -> int:
        """Bytes moved over DRAM: whole beats under AXI, payload under the serial model."""
        if self.design.memory_model != "axi":
            return self.read_bytes + self.write_bytes
        db = self.design.axi_data_bytes or 16
        return sum((core.read.beats + core.write.beats) * db for core in self.cores)


class _Core:
    """One core pipeline: loader, compute engine and storer around ``buffer_slots`` tile
    slots of up to ``panel_buffers`` reduction-panel buffers each. A slot is taken by a
    tile's first load and freed by its store; a panel load waits for the buffer it reuses
    to be consumed; compute and store run in tile order."""

    def __init__(self, design: Design, index: int, tiles: list[Tile]):
        self.design = design
        self.out = CoreSchedule(index, len(tiles))
        self.tiles = tiles
        self.blocks = design.reduction_blocks
        self.buffers = min(design.panel_buffers, self.blocks)
        self.cache = _PanelCache(design.weight_cache_bytes)
        self.activation_cache = _PanelCache(design.activation_cache_bytes)
        self.next_load = 0                     # flat panel index: tile * blocks + block
        self.loading = False
        self.load_free = 0
        self.compute_free = 0
        self.compute_end: dict[int, int] = {}  # flat panel index -> compute end
        self.next_store = 0
        self.storing = False
        self.store_free = 0
        self.store_done: list[int | None] = [None] * len(tiles)
        self.channels = ((_AxiChannel(design, READ), _AxiChannel(design, WRITE))
                         if design.memory_model == "axi" else None)

    def load_request(self) -> tuple[int, list[tuple[int, int]], int] | None:
        """(arrival, runs, payload) of the next panel load once its slot and buffer are
        free, or None while it waits on a store that has not completed."""
        if self.loading or self.next_load >= len(self.tiles) * self.blocks:
            return None
        index = self.next_load
        tile_index, block = divmod(index, self.blocks)
        arrival = self.load_free
        if block == 0 and tile_index >= self.design.buffer_slots:
            freed = self.store_done[tile_index - self.design.buffer_slots]
            if freed is None:
                return None
            arrival = max(arrival, freed)
        if block >= self.buffers:
            arrival = max(arrival, self.compute_end[index - self.buffers])
        tile = self.tiles[tile_index]
        size = weight_bytes(self.design, block_extent(self.design, block)[1], tile.cols)
        hit = self.cache.access((tile.group, tile.col, block), size)
        if not hit:
            self.out.weight_read_bytes += size
        act = activation_bytes(self.design, tile.row, tile.rows, tile.cols, block)
        act_hit = self.activation_cache.access(activation_key(self.design, tile, block), act)
        if not act_hit:
            self.out.activation_read_bytes += act
        runs = panel_read_runs(self.design, tile, block, hit, act_hit)
        self.loading = True
        return arrival, runs, sum(length for _, length in runs)

    def load_done(self, completion: int) -> None:
        index = self.next_load
        tile_index, block = divmod(index, self.blocks)
        tile = self.tiles[tile_index]
        cycles = panel_cycles(self.design, tile, block)
        self.compute_free = max(completion, self.compute_free) + cycles
        self.compute_end[index] = self.compute_free
        self.out.compute_busy_cycles += cycles
        if block == 0:
            self.out.macs += tile.rows * tile.cols * self.design.k
            if self.design.channel_wise:
                self.out.depthwise_macs += tile.rows * tile.cols * self.design.k
        self.loading = False
        self.load_free = completion
        self.next_load += 1

    def store_request(self) -> tuple[int, list[tuple[int, int]], int] | None:
        if self.storing or self.next_store >= len(self.tiles):
            return None
        last = (self.next_store + 1) * self.blocks - 1
        if last >= self.next_load:
            return None
        runs = tile_store_runs(self.design, self.tiles[self.next_store])
        self.storing = True
        return (max(self.compute_end[last], self.store_free), runs,
                sum(length for _, length in runs))

    def store_done_at(self, completion: int) -> None:
        self.store_done[self.next_store] = completion
        # Panels of stored tiles are never consulted again.
        for index in range(self.next_store * self.blocks, (self.next_store + 1) * self.blocks):
            self.compute_end.pop(index, None)
        self.storing = False
        self.store_free = completion
        self.next_store += 1
        self.out.cycles = max(self.out.cycles, completion)


def _union_cycles(intervals: list[tuple[int, int]]) -> int:
    total, end = 0, -1
    for begin, finish in sorted(intervals):
        if finish <= end:
            continue
        total += finish - max(begin, end)
        end = finish
    return total


def _run_axi(design: Design, cores: list[_Core]) -> None:
    """Grant bursts in issue order across every channel of every core; same-cycle bursts
    go in core order, read before write, as npu-sim's arbiter grants them."""
    dram = DramTimeline() if design.dram_bytes_per_cycle else None
    heap: list[tuple[int, int, int]] = []

    def start(core: _Core, direction: int) -> None:
        request = core.load_request() if direction == READ else core.store_request()
        if request is None:
            return
        arrival, runs, payload = request
        channel = core.channels[direction]  # type: ignore[index]
        if direction == READ:
            core.out.read_bytes += payload
        else:
            core.out.write_bytes += payload
        core.out.dma_requests += 1
        if not payload:
            # A load whose panels the caches all hold moves nothing: it completes on
            # arrival without a burst.
            channel.arrival = arrival
            channel.finish(arrival)
            core.load_done(arrival)
            start(core, READ)
            start(core, WRITE)
            return
        heapq.heappush(heap, (channel.begin(arrival, runs), core.out.core, direction))

    for core in cores:
        start(core, READ)
    while heap:
        issue, index, direction = heapq.heappop(heap)
        core = cores[index]
        channel = core.channels[direction]  # type: ignore[index]
        if dram is not None:
            dram.release(issue * design.dram_bytes_per_cycle)
        completion = channel.grant(issue, dram)
        if not channel.done:
            heapq.heappush(heap, (channel.next_issue(), index, direction))
            continue
        channel.finish(completion)
        if direction == READ:
            core.load_done(completion)
        else:
            core.store_done_at(completion)
        start(core, READ)
        start(core, WRITE)
    for core in cores:
        read, write = core.channels  # type: ignore[misc]
        core.out.read, core.out.write = read.stats, write.stats
        core.out.dma_bursts = read.stats.bursts + write.stats.bursts
        core.out.dma_busy_cycles = _union_cycles(read.intervals + write.intervals)


def _run_serial(design: Design, core: _Core) -> None:
    """One non-preemptive DMA server: when a load and a store both wait it alternates,
    a load first; a sole waiting request proceeds."""
    server_free = 0
    last = WRITE
    load = store = None
    while True:
        if load is None:
            load = core.load_request()
        if store is None:
            store = core.store_request()
        if load is None and store is None:
            break
        now = max(server_free, min(r[0] for r in (load, store) if r is not None))
        ready = [(kind, r) for kind, r in ((READ, load), (WRITE, store))
                 if r is not None and r[0] <= now]
        kind, request = ready[1] if len(ready) == 2 and last == READ else ready[0]
        payload = request[2]
        cycles = dma_cycles(design, payload)
        server_free = now + cycles
        last = kind
        core.out.dma_requests += 1
        core.out.dma_bursts += dma_bursts(design, payload)
        core.out.dma_busy_cycles += cycles
        if kind == READ:
            core.out.read_bytes += payload
            core.load_done(server_free)
            load = None
        else:
            core.out.write_bytes += payload
            core.store_done_at(server_free)
            store = None


def schedule(design: Design) -> Schedule:
    """Replay ``design`` request by request; see the section comment above."""
    if design.weight_encoding not in ("dense", "palette"):
        raise ValueError("the schedule covers dense and palette weights")
    if design.memory_model not in ("serial", "axi"):
        raise ValueError("memory model must be serial or axi")
    if design.cores > 1 and design.memory_model != "axi":
        raise ValueError("more than one core requires the AXI memory model")
    plan, predicted = choose_plan(design)
    cores = [_Core(design, index, tiles)
             for index, tiles in enumerate(core_tiles(design, plan))]
    if design.memory_model == "axi":
        _run_axi(design, cores)
    else:
        _run_serial(design, cores[0])
    for core in cores:
        core.out.weight_cache_hits = core.cache.hits
        core.out.weight_cache_misses = core.cache.misses
        cache = core.activation_cache
        core.out.activation_cache_hits = cache.hits
        core.out.activation_cache_misses = cache.misses
        core.out.activation_bytes_saved = cache.saved
        core.out.peak_cached_activation_bytes = cache.peak
    return Schedule(design, plan, predicted, max(core.out.cycles for core in cores),
                    [core.out for core in cores])


# ------------------------------------------------------------------ simulator results --

@dataclass(frozen=True)
class Observed:
    """The quantities ``analyze`` predicts, read from an npu-sim result."""

    cycles: int
    macs: int
    read_bytes: int
    write_bytes: int
    dram_bytes: int
    compute_cycles_per_core: int
    memory_cycles: int
    memory_data_cycles: int
    partition: str
    split: str
    cores: int
    imbalance: float
    effective_macs_per_cycle: int
    peak_memory_bytes_per_cycle: float

    @property
    def bound(self) -> str:
        return classify(self.compute_cycles_per_core, self.memory_cycles,
                        self.memory_data_cycles)

    @property
    def mac_utilization(self) -> float:
        return self.macs / (self.cycles * self.effective_macs_per_cycle)

    @property
    def arithmetic_intensity(self) -> float:
        return self.macs / self.dram_bytes


def observed(result: Mapping[str, Any]) -> Observed:
    """Read a schema_version 2 result: its per-core "cores" entries, their "memory"
    channels and the "cluster" split.

    Memory time per core is the time a request is in flight on its busier channel (an
    issuing thread waits for each request, so a channel's request latencies add up to its
    in-flight time; the serial server reports dma_busy_cycles), and payload time is that
    channel's data cycles. Both are at least the DRAM time of every byte moved."""
    config = result["config"]
    model = config_value(config, "memory_model")
    cores = result.get("cores") or []
    if not cores:
        raise ValueError("result has no per-core statistics (schema_version 2)")
    compute = max(core["compute_busy_cycles"] for core in cores)
    if model == "axi":
        def in_flight(channel: Mapping[str, Any]) -> int:
            return round(channel["mean_latency_cycles"] * channel["requests"])
        memory = max(max(in_flight(c["memory"]["read"]), in_flight(c["memory"]["write"]))
                     for c in cores)
        data = max(max(c["memory"]["read"]["busy_cycles"], c["memory"]["write"]["busy_cycles"])
                   for c in cores)
        dram_bytes = result["memory"]["dram"]["bytes"]
        rate = config_value(config, "dram_bytes_per_cycle")
        dram = ceil_div(dram_bytes, rate) if rate else 0
        memory, data = max(memory, dram), max(data, dram)
    else:
        memory = cores[0]["dma_busy_cycles"]
        channels = result["memory"]
        data = channels["read"]["busy_cycles"] + channels["write"]["busy_cycles"]
        dram_bytes = result["read_bytes"] + result["write_bytes"]
    cluster = result.get("cluster", {})
    return Observed(result["cycles"], result["macs"], result["read_bytes"],
                    result["write_bytes"], dram_bytes, compute, memory, data,
                    cluster.get("partition", "n"), cluster.get("split", "groups"),
                    cluster.get("cores", 1), float(cluster.get("imbalance", 1.0)),
                    result["effective_macs_per_cycle"],
                    float(result["peak_memory_bytes_per_cycle"]))


def observed_bound(result: Mapping[str, Any]) -> str:
    return observed(result).bound
