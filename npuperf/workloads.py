"""Canonical network workloads (workloads/*.json) lowered to npu-sim GEMM calls.

A layer becomes one ``Gemm``: an M x N x K GEMM per group, G groups (npu-sim --groups G),
run ``calls`` times. Convolutions are M = out_h*out_w, N = out_c/G, K = kh*kw*in_c/G and
carry their geometry (``Gemm.conv``): run with it, the activation operand is the NHWC
input's im2col view and a panel load reads the input footprint (npu-sim --conv-*); run
without it (``flags(conv=False)``), the operand is the materialized im2col matrix.
Depthwise is G = channels and N = 1, which a design with a depthwise datapath runs
channel-wise. Attention matmuls run per KV head.
``split`` cuts a call only when it exceeds a per-call simulator limit, along groups first,
then N, then M. A reduction too deep for one call is an error, since npu-sim accumulates
no partial sums across calls; a split piece of a convolution is a plain GEMM.
``call_options(binary)`` gives the limits of one npu-sim build: a build without
--timing-only takes LEGACY_LIMITS, a build without --groups runs a grouped layer as one call
per group, and a build without --conv-input-h runs every convolution on its im2col matrix.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import analytic, jsonio, sim
from .binary import ROOT

WORKLOAD_DIR = ROOT / "workloads"

# Per-call limits of npu-sim (include/npu/core.hpp). Functional runs generate and check
# every tensor; --timing-only runs skip tensors and numerics.
MAX_DIMENSION = 1 << 20
MAX_FUNCTIONAL_MACS = 4_000_000_000
MAX_TIMING_MACS = 10_000_000_000_000
MAX_HOST_BYTES = 256 * 1024 * 1024
# Builds without --timing-only accept dimensions up to 4096 and 1e8 MACs per call.
LEGACY_LIMITS = {"max_dimension": 4096, "max_macs": 100_000_000}


@dataclass(frozen=True)
class Gemm:
    name: str
    m: int
    n: int
    k: int
    groups: int = 1
    calls: int = 1
    op: str = "dense"
    # analytic.CONV_FIELDS values of a convolution's geometry; () for a plain operand.
    conv: tuple[int, ...] = ()

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k * self.groups

    @property
    def total_macs(self) -> int:
        return self.macs * self.calls

    def conv_flags(self) -> dict[str, int]:
        """npu-sim --conv-* flags of the geometry, those that differ from the defaults."""
        defaults = analytic.Design()
        return {
            name: value
            for name, value in zip(analytic.CONV_FIELDS, self.conv, strict=False)
            if value != getattr(defaults, name)
        }

    def flags(self, conv: bool = True) -> dict[str, int]:
        """npu-sim shape flags; with ``conv`` the convolution geometry too."""
        flags = {"m": self.m, "n": self.n, "k": self.k}
        if self.groups != 1:
            flags["groups"] = self.groups
        if conv:
            flags.update(self.conv_flags())
        return flags

    def plain(self) -> Gemm:
        """The same GEMM on its im2col operand, without the convolution geometry."""
        return replace(self, conv=())

    @property
    def pointwise(self) -> bool:
        """A 1x1 convolution: its im2col rows are input pixels, so the plain operand reads
        each needed pixel once, as one run per output pixel."""
        fields = dict(zip(analytic.CONV_FIELDS, self.conv, strict=False))
        return bool(self.conv) and fields["conv_kernel_h"] == fields["conv_kernel_w"] == 1

    @property
    def op_class(self) -> str:
        """Coarse class for plots: conv, depthwise or matmul (dense, GEMV, attention)."""
        if self.op == "depthwise_conv2d":
            return "depthwise"
        return "conv" if self.op == "conv2d" else "matmul"


def names(directory: Path = WORKLOAD_DIR) -> list[str]:
    return sorted(path.stem for path in directory.glob("*.json"))


def load(name: str | Path, directory: Path = WORKLOAD_DIR) -> dict[str, Any]:
    path = Path(name)
    if not (path.suffix == ".json" and path.is_file()):
        path = directory / (str(name) + ".json")
    if not path.is_file():
        raise FileNotFoundError(f"no workload {name!r} (known: {', '.join(names(directory))})")
    document = jsonio.load_strict(path)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(document.get("layers"), list)
    ):
        raise ValueError(f"{path} is not a schema_version 1 workload")
    return document


def conv_geometry(layer: dict[str, Any]) -> tuple[int, ...]:
    """analytic.CONV_FIELDS values of a convolution layer, or () when it does not give its
    input extents, stride and padding."""
    if not all(key in layer for key in ("in_h", "in_w", "stride", "pad")):
        return ()
    pad = layer["pad"]
    top, bottom, left, right = layer.get("pad_tblr", (pad, pad, pad, pad))
    geometry = {
        "conv_input_h": layer["in_h"],
        "conv_input_w": layer["in_w"],
        "conv_input_c": layer["in_c"],
        "conv_kernel_h": layer["kernel_h"],
        "conv_kernel_w": layer["kernel_w"],
        "conv_stride_h": layer["stride"],
        "conv_stride_w": layer["stride"],
        "conv_dilation_h": 1,
        "conv_dilation_w": 1,
        "conv_pad_top": top,
        "conv_pad_left": left,
        "conv_pad_bottom": bottom,
        "conv_pad_right": right,
    }
    return tuple(geometry[name] for name in analytic.CONV_FIELDS)


def lower(layer: dict[str, Any]) -> Gemm:
    """The npu-sim call for one layer, recomputed from its shape fields."""
    op = layer["op"]
    if op in ("conv2d", "depthwise_conv2d"):
        groups = layer["groups"]
        gemm = Gemm(
            layer["name"],
            layer["out_h"] * layer["out_w"],
            layer["out_c"] // groups,
            layer["kernel_h"] * layer["kernel_w"] * (layer["in_c"] // groups),
            groups,
            layer["repeat"],
            op,
            conv_geometry(layer),
        )
    elif op in ("dense", "matmul"):
        record = layer.get("lowered", {})
        groups = record.get("G", 1)
        calls = record.get("calls", layer["repeat"])
        m = record.get("M", layer["M"])
        gemm = Gemm(layer["name"], m, layer["N"], layer["K"], groups, calls, op)
    else:
        raise ValueError(f"cannot lower layer op {op!r}")
    if gemm.total_macs != layer["macs"] * layer["repeat"]:
        raise ValueError(f"{layer['name']}: lowering does not preserve the layer's MACs")
    return gemm


def gemms(workload: dict[str, Any]) -> list[Gemm]:
    return [lower(layer) for layer in workload["layers"]]


def host_bytes(gemm: Gemm, tile_m: int = 8, tile_n: int = 8, tile_k: int = 0) -> int:
    """Simulator working storage for a functional run: tensors, reference and simulated
    outputs (10 B per output), INT64 tile scratch, and the per-panel trace budget (512 B)."""
    g = gemm.groups
    tiles = math.ceil(gemm.m / tile_m) * math.ceil(gemm.n / tile_n) * g
    panels = math.ceil(gemm.k / tile_k) if tile_k else 1
    fields = dict(zip(analytic.CONV_FIELDS, gemm.conv, strict=False))
    # Conv geometry also holds the NHWC input the operand is built from.
    image = (
        fields["conv_input_h"] * fields["conv_input_w"] * fields["conv_input_c"] if gemm.conv else 0
    )
    return (
        gemm.m * g * gemm.k
        + g * gemm.k * gemm.n
        + 4 * g * gemm.n
        + 10 * gemm.m * g * gemm.n
        + 8 * min(gemm.m, tile_m) * min(gemm.n, tile_n)
        + 512 * tiles * panels
        + image
    )


def fits(
    gemm: Gemm,
    timing_only: bool = True,
    max_dimension: int = MAX_DIMENSION,
    max_macs: int | None = None,
    max_host: int = MAX_HOST_BYTES,
    tile_m: int = 8,
    tile_n: int = 8,
    tile_k: int = 0,
) -> bool:
    """True when one npu-sim call runs ``gemm``. The activation row (G*K) and output row
    (G*N) obey the dimension limit as well as M, N and K."""
    limit = (
        max_macs
        if max_macs is not None
        else (MAX_TIMING_MACS if timing_only else MAX_FUNCTIONAL_MACS)
    )
    widest = max(gemm.m, gemm.groups * gemm.n, gemm.groups * gemm.k)
    if widest > max_dimension or gemm.macs > limit:
        return False
    return timing_only or host_bytes(gemm, tile_m, tile_n, tile_k) <= max_host


def _halves(total: int, parts: int) -> list[tuple[int, int]]:
    """(size, count) classes splitting ``total`` into ``parts`` near-equal pieces."""
    base, extra = divmod(total, parts)
    return [(size, count) for size, count in ((base + 1, extra), (base, parts - extra)) if count]


def split(gemm: Gemm, timing_only: bool = True, **limits: Any) -> list[Gemm]:
    """Cut ``gemm`` into the fewest even pieces that each fit the per-call limits."""
    if fits(gemm, timing_only, **limits):
        return [gemm]
    gemm = gemm.plain()
    max_dimension = limits.get("max_dimension", MAX_DIMENSION)
    if gemm.k > max_dimension:
        raise ValueError(
            f"{gemm.name}: K={gemm.k} exceeds the per-call limit of "
            f"{max_dimension}; a split K needs partial-sum accumulation across "
            "calls"
        )
    for axis in ("n", "m"):
        extent = getattr(gemm, axis)
        if extent > max_dimension:
            pieces = [
                replace(gemm, **{axis: size}, calls=gemm.calls * count)
                for size, count in _halves(extent, math.ceil(extent / max_dimension))
            ]
            return [part for piece in pieces for part in split(piece, timing_only, **limits)]
    for axis in ("groups", "n", "m"):
        extent = getattr(gemm, axis)
        if extent == 1:
            continue
        for parts in range(2, extent + 1):
            pieces = [
                replace(gemm, **{axis: size}, calls=gemm.calls * count)
                for size, count in _halves(extent, parts)
            ]
            if all(fits(piece, timing_only, **limits) for piece in pieces):
                return pieces
        unit = replace(gemm, **{axis: 1}, calls=gemm.calls * extent)
        return split(unit, timing_only, **limits)
    raise ValueError(f"{gemm.name}: a single output element exceeds the per-call limits")


def ungroup(gemm: Gemm) -> Gemm:
    """The same work as one call per group, for builds without --groups."""
    return replace(gemm, groups=1, calls=gemm.calls * gemm.groups, conv=())


def call_options(binary: Path | str) -> dict[str, Any]:
    """``iter_calls`` options for one npu-sim build, probed from its --help."""
    timing_only = sim.supports(binary, "--timing-only")
    options: dict[str, Any] = {
        "timing_only": timing_only,
        "grouped": sim.supports(binary, "--groups"),
        "conv": sim.supports(binary, "--conv-input-h"),
    }
    if not timing_only:
        options.update(LEGACY_LIMITS)
    return options


def iter_calls(
    workload: dict[str, Any],
    timing_only: bool = True,
    grouped: bool = True,
    conv: bool = True,
    **limits: Any,
) -> Iterator[tuple[dict[str, Any], Gemm]]:
    for layer in workload["layers"]:
        gemm = lower(layer)
        if not conv:
            gemm = gemm.plain()
        if not grouped and gemm.groups > 1:
            gemm = ungroup(gemm)
        for piece in split(gemm, timing_only, **limits):
            yield layer, piece
