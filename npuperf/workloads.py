"""Canonical network workloads (workloads/*.json) lowered to npu-sim GEMM calls.

A layer becomes one ``Gemm``: an M x N x K GEMM per group, G groups (npu-sim --groups G),
run ``calls`` times. Convolutions lower by im2col (M = out_h*out_w, N = out_c/G,
K = kh*kw*in_c/G); depthwise is G = channels; attention matmuls run per KV head.
``split`` cuts a call only when it exceeds a per-call simulator limit, along groups first,
then N, then M; a reduction too deep for one call cannot be split (the simulator does not
accumulate partial sums across calls).
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import jsonio
from .binary import ROOT

WORKLOAD_DIR = ROOT / "workloads"

# Per-call limits from the simulator spec. Functional runs generate and check every
# tensor; --timing-only runs skip tensors and numerics.
MAX_DIMENSION = 1 << 20
MAX_FUNCTIONAL_MACS = 4_000_000_000
MAX_TIMING_MACS = 10_000_000_000_000
MAX_HOST_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Gemm:
    name: str
    m: int
    n: int
    k: int
    groups: int = 1
    calls: int = 1
    op: str = "dense"

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k * self.groups

    @property
    def total_macs(self) -> int:
        return self.macs * self.calls

    def flags(self) -> dict[str, int]:
        flags = {"m": self.m, "n": self.n, "k": self.k}
        if self.groups != 1:
            flags["groups"] = self.groups
        return flags

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
    if not isinstance(document, dict) or document.get("schema_version") != 1 \
            or not isinstance(document.get("layers"), list):
        raise ValueError(f"{path} is not a schema_version 1 workload")
    return document


def lower(layer: dict[str, Any]) -> Gemm:
    """The npu-sim call for one layer, recomputed from its shape fields."""
    op = layer["op"]
    if op in ("conv2d", "depthwise_conv2d"):
        groups = layer["groups"]
        gemm = Gemm(layer["name"], layer["out_h"] * layer["out_w"], layer["out_c"] // groups,
                    layer["kernel_h"] * layer["kernel_w"] * (layer["in_c"] // groups), groups,
                    layer["repeat"], op)
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


def host_bytes(gemm: Gemm, tile_m: int = 8, tile_n: int = 8) -> int:
    """Simulator working storage for a functional run: tensors, reference and simulated
    outputs (10 B per output), and the per-panel trace budget (512 B)."""
    g = gemm.groups
    tiles = math.ceil(gemm.m / tile_m) * math.ceil(gemm.n / tile_n) * g
    return (gemm.m * g * gemm.k + g * gemm.k * gemm.n + 4 * g * gemm.n
            + 10 * gemm.m * g * gemm.n + 512 * tiles)


def fits(gemm: Gemm, timing_only: bool = True, max_dimension: int = MAX_DIMENSION,
         max_macs: int | None = None, max_host: int = MAX_HOST_BYTES) -> bool:
    limit = max_macs if max_macs is not None else (MAX_TIMING_MACS if timing_only
                                                     else MAX_FUNCTIONAL_MACS)
    if max(gemm.m, gemm.n, gemm.k) > max_dimension or gemm.macs > limit:
        return False
    return timing_only or host_bytes(gemm) <= max_host


def _halves(total: int, parts: int) -> list[tuple[int, int]]:
    """(size, count) classes splitting ``total`` into ``parts`` near-equal pieces."""
    base, extra = divmod(total, parts)
    return [(size, count) for size, count in ((base + 1, extra), (base, parts - extra)) if count]


def split(gemm: Gemm, timing_only: bool = True, **limits: Any) -> list[Gemm]:
    """Cut ``gemm`` into the fewest even pieces that each fit the per-call limits."""
    if fits(gemm, timing_only, **limits):
        return [gemm]
    max_dimension = limits.get("max_dimension", MAX_DIMENSION)
    if gemm.k > max_dimension:
        raise ValueError(f"{gemm.name}: K={gemm.k} exceeds the per-call limit; a split K "
                         "needs partial-sum accumulation across calls")
    for axis in ("n", "m"):
        extent = getattr(gemm, axis)
        if extent > max_dimension:
            pieces = [replace(gemm, **{axis: size}, calls=gemm.calls * count)
                      for size, count in _halves(extent, math.ceil(extent / max_dimension))]
            return [part for piece in pieces for part in split(piece, timing_only, **limits)]
    for axis in ("groups", "n", "m"):
        extent = getattr(gemm, axis)
        if extent == 1:
            continue
        for parts in range(2, extent + 1):
            pieces = [replace(gemm, **{axis: size}, calls=gemm.calls * count)
                      for size, count in _halves(extent, parts)]
            if all(fits(piece, timing_only, **limits) for piece in pieces):
                return pieces
        unit = replace(gemm, **{axis: 1}, calls=gemm.calls * extent)
        return split(unit, timing_only, **limits)
    raise ValueError(f"{gemm.name}: a single output element exceeds the per-call limits")


def iter_calls(workload: dict[str, Any], timing_only: bool = True,
               **limits: Any) -> Iterator[tuple[dict[str, Any], Gemm]]:
    for layer in workload["layers"]:
        for gemm in split(lower(layer), timing_only, **limits):
            yield layer, gemm
