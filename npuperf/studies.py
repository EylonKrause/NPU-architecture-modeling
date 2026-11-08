"""The studies that generate the tables and figures of docs/results.md.

Each study returns markdown tables (keyed by their docs/results.md marker), SVG figures
(docs/figures/) and raw records (results/studies/<id>.json). ``python -m npuperf.reproduce``
runs them; see that module for the command line.

Conventions shared by the studies:
  - Design points are configs/*.json (hypothetical configurations sized from public data,
    docs/design-points.md). A layer runs on its point with the tiles npuperf.mapper picks.
  - Only the GEMM layers of a workload run; vector operators (softmax, normalization,
    pooling, residual adds) are not modeled. A layer with ``calls`` identical instances
    runs once and counts ``calls`` times. A convolution runs on its NHWC input footprint
    (npuperf.workloads conv geometry), or a 1x1 convolution on its plain operand when the
    mapper picks it; depthwise convolution runs on each point's depthwise datapath, its
    rate sized by S9.
  - A call runs bit-exact (tensors, numerics, reference) when it fits the functional limits
    and FUNCTIONAL_MACS; otherwise --timing-only. Dense timing does not depend on the data.
  - External-memory bytes are the bytes the DRAM timeline charges: whole AXI beats, so a
    partial beat counts in full.
  - MAC utilization is MACs over cycles times the array's peak, the depthwise layers'
    MACs included.
  - One MAC is two operations (TOP/s).
  - A result that breaks a physical limit (above a roof, utilization above 1, speedup above
    the core count, bandwidth above the bus) stops the run with a ``SanityError``.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import math
import random
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import analytic, autotune, designs, mapper, roofline, sim, svg, workloads
from .binary import ROOT

FUNCTIONAL_MACS = 400_000_000
OPS_PER_MAC = 2
RESULTS_DIR = ROOT / "results" / "studies"
FIGURE_DIR = ROOT / "docs" / "figures"

NETWORKS = {
    "mobilenet_v2_1.0_224": "MobileNetV2",
    "resnet50_v1.5_224": "ResNet-50 v1.5",
    "bert_base_encoder_layer_s128": "BERT-base layer",
}
DECODE = "llama3.2_1b_decode_kv129"
PREFILL = "llama3.2_1b_prefill_s128"
NETWORK_ORDER = (
    "MobileNetV2",
    "ResNet-50 v1.5",
    "BERT-base layer",
    "Llama decode",
    "Llama prefill",
)


class SanityError(RuntimeError):
    """A simulator result breaks a physical limit."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SanityError(message)


# ---------------------------------------------------------------------------- running --


@dataclass
class Context:
    """The simulator binary and caches of results keyed by their flags and of mapped calls
    keyed by their task, shared by every study of one reproduce run, so a run or a mapping
    two studies need happens once."""

    binary: Path
    jobs: int = 1
    timeout: float = 900
    work: Path = RESULTS_DIR
    cache: dict[str, dict] = field(default_factory=dict)
    extra_runs: int = 0  # runs made outside run_all (the autotune search)
    mapped: dict = field(default_factory=dict)

    def parallel(self, function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        """``function`` of every item, in order; up to ``jobs`` at a time, each in its own
        process (the mapper and the analytic model are pure Python). Results are cached by
        function and item."""
        name = function.__qualname__
        missing = [item for item in dict.fromkeys(items) if (name, item) not in self.mapped]
        if len(missing) > 1 and self.jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(min(self.jobs, len(missing))) as pool:
                for item, value in zip(missing, pool.map(function, missing), strict=True):
                    self.mapped[(name, item)] = value
        else:
            for item in missing:
                self.mapped[(name, item)] = function(item)
        return [self.mapped[(name, item)] for item in items]

    def layer_flags(self, tasks: Sequence[Task]) -> list[dict[str, Any]]:
        """layer_flags of every task, in order, mapped in parallel."""
        return [dict(flags) for flags in self.parallel(_task_flags, tasks)]

    @staticmethod
    def key(flags: Mapping[str, Any]) -> str:
        return json.dumps(flags, sort_keys=True)

    def run_all(self, runs: Sequence[Mapping[str, Any]]) -> list[dict]:
        """npu-sim results of ``runs`` (flag dicts), in order; at most ``jobs`` at a time."""
        pending = {}
        for flags in runs:
            key = self.key(flags)
            if key not in self.cache and key not in pending:
                pending[key] = flags

        def one(flags: Mapping[str, Any]) -> dict:
            return sim.run(dict(flags), self.binary, self.timeout).raw

        with concurrent.futures.ThreadPoolExecutor(max(1, self.jobs)) as pool:
            for key, result in zip(pending, pool.map(one, pending.values()), strict=True):
                self.cache[key] = result
        return [self.cache[self.key(flags)] for flags in runs]


@dataclass(frozen=True)
class Point:
    """A design point: its config file, the analytic.Design it describes and the memory
    behind its port."""

    name: str
    path: Path
    design: analytic.Design
    min_depth: int
    memory: str = "DRAM"

    @property
    def frequency_mhz(self) -> float:
        return self.design.frequency_mhz

    @property
    def runs_llm(self) -> bool:
        """The LLM workloads run only on points that stream their weights from DRAM."""
        return self.memory == "DRAM"

    def flags(self, **extra: Any) -> dict[str, Any]:
        """--config first, so the flags after it override the file."""
        return {"config": str(self.path), **extra}


def design_points(names: Iterable[str] = designs.DESIGN_POINTS) -> list[Point]:
    points = []
    for name in names:
        document = designs.load(name)
        points.append(
            Point(
                name,
                designs.config_path(name),
                designs.design(document),
                designs.min_panel_depth(name),
                designs.external_memory(name),
            )
        )
    return points


def functional(gemm: workloads.Gemm, mapping: mapper.Mapping) -> bool:
    return gemm.macs <= FUNCTIONAL_MACS and workloads.fits(
        gemm, timing_only=False, tile_m=mapping.tile_m, tile_n=mapping.tile_n, tile_k=mapping.tile_k
    )


@dataclass(frozen=True)
class Task:
    """One call to map: ``layer_flags(point, gemm, weight_bits, **dict(changes))``."""

    point: Point
    gemm: workloads.Gemm
    weight_bits: int = 8
    changes: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def of(cls, point: Point, gemm: workloads.Gemm, weight_bits: int = 8, **changes: Any) -> Task:
        # The mapping depends on the shape and geometry only, so repeated layers map once.
        shape = dataclasses.replace(gemm, name="", calls=1, op="")
        return cls(point, shape, weight_bits, tuple(sorted(changes.items())))


def _task_flags(task: Task) -> dict[str, Any]:
    return layer_flags(task.point, task.gemm, task.weight_bits, **dict(task.changes))


def layer_flags(
    point: Point, gemm: workloads.Gemm, weight_bits: int = 8, **design_changes: Any
) -> dict[str, Any]:
    """npu-sim flags of one call of ``gemm`` on ``point`` with the mapper's tiles and
    lowering."""
    design = point.design.replace(weight_bits=weight_bits, **design_changes)
    mapping = mapper.map_call(design, gemm, point.min_depth)
    flags = point.flags(**{key: value for key, value in design_changes.items()})
    flags.update(gemm.flags(conv=mapping.conv))
    flags.update(mapping.flags())
    if weight_bits != 8:
        flags["weight_bits"] = weight_bits
    if not functional(gemm, mapping):
        flags["timing_only"] = True
    return flags


def dram_bytes(result: Mapping[str, Any]) -> int:
    memory = result.get("memory", {})
    if "dram" in memory:
        return memory["dram"]["bytes"]
    return result["read_bytes"] + result["write_bytes"]


def check_result(result: Mapping[str, Any], label: str) -> None:
    """Physical limits of one run: utilization, the compute and DRAM roofs, the bus."""
    cycles, macs = result["cycles"], result["macs"]
    peak = result["effective_macs_per_cycle"]
    require(macs <= peak * cycles, f"{label}: {macs} MACs exceed {peak} MAC/cycle x {cycles}")
    rate = result["peak_memory_bytes_per_cycle"]
    moved = dram_bytes(result)
    require(moved <= rate * cycles, f"{label}: {moved} DRAM bytes exceed {rate} B/cycle x {cycles}")
    config = result["config"]
    if sim.config_value(config, "memory_model") == "axi":
        bus = sim.config_value(config, "axi_data_bytes")
        for core in result["cores"]:
            for direction in ("read", "write"):
                channel = core["memory"][direction]
                require(
                    channel["beats"] * bus <= bus * cycles
                    and channel["payload_bytes"] <= bus * cycles,
                    f"{label}: core {core['core']} {direction} channel above its bus",
                )


# ------------------------------------------------------------------------- formatting --


def markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], numeric: Sequence[bool] | None = None
) -> str:
    """A GitHub table; numeric columns align right."""
    numeric = numeric or [False] + [True] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" if right else "---" for right in numeric) + "|",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def fmt(value: float, digits: int = 3) -> str:
    """``digits`` significant figures, thousands separated, never exponent notation."""
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, digits - 1 - magnitude)
    rounded = round(value, decimals)
    return f"{rounded:,.{decimals}f}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def pct(value: float, decimals: int = 1) -> str:
    return f"{100 * value:.{decimals}f}%"


def percent_tick(value: float) -> str:
    """Axis label of a fraction, in percent."""
    return f"{100 * value:g}%"


def kib(byte_count: float) -> str:
    """Bytes in KiB with two decimals, the fixed precision of the KiB table columns."""
    return f"{byte_count / 1024:,.2f}"


def figure_path(name: str) -> str:
    return f"figures/{name}.svg"


@dataclass
class StudyOutput:
    tables: dict[str, str] = field(default_factory=dict)
    figures: dict[str, svg.Figure] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------- network workloads --


@dataclass(frozen=True)
class Call:
    network: str
    layer: dict
    gemm: workloads.Gemm
    flags: dict


@dataclass(frozen=True)
class Planned:
    """A call before its mapping: ``task`` gives its flags."""

    network: str
    layer: dict
    gemm: workloads.Gemm
    task: Task


def plan_calls(
    point: Point,
    workload_name: str,
    network: str,
    weight_bits: int = 8,
    plain_depthwise: bool = False,
    **changes: Any,
) -> list[Planned]:
    """Every call of a workload on ``point``. INT4 applies to weight operands only; an
    attention matmul's B operand is the KV cache and stays INT8. ``plain_depthwise`` runs
    the depthwise layers on their im2col operand (the lowering without conv geometry)."""
    workload = workloads.load(workload_name)
    calls = []
    for layer, gemm in workloads.iter_calls(workload):
        bits = weight_bits if layer.get("b_operand", "weights") == "weights" else 8
        if plain_depthwise and layer["op"] == "depthwise_conv2d":
            gemm = gemm.plain()
        calls.append(Planned(network, layer, gemm, Task.of(point, gemm, bits, **changes)))
    return calls


def resolve(ctx: Context, plans: Mapping[Any, Sequence[Planned]]) -> dict[Any, list[Call]]:
    """The Calls of every plan, mapped in one batch."""
    flat = [planned for calls in plans.values() for planned in calls]
    flags = iter(ctx.layer_flags([planned.task for planned in flat]))
    return {
        key: [Call(p.network, p.layer, p.gemm, next(flags)) for p in calls]
        for key, calls in plans.items()
    }


def network_calls(
    ctx: Context, point: Point, workload_name: str, network: str, weight_bits: int = 8
) -> list[Call]:
    return resolve(ctx, {0: plan_calls(point, workload_name, network, weight_bits)})[0]


@dataclass(frozen=True)
class Totals:
    cycles: int
    macs: int
    dram_bytes: int
    bound_calls: dict[str, int]
    calls: int
    bit_exact_calls: int

    def latency_ms(self, frequency_mhz: float) -> float:
        return self.cycles / (frequency_mhz * 1e3)


def totals(calls: Sequence[Call], results: Sequence[Mapping[str, Any]]) -> Totals:
    cycles = macs = moved = exact = count = 0
    bounds = {"compute": 0, "bandwidth": 0, "latency": 0}
    for call, result in zip(calls, results, strict=True):
        n = call.gemm.calls
        cycles += n * result["cycles"]
        macs += n * result["macs"]
        moved += n * dram_bytes(result)
        bounds[analytic.observed_bound(result)] += n
        count += n
        exact += n if result.get("numerics") == "bit-exact" else 0
    return Totals(cycles, macs, moved, bounds, count, exact)


def run_networks(ctx: Context, points: Sequence[Point]) -> dict[tuple[str, str, int], tuple]:
    """(point, network, weight bits) -> (calls, results) for the S1 and S2 workloads."""
    plans: dict[tuple[str, str, int], list[Planned]] = {}
    for point in points:
        for name, network in NETWORKS.items():
            plans[(point.name, network, 8)] = plan_calls(point, name, network)
        if not point.runs_llm:
            continue
        for bits in (8, 4):
            plans[(point.name, "Llama decode", bits)] = plan_calls(
                point, DECODE, "Llama decode", bits
            )
            plans[(point.name, "Llama prefill", bits)] = plan_calls(
                point, PREFILL, "Llama prefill", bits
            )
    plan = resolve(ctx, plans)
    flat = [call.flags for calls in plan.values() for call in calls]
    results = iter(ctx.run_all(flat))
    out = {}
    for key, calls in plan.items():
        chunk = [next(results) for _ in calls]
        for call, result in zip(calls, chunk, strict=True):
            check_result(result, f"{key[0]} {call.network} {call.layer['name']}")
        out[key] = (calls, chunk)
    return out


def network_weight_bytes(name: str) -> int:
    """INT8 bytes of every weight operand of a workload, each once. An attention matmul's B
    operand is the KV cache and counts as an activation."""
    total = 0
    for layer in workloads.load(name)["layers"]:
        if layer.get("b_operand", "weights") == "weights":
            gemm = workloads.lower(layer)
            total += gemm.calls * gemm.groups * gemm.k * gemm.n
    return total


def burst_efficiency(pairs: Sequence[tuple[Call, Mapping[str, Any]]], direction: str) -> float:
    """Payload bytes over whole-beat bytes of one AXI direction, every call counted."""
    payload = moved = 0
    for call, result in pairs:
        channel = result["memory"][direction]
        payload += call.gemm.calls * channel["payload_bytes"]
        moved += (
            call.gemm.calls
            * channel["beats"]
            * sim.config_value(result["config"], "axi_data_bytes")
        )
    return payload / moved


def depthwise_row(
    point: Point, calls: Sequence[Call], results: Sequence[Mapping[str, Any]]
) -> tuple[list[str], dict[str, Any]]:
    """MobileNetV2 with and without its depthwise layers on ``point``."""
    pairs = list(zip(calls, results, strict=True))
    depthwise = [pair for pair in pairs if pair[0].layer["op"] == "depthwise_conv2d"]
    rest = [pair for pair in pairs if pair[0].layer["op"] != "depthwise_conv2d"]
    whole, dw = totals(calls, results), totals(*zip(*depthwise, strict=True))
    other = totals(*zip(*rest, strict=True))
    util = other.macs / (other.cycles * point.design.array_macs_per_cycle)
    record = {
        "point": point.name,
        "cycles": whole.cycles,
        "depthwise_cycles": dw.cycles,
        "depthwise_macs_per_cycle": point.design.depthwise_macs_per_cycle,
        "dram_bytes": whole.dram_bytes,
        "depthwise_dram_bytes": dw.dram_bytes,
        "depthwise_read_efficiency": burst_efficiency(depthwise, "read"),
        "depthwise_write_efficiency": burst_efficiency(depthwise, "write"),
        "other_cycles": other.cycles,
        "other_macs": other.macs,
        "other_mac_utilisation": util,
    }
    row = [
        f"`{point.name}`",
        f"{point.design.depthwise_macs_per_cycle:,}",
        fmt(whole.latency_ms(point.frequency_mhz)),
        pct(dw.cycles / whole.cycles),
        pct(dw.dram_bytes / whole.dram_bytes),
        pct(record["depthwise_read_efficiency"]),
        pct(record["depthwise_write_efficiency"]),
        fmt(other.latency_ms(point.frequency_mhz)),
        pct(util),
    ]
    return row, record


def half_sram_decode(ctx: Context, points: Sequence[Point]) -> dict[str, Totals]:
    """INT8 decode totals of every LLM point with half its SRAM per core."""
    plans = {
        point.name: plan_calls(
            point, DECODE, "Llama decode", 8, sram_bytes=point.design.sram_bytes // 2
        )
        for point in points
        if point.runs_llm
    }
    planned = resolve(ctx, plans)
    results = iter(ctx.run_all([call.flags for calls in planned.values() for call in calls]))
    out = {}
    for name, calls in planned.items():
        chunk = [next(results) for _ in calls]
        for call, result in zip(calls, chunk, strict=True):
            check_result(result, f"{name} half-SRAM decode {call.layer['name']}")
        out[name] = totals(calls, chunk)
    return out


def s1_networks(ctx: Context) -> StudyOutput:
    points = design_points()
    runs = run_networks(ctx, points)
    half = half_sram_decode(ctx, points)
    out = StudyOutput()
    rows, llm, depthwise, raw = [], [], [], []
    weights = {network: network_weight_bytes(name) for name, network in NETWORKS.items()}
    for point in points:
        peak = point.design.array_macs_per_cycle
        latency = {}
        for network in NETWORKS.values():
            calls, results = runs[(point.name, network, 8)]
            t = totals(calls, results)
            ms = t.latency_ms(point.frequency_mhz)
            latency[network] = ms
            util = t.macs / (t.cycles * peak)
            require(util <= 1, f"{point.name} {network}: utilization {util}")
            memory = t.bound_calls["bandwidth"] + t.bound_calls["latency"]
            rows.append(
                [
                    f"`{point.name}`",
                    network,
                    fmt(weights[network] / 1e6),
                    fmt(ms),
                    fmt(1e3 / ms),
                    fmt(OPS_PER_MAC * t.macs / (ms * 1e9)),
                    pct(util),
                    fmt(t.dram_bytes / 1e6),
                    pct(t.bound_calls["compute"] / t.calls, 0),
                    pct(memory / t.calls, 0),
                ]
            )
            raw.append(
                {
                    "point": point.name,
                    "network": network,
                    "weight_bytes": weights[network],
                    "cycles": t.cycles,
                    "macs": t.macs,
                    "dram_bytes": t.dram_bytes,
                    "latency_ms": ms,
                    "bound_calls": t.bound_calls,
                    "calls": t.calls,
                    "bit_exact_calls": t.bit_exact_calls,
                }
            )
        # MobileNetV2 has 13.6x fewer MACs and 7.4x fewer weight bytes than ResNet-50
        # (workloads/*.json totals).
        require(
            latency["MobileNetV2"] < latency["ResNet-50 v1.5"],
            f"{point.name}: MobileNetV2 {latency['MobileNetV2']} ms is not below ResNet-50 "
            f"{latency['ResNet-50 v1.5']} ms",
        )
        row, record = depthwise_row(point, *runs[(point.name, "MobileNetV2", 8)])
        depthwise.append(row)
        raw.append({"network": "MobileNetV2 depthwise split", **record})
        if not point.runs_llm:
            continue
        decode = {bits: totals(*runs[(point.name, "Llama decode", bits)]) for bits in (8, 4)}
        prefill = {bits: totals(*runs[(point.name, "Llama prefill", bits)]) for bits in (8, 4)}
        tps = {bits: 1e3 / decode[bits].latency_ms(point.frequency_mhz) for bits in (8, 4)}
        gbps = {bits: decode[bits].dram_bytes * tps[bits] / 1e9 for bits in (8, 4)}
        peak_gbps = point.design.bandwidth * point.frequency_mhz / 1e3
        for bits in (8, 4):
            require(
                gbps[bits] <= peak_gbps * (1 + 1e-9),
                f"{point.name} decode INT{bits}: {gbps[bits]} GB/s above {peak_gbps}",
            )
        # Mapping checks. Every candidate tiling of half the SRAM exists with the full SRAM,
        # its caches larger, and the mapper keeps the fastest replay of its candidates; INT4
        # halves only the weight bytes of a decode that streams them.
        require(
            decode[8].cycles <= half[point.name].cycles,
            f"{point.name} decode INT8: {decode[8].cycles} cycles, slower than "
            f"{half[point.name].cycles} with half the SRAM",
        )
        require(
            decode[8].cycles <= 2 * decode[4].cycles,
            f"{point.name} decode: INT4 speedup {decode[8].cycles / decode[4].cycles:.3f}"
            f"x above the 2x of halving the weight bytes",
        )
        llm.append(
            [
                f"`{point.name}`",
                fmt(tps[8]),
                fmt(tps[4]),
                f"{tps[4] / tps[8]:.2f}x",
                fmt(decode[8].dram_bytes / 1e6),
                pct(gbps[8] / peak_gbps),
                fmt(prefill[8].latency_ms(point.frequency_mhz)),
                fmt(prefill[4].latency_ms(point.frequency_mhz)),
                pct(prefill[8].macs / (prefill[8].cycles * point.design.array_macs_per_cycle)),
            ]
        )
        raw.append(
            {
                "point": point.name,
                "network": "Llama decode, half the SRAM",
                "weight_bits": 8,
                "sram_bytes": point.design.sram_bytes // 2,
                "cycles": half[point.name].cycles,
                "dram_bytes": half[point.name].dram_bytes,
            }
        )
        for kind, table in (("decode", decode), ("prefill", prefill)):
            for bits, t in table.items():
                raw.append(
                    {
                        "point": point.name,
                        "network": f"Llama {kind}",
                        "weight_bits": bits,
                        "cycles": t.cycles,
                        "macs": t.macs,
                        "dram_bytes": t.dram_bytes,
                        "latency_ms": t.latency_ms(point.frequency_mhz),
                        "bound_calls": t.bound_calls,
                        "calls": t.calls,
                        "bit_exact_calls": t.bit_exact_calls,
                    }
                )
    out.tables["s1-networks"] = markdown_table(
        [
            "Design point",
            "Network",
            "Weights (MB)",
            "Latency (ms)",
            "Inferences/s",
            "Achieved TOP/s",
            "MAC utilization",
            "DRAM MB/inference",
            "Compute-bound layers",
            "Memory-bound layers",
        ],
        rows,
        [False, False] + [True] * 8,
    )
    out.tables["s1-depthwise"] = markdown_table(
        [
            "Design point",
            "Depthwise MAC/cycle per core",
            "MobileNetV2 latency (ms)",
            "Depthwise share of cycles",
            "Depthwise share of DRAM bytes",
            "Depthwise read burst efficiency",
            "Depthwise write burst efficiency",
            "Latency without depthwise (ms)",
            "MAC utilization without depthwise",
        ],
        depthwise,
    )
    out.tables["s1-llm"] = markdown_table(
        [
            "Design point",
            "Decode tokens/s INT8",
            "Decode tokens/s INT4",
            "INT4 speedup",
            "Decode DRAM MB/token INT8",
            "Decode DRAM rate used INT8",
            "Prefill TTFT INT8 (ms)",
            "Prefill TTFT INT4 (ms)",
            "Prefill MAC utilization INT8",
        ],
        llm,
    )
    out.raw["networks"] = raw
    out.raw["calls"] = [
        {
            "point": key[0],
            "network": key[1],
            "weight_bits": key[2],
            "layer": call.layer["name"],
            "calls": call.gemm.calls,
            "flags": call.flags,
            "result": result,
        }
        for key, (calls, results) in runs.items()
        for call, result in zip(calls, results, strict=True)
    ]
    return out


def s2_roofline(ctx: Context) -> StudyOutput:
    points = design_points()
    runs = run_networks(ctx, points)
    out = StudyOutput()
    raw = []
    for point in points:
        marks = []
        networks = [network for network in NETWORK_ORDER if (point.name, network, 8) in runs]
        for network in networks:
            calls, results = runs[(point.name, network, 8)]
            for call, result in zip(calls, results, strict=True):
                intensity = result["macs"] / dram_bytes(result)
                achieved = result["macs"] / result["cycles"]
                # The roof of the datapath the call ran on: the depthwise datapath's peak
                # for a channel-wise call.
                roof = analytic.attainable(analytic.Design.from_result(result), intensity)
                require(
                    achieved <= roof * (1 + 1e-12),
                    f"{point.name} {network} {call.layer['name']}: {achieved} MAC/cycle "
                    f"above the {roof} roof",
                )
                marks.append(roofline.Point(call.layer["name"], intensity, achieved, network))
                raw.append(
                    {
                        "point": point.name,
                        "network": network,
                        "layer": call.layer["name"],
                        "intensity": intensity,
                        "macs_per_cycle": achieved,
                        "attainable": roof,
                    }
                )
        d = point.design
        array = f"{d.array_rows}x{d.array_cols} {d.dataflow.upper()}"
        cores = f"{d.cores} cores of {array}" if d.cores > 1 else f"one {array} array"
        subtitle = (
            f"{cores}, {d.peak_macs_per_cycle:,} MAC/cycle; {point.memory} "
            f"{d.dram_bytes_per_cycle:g} B/cycle; every GEMM layer of "
            f"{len(networks)} workloads, INT8"
        )
        fig = roofline.figure(
            d,
            marks,
            title=f"Roofline: {point.name}",
            subtitle=subtitle,
            label_points=False,
            width=820,
            height=500,
            group_order=NETWORK_ORDER,
            hollow=True,
        )
        depthwise = d.depthwise_macs_per_cycle * d.cores
        fig.hline(
            depthwise,
            f"depthwise datapath {depthwise:,} MAC/cycle",
            color=svg.INK_SECONDARY,
            dash="2 3",
            width=1,
            text_anchor="start",
        )
        fig.x.label = f"arithmetic intensity (MAC/B of {point.memory} traffic)"
        fig.legend_position = "right"
        out.figures[f"s2-roofline-{point.name}"] = fig
    out.raw["points"] = raw
    return out


# --------------------------------------------------------------------- AXI outstanding --

GEMV = {
    "m": 1,
    "n": 4096,
    "k": 4096,
    "weight_bits": 4,
    "memory_model": "axi",
    "macs_per_cycle": 512,
    "axi_data_bytes": 16,
    "axi_burst_beats": 16,
    "timing_only": True,
}
S3_LATENCIES = (100, 400)
S3_OUTSTANDING = tuple(range(1, 33))


def s3_outstanding(ctx: Context) -> StudyOutput:
    runs = [
        {
            **GEMV,
            "axi_read_latency": latency,
            "axi_write_latency": latency,
            "axi_read_outstanding": n,
        }
        for latency in S3_LATENCIES
        for n in S3_OUTSTANDING
    ]
    results = ctx.run_all(runs)
    bus = GEMV["axi_data_bytes"]
    beats = GEMV["axi_burst_beats"]
    burst = bus * beats
    out = StudyOutput()
    fig = svg.Figure(
        820,
        480,
        title="Read bandwidth against outstanding AXI reads",
        subtitle="INT4 GEMV 1x4096x4096 streaming weights; 16 B beats, 16-beat "
        "bursts, DRAM unlimited",
        x_label="read transactions outstanding (N)",
        y_label="read bandwidth (B/cycle)",
        x_range=(0, 33),
        y_range=(0, 17),
        x_ticks=[1, 4, 8, 12, 16, 20, 24, 28, 32],
        legend="lower right",
        id_prefix="s3",
    )
    rows: dict[int, list] = {n: [n] for n in (1, 2, 4, 8, 16, 24, 32)}
    raw = []
    for index, latency in enumerate(S3_LATENCIES):
        color = svg.PALETTE[index]
        chunk = results[index * len(S3_OUTSTANDING) : (index + 1) * len(S3_OUTSTANDING)]
        measured, formula = [], []
        for n, result in zip(S3_OUTSTANDING, chunk, strict=True):
            check_result(result, f"S3 N={n} latency {latency}")
            rate = result["memory"]["read"]["payload_bytes"] / result["cycles"]
            ideal = min(bus, n * burst / (latency + beats))
            require(rate <= bus, f"S3: {rate} B/cycle above the {bus} B bus")
            measured.append(rate)
            formula.append(ideal)
            model = analytic.analyze(analytic.Design.from_result(result))
            raw.append(
                {
                    "latency": latency,
                    "outstanding": n,
                    "cycles": result["cycles"],
                    "read_bytes_per_cycle": rate,
                    "steady_state": ideal,
                    "closed_form_cycles": model.estimate_cycles,
                }
            )
            if n in rows:
                rows[n] += [f"{rate:.2f}", f"{ideal:.2f}"]
        fig.line(
            S3_OUTSTANDING,
            formula,
            color=color,
            width=1.5,
            dash="6 4",
            label=f"min(D, N·b·D / (L + b)), L = {latency}",
        )
        fig.scatter(
            S3_OUTSTANDING,
            measured,
            color=color,
            marker=svg.MARKERS[index],
            label=f"npu-sim, read latency L = {latency}",
        )
    fig.hline(bus, f"bus {bus} B/cycle", color=svg.INK_MUTED)
    out.figures["s3-outstanding"] = fig
    headers = ["$N_r$"]
    for latency in S3_LATENCIES:
        headers += [
            f"`npu-sim`, $L_r$ = {latency} (B/cycle)",
            f"Formula, $L_r$ = {latency} (B/cycle)",
        ]
    out.tables["s3-outstanding"] = markdown_table(headers, list(rows.values()))
    out.raw["runs"] = raw
    return out


# ----------------------------------------------------------------------- multi-core --

S4_CORES = (1, 2, 4, 8, 12, 16, 24, 32)
S4_RATES = {"client-16c": 100, "workstation-32c": 667}


def compulsory_bytes(gemm: workloads.Gemm, weight_bits: int) -> int:
    """Bytes any schedule moves: every input, weight, bias and output once. The input is
    the activation operand, or a convolution's NHWC input tensor."""
    g = gemm.groups
    fields = dict(zip(analytic.CONV_FIELDS, gemm.conv, strict=False))
    inputs = (
        fields["conv_input_h"] * fields["conv_input_w"] * fields["conv_input_c"]
        if gemm.conv
        else gemm.m * g * gemm.k
    )
    return (
        inputs + -(-g * gemm.k * gemm.n * weight_bits // 8) + 4 * g * gemm.n + gemm.m * g * gemm.n
    )


def s4_layers() -> list[tuple[str, workloads.Gemm, int]]:
    """(label, GEMM, weight bits): ResNet-50 res3 3x3 and the Llama decode gate projection."""
    resnet = {
        layer["name"]: workloads.lower(layer)
        for layer in workloads.load("resnet50_v1.5_224")["layers"]
    }
    decode = {layer["name"]: workloads.lower(layer) for layer in workloads.load(DECODE)["layers"]}
    return [
        ("ResNet-50 res3 3x3 conv (784x128x1152)", resnet["layer2.1-3.conv2"], 8),
        ("Llama-3.2-1B gate_proj GEMV INT4 (1x8192x2048)", decode["gate_proj"], 4),
    ]


def coincident(curves: Sequence[Sequence[float]]) -> list[list[int]]:
    """Indexes of ``curves`` grouped by equal values (to 1e-9), in order of first appearance."""
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, values in enumerate(curves):
        groups.setdefault(tuple(round(value, 9) for value in values), []).append(index)
    return list(groups.values())


def s4_scaling(ctx: Context) -> StudyOutput:
    point = design_points(["client-16c"])[0]
    d = point.design
    core = (
        f"{d.array_rows}x{d.array_cols} {d.dataflow.upper()}, {8 * d.axi_data_bytes}-bit "
        f"AXI, {d.sram_bytes / 1024:g} KiB SRAM"
    )
    out = StudyOutput()
    rows, raw = [], []
    for slug, (label, gemm, bits) in zip(("conv", "gemv"), s4_layers(), strict=True):
        # The mapper picks the tiles of every configuration, so a partition is not held to
        # the unit count of another configuration's tiles.
        runs = ctx.layer_flags(
            [
                Task.of(
                    point, gemm, bits, cores=cores, partition=partition, dram_bytes_per_cycle=rate
                )
                for rate in S4_RATES.values()
                for partition in ("m", "n")
                for cores in S4_CORES
            ]
        )
        outcomes = ctx.run_all(runs)
        results = iter(zip(outcomes, runs, strict=True))
        floor = compulsory_bytes(gemm, bits)
        # Every series starts from the same one-core run; the y axis reaches the highest
        # ceiling so each ceiling line is drawn.
        top = max(outcomes[0]["cycles"] / math.ceil(floor / rate) for rate in S4_RATES.values())
        top = max(64, 2 ** math.ceil(math.log2(top * 1.15)))
        ticks = [2**e for e in range(int(math.log2(top)) + 1)]
        fig = svg.Figure(
            820,
            500,
            title=f"Cluster speedup: {label}",
            subtitle=f"cores of the {point.name} point ({core}); tiles from "
            "npuperf.mapper per configuration",
            x_label="cores",
            y_label="speedup over one core",
            x_scale="log",
            y_scale="log",
            x_range=(1, 32),
            y_range=(0.8, top),
            x_ticks=list(S4_CORES),
            y_ticks=ticks,
            legend="right",
            id_prefix=f"s4-{slug}",
        )
        fig.x.tick_format = lambda v: f"{v:g}"
        fig.y.tick_format = lambda v: f"{v:g}"
        fig.line(S4_CORES, S4_CORES, color=svg.INK_MUTED, width=1.25, label="linear speedup")
        curves: list[tuple[str, int, int, list[float]]] = []  # partition, rate, units, speedups
        for rate in S4_RATES.values():
            for partition in ("m", "n"):
                chunk = [next(results) for _ in S4_CORES]
                one = chunk[0][0]["cycles"]
                # No schedule moves fewer bytes than the compulsory traffic, so no cluster
                # beats one-core cycles / (compulsory bytes / DRAM rate).
                ceiling = one / math.ceil(floor / rate)
                speedups = []
                for cores, (result, flags) in zip(S4_CORES, chunk, strict=True):
                    check_result(result, f"S4 {slug} {partition} P={cores} R={rate}")
                    speedup = one / result["cycles"]
                    require(
                        speedup <= cores * (1 + 1e-9),
                        f"S4 {slug}: speedup {speedup} above {cores} cores",
                    )
                    require(
                        speedup <= ceiling * (1 + 1e-9),
                        f"S4 {slug}: speedup {speedup} above the DRAM ceiling {ceiling}",
                    )
                    speedups.append(speedup)
                    raw.append(
                        {
                            "layer": slug,
                            "dram_bytes_per_cycle": rate,
                            "partition": partition,
                            "cores": cores,
                            "tile": [flags["tile_m"], flags["tile_n"], flags["tile_k"]],
                            "cycles": result["cycles"],
                            "speedup": speedup,
                            "dram_ceiling": ceiling,
                            "dram_bytes": dram_bytes(result),
                            "units": result["cluster"]["units"],
                            "imbalance": result["cluster"]["imbalance"],
                        }
                    )
                units = max(result["cluster"]["units"] for result, _ in chunk[1:])
                curves.append((partition, rate, units, speedups))
                if partition == "m":
                    fig.hline(
                        ceiling,
                        f"DRAM ceiling at {rate} B/cycle: {ceiling:.1f}",
                        color=svg.INK_SECONDARY,
                        dash="2 3",
                        width=1,
                    )
                rows.append(
                    [label.split(" (")[0], partition, f"{rate}"]
                    + [
                        f"{s:.2f}"
                        for s, c in zip(speedups, S4_CORES, strict=True)
                        if c in (2, 4, 8, 16, 32)
                    ]
                    + [f"{ceiling:.2f}"]
                )
        # Curves that coincide (the m split of a one-row GEMV leaves one busy core at every
        # DRAM rate) are drawn once under a joint label.
        for members in coincident([speedups for *_, speedups in curves]):
            first = members[0]
            _, rate, _, speedups = curves[first]
            partitions = "/".join(sorted({curves[i][0] for i in members}))
            rates = " and ".join(str(curves[i][1]) for i in members)
            dash = None if rate == S4_RATES["client-16c"] else "7 4"
            fig.line(
                S4_CORES,
                speedups,
                color=svg.PALETTE[first],
                dash=dash,
                marker=svg.MARKERS[first],
                label=f"partition {partitions}, {rates} B/cycle",
            )
            if all(curves[i][2] == 1 for i in members):
                fig.annotate(
                    S4_CORES[1],
                    speedups[1],
                    f"partition {partitions}: one tile band, one busy core",
                    dx=4,
                    dy=-10,
                    color=svg.PALETTE[first],
                )
        out.figures[f"s4-scaling-{slug}"] = fig
    out.tables["s4-scaling"] = markdown_table(
        [
            "Layer",
            "Partition",
            "DRAM B/cycle",
            "2 cores",
            "4 cores",
            "8 cores",
            "16 cores",
            "32 cores",
            "DRAM ceiling",
        ],
        rows,
        [False, False, True] + [True] * 6,
    )
    out.raw["runs"] = raw
    return out


# ------------------------------------------------------------------------- dataflow --


def layer_type(layer: Mapping[str, Any]) -> str:
    if layer["op"] == "depthwise_conv2d":
        return "depthwise"
    if layer["op"] in ("dense", "matmul"):
        return "FC"
    return f"{layer['kernel_h']}x{layer['kernel_w']}" if layer["kernel_h"] > 1 else "pointwise"


TYPE_ORDER = ("pointwise", "3x3", "7x7", "FC")


def s5_dataflow(ctx: Context) -> StudyOutput:
    point = design_points(["client-16c"])[0]
    calls = []
    for name in ("mobilenet_v2_1.0_224", "resnet50_v1.5_224"):
        for layer, gemm in workloads.iter_calls(workloads.load(name)):
            # Depthwise layers run on the depthwise datapath, which has no dataflow.
            if layer["op"] != "depthwise_conv2d":
                calls.append((NETWORKS[name], layer, gemm))
    runs = ctx.layer_flags(
        [
            Task.of(point, gemm, cores=1, dataflow=dataflow)
            for dataflow in ("os", "ws")
            for _, _, gemm in calls
        ]
    )
    results = ctx.run_all(runs)
    os_results, ws_results = results[: len(calls)], results[len(calls) :]
    array = point.design.array_rows * point.design.array_cols
    out = StudyOutput()
    by_type: dict[str, list] = {kind: [] for kind in TYPE_ORDER}
    raw = []
    for (network, layer, gemm), os_r, ws_r in zip(calls, os_results, ws_results, strict=True):
        values = []
        for result in (os_r, ws_r):
            check_result(result, f"S5 {layer['name']}")
            array_util = result["macs"] / (result["compute_busy_cycles"] * array)
            overall = result["macs"] / (result["cycles"] * array)
            require(array_util <= 1 and overall <= 1, f"S5 {layer['name']}: utilization")
            values.append((array_util, overall))
        kind = layer_type(layer)
        by_type[kind].append((gemm, values))
        raw.append(
            {
                "network": network,
                "layer": layer["name"],
                "type": kind,
                "calls": gemm.calls,
                "m": gemm.m,
                "n": gemm.n,
                "k": gemm.k,
                "groups": gemm.groups,
                "os": values[0],
                "ws": values[1],
            }
        )
    # Linear axes: the dense layers sit between 0.4 and 1, where the OS-WS differences of
    # the table are; the FC layers collapse near the origin.
    fig = svg.Figure(
        820,
        600,
        title="Array utilization, output-stationary against weight-stationary",
        subtitle="32x32 array, one core of client-16c; MACs over compute-busy cycles "
        "× 1,024; one mark per distinct layer shape",
        x_label="output-stationary array utilization (%)",
        y_label="weight-stationary array utilization (%)",
        x_range=(0, 1),
        y_range=(0, 1),
        legend="upper left",
        id_prefix="s5",
    )
    fig.x.tick_format = fig.y.tick_format = percent_tick
    fig.line([0, 1], [0, 1], color=svg.INK_MUTED, width=1, dash="4 4")
    fig.annotate(0.3, 0.3, "equal", dx=8, dy=14, color=svg.INK_MUTED)
    rows = []
    for index, kind in enumerate(TYPE_ORDER):
        members = by_type[kind]
        if not members:
            continue
        # Layers count every call; workload entries that repeat a shape plot once.
        layers = sum(gemm.calls for gemm, _ in members)
        shapes = sorted({(gemm.m, gemm.n, gemm.k, gemm.groups): v for gemm, v in members}.items())
        fig.scatter(
            [v[0][0] for _, v in shapes],
            [v[1][0] for _, v in shapes],
            color=svg.PALETTE[index],
            marker=svg.MARKERS[index],
            label=f"{kind} ({plural(layers, 'layer')}, {plural(len(shapes), 'shape')})",
            hollow=True,
        )
        # MAC-weighted means of (OS, WS) x (array, end-to-end) utilization.
        weights = [g.total_macs for g, _ in members]
        means = [
            pct(
                sum(w * v[flow][measure] for w, (_, v) in zip(weights, members, strict=True))
                / sum(weights)
            )
            for measure in (0, 1)
            for flow in (0, 1)
        ]
        rows.append([kind, str(layers), str(len(shapes)), fmt(sum(weights) / 1e6), *means])
    out.figures["s5-dataflow"] = fig
    out.tables["s5-dataflow"] = markdown_table(
        [
            "Layer type",
            "Layers",
            "Distinct shapes",
            "MACs (M)",
            "OS array utilization",
            "WS array utilization",
            "OS end-to-end",
            "WS end-to-end",
        ],
        rows,
    )
    out.raw["layers"] = raw
    return out


# ------------------------------------------------------------------ model vs SystemC --

S6_SEED = 20261107
S6_COUNT = {"serial": 120, "axi": 120, "conv": 80}


def s6_configs(seed: int = S6_SEED) -> list[dict[str, Any]]:
    """Seeded random dense configurations that fit their SRAM; the AXI ones include
    clusters of 2 to 8 cores, and the conv ones implicit convolutions, the depthwise
    datapath and the activation cache. Draw order is fixed, so the set is the same
    everywhere."""
    rng = random.Random(seed)
    configs = []
    for model, count in S6_COUNT.items():
        made = 0
        while made < count:
            flags = _random_config(rng, model)
            design = analytic.Design.from_flags(flags)
            if analytic.scratchpad_bytes(design) > design.sram_bytes:
                continue
            configs.append(flags)
            made += 1
    return configs


def _random_conv(rng: random.Random) -> dict[str, Any]:
    """An AXI convolution on its input footprint: a third depthwise on the depthwise
    datapath, half with an activation cache."""
    depthwise = rng.random() < 1 / 3
    while True:
        size, kernel = rng.choice([7, 14, 28, 56]), rng.choice([1, 3, 3, 5])
        stride, pad = rng.choice([1, 1, 2]), kernel // 2
        groups = rng.choice([16, 32, 96]) if depthwise else rng.choice([1, 1, 1, 2])
        channels = groups if depthwise else groups * rng.choice([3, 8, 16, 32, 64])
        out = (size + 2 * pad - kernel) // stride + 1
        if out >= 1:
            break
    row_k = kernel * channels // groups
    flags: dict[str, Any] = {
        "m": out * out,
        "n": 1 if depthwise else rng.choice([8, 32, 64]),
        "k": kernel * row_k,
        "groups": groups,
        "conv_input_h": size,
        "conv_input_w": size,
        "conv_input_c": channels,
        "conv_kernel_h": kernel,
        "conv_kernel_w": kernel,
        "conv_stride_h": stride,
        "conv_stride_w": stride,
        "conv_pad_top": pad,
        "conv_pad_left": pad,
        "conv_pad_bottom": pad,
        "conv_pad_right": pad,
        "tile_m": rng.choice([16, 32, 64, out, 2 * out]),
        "tile_n": rng.choice([8, 16, 32, 64]),
        "tile_k": rng.randrange(kernel) * row_k,
        "buffer_slots": rng.choice([1, 2, 2]),
        "panel_buffers": rng.choice([1, 2, 2]),
        "pipeline_cycles": 4,
        "memory_model": "axi",
        "timing_only": True,
        "array_rows": 16,
        "array_cols": 16,
        "axi_data_bytes": rng.choice([8, 32]),
        "axi_burst_beats": rng.choice([8, 16, 64]),
        "axi_read_outstanding": rng.choice([4, 12, 64]),
        "axi_write_outstanding": 4,
        "axi_read_latency": rng.choice([32, 500]),
        "axi_write_latency": rng.choice([32, 250]),
        "dram_bytes_per_cycle": rng.choice([4, 24, 100]),
        "sram_bytes": 1 << 20,
    }
    if depthwise:
        flags["depthwise_macs_per_cycle"] = rng.choice([16, 64, 256])
    if rng.random() < 0.5:
        flags["activation_cache_bytes"] = rng.choice([4096, 32768, 131072])
    if rng.random() < 0.3:
        flags["cores"] = rng.choice([2, 4])
        flags["partition"] = rng.choice(["m", "n", "auto"])
    return flags


def _random_config(rng: random.Random, model: str) -> dict[str, Any]:
    if model == "conv":
        return _random_conv(rng)
    flags: dict[str, Any] = {
        "m": rng.choice([1, 7, 16, 33, 64, 128, 196, 300]),
        "n": rng.choice([1, 8, 24, 64, 100, 256]),
        "k": rng.choice([9, 32, 75, 128, 300, 576]),
        "groups": rng.choice([1, 1, 1, 1, 2, 4]),
        "tile_m": rng.choice([4, 8, 16, 32, 64]),
        "tile_n": rng.choice([4, 8, 16, 32, 64]),
        "tile_k": rng.choice([0, 0, 16, 32, 64, 128]),
        "buffer_slots": rng.choice([1, 2, 2, 3]),
        "panel_buffers": rng.choice([1, 2, 2]),
        "weight_bits": rng.choice([8, 8, 4]),
        "pipeline_cycles": rng.choice([0, 4, 8]),
        "memory_model": model,
        "timing_only": True,
    }
    if rng.random() < 0.4:
        flags["macs_per_cycle"] = rng.choice([32, 64, 256, 1024])
    else:
        flags["array_rows"], flags["array_cols"] = rng.choice(
            [(8, 8), (16, 16), (32, 32), (16, 32)]
        )
        flags["dataflow"] = rng.choice(["os", "ws"])
    if rng.random() < 0.25:
        flags["weight_cache_bytes"] = rng.choice([1024, 8192, 32768])
    if model == "serial":
        flags["memory_bytes_per_cycle"] = rng.choice([4, 8, 16, 32, 64])
        flags["memory_latency"] = rng.choice([0, 12, 40, 100])
        flags["dma_burst_bytes"] = rng.choice([0, 0, 64, 256])
    else:
        flags["axi_data_bytes"] = rng.choice([8, 16, 32])
        flags["axi_burst_beats"] = rng.choice([4, 8, 16, 64])
        flags["axi_read_outstanding"] = rng.choice([1, 2, 4, 8, 16])
        flags["axi_write_outstanding"] = rng.choice([1, 2, 4])
        flags["axi_read_latency"] = rng.choice([12, 50, 100, 300])
        flags["axi_write_latency"] = rng.choice([12, 50, 100])
        flags["dram_bytes_per_cycle"] = rng.choice([0, 8, 16, 32, 64])
        if rng.random() < 0.35:
            flags["cores"] = rng.choice([2, 3, 4, 8])
            flags["partition"] = rng.choice(["m", "n", "auto"])
    return flags


def s6_entry(flags: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """analyze() and schedule() of one npu-sim run, against its result."""
    design = analytic.Design.from_result(result)
    model = analytic.analyze(design)
    replay = analytic.schedule(design)
    cycles = result["cycles"]
    require(
        model.lower_bound_cycles <= cycles,
        f"S6: lower bound {model.lower_bound_cycles} above {cycles} for {flags}",
    )
    traffic = (model.read_bytes, model.write_bytes, model.compute_busy_cycles) == (
        result["read_bytes"],
        result["write_bytes"],
        result["compute_busy_cycles"],
    )
    return {
        "flags": dict(flags),
        "cycles": cycles,
        "estimate": model.estimate_cycles,
        "lower_bound": model.lower_bound_cycles,
        "schedule": replay.cycles,
        "bound": model.bound,
        "observed_bound": analytic.observed_bound(result),
        "traffic_exact": traffic,
    }


def s6_design_point_runs(ctx: Context) -> dict[str, list[tuple[dict, dict]]]:
    """Point name -> the distinct (flags, result) runs of its S1 network and LLM layers,
    each with the tiling npuperf.mapper picked by its analyze() estimate."""
    runs = run_networks(ctx, design_points())
    out: dict[str, list[tuple[dict, dict]]] = {}
    seen: set[str] = set()
    for (point, _, _), (calls, results) in runs.items():
        for call, result in zip(calls, results, strict=True):
            key = Context.key(call.flags)
            if key not in seen:
                seen.add(key)
                out.setdefault(point, []).append((call.flags, result))
    return out


def _s6_entry(pair: tuple[Mapping[str, Any], Mapping[str, Any]]) -> dict[str, Any]:
    return s6_entry(*pair)


def s6_entries(ctx: Context, pairs: Sequence[tuple[dict, dict]]) -> list[dict[str, Any]]:
    """s6_entry of every (flags, result), up to ``jobs`` at a time in their own processes."""
    if ctx.jobs > 1 and len(pairs) > 1:
        with concurrent.futures.ProcessPoolExecutor(min(ctx.jobs, len(pairs))) as pool:
            return list(pool.map(_s6_entry, pairs))
    return [s6_entry(*pair) for pair in pairs]


def s6_analytic(ctx: Context) -> StudyOutput:
    configs = s6_configs()
    results = ctx.run_all(configs)
    out = StudyOutput()
    raw = []
    groups: dict[str, list[dict]] = {
        "random serial": [],
        "random AXI": [],
        "random AXI cluster": [],
        "random conv": [],
    }
    for flags, result in zip(configs, results, strict=True):
        check_result(result, f"S6 {flags}")
    for flags, entry in zip(
        configs, s6_entries(ctx, list(zip(configs, results, strict=True))), strict=True
    ):
        raw.append(entry)
        memory = "AXI" if flags["memory_model"] == "axi" else flags["memory_model"]
        kind = "random " + memory + (" cluster" if flags.get("cores", 1) > 1 else "")
        if "conv_input_h" in flags:
            kind = "random conv"
        groups[kind].append(entry)
    design_runs = s6_design_point_runs(ctx)
    pairs = [pair for runs in design_runs.values() for pair in runs]
    entries = iter(s6_entries(ctx, pairs))
    for point, runs in design_runs.items():
        kind = f"`{point}` network layers"
        for _ in runs:
            entry = next(entries)
            raw.append({"point": point, **entry})
            groups.setdefault(kind, []).append(entry)
    rows = []
    # The ratio on a linear axis: on log-log axes spanning six decades a 30% error is a
    # few pixels off the diagonal.
    lo = 10 ** math.floor(math.log10(min(e["cycles"] for e in raw)))
    hi = 10 ** math.ceil(math.log10(max(e["cycles"] for e in raw)))
    ratios = [e["estimate"] / e["cycles"] for e in raw]
    conv_count = len(groups["random conv"])
    dense_count = sum(
        len(groups[kind]) for kind in groups if kind.startswith("random") and kind != "random conv"
    )
    fig = svg.Figure(
        820,
        600,
        title="Analytic-model cycle estimate against npu-sim",
        subtitle=f"{dense_count} random dense and {conv_count} random conv "
        f"configurations (seed {S6_SEED}) and the distinct S1 layer runs "
        "of each design point; 1 = exact",
        x_label="npu-sim cycles",
        y_label="estimated cycles / simulated cycles",
        x_scale="log",
        x_range=(lo, hi),
        y_range=(math.floor(min(ratios) * 10) / 10, math.ceil(max(ratios) * 10) / 10),
        legend="right",
        id_prefix="s6",
    )
    fig.hline(1.0, color=svg.INK_MUTED, dash="4 4", width=1)
    for index, (kind, members) in enumerate(groups.items()):
        if not members:
            continue
        errors = sorted(abs(e["estimate"] - e["cycles"]) / e["cycles"] for e in members)
        low_count = sum(e["estimate"] < e["cycles"] for e in members)
        agree = sum(e["bound"] == e["observed_bound"] for e in members)
        exact = sum(e["schedule"] == e["cycles"] for e in members)
        traffic = sum(e["traffic_exact"] for e in members)
        count = len(members)
        rows.append(
            [
                kind,
                str(count),
                pct(statistics.median(errors)),
                pct(_percentile(errors, 0.95)),
                pct(errors[-1]),
                f"{low_count}/{count}",
                f"{agree}/{count}",
                f"{exact}/{count}",
                f"{traffic}/{count}",
            ]
        )
        fig.scatter(
            [e["cycles"] for e in members],
            [e["estimate"] / e["cycles"] for e in members],
            color=svg.PALETTE[index],
            marker=svg.MARKERS[index],
            label=f"{kind.replace('`', '')} ({count})",
            hollow=True,
            size=4,
        )
    out.figures["s6-analytic"] = fig
    out.tables["s6-analytic"] = markdown_table(
        [
            "Configurations",
            "Count",
            "Median error",
            "p95 error",
            "Max error",
            "Estimate below npu-sim",
            "Bound class agrees",
            "schedule() exact",
            "Traffic and compute exact",
        ],
        rows,
    )
    out.raw["configs"] = raw
    return out


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an ascending list."""
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


# ------------------------------------------------------------------ compression --

S7_DECODER_RATES = (4, 8, 16, 32, 64, 128, 256)
PALETTE_BITS = 4
# A palette run is functional: its input file is the panel data. 18 MiB of host storage
# keeps every piece of the 128256-column lm_head near 16 M weights.
PALETTE_HOST_BYTES = 18 * 1024 * 1024


def palette_input(ctx: Context, gemm: workloads.Gemm) -> Path:
    """An NPU1 file whose weights cycle through all 16 4-bit LUT values along every row
    and column, so every panel's LUT is full (the encoded size analytic.weight_bytes
    assumes). Row r of the weights is the value sequence rotated by r."""
    path = ctx.work / "inputs" / f"palette{PALETTE_BITS}_{gemm.m}x{gemm.n}x{gemm.k}.npu"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(range(-(1 << (PALETTE_BITS - 1)), 1 << (PALETTE_BITS - 1)))
    period = len(values)
    tokens = [str(values[j % period]) for j in range(gemm.n + period)]
    rows = [" ".join(tokens[shift : shift + gemm.n]) for shift in range(period)]
    activations = " ".join(str((i * 37) % 255 - 127) for i in range(gemm.m * gemm.k))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"NPU1 {gemm.m} {gemm.n} {gemm.k}\n{activations}\n")
        for row in range(gemm.k):
            handle.write(rows[row % period])
            handle.write("\n")
        handle.write(" ".join(["0"] * gemm.n) + "\n")
    temporary.replace(path)
    return path


def _palette_mapping(item: tuple[Point, workloads.Gemm, int]) -> mapper.Mapping:
    point, piece, rate = item
    return mapper.map_gemm(
        point.design,
        piece.m,
        piece.n,
        piece.k,
        piece.groups,
        point.min_depth,
        weight_encoding="palette",
        palette_bits=PALETTE_BITS,
        weight_decode_values_per_cycle=rate,
    )


def decode_rate(
    point: Point,
    pieces: Sequence[workloads.Gemm],
    results: Sequence[Mapping[str, Any]],
    attention_cycles: int,
    label: str,
) -> tuple[int, float]:
    """(cycles, tokens/s) of one decode token: the attention matmuls plus every weight
    piece's calls."""
    cycles = attention_cycles
    for piece, result in zip(pieces, results, strict=True):
        check_result(result, f"S7 {point.name} {piece.name} {label}")
        cycles += piece.calls * result["cycles"]
    return cycles, 1e3 / (cycles / (point.frequency_mhz * 1e3))


def s7_compression(ctx: Context) -> StudyOutput:
    points = [point for point in design_points() if point.runs_llm]
    decode = workloads.load(DECODE)
    out = StudyOutput()
    rows, raw = [], []
    # Every weight GEMM is split to the functional host limit of the palette runs, and the
    # dense baselines run the same pieces, so per-call fill and drain costs compare like for
    # like (S1 runs the unsplit layers).
    pieces = [
        piece
        for layer in decode["layers"]
        if layer.get("b_operand") == "weights"
        for piece in workloads.split(
            workloads.lower(layer), timing_only=False, max_host=PALETTE_HOST_BYTES
        )
    ]
    curves = []
    items = [
        (point, piece, rate) for point in points for rate in S7_DECODER_RATES for piece in pieces
    ]
    mappings = dict(zip(items, ctx.parallel(_palette_mapping, items), strict=True))
    for point in points:
        # Attention matmuls read the KV cache, which stays INT8 in every column.
        attention = [
            c
            for c in network_calls(ctx, point, DECODE, "Llama decode", 8)
            if c.layer.get("b_operand") != "weights"
        ]
        attention_results = ctx.run_all([c.flags for c in attention])
        for call, result in zip(attention, attention_results, strict=True):
            check_result(result, f"S7 {point.name} {call.layer['name']}")
        attention_cycles = sum(
            c.gemm.calls * r["cycles"] for c, r in zip(attention, attention_results, strict=True)
        )
        dense = {}
        for bits in (8, 4):
            results = ctx.run_all(
                ctx.layer_flags([Task.of(point, piece, bits) for piece in pieces])
            )
            cycles, speed = decode_rate(point, pieces, results, attention_cycles, f"INT{bits}")
            dense[f"INT{bits}"] = speed
            raw.append(
                {
                    "point": point.name,
                    "weights": f"dense INT{bits}",
                    "cycles": cycles,
                    "tokens_per_s": speed,
                }
            )
        # The S1 mapping check on the same pieces: INT4 halves only the weight bytes.
        require(
            dense["INT4"] <= 2 * dense["INT8"],
            f"S7 {point.name}: INT4 speedup {dense['INT4'] / dense['INT8']:.3f}x above 2x",
        )
        palette = []
        for rate in S7_DECODER_RATES:
            runs = []
            for piece in pieces:
                mapping = mappings[(point, piece, rate)]
                # The input file's header gives M, N and K; only the group count is a flag.
                shape = {
                    key: value for key, value in piece.flags().items() if key not in ("m", "n", "k")
                }
                runs.append(
                    point.flags(
                        **shape,
                        **mapping.flags(),
                        input=str(palette_input(ctx, piece)),
                        weight_encoding="palette",
                        palette_bits=PALETTE_BITS,
                        weight_decode_values_per_cycle=rate,
                    )
                )
            results = ctx.run_all(runs)
            for piece, result in zip(pieces, results, strict=True):
                require(result["numerics"] == "bit-exact", f"S7 {piece.name}: not bit-exact")
            cycles, speed = decode_rate(
                point, pieces, results, attention_cycles, f"palette rate {rate}"
            )
            palette.append(speed)
            raw.append(
                {
                    "point": point.name,
                    "weights": "palette",
                    "decoder_values_per_cycle": rate,
                    "cycles": cycles,
                    "tokens_per_s": speed,
                }
            )
        curves.append((point, dense, palette))
        rows.append(
            [f"`{point.name}`", fmt(dense["INT8"]), fmt(dense["INT4"])] + [fmt(p) for p in palette]
        )
    # Relative to dense INT4, so one reference line serves every point.
    int8 = [dense["INT8"] / dense["INT4"] for _, dense, _ in curves]
    top = max(p / dense["INT4"] for _, dense, palette in curves for p in palette)
    fig = svg.Figure(
        820,
        500,
        title="Llama-3.2-1B decode: 4-bit palette against dense INT4 weights",
        subtitle=f"every weight GEMM in the same pieces for all three formats; "
        f"dense INT8 runs at {min(int8):.2f} to {max(int8):.2f} of dense "
        "INT4; KV-cache matmuls stay INT8",
        x_label="weight decoder throughput (values per cycle per core)",
        y_label="palette decode rate relative to dense INT4",
        x_scale="log",
        x_range=(4, 256),
        y_range=(0, max(1.2, math.ceil(top * 10) / 10)),
        x_ticks=list(S7_DECODER_RATES),
        legend="lower right",
        id_prefix="s7",
    )
    fig.x.tick_format = lambda v: f"{v:g}"
    fig.hline(1.0, "dense INT4, same pieces", color=svg.INK_MUTED, text_anchor="start")
    for index, (point, dense, palette) in enumerate(curves):
        fig.line(
            S7_DECODER_RATES,
            [p / dense["INT4"] for p in palette],
            color=svg.PALETTE[index],
            marker=svg.MARKERS[index],
            label=f"{point.name} palette",
        )
    out.figures["s7-compression"] = fig
    out.tables["s7-compression"] = markdown_table(
        ["Design point", "Dense INT8", "Dense INT4"]
        + [f"Palette, {rate}/cycle" for rate in S7_DECODER_RATES],
        rows,
    )
    out.raw["runs"] = raw
    return out


# ------------------------------------------------------------------- mapping search --

S8_POINT = "mobile-2k"
S8_SRAM = 1 << 20
# tile_k is 0 or whole kernel rows of the 3x3 convolution: 3 x 128 channels = 384.
S8_GRID = ((32, 64, 128, 256), (64, 128), (0, 384, 768), (1, 2))


def s8_pareto(ctx: Context) -> StudyOutput:
    point = design_points([S8_POINT])[0]
    layer = next(
        layer
        for layer in workloads.load("resnet50_v1.5_224")["layers"]
        if layer["name"] == "layer2.1-3.conv2"
    )
    gemm = workloads.lower(layer)
    # The mapper under the point's own SRAM: without the weight cache, the condition of the
    # search, and with the free SRAM as weight cache, as the network studies run it.
    plain = mapper.map_call(point.design, gemm, point.min_depth, cache=False)
    cached = mapper.map_call(point.design, gemm, point.min_depth)
    require(
        plain.conv and cached.conv and plain.panel_buffers == point.design.panel_buffers,
        "S8: the mapper left the grid's lowering",
    )
    pick = (plain.tile_m, plain.tile_n, plain.tile_k, plain.buffer_slots)
    candidates = sorted(
        {
            (m, n, k, s)
            for m in S8_GRID[0]
            for n in S8_GRID[1]
            for k in S8_GRID[2]
            for s in S8_GRID[3]
        }
        | {pick}
    )
    base = point.flags(
        **gemm.flags(), sram_bytes=S8_SRAM, weight_cache_bytes=0, activation_cache_bytes=0
    )
    report, rows = autotune.search(
        ctx.binary, [gemm.m, gemm.n, gemm.k], candidates, base, timeout=ctx.timeout
    )
    ctx.extra_runs += sum(1 for row in rows if row["feasible"])
    feasible = [row for row in rows if row["feasible"]]
    hashes = {row["output_hash"] for row in feasible}
    require(len(hashes) == 1, "S8: mappings disagree on the output")
    with_cache = ctx.run_all([point.flags(**gemm.flags(), **cached.flags())])[0]
    check_result(with_cache, "S8 mapper with weight cache")
    pareto = report["pareto"]
    mine = next(
        row
        for row in feasible
        if (row["tile_m"], row["tile_n"], row["tile_k"], row["buffer_slots"]) == pick
    )
    choices = (
        f"{len(candidates)} tile and slot choices, all feasible and bit-exact"
        if len(feasible) == len(candidates)
        else f"{len(feasible)} feasible tile and slot choices of {len(candidates)}, all bit-exact"
    )
    within = [row for row in feasible if row["scratchpad_bytes"] <= point.design.sram_bytes]
    best = min(within, key=lambda row: (row["cycles"], row["scratchpad_bytes"]))
    out = StudyOutput()
    fig = svg.Figure(
        820,
        500,
        title="Mapping search: ResNet-50 res3 3x3 conv on mobile-2k",
        subtitle=f"784x128x1152 on its input footprint; {choices}; caches off unless noted",
        x_label="SRAM reserved (KiB)",
        y_label="cycles",
        y_include_zero=True,
        legend="upper right",
        id_prefix="s8",
    )
    fig.y.tick_format = lambda v: f"{v:,.0f}"
    fig.scatter(
        [r["scratchpad_bytes"] / 1024 for r in feasible],
        [r["cycles"] for r in feasible],
        color=svg.INK_MUTED,
        marker="circle",
        hollow=True,
        size=3.5,
        label="feasible mapping",
    )
    fig.step(
        [r["scratchpad_bytes"] / 1024 for r in pareto],
        [r["cycles"] for r in pareto],
        color=svg.PALETTE[0],
        label="Pareto front",
    )
    fig.scatter(
        [r["scratchpad_bytes"] / 1024 for r in pareto],
        [r["cycles"] for r in pareto],
        color=svg.PALETTE[0],
        marker="circle",
        size=4.5,
    )
    fig.scatter(
        [mine["scratchpad_bytes"] / 1024],
        [mine["cycles"]],
        color=svg.PALETTE[1],
        marker="diamond",
        size=7,
        label="npuperf.mapper, caches off",
    )
    fig.scatter(
        [with_cache["scratchpad_bytes"] / 1024],
        [with_cache["cycles"]],
        color=svg.PALETTE[2],
        marker="triangle",
        size=7,
        label="npuperf.mapper, free SRAM as caches",
    )
    fig.vline(
        point.design.sram_bytes / 1024,
        f"{point.name} SRAM {point.design.sram_bytes / 1024:g} KiB",
        color=svg.INK_SECONDARY,
    )
    out.figures["s8-pareto"] = fig

    def describe(row: Mapping[str, Any]) -> str:
        return f"{row['tile_m']}x{row['tile_n']}x{row['tile_k'] or gemm.k}"

    table_rows = [
        [
            describe(row),
            str(row["buffer_slots"]),
            kib(0),
            kib(0),
            kib(row["scratchpad_bytes"]),
            f"{row['cycles']:,}",
            "Pareto front",
        ]
        for row in pareto
    ]
    table_rows.append(
        [
            describe(best),
            str(best["buffer_slots"]),
            kib(0),
            kib(0),
            kib(best["scratchpad_bytes"]),
            f"{best['cycles']:,}",
            f"fastest within {point.design.sram_bytes / 1024:g} KiB",
        ]
    )
    table_rows.append(
        [
            describe(mine),
            str(mine["buffer_slots"]),
            kib(0),
            kib(0),
            kib(mine["scratchpad_bytes"]),
            f"{mine['cycles']:,}",
            "npuperf.mapper, caches off",
        ]
    )
    table_rows.append(
        [
            f"{cached.tile_m}x{cached.tile_n}x{cached.tile_k or gemm.k}",
            str(cached.buffer_slots),
            kib(cached.weight_cache_bytes),
            kib(cached.activation_cache_bytes),
            kib(with_cache["scratchpad_bytes"]),
            f"{with_cache['cycles']:,}",
            "npuperf.mapper, free SRAM as caches",
        ]
    )
    out.tables["s8-pareto"] = markdown_table(
        [
            "Tile (m x n x k)",
            "Slots",
            "Weight cache (KiB)",
            "Activation cache (KiB)",
            "SRAM (KiB)",
            "Cycles",
            "Row",
        ],
        table_rows,
        [False, True, True, True, True, True, False],
    )
    out.raw["rows"] = rows
    out.raw["mapper"] = {
        "cache_off": asdict(plain),
        "cache_on": asdict(cached),
        "cache_on_result": with_cache,
    }
    return out


# ---------------------------------------------------------------- depthwise datapath --

S9_NETWORK = "mobilenet_v2_1.0_224"
# The sizing rule: the smallest swept rate whose MobileNetV2 latency is within this fraction
# of the latency with a depthwise datapath as wide as the core's array.
S9_TOLERANCE = 0.05
S9_LOWEST = 1


def s9_sweep(point: Point) -> list[int]:
    """Powers of two from S9_LOWEST to the core's array MACs: the rates the rule picks from."""
    top = point.design.array_macs_per_cycle // point.design.cores
    rates, rate = [], S9_LOWEST
    while rate <= top:
        rates.append(rate)
        rate *= 2
    return rates


def s9_rates(point: Point) -> list[int]:
    """0 (no depthwise datapath), the sweep, and the configured rate when the sweep lacks it
    (an Arm-model rate, designs.vela_depthwise_rate)."""
    return sorted({0, point.design.depthwise_macs_per_cycle, *s9_sweep(point)})


def s9_pick(latency: Mapping[int, float]) -> int:
    """The S9 rule on {rate: MobileNetV2 latency} over the swept rates: the smallest rate
    within S9_TOLERANCE of the latency at the largest rate."""
    best = latency[max(latency)]
    return min(rate for rate, value in latency.items() if value <= (1 + S9_TOLERANCE) * best)


def s9_source(point: Point) -> tuple[int, str]:
    """(rate, source) the point's config must set: the Arm-model rate of the part it is sized
    from, or the S9 rule's pick where there is none (returned as 0 here)."""
    rate = designs.vela_depthwise_rate(point.name)
    if rate is None:
        return 0, "S9 rule"
    model = "Vela 5.2.0" if point.name == "edge-256" else "Regor (Vela 5.2.0)"
    return rate, f"{model} 3x3 rate"


def is_depthwise(call: Call) -> bool:
    return call.layer["op"] == "depthwise_conv2d"


def s9_depthwise(ctx: Context) -> StudyOutput:
    points = design_points()
    plans = {
        (point.name, rate): plan_calls(
            point,
            S9_NETWORK,
            "MobileNetV2",
            plain_depthwise=rate == 0,
            depthwise_macs_per_cycle=rate,
        )
        for point in points
        for rate in s9_rates(point)
    }
    planned = resolve(ctx, plans)
    results = iter(ctx.run_all([call.flags for calls in planned.values() for call in calls]))
    out = StudyOutput()
    raw, summary, curves, choice = [], [], [], []
    cells: dict[tuple[int, str], str] = {}
    for point in points:
        latency, share = {}, {}
        for rate in s9_rates(point):
            calls = planned[(point.name, rate)]
            chunk = [next(results) for _ in calls]
            for call, result in zip(calls, chunk, strict=True):
                check_result(result, f"S9 {point.name} rate {rate} {call.layer['name']}")
            whole = totals(calls, chunk)
            dw = totals(
                *zip(
                    *[pair for pair in zip(calls, chunk, strict=True) if is_depthwise(pair[0])],
                    strict=True,
                )
            )
            latency[rate] = whole.latency_ms(point.frequency_mhz)
            share[rate] = dw.cycles / whole.cycles
            cells[(rate, point.name)] = f"{fmt(latency[rate])} ({pct(share[rate], 0)})"
            raw.append(
                {
                    "point": point.name,
                    "depthwise_macs_per_cycle": rate,
                    "cycles": whole.cycles,
                    "depthwise_cycles": dw.cycles,
                    "latency_ms": latency[rate],
                    "depthwise_share": share[rate],
                    "dram_bytes": whole.dram_bytes,
                    "mac_utilisation": whole.macs
                    / (whole.cycles * point.design.array_macs_per_cycle),
                }
            )
        pick = s9_pick({rate: latency[rate] for rate in s9_sweep(point)})
        configured = point.design.depthwise_macs_per_cycle
        model_rate, source = s9_source(point)
        want = model_rate or pick
        require(
            configured == want,
            f"configs/{point.name}.json sets depthwise_macs_per_cycle {configured}; "
            f"the {source} is {want}",
        )
        top = max(latency)
        summary.append(
            [
                f"`{point.name}`",
                f"{configured:,}",
                source,
                f"{pick:,}",
                fmt(latency[configured]),
                fmt(latency[pick]),
                pct(share[configured]),
                f"{top:,}",
                fmt(latency[top]),
                fmt(latency[0]),
                f"{latency[0] / latency[configured]:.1f}x",
            ]
        )
        choice.append(
            {"point": point.name, "configured": configured, "source": source, "rule_pick": pick}
        )
        curves.append((point, latency, pick))
    rates = sorted({rate for point in points for rate in s9_rates(point)})
    rows = [
        ["none: grouped GEMM on the array, im2col" if rate == 0 else f"{rate:,}"]
        + [cells.get((rate, point.name), "–") for point in points]
        for rate in rates
    ]
    out.tables["s9-depthwise"] = markdown_table(
        ["Depthwise MAC/cycle per core"] + [f"`{point.name}`" for point in points], rows
    )
    out.tables["s9-choice"] = markdown_table(
        [
            "Design point",
            "Configured rate (MAC/cycle per core)",
            "Source",
            "Rule's rate",
            "Latency at the configured rate (ms)",
            "Latency at the rule's rate (ms)",
            "Depthwise share of cycles",
            "Array-wide rate",
            "Latency at the array-wide rate (ms)",
            "Latency without the datapath (ms)",
            "Speedup over no datapath",
        ],
        summary,
        [False, True, False] + [True] * 8,
    )
    top = max(
        latency[rate] / latency[max(latency)]
        for _, latency, _ in curves
        for rate in latency
        if rate
    )
    fig = svg.Figure(
        820,
        500,
        title="MobileNetV2 latency against depthwise datapath rate",
        subtitle="MobileNetV2 on every design point; tiles from npuperf.mapper at every rate",
        x_label="depthwise MAC/cycle per core",
        y_label="latency relative to the array-wide rate",
        x_scale="log",
        y_scale="log",
        x_range=(S9_LOWEST, max(rates)),
        y_range=(0.9, 2 ** math.ceil(math.log2(top))),
        x_ticks=[rate for rate in rates if rate and not rate & (rate - 1)],
        y_ticks=[2**e for e in range(math.ceil(math.log2(top)) + 1)],
        legend="upper right",
        id_prefix="s9",
    )
    fig.x.tick_format = lambda v: f"{v:,g}"
    fig.y.tick_format = lambda v: f"{v:g}"
    fig.hline(
        1 + S9_TOLERANCE,
        f"rule: within {pct(S9_TOLERANCE, 0)} of the array-wide rate",
        color=svg.INK_MUTED,
        dash="4 4",
        width=1,
        text_anchor="end",
    )
    for index, (point, latency, pick) in enumerate(curves):
        swept = s9_sweep(point)
        base = latency[max(latency)]
        configured = point.design.depthwise_macs_per_cycle
        label = f"{point.name} (rule picks {pick:,}"
        label += ")" if configured == pick else f"; configured {configured:,})"
        fig.line(
            swept,
            [latency[rate] / base for rate in swept],
            color=svg.PALETTE[index],
            marker=svg.MARKERS[index],
            label=label,
        )
        # Picks of several points can share a rate and nearly a latency: hollow rings that
        # shrink with the draw order keep every one visible.
        fig.scatter(
            [pick],
            [latency[pick] / base],
            color=svg.PALETTE[index],
            marker="circle",
            size=12 - 2 * index,
            hollow=True,
        )
        if configured != pick:
            fig.scatter(
                [configured],
                [latency[configured] / base],
                color=svg.PALETTE[index],
                marker="down",
                size=9,
                hollow=True,
            )
    for label, marker in (("S9 rule's pick", "circle"), ("configured Arm-model rate", "down")):
        fig.scatter([], [], label=label, color=svg.INK_SECONDARY, marker=marker, hollow=True)
    out.figures["s9-depthwise"] = fig
    out.raw["runs"] = raw
    out.raw["choice"] = choice
    return out


# -------------------------------------------------------------------- per-core SRAM --

S10_POINTS = ("client-16c", "workstation-32c")
S10_NETWORKS = {"resnet50_v1.5_224": "ResNet-50 v1.5", "mobilenet_v2_1.0_224": "MobileNetV2"}
S10_SIZES_KIB = (64, 128, 256, 512, 1024, 2048, 4096)
# The sizing rule: on each point, the smallest swept size at which every network reaches
# this fraction of its highest MAC utilization in the sweep; the two points share one core
# design, so both take the larger of their picks.
S10_SHARE = 0.9


def s10_pick(utilization: Mapping[int, float]) -> int:
    """The smallest size (KiB) whose utilization reaches S10_SHARE of the sweep's highest."""
    best = max(utilization.values())
    return min(size for size, value in utilization.items() if value >= S10_SHARE * best)


def s10_sram(ctx: Context) -> StudyOutput:
    points = design_points(S10_POINTS)
    plans = {
        (point.name, name, kib): plan_calls(point, name, network, sram_bytes=kib * 1024)
        for point in points
        for name, network in S10_NETWORKS.items()
        for kib in S10_SIZES_KIB
    }
    planned = resolve(ctx, plans)
    results = iter(ctx.run_all([call.flags for calls in planned.values() for call in calls]))
    out = StudyOutput()
    raw = []
    utilization: dict[tuple[str, str], dict[int, float]] = {}
    latency: dict[tuple[str, str], dict[int, float]] = {}
    for (name, network_file, kib), calls in planned.items():
        chunk = [next(results) for _ in calls]
        network = S10_NETWORKS[network_file]
        for call, result in zip(calls, chunk, strict=True):
            check_result(result, f"S10 {name} {kib} KiB {call.layer['name']}")
            require(result["scratchpad_bytes"] <= kib * 1024, f"S10 {name}: SRAM overrun")
        point = next(point for point in points if point.name == name)
        t = totals(calls, chunk)
        util = t.macs / (t.cycles * point.design.array_macs_per_cycle)
        utilization.setdefault((name, network), {})[kib] = util
        latency.setdefault((name, network), {})[kib] = t.latency_ms(point.frequency_mhz)
        raw.append(
            {
                "point": name,
                "network": network,
                "sram_kib": kib,
                "cycles": t.cycles,
                "macs": t.macs,
                "dram_bytes": t.dram_bytes,
                "mac_utilisation": util,
            }
        )
    picks = {
        point.name: max(
            s10_pick(utilization[(point.name, network)]) for network in S10_NETWORKS.values()
        )
        for point in points
    }
    chosen = max(picks.values())
    for point in points:
        require(
            point.design.sram_bytes == chosen * 1024,
            f"configs/{point.name}.json sets sram_bytes {point.design.sram_bytes}; the "
            f"S10 rule picks {chosen} KiB",
        )
    keys = [(point.name, network) for point in points for network in S10_NETWORKS.values()]
    rows = [
        [f"{kib:,}"]
        + [f"{pct(utilization[key][kib])} ({fmt(latency[key][kib])} ms)" for key in keys]
        for kib in S10_SIZES_KIB
    ]
    out.tables["s10-sram"] = markdown_table(
        ["SRAM per core (KiB)"] + [f"`{name}` {network}" for name, network in keys], rows
    )
    choice = [
        [f"`{point.name}`"]
        + [f"{s10_pick(utilization[(point.name, network)]):,}" for network in S10_NETWORKS.values()]
        + [f"{picks[point.name]:,}"]
        for point in points
    ]
    out.tables["s10-choice"] = markdown_table(
        ["Design point"]
        + [f"{network} pick (KiB)" for network in S10_NETWORKS.values()]
        + ["Point's pick (KiB)"],
        choice,
    )
    top = max(value for values in utilization.values() for value in values.values())
    fig = svg.Figure(
        820,
        500,
        title="MAC utilization against per-core SRAM",
        subtitle="cores of client-16c and workstation-32c; tiles and cache split "
        "from npuperf.mapper per size",
        x_label="SRAM per core (KiB)",
        y_label="MAC utilization (%)",
        x_scale="log",
        x_range=(S10_SIZES_KIB[0], S10_SIZES_KIB[-1]),
        y_range=(0, math.ceil(top * 10) / 10),
        x_ticks=list(S10_SIZES_KIB),
        legend="right",
        id_prefix="s10",
    )
    fig.x.tick_format = lambda v: f"{v:,g}"
    fig.y.tick_format = percent_tick
    fig.vline(chosen, f"rule picks {chosen:,} KiB", color=svg.INK_SECONDARY)
    for index, network in enumerate(S10_NETWORKS.values()):
        for point in points:
            values = utilization[(point.name, network)]
            fig.line(
                S10_SIZES_KIB,
                [values[kib] for kib in S10_SIZES_KIB],
                color=svg.PALETTE[index],
                dash=None if point.name == S10_POINTS[0] else "7 4",
                marker=svg.MARKERS[index],
                label=f"{network}, {point.name}",
            )
    out.figures["s10-sram"] = fig
    out.raw["runs"] = raw
    out.raw["picks"] = {"points": picks, "chosen_kib": chosen}
    return out


# --------------------------------------------------------------------------- registry --


@dataclass(frozen=True)
class Study:
    id: str
    title: str
    quick: bool
    run: Callable[[Context], StudyOutput]


STUDIES = {
    study.id: study
    for study in (
        Study("S1", "Network results per design point", False, s1_networks),
        Study("S2", "Roofline per design point", False, s2_roofline),
        Study("S3", "AXI outstanding transactions", True, s3_outstanding),
        Study("S4", "Multi-core scaling", True, s4_scaling),
        Study("S5", "Dataflow", True, s5_dataflow),
        Study("S6", "Analytic model against npu-sim", True, s6_analytic),
        Study("S7", "Weight compression for LLM decode", False, s7_compression),
        Study("S8", "Mapping search Pareto front", True, s8_pareto),
        Study("S9", "Depthwise datapath rate", False, s9_depthwise),
        Study("S10", "Per-core SRAM of the cluster core", False, s10_sram),
    )
}
