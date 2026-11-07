"""The generated studies of docs/results.md. Every number comes from an npu-sim run.

Each study returns markdown tables (keyed by their docs/results.md marker), SVG figures
(docs/figures/) and raw records (results/studies/<id>.json). ``python -m npuperf.reproduce``
runs them; see that module for the command line.

Conventions shared by the studies:
  - Design points are configs/*.json (hypothetical configurations sized from public data,
    docs/design-points.md). A layer runs on its point with the tiles npuperf.mapper picks.
  - Only the GEMM layers of a workload run; vector operators (softmax, normalisation,
    pooling, residual adds) are not modeled. A layer with ``calls`` identical instances
    runs once and counts ``calls`` times. Depthwise convolution is the im2col lowering of
    npuperf.workloads (G groups of N = 1, K = kh*kw): every activation-panel row is a
    kh*kw-byte run and every output row a 1-byte run per group.
  - A call runs bit-exact (tensors, numerics, reference) when it fits the functional limits
    and FUNCTIONAL_MACS; otherwise --timing-only. Dense timing does not depend on the data.
  - External-memory bytes are the bytes the DRAM timeline charges (system SRAM on
    edge-256): whole AXI beats, so a partial beat counts in full.
  - The LLM workloads run only on points whose external memory is DRAM (Point.runs_llm).
  - One MAC is two operations (TOP/s).
  - A result that breaks a physical limit (above a roof, utilisation above 1, speedup above
    the core count, bandwidth above the bus) stops the run with a ``SanityError``.
"""
from __future__ import annotations

import concurrent.futures
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

NETWORKS = {"mobilenet_v2_1.0_224": "MobileNetV2", "resnet50_v1.5_224": "ResNet-50 v1.5",
            "bert_base_encoder_layer_s128": "BERT-base layer"}
DECODE = "llama3.2_1b_decode_kv129"
PREFILL = "llama3.2_1b_prefill_s128"
NETWORK_ORDER = ("MobileNetV2", "ResNet-50 v1.5", "BERT-base layer", "Llama decode",
                 "Llama prefill")


class SanityError(RuntimeError):
    """A simulator result breaks a physical limit."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SanityError(message)


# ---------------------------------------------------------------------------- running --

@dataclass
class Context:
    """The simulator binary and a cache of results keyed by their flags, shared by every
    study of one reproduce run, so a run two studies need executes once."""

    binary: Path
    jobs: int = 1
    timeout: float = 900
    work: Path = RESULTS_DIR
    cache: dict[str, dict] = field(default_factory=dict)
    extra_runs: int = 0       # runs made outside run_all (the autotune search)

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
            for key, result in zip(pending, pool.map(one, pending.values())):
                self.cache[key] = result
        return [self.cache[self.key(flags)] for flags in runs]


@dataclass(frozen=True)
class Point:
    """A design point: its config file, the analytic.Design it describes and the memory
    behind its port ("DRAM" or "system SRAM")."""

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
        """A billion-parameter model's weights fit no microcontroller-class system SRAM,
        so the LLM workloads run only on points that stream from DRAM."""
        return self.memory == "DRAM"

    def flags(self, **extra: Any) -> dict[str, Any]:
        """--config first, so the flags after it override the file."""
        return {"config": str(self.path), **extra}


def design_points(names: Iterable[str] = designs.DESIGN_POINTS) -> list[Point]:
    points = []
    for name in names:
        document = designs.load(name)
        points.append(Point(name, designs.config_path(name), designs.design(document),
                            designs.min_panel_depth(name), designs.external_memory(name)))
    return points


def functional(gemm: workloads.Gemm, mapping: mapper.Mapping) -> bool:
    return gemm.macs <= FUNCTIONAL_MACS and workloads.fits(
        gemm, timing_only=False, tile_m=mapping.tile_m, tile_n=mapping.tile_n,
        tile_k=mapping.tile_k)


def layer_flags(point: Point, gemm: workloads.Gemm, weight_bits: int = 8,
                **design_changes: Any) -> dict[str, Any]:
    """npu-sim flags of one call of ``gemm`` on ``point`` with the mapper's tiles."""
    design = point.design.replace(weight_bits=weight_bits, **design_changes)
    mapping = mapper.map_gemm(design, gemm.m, gemm.n, gemm.k, gemm.groups, point.min_depth)
    flags = point.flags(**{key: value for key, value in design_changes.items()})
    flags.update(gemm.flags())
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
    """Physical limits of one run: utilisation, the compute and DRAM roofs, the bus."""
    cycles, macs = result["cycles"], result["macs"]
    peak = result["effective_macs_per_cycle"]
    require(macs <= peak * cycles, f"{label}: {macs} MACs exceed {peak} MAC/cycle x {cycles}")
    rate = result["peak_memory_bytes_per_cycle"]
    moved = dram_bytes(result)
    require(moved <= rate * cycles, f"{label}: {moved} DRAM bytes exceed {rate} B/cycle x "
                                    f"{cycles}")
    config = result["config"]
    if sim.config_value(config, "memory_model") == "axi":
        bus = sim.config_value(config, "axi_data_bytes")
        for core in result["cores"]:
            for direction in ("read", "write"):
                channel = core["memory"][direction]
                require(channel["beats"] * bus <= bus * cycles
                        and channel["payload_bytes"] <= bus * cycles,
                        f"{label}: core {core['core']} {direction} channel above its bus")


# ------------------------------------------------------------------------- formatting --

def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
                   numeric: Sequence[bool] | None = None) -> str:
    """A GitHub table; numeric columns align right."""
    numeric = numeric or [False] + [True] * (len(headers) - 1)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---:" if right else "---" for right in numeric) + "|"]
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


def network_calls(point: Point, workload_name: str, network: str,
                  weight_bits: int = 8) -> list[Call]:
    """Every call of a workload on ``point``. INT4 applies to weight operands only; an
    attention matmul's B operand is the KV cache and stays INT8."""
    workload = workloads.load(workload_name)
    calls = []
    for layer, gemm in workloads.iter_calls(workload):
        bits = weight_bits if layer.get("b_operand", "weights") == "weights" else 8
        calls.append(Call(network, layer, gemm, layer_flags(point, gemm, bits)))
    return calls


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
    for call, result in zip(calls, results):
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
    plan: dict[tuple[str, str, int], list[Call]] = {}
    for point in points:
        for name, network in NETWORKS.items():
            plan[(point.name, network, 8)] = network_calls(point, name, network)
        if not point.runs_llm:
            continue
        for bits in (8, 4):
            plan[(point.name, "Llama decode", bits)] = network_calls(point, DECODE,
                                                                    "Llama decode", bits)
            plan[(point.name, "Llama prefill", bits)] = network_calls(point, PREFILL,
                                                                     "Llama prefill", bits)
    flat = [call.flags for calls in plan.values() for call in calls]
    results = iter(ctx.run_all(flat))
    out = {}
    for key, calls in plan.items():
        chunk = [next(results) for _ in calls]
        for call, result in zip(calls, chunk):
            check_result(result, f"{key[0]} {call.network} {call.layer['name']}")
        out[key] = (calls, chunk)
    return out


def network_weight_bytes(name: str) -> int:
    """INT8 bytes of every weight operand of a workload, each once; an attention matmul's B
    operand is the KV cache, not a weight."""
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
        moved += call.gemm.calls * channel["beats"] * sim.config_value(result["config"],
                                                                       "axi_data_bytes")
    return payload / moved


def depthwise_row(point: Point, calls: Sequence[Call],
                  results: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """MobileNetV2 with and without its depthwise layers on ``point``."""
    pairs = list(zip(calls, results))
    depthwise = [pair for pair in pairs if pair[0].layer["op"] == "depthwise_conv2d"]
    rest = [pair for pair in pairs if pair[0].layer["op"] != "depthwise_conv2d"]
    whole, dw = totals(calls, results), totals(*zip(*depthwise))
    other = totals(*zip(*rest))
    util = other.macs / (other.cycles * point.design.peak_macs_per_cycle)
    record = {"point": point.name, "cycles": whole.cycles, "depthwise_cycles": dw.cycles,
              "dram_bytes": whole.dram_bytes, "depthwise_dram_bytes": dw.dram_bytes,
              "depthwise_read_efficiency": burst_efficiency(depthwise, "read"),
              "depthwise_write_efficiency": burst_efficiency(depthwise, "write"),
              "other_cycles": other.cycles, "other_macs": other.macs,
              "other_mac_utilisation": util}
    row = [f"`{point.name}`", fmt(whole.latency_ms(point.frequency_mhz)),
           pct(dw.cycles / whole.cycles), pct(dw.dram_bytes / whole.dram_bytes),
           pct(record["depthwise_read_efficiency"]), pct(record["depthwise_write_efficiency"]),
           fmt(other.latency_ms(point.frequency_mhz)), pct(util)]
    return row, record


def s1_networks(ctx: Context) -> StudyOutput:
    points = design_points()
    runs = run_networks(ctx, points)
    out = StudyOutput()
    rows, llm, depthwise, raw = [], [], [], []
    weights = {network: network_weight_bytes(name) for name, network in NETWORKS.items()}
    for point in points:
        peak = point.design.peak_macs_per_cycle
        for network in NETWORKS.values():
            calls, results = runs[(point.name, network, 8)]
            t = totals(calls, results)
            ms = t.latency_ms(point.frequency_mhz)
            util = t.macs / (t.cycles * peak)
            require(util <= 1, f"{point.name} {network}: utilisation {util}")
            memory = t.bound_calls["bandwidth"] + t.bound_calls["latency"]
            rows.append([f"`{point.name}`", network, point.memory,
                         fmt(weights[network] / 1e6), fmt(ms), fmt(1e3 / ms),
                         fmt(OPS_PER_MAC * t.macs / (ms * 1e9)), pct(util),
                         fmt(t.dram_bytes / 1e6), pct(t.bound_calls["compute"] / t.calls, 0),
                         pct(memory / t.calls, 0)])
            raw.append({"point": point.name, "network": network, "memory": point.memory,
                        "weight_bytes": weights[network], "cycles": t.cycles,
                        "macs": t.macs, "dram_bytes": t.dram_bytes, "latency_ms": ms,
                        "bound_calls": t.bound_calls, "calls": t.calls,
                        "bit_exact_calls": t.bit_exact_calls})
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
            require(gbps[bits] <= peak_gbps * (1 + 1e-9),
                    f"{point.name} decode INT{bits}: {gbps[bits]} GB/s above {peak_gbps}")
        llm.append([f"`{point.name}`", fmt(tps[8]), fmt(tps[4]), f"{tps[4] / tps[8]:.2f}x",
                    fmt(decode[8].dram_bytes / 1e6), pct(gbps[8] / peak_gbps),
                    fmt(prefill[8].latency_ms(point.frequency_mhz)),
                    fmt(prefill[4].latency_ms(point.frequency_mhz)),
                    pct(prefill[8].macs / (prefill[8].cycles * point.design.peak_macs_per_cycle))])
        for kind, table in (("decode", decode), ("prefill", prefill)):
            for bits, t in table.items():
                raw.append({"point": point.name, "network": f"Llama {kind}",
                            "weight_bits": bits, "cycles": t.cycles, "macs": t.macs,
                            "dram_bytes": t.dram_bytes,
                            "latency_ms": t.latency_ms(point.frequency_mhz),
                            "bound_calls": t.bound_calls, "calls": t.calls,
                            "bit_exact_calls": t.bit_exact_calls})
    out.tables["s1-networks"] = markdown_table(
        ["Design point", "Network", "External memory", "Weights (MB)", "Latency (ms)",
         "Inferences/s", "Achieved TOP/s", "MAC utilisation", "External memory MB/inference",
         "Compute-bound layers", "Memory-bound layers"], rows,
        [False, False, False] + [True] * 8)
    out.tables["s1-depthwise"] = markdown_table(
        ["Design point", "MobileNetV2 latency (ms)", "Depthwise share of cycles",
         "Depthwise share of external-memory bytes", "Depthwise read burst efficiency",
         "Depthwise write burst efficiency", "Latency without depthwise (ms)",
         "MAC utilisation without depthwise"], depthwise)
    out.tables["s1-llm"] = markdown_table(
        ["Design point", "Decode tokens/s INT8", "Decode tokens/s INT4", "INT4 speedup",
         "Decode DRAM MB/token INT8", "Decode DRAM rate used INT8", "Prefill TTFT INT8 (ms)",
         "Prefill TTFT INT4 (ms)", "Prefill MAC utilisation INT8"], llm)
    out.raw["networks"] = raw
    out.raw["calls"] = [{"point": key[0], "network": key[1], "weight_bits": key[2],
                         "layer": call.layer["name"], "calls": call.gemm.calls,
                         "flags": call.flags, "result": result}
                        for key, (calls, results) in runs.items()
                        for call, result in zip(calls, results)]
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
            for call, result in zip(calls, results):
                intensity = result["macs"] / dram_bytes(result)
                achieved = result["macs"] / result["cycles"]
                roof = analytic.attainable(point.design, intensity)
                require(achieved <= roof * (1 + 1e-12),
                        f"{point.name} {network} {call.layer['name']}: {achieved} MAC/cycle "
                        f"above the {roof} roof")
                marks.append(roofline.Point(call.layer["name"], intensity, achieved, network))
                raw.append({"point": point.name, "network": network,
                            "layer": call.layer["name"], "intensity": intensity,
                            "macs_per_cycle": achieved, "attainable": roof})
        d = point.design
        array = f"{d.array_rows}x{d.array_cols} {d.dataflow.upper()}"
        cores = f"{d.cores} cores of {array}" if d.cores > 1 else f"one {array} array"
        subtitle = (f"{cores}, {d.peak_macs_per_cycle:,} MAC/cycle; {point.memory} "
                    f"{d.dram_bytes_per_cycle:g} B/cycle; every GEMM layer of "
                    f"{len(networks)} workloads, INT8")
        fig = roofline.figure(d, marks, title=f"Roofline: {point.name}", subtitle=subtitle,
                              label_points=False, width=820, height=500,
                              group_order=NETWORK_ORDER, hollow=True)
        fig.x.label = f"arithmetic intensity (MAC per {point.memory} byte moved)"
        fig.legend_position = "right"
        out.figures[f"s2-roofline-{point.name}"] = fig
    out.raw["points"] = raw
    return out


# --------------------------------------------------------------------- AXI outstanding --

GEMV = {"m": 1, "n": 4096, "k": 4096, "weight_bits": 4, "memory_model": "axi",
        "macs_per_cycle": 512, "axi_data_bytes": 16, "axi_burst_beats": 16,
        "timing_only": True}
S3_LATENCIES = (100, 400)
S3_OUTSTANDING = tuple(range(1, 33))


def s3_outstanding(ctx: Context) -> StudyOutput:
    runs = [{**GEMV, "axi_read_latency": latency, "axi_write_latency": latency,
             "axi_read_outstanding": n} for latency in S3_LATENCIES for n in S3_OUTSTANDING]
    results = ctx.run_all(runs)
    bus = GEMV["axi_data_bytes"]
    beats = GEMV["axi_burst_beats"]
    burst = bus * beats
    out = StudyOutput()
    fig = svg.Figure(820, 480, title="Read bandwidth vs outstanding AXI reads",
                     subtitle="INT4 GEMV 1x4096x4096 streaming weights; 16 B beats, 16-beat "
                              "bursts, DRAM unlimited",
                     x_label="read transactions outstanding (N)",
                     y_label="read bandwidth (B/cycle)", x_range=(0, 33), y_range=(0, 17),
                     x_ticks=[1, 4, 8, 12, 16, 20, 24, 28, 32], legend="lower right",
                     id_prefix="s3")
    rows: dict[int, list] = {n: [n] for n in (1, 2, 4, 8, 16, 24, 32)}
    raw = []
    for index, latency in enumerate(S3_LATENCIES):
        color = svg.PALETTE[index]
        chunk = results[index * len(S3_OUTSTANDING):(index + 1) * len(S3_OUTSTANDING)]
        measured, formula = [], []
        for n, result in zip(S3_OUTSTANDING, chunk):
            check_result(result, f"S3 N={n} latency {latency}")
            rate = result["memory"]["read"]["payload_bytes"] / result["cycles"]
            ideal = min(bus, n * burst / (latency + beats))
            require(rate <= bus, f"S3: {rate} B/cycle above the {bus} B bus")
            measured.append(rate)
            formula.append(ideal)
            model = analytic.analyze(analytic.Design.from_result(result))
            raw.append({"latency": latency, "outstanding": n, "cycles": result["cycles"],
                        "read_bytes_per_cycle": rate, "steady_state": ideal,
                        "closed_form_cycles": model.estimate_cycles})
            if n in rows:
                rows[n] += [f"{rate:.2f}", f"{ideal:.2f}"]
        fig.line(S3_OUTSTANDING, formula, color=color, width=1.5, dash="6 4",
                 label=f"min(bus, N x burst / (L + beats)), L = {latency}")
        fig.scatter(S3_OUTSTANDING, measured, color=color,
                    marker=svg.MARKERS[index], label=f"npu-sim, read latency L = {latency}")
    fig.hline(bus, f"bus {bus} B/cycle", color=svg.INK_MUTED)
    out.figures["s3-outstanding"] = fig
    headers = ["Outstanding N"]
    for latency in S3_LATENCIES:
        headers += [f"L={latency} npu-sim (B/cycle)", f"L={latency} formula (B/cycle)"]
    out.tables["s3-outstanding"] = markdown_table(headers, list(rows.values()))
    out.raw["runs"] = raw
    return out


# ----------------------------------------------------------------------- multi-core --

S4_CORES = (1, 2, 4, 8, 12, 16, 24, 32)
S4_RATES = {"client-16c": 100, "workstation-32c": 667}


def compulsory_bytes(gemm: workloads.Gemm, weight_bits: int) -> int:
    """Bytes any schedule moves: every activation, weight, bias and output once."""
    g = gemm.groups
    return (gemm.m * g * gemm.k + -(-g * gemm.k * gemm.n * weight_bits // 8) + 4 * g * gemm.n
            + gemm.m * g * gemm.n)


def s4_layers() -> list[tuple[str, workloads.Gemm, int]]:
    """(label, GEMM, weight bits): ResNet-50 res3 3x3 and the Llama decode gate projection."""
    resnet = {layer["name"]: workloads.lower(layer)
              for layer in workloads.load("resnet50_v1.5_224")["layers"]}
    decode = {layer["name"]: workloads.lower(layer) for layer in workloads.load(DECODE)["layers"]}
    return [("ResNet-50 res3 3x3 conv (784x128x1152)", resnet["layer2.1-3.conv2"], 8),
            ("Llama-3.2-1B gate_proj GEMV INT4 (1x8192x2048)", decode["gate_proj"], 4)]


def coincident(curves: Sequence[Sequence[float]]) -> list[list[int]]:
    """Indexes of ``curves`` grouped by equal values (to 1e-9), in order of first appearance."""
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, values in enumerate(curves):
        groups.setdefault(tuple(round(value, 9) for value in values), []).append(index)
    return list(groups.values())


def s4_scaling(ctx: Context) -> StudyOutput:
    point = design_points(["client-16c"])[0]
    out = StudyOutput()
    rows, raw = [], []
    for slug, (label, gemm, bits) in zip(("conv", "gemv"), s4_layers()):
        # The mapper picks the tiles of every configuration, so a partition is not held to
        # the unit count of another configuration's tiles.
        runs, mappings = [], []
        for rate in S4_RATES.values():
            for partition in ("m", "n"):
                for cores in S4_CORES:
                    design = point.design.replace(cores=cores, partition=partition,
                                                  dram_bytes_per_cycle=rate, weight_bits=bits)
                    mapping = mapper.map_gemm(design, gemm.m, gemm.n, gemm.k, gemm.groups,
                                              point.min_depth)
                    flags = point.flags(**gemm.flags(), **mapping.flags(), cores=cores,
                                        partition=partition, dram_bytes_per_cycle=rate,
                                        timing_only=True)
                    if bits != 8:
                        flags["weight_bits"] = bits
                    runs.append(flags)
                    mappings.append(mapping)
        outcomes = ctx.run_all(runs)
        results = iter(zip(outcomes, mappings))
        floor = compulsory_bytes(gemm, bits)
        # Every series starts from the same one-core run; the y axis reaches the highest
        # ceiling so each ceiling line is drawn.
        top = max(outcomes[0]["cycles"] / math.ceil(floor / rate)
                  for rate in S4_RATES.values())
        top = max(64, 2 ** math.ceil(math.log2(top * 1.15)))
        ticks = [2 ** e for e in range(int(math.log2(top)) + 1)]
        fig = svg.Figure(820, 500, title=f"Cluster speedup: {label}",
                         subtitle="cores of the client-16c point (32x32 OS, 256-bit AXI, "
                                  "133.5 KiB SRAM); tiles from npuperf.mapper per configuration",
                         x_label="cores", y_label="speedup over one core", x_scale="log",
                         y_scale="log", x_range=(1, 32), y_range=(0.8, top),
                         x_ticks=list(S4_CORES), y_ticks=ticks,
                         legend="right", id_prefix=f"s4-{slug}")
        fig.x.tick_format = lambda v: f"{v:g}"
        fig.y.tick_format = lambda v: f"{v:g}"
        fig.line(S4_CORES, S4_CORES, color=svg.INK_MUTED, width=1.25, label="ideal (= cores)")
        curves: list[tuple[str, int, int, list[float]]] = []   # partition, rate, units, speedups
        for rate in S4_RATES.values():
            for partition in ("m", "n"):
                chunk = [next(results) for _ in S4_CORES]
                one = chunk[0][0]["cycles"]
                # No schedule moves fewer bytes than the compulsory traffic, so no cluster
                # beats one-core cycles / (compulsory bytes / DRAM rate).
                ceiling = one / math.ceil(floor / rate)
                speedups = []
                for cores, (result, mapping) in zip(S4_CORES, chunk):
                    check_result(result, f"S4 {slug} {partition} P={cores} R={rate}")
                    speedup = one / result["cycles"]
                    require(speedup <= cores * (1 + 1e-9),
                            f"S4 {slug}: speedup {speedup} above {cores} cores")
                    require(speedup <= ceiling * (1 + 1e-9),
                            f"S4 {slug}: speedup {speedup} above the DRAM ceiling {ceiling}")
                    speedups.append(speedup)
                    raw.append({"layer": slug, "dram_bytes_per_cycle": rate,
                                "partition": partition, "cores": cores,
                                "tile": [mapping.tile_m, mapping.tile_n, mapping.tile_k],
                                "cycles": result["cycles"], "speedup": speedup,
                                "dram_ceiling": ceiling, "dram_bytes": dram_bytes(result),
                                "units": result["cluster"]["units"],
                                "imbalance": result["cluster"]["imbalance"]})
                units = max(result["cluster"]["units"] for result, _ in chunk[1:])
                curves.append((partition, rate, units, speedups))
                if partition == "m":
                    fig.hline(ceiling, f"DRAM ceiling at {rate} B/cycle: {ceiling:.1f}",
                              color=svg.INK_SECONDARY, dash="2 3", width=1)
                rows.append([label.split(" (")[0], partition, f"{rate}"]
                            + [f"{s:.2f}" for s, c in zip(speedups, S4_CORES)
                               if c in (2, 4, 8, 16, 32)]
                            + [f"{ceiling:.2f}"])
        # Curves that coincide (the m split of a one-row GEMV leaves one busy core at every
        # DRAM rate) are drawn once under a joint label.
        for members in coincident([speedups for *_, speedups in curves]):
            first = members[0]
            _, rate, _, speedups = curves[first]
            partitions = "/".join(sorted({curves[i][0] for i in members}))
            rates = " and ".join(str(curves[i][1]) for i in members)
            dash = None if rate == S4_RATES["client-16c"] else "7 4"
            fig.line(S4_CORES, speedups, color=svg.PALETTE[first], dash=dash,
                     marker=svg.MARKERS[first], label=f"partition {partitions}, {rates} B/cycle")
            if all(curves[i][2] == 1 for i in members):
                fig.annotate(S4_CORES[1], speedups[1], f"partition {partitions}: one tile band, "
                             "one busy core", dx=4, dy=-10, color=svg.PALETTE[first])
        out.figures[f"s4-scaling-{slug}"] = fig
    out.tables["s4-scaling"] = markdown_table(
        ["Layer", "Partition", "DRAM B/cycle", "2 cores", "4 cores", "8 cores", "16 cores",
         "32 cores", "DRAM ceiling"], rows, [False, False, True] + [True] * 6)
    out.raw["runs"] = raw
    return out


# ------------------------------------------------------------------------- dataflow --

def layer_type(layer: Mapping[str, Any]) -> str:
    if layer["op"] == "depthwise_conv2d":
        return "depthwise"
    if layer["op"] in ("dense", "matmul"):
        return "FC"
    return f"{layer['kernel_h']}x{layer['kernel_w']}" if layer["kernel_h"] > 1 else "pointwise"


TYPE_ORDER = ("pointwise", "3x3", "7x7", "depthwise", "FC")


def s5_dataflow(ctx: Context) -> StudyOutput:
    point = design_points(["client-16c"])[0]
    calls = []
    for name in ("mobilenet_v2_1.0_224", "resnet50_v1.5_224"):
        for layer, gemm in workloads.iter_calls(workloads.load(name)):
            calls.append((NETWORKS[name], layer, gemm))
    runs = [layer_flags(point, gemm, cores=1, dataflow=dataflow)
            for dataflow in ("os", "ws") for _, _, gemm in calls]
    results = ctx.run_all(runs)
    os_results, ws_results = results[:len(calls)], results[len(calls):]
    array = point.design.array_rows * point.design.array_cols
    out = StudyOutput()
    by_type: dict[str, list] = {kind: [] for kind in TYPE_ORDER}
    raw = []
    for (network, layer, gemm), os_r, ws_r in zip(calls, os_results, ws_results):
        values = []
        for result in (os_r, ws_r):
            check_result(result, f"S5 {layer['name']}")
            array_util = result["macs"] / (result["compute_busy_cycles"] * array)
            overall = result["macs"] / (result["cycles"] * array)
            require(array_util <= 1 and overall <= 1, f"S5 {layer['name']}: utilisation")
            values.append((array_util, overall))
        kind = layer_type(layer)
        by_type[kind].append((gemm, values))
        raw.append({"network": network, "layer": layer["name"], "type": kind,
                    "calls": gemm.calls, "m": gemm.m, "n": gemm.n, "k": gemm.k,
                    "groups": gemm.groups, "os": values[0], "ws": values[1]})
    # Linear axes: the dense layers sit between 0.4 and 1, where the OS-WS differences of
    # the table are; the depthwise and FC layers collapse near the origin.
    fig = svg.Figure(820, 600, title="Array utilisation: output- vs weight-stationary",
                     subtitle="32x32 array, one core of client-16c; MACs / (compute-busy "
                              "cycles x 1024); one mark per distinct layer shape",
                     x_label="output-stationary utilisation", y_label="weight-stationary "
                     "utilisation", x_range=(0, 1), y_range=(0, 1), legend="upper left",
                     id_prefix="s5")
    fig.line([0, 1], [0, 1], color=svg.INK_MUTED, width=1, dash="4 4")
    fig.annotate(0.3, 0.3, "equal", dx=8, dy=14, color=svg.INK_MUTED)
    rows = []
    for index, kind in enumerate(TYPE_ORDER):
        members = by_type[kind]
        if not members:
            continue
        # Layers count every call; workload entries that repeat a shape plot once.
        layers = sum(gemm.calls for gemm, _ in members)
        shapes = sorted({(gemm.m, gemm.n, gemm.k, gemm.groups): v
                         for gemm, v in members}.items())
        fig.scatter([v[0][0] for _, v in shapes], [v[1][0] for _, v in shapes],
                    color=svg.PALETTE[index], marker=svg.MARKERS[index],
                    label=f"{kind} ({plural(layers, 'layer')}, "
                          f"{plural(len(shapes), 'shape')})", hollow=True)
        # MAC-weighted means of (OS, WS) x (array, end-to-end) utilisation.
        weights = [g.total_macs for g, _ in members]
        means = [pct(sum(w * v[flow][measure] for w, (_, v) in zip(weights, members))
                     / sum(weights)) for measure in (0, 1) for flow in (0, 1)]
        rows.append([kind, str(layers), str(len(shapes)), fmt(sum(weights) / 1e6), *means])
    out.figures["s5-dataflow"] = fig
    out.tables["s5-dataflow"] = markdown_table(
        ["Layer type", "Layers", "Distinct shapes", "MACs (M)", "OS array utilisation",
         "WS array utilisation", "OS end-to-end", "WS end-to-end"], rows)
    out.raw["layers"] = raw
    return out


# ------------------------------------------------------------------ model vs SystemC --

S6_SEED = 20261107
S6_COUNT = {"serial": 120, "axi": 120}


def s6_configs(seed: int = S6_SEED) -> list[dict[str, Any]]:
    """Seeded random dense configurations that fit their SRAM; the AXI third includes
    clusters of 2 to 8 cores. Draw order is fixed, so the set is the same everywhere."""
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


def _random_config(rng: random.Random, model: str) -> dict[str, Any]:
    flags: dict[str, Any] = {
        "m": rng.choice([1, 7, 16, 33, 64, 128, 196, 300]),
        "n": rng.choice([1, 8, 24, 64, 100, 256]),
        "k": rng.choice([9, 32, 75, 128, 300, 576]),
        "groups": rng.choice([1, 1, 1, 1, 2, 4]),
        "tile_m": rng.choice([4, 8, 16, 32, 64]), "tile_n": rng.choice([4, 8, 16, 32, 64]),
        "tile_k": rng.choice([0, 0, 16, 32, 64, 128]),
        "buffer_slots": rng.choice([1, 2, 2, 3]), "panel_buffers": rng.choice([1, 2, 2]),
        "weight_bits": rng.choice([8, 8, 4]), "pipeline_cycles": rng.choice([0, 4, 8]),
        "memory_model": model, "timing_only": True}
    if rng.random() < 0.4:
        flags["macs_per_cycle"] = rng.choice([32, 64, 256, 1024])
    else:
        flags["array_rows"], flags["array_cols"] = rng.choice([(8, 8), (16, 16), (32, 32),
                                                              (16, 32)])
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
    require(model.lower_bound_cycles <= cycles,
            f"S6: lower bound {model.lower_bound_cycles} above {cycles} for {flags}")
    traffic = (model.read_bytes, model.write_bytes, model.compute_busy_cycles) == (
        result["read_bytes"], result["write_bytes"], result["compute_busy_cycles"])
    return {"flags": dict(flags), "cycles": cycles, "estimate": model.estimate_cycles,
            "lower_bound": model.lower_bound_cycles, "schedule": replay.cycles,
            "bound": model.bound, "observed_bound": analytic.observed_bound(result),
            "traffic_exact": traffic}


def s6_design_point_runs(ctx: Context) -> dict[str, list[tuple[dict, dict]]]:
    """Point name -> the distinct (flags, result) runs of its S1 network and LLM layers,
    each with the tiling npuperf.mapper picked by its analyze() estimate."""
    runs = run_networks(ctx, design_points())
    out: dict[str, list[tuple[dict, dict]]] = {}
    seen: set[str] = set()
    for (point, _, _), (calls, results) in runs.items():
        for call, result in zip(calls, results):
            key = Context.key(call.flags)
            if key not in seen:
                seen.add(key)
                out.setdefault(point, []).append((call.flags, result))
    return out


def s6_analytic(ctx: Context) -> StudyOutput:
    configs = s6_configs()
    results = ctx.run_all(configs)
    out = StudyOutput()
    raw = []
    groups: dict[str, list[dict]] = {"random serial": [], "random axi": [],
                                     "random axi cluster": []}
    for flags, result in zip(configs, results):
        check_result(result, f"S6 {flags}")
        entry = s6_entry(flags, result)
        raw.append(entry)
        kind = "random " + flags["memory_model"] + (" cluster" if flags.get("cores", 1) > 1
                                                     else "")
        groups[kind].append(entry)
    for point, runs in s6_design_point_runs(ctx).items():
        kind = f"`{point}` network layers"
        for flags, result in runs:
            entry = s6_entry(flags, result)
            raw.append({"point": point, **entry})
            groups.setdefault(kind, []).append(entry)
    rows = []
    # The ratio on a linear axis: on log-log axes spanning six decades a 30% error is a
    # few pixels off the diagonal.
    lo = 10 ** math.floor(math.log10(min(e["cycles"] for e in raw)))
    hi = 10 ** math.ceil(math.log10(max(e["cycles"] for e in raw)))
    ratios = [e["estimate"] / e["cycles"] for e in raw]
    random_count = sum(len(groups[kind]) for kind in groups if kind.startswith("random"))
    fig = svg.Figure(820, 600, title="Closed-form estimate / npu-sim cycles",
                     subtitle=f"{random_count} seeded random dense configurations (seed "
                              f"{S6_SEED}) and the distinct S1 layer runs of each design "
                              "point; 1 = exact",
                     x_label="npu-sim cycles", y_label="analytic.analyze estimate / npu-sim "
                     "cycles", x_scale="log", x_range=(lo, hi),
                     y_range=(math.floor(min(ratios) * 10) / 10, math.ceil(max(ratios) * 10) / 10),
                     legend="right", id_prefix="s6")
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
        rows.append([kind, str(count), pct(statistics.median(errors)),
                     pct(_percentile(errors, 0.95)), pct(errors[-1]), f"{low_count}/{count}",
                     f"{agree}/{count}", f"{exact}/{count}", f"{traffic}/{count}"])
        fig.scatter([e["cycles"] for e in members],
                    [e["estimate"] / e["cycles"] for e in members],
                    color=svg.PALETTE[index], marker=svg.MARKERS[index],
                    label=f"{kind.replace('`', '')} ({count})", hollow=True, size=4)
    out.figures["s6-analytic"] = fig
    out.tables["s6-analytic"] = markdown_table(
        ["Configurations", "Count", "Median error", "p95 error", "Max error",
         "Estimate below npu-sim", "Bound class agrees", "schedule() exact",
         "Traffic and compute exact"], rows)
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
    rows = [" ".join(tokens[shift:shift + gemm.n]) for shift in range(period)]
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


def decode_rate(point: Point, pieces: Sequence[workloads.Gemm],
                results: Sequence[Mapping[str, Any]], attention_cycles: int,
                label: str) -> tuple[int, float]:
    """(cycles, tokens/s) of one decode token: the attention matmuls plus every weight
    piece's calls."""
    cycles = attention_cycles
    for piece, result in zip(pieces, results):
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
    pieces = [piece for layer in decode["layers"] if layer.get("b_operand") == "weights"
              for piece in workloads.split(workloads.lower(layer), timing_only=False,
                                           max_host=PALETTE_HOST_BYTES)]
    curves = []
    for point in points:
        # Attention matmuls read the KV cache, which stays INT8 in every column.
        attention = [c for c in network_calls(point, DECODE, "Llama decode", 8)
                     if c.layer.get("b_operand") != "weights"]
        attention_results = ctx.run_all([c.flags for c in attention])
        for call, result in zip(attention, attention_results):
            check_result(result, f"S7 {point.name} {call.layer['name']}")
        attention_cycles = sum(c.gemm.calls * r["cycles"]
                               for c, r in zip(attention, attention_results))
        dense = {}
        for bits in (8, 4):
            results = ctx.run_all([layer_flags(point, piece, bits) for piece in pieces])
            cycles, speed = decode_rate(point, pieces, results, attention_cycles, f"INT{bits}")
            dense[f"INT{bits}"] = speed
            raw.append({"point": point.name, "weights": f"dense INT{bits}", "cycles": cycles,
                        "tokens_per_s": speed})
        palette = []
        for rate in S7_DECODER_RATES:
            runs = []
            for piece in pieces:
                mapping = mapper.map_gemm(point.design, piece.m, piece.n, piece.k,
                                          piece.groups, point.min_depth,
                                          weight_encoding="palette", palette_bits=PALETTE_BITS,
                                          weight_decode_values_per_cycle=rate)
                # The input file's header gives M, N and K; only the group count is a flag.
                shape = {key: value for key, value in piece.flags().items()
                         if key not in ("m", "n", "k")}
                runs.append(point.flags(**shape, **mapping.flags(),
                                        input=str(palette_input(ctx, piece)),
                                        weight_encoding="palette", palette_bits=PALETTE_BITS,
                                        weight_decode_values_per_cycle=rate))
            results = ctx.run_all(runs)
            for piece, result in zip(pieces, results):
                require(result["numerics"] == "bit-exact", f"S7 {piece.name}: not bit-exact")
            cycles, speed = decode_rate(point, pieces, results, attention_cycles,
                                        f"palette rate {rate}")
            palette.append(speed)
            raw.append({"point": point.name, "weights": "palette",
                        "decoder_values_per_cycle": rate, "cycles": cycles,
                        "tokens_per_s": speed})
        curves.append((point, dense, palette))
        rows.append([f"`{point.name}`", fmt(dense["INT8"]), fmt(dense["INT4"])]
                    + [fmt(p) for p in palette])
    # Relative to dense INT4, so one reference line serves every point.
    int8 = [dense["INT8"] / dense["INT4"] for _, dense, _ in curves]
    top = max(p / dense["INT4"] for _, dense, palette in curves for p in palette)
    fig = svg.Figure(820, 500, title="Llama-3.2-1B decode: 4-bit palette vs dense INT4 weights",
                     subtitle=f"every weight GEMM in the same pieces for all three formats; "
                              f"dense INT8 runs at {min(int8):.2f} to {max(int8):.2f} of dense "
                              "INT4; KV-cache matmuls stay INT8",
                     x_label="weight decoder throughput (values per cycle per core)",
                     y_label="palette tokens/s / dense INT4 tokens/s", x_scale="log",
                     x_range=(4, 256), y_range=(0, max(1.2, math.ceil(top * 10) / 10)),
                     x_ticks=list(S7_DECODER_RATES), legend="lower right", id_prefix="s7")
    fig.x.tick_format = lambda v: f"{v:g}"
    fig.hline(1.0, "dense INT4, same pieces", color=svg.INK_MUTED, text_anchor="start")
    for index, (point, dense, palette) in enumerate(curves):
        fig.line(S7_DECODER_RATES, [p / dense["INT4"] for p in palette],
                 color=svg.PALETTE[index], marker=svg.MARKERS[index],
                 label=f"{point.name} palette")
    out.figures["s7-compression"] = fig
    out.tables["s7-compression"] = markdown_table(
        ["Design point", "Dense INT8", "Dense INT4"]
        + [f"Palette, {rate}/cycle" for rate in S7_DECODER_RATES], rows)
    out.raw["runs"] = raw
    return out


# ------------------------------------------------------------------- mapping search --

S8_POINT = "mobile-2k"
S8_SRAM = 1 << 20
S8_GRID = ((32, 64, 128, 256), (64, 128), (0, 128, 256, 512), (1, 2))


def s8_pareto(ctx: Context) -> StudyOutput:
    point = design_points([S8_POINT])[0]
    layer = next(layer for layer in workloads.load("resnet50_v1.5_224")["layers"]
                 if layer["name"] == "layer2.1-3.conv2")
    gemm = workloads.lower(layer)
    # The mapper under the point's own SRAM: without the weight cache, the condition of the
    # search, and with the free SRAM as weight cache, as the network studies run it.
    plain = mapper.map_gemm(point.design, gemm.m, gemm.n, gemm.k, gemm.groups,
                            point.min_depth, cache=False)
    cached = mapper.map_gemm(point.design, gemm.m, gemm.n, gemm.k, gemm.groups,
                             point.min_depth)
    pick = (plain.tile_m, plain.tile_n, plain.tile_k, plain.buffer_slots)
    candidates = sorted({(m, n, k, s) for m in S8_GRID[0] for n in S8_GRID[1]
                         for k in S8_GRID[2] for s in S8_GRID[3]} | {pick})
    base = point.flags(**gemm.flags(), sram_bytes=S8_SRAM, weight_cache_bytes=0)
    report, rows = autotune.search(ctx.binary, [gemm.m, gemm.n, gemm.k], candidates, base,
                                   timeout=ctx.timeout)
    ctx.extra_runs += sum(1 for row in rows if row["feasible"])
    feasible = [row for row in rows if row["feasible"]]
    hashes = {row["output_hash"] for row in feasible}
    require(len(hashes) == 1, "S8: mappings disagree on the output")
    with_cache = ctx.run_all([point.flags(**gemm.flags(), **cached.flags())])[0]
    check_result(with_cache, "S8 mapper with weight cache")
    pareto = report["pareto"]
    mine = next(row for row in feasible
                if (row["tile_m"], row["tile_n"], row["tile_k"], row["buffer_slots"]) == pick)
    within = [row for row in feasible if row["scratchpad_bytes"] <= point.design.sram_bytes]
    best = min(within, key=lambda row: (row["cycles"], row["scratchpad_bytes"]))
    out = StudyOutput()
    fig = svg.Figure(820, 500, title="Mapping search: ResNet-50 res3 3x3 conv on mobile-2k",
                     subtitle=f"784x128x1152; {len(feasible)} feasible tile and slot choices of "
                              f"{len(candidates)}, all bit-exact; weight cache off unless "
                              "noted",
                     x_label="SRAM reserved (KiB)", y_label="cycles", y_include_zero=True,
                     legend="upper right", id_prefix="s8")
    fig.scatter([r["scratchpad_bytes"] / 1024 for r in feasible],
                [r["cycles"] for r in feasible], color=svg.INK_MUTED, marker="circle",
                hollow=True, size=3.5, label="feasible mapping")
    fig.step([r["scratchpad_bytes"] / 1024 for r in pareto], [r["cycles"] for r in pareto],
             color=svg.PALETTE[0], label="Pareto front")
    fig.scatter([r["scratchpad_bytes"] / 1024 for r in pareto], [r["cycles"] for r in pareto],
                color=svg.PALETTE[0], marker="circle", size=4.5)
    fig.scatter([mine["scratchpad_bytes"] / 1024], [mine["cycles"]], color=svg.PALETTE[1],
                marker="diamond", size=7, label="npuperf.mapper, cache off")
    fig.scatter([with_cache["scratchpad_bytes"] / 1024], [with_cache["cycles"]],
                color=svg.PALETTE[2], marker="triangle", size=7,
                label="npuperf.mapper, free SRAM as weight cache")
    fig.vline(point.design.sram_bytes / 1024, f"{point.name} SRAM "
              f"{point.design.sram_bytes / 1024:g} KiB", color=svg.INK_SECONDARY)
    out.figures["s8-pareto"] = fig

    def describe(row: Mapping[str, Any]) -> str:
        return f"{row['tile_m']}x{row['tile_n']}x{row['tile_k'] or gemm.k}"
    table_rows = [[describe(row), str(row["buffer_slots"]), "0",
                   fmt(row["scratchpad_bytes"] / 1024, 4), f"{row['cycles']:,}",
                   "Pareto front"] for row in pareto]
    table_rows.append([describe(best), str(best["buffer_slots"]), "0",
                       fmt(best["scratchpad_bytes"] / 1024, 4), f"{best['cycles']:,}",
                       f"fastest within {point.design.sram_bytes / 1024:g} KiB"])
    table_rows.append([describe(mine), str(mine["buffer_slots"]), "0",
                       fmt(mine["scratchpad_bytes"] / 1024, 4), f"{mine['cycles']:,}",
                       "npuperf.mapper, cache off"])
    table_rows.append([f"{cached.tile_m}x{cached.tile_n}x{cached.tile_k or gemm.k}",
                       str(cached.buffer_slots), fmt(cached.weight_cache_bytes / 1024, 4),
                       fmt(with_cache["scratchpad_bytes"] / 1024, 4),
                       f"{with_cache['cycles']:,}", "npuperf.mapper, free SRAM as cache"])
    out.tables["s8-pareto"] = markdown_table(
        ["Tile (m x n x k)", "Slots", "Weight cache (KiB)", "SRAM (KiB)", "Cycles", "Row"],
        table_rows, [False, True, True, True, True, False])
    out.raw["rows"] = rows
    out.raw["mapper"] = {"cache_off": asdict(plain), "cache_on": asdict(cached),
                         "cache_on_result": with_cache}
    return out


# --------------------------------------------------------------------------- registry --

@dataclass(frozen=True)
class Study:
    id: str
    title: str
    quick: bool
    run: Callable[[Context], StudyOutput]


STUDIES = {study.id: study for study in (
    Study("S1", "Network results per design point", False, s1_networks),
    Study("S2", "Roofline per design point", False, s2_roofline),
    Study("S3", "AXI outstanding transactions", True, s3_outstanding),
    Study("S4", "Multi-core scaling", True, s4_scaling),
    Study("S5", "Dataflow: output- vs weight-stationary", True, s5_dataflow),
    Study("S6", "Analytic model vs SystemC", True, s6_analytic),
    Study("S7", "Weight compression for LLM decode", False, s7_compression),
    Study("S8", "Mapping search Pareto front", True, s8_pareto),
)}
