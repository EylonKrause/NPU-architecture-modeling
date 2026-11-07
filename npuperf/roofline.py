"""Roofline points for workloads on one design point, from the analytic model or npu-sim.

    python -m npuperf.roofline --workload resnet50_v1.5_224 \\
        --set macs_per_cycle=256 --set memory_bytes_per_cycle=32 --set tile_m=16 --out roof.svg

x is arithmetic intensity (MACs per DRAM byte moved, after the mapping's re-reads), y is
achieved MACs per cycle. Roofs: peak MACs per cycle and peak DRAM bytes per cycle; under the
serial memory model a dashed roof shows the bandwidth left after per-request latency.
Every point satisfies y <= min(peak, bandwidth * x); the tests check that invariant.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import analytic, cli, jsonio, sim, svg, table, workloads

GROUP_ORDER = ("conv", "depthwise", "matmul", "simulated")


@dataclass(frozen=True)
class Point:
    label: str
    intensity: float
    macs_per_cycle: float
    group: str = ""


def from_result(result: dict | sim.SimResult, label: str, group: str = "simulated") -> Point:
    traffic = result["read_bytes"] + result["write_bytes"]
    return Point(label, result["macs"] / traffic, result["macs"] / result["cycles"], group)


def from_analysis(analysis: analytic.Analysis, label: str, group: str = "") -> Point:
    return Point(label, analysis.arithmetic_intensity, analysis.macs / analysis.estimate_cycles,
                 group)


def effective_bandwidth(analyses: Sequence[analytic.Analysis]) -> float | None:
    """DRAM bytes per DMA-busy cycle: the serial model's bandwidth after request latency."""
    busy = sum(a.dma_busy_cycles or 0 for a in analyses)
    return sum(a.dram_bytes for a in analyses) / busy if busy else None


def _decade_floor(value: float) -> float:
    return 10 ** math.floor(math.log10(value))


def _decade_ceil(value: float) -> float:
    return 10 ** math.ceil(math.log10(value))


def figure(design: analytic.Design, points: Sequence[Point], title: str = "Roofline",
           subtitle: str | None = None, effective_bw: float | None = None,
           label_points: Sequence[str] | bool = True, width: int = 760,
           height: int = 480, group_order: Sequence[str] = GROUP_ORDER,
           hollow: bool = False) -> svg.Figure:
    """``group_order`` fixes the colour of each group (others follow, sorted); ``hollow``
    draws every group's markers as outlines, so coinciding points of different groups
    stay visible."""
    peak, bw = float(design.peak_macs_per_cycle), design.bandwidth
    ridge = peak / bw
    xs = [p.intensity for p in points] + [ridge]
    x_lo, x_hi = _decade_floor(min(xs) / 1.5), _decade_ceil(max(xs) * 1.5)
    ys = [p.macs_per_cycle for p in points] + [peak, bw * x_lo]
    y_lo, y_hi = _decade_floor(min(ys) / 1.5), _decade_ceil(peak * 2.5)
    fig = svg.Figure(width, height, title=title, subtitle=subtitle,
                     x_label="arithmetic intensity (MAC per DRAM byte)",
                     y_label="MAC per cycle", x_scale="log", y_scale="log",
                     x_range=(x_lo, x_hi), y_range=(y_lo, y_hi), id_prefix="roofline")
    roof = [(x_lo, bw * x_lo), (ridge, peak), (x_hi, peak)]
    fig.line([x for x, _ in roof], [y for _, y in roof], color=svg.INK, width=2)
    fig.vline(ridge, color=svg.INK_MUTED, dash="2 3", width=1)
    fig.annotate(ridge, y_lo, f"ridge {_fmt(ridge)} MAC/B", dx=5, dy=-6, color=svg.INK_MUTED)
    fig.annotate(x_hi, peak, f"peak {_fmt(peak)} MAC/cycle", dx=-6, dy=-7, anchor="end",
                 color=svg.INK)
    fig.annotate_along((x_lo, bw * x_lo), (ridge, peak), f"{_fmt(bw)} B/cycle",
                       at=0.3, offset=-8, color=svg.INK)
    if effective_bw and effective_bw < bw * 0.98:
        knee = peak / effective_bw
        end = min(knee, x_hi)
        fig.line([x_lo, end], [effective_bw * x_lo, effective_bw * end], color=svg.INK_MUTED,
                 width=1.25, dash="6 4")
        fig.annotate_along((x_lo, effective_bw * x_lo), (end, effective_bw * end),
                           f"{_fmt(effective_bw)} B/cycle after DMA latency", at=0.55,
                           offset=15, color=svg.INK_MUTED)
    # Model points take categorical colours by op class; simulated points are hollow ink
    # rings drawn last, so a simulated point on top of its model point leaves both visible.
    groups = [g for g in group_order if any(p.group == g for p in points) and g != "simulated"]
    groups += sorted({p.group for p in points} - set(groups) - {"simulated"})
    if any(p.group == "simulated" for p in points):
        groups.append("simulated")
    chosen = set(label_points) if not isinstance(label_points, bool) else None
    index = 0
    for group in groups:
        members = [p for p in points if p.group == group]
        labels = [p.label if (label_points is True or (chosen and p.label in chosen)) else None
                  for p in members]
        simulated = group == "simulated"
        fig.scatter([p.intensity for p in members], [p.macs_per_cycle for p in members],
                    label=(group + " (npu-sim)" if simulated else group) or None,
                    color=svg.INK if simulated else svg.PALETTE[index % len(svg.PALETTE)],
                    marker="circle" if simulated else svg.MARKERS[index % len(svg.MARKERS)],
                    point_labels=labels, hollow=simulated or hollow,
                    size=5 if simulated else 4.5)
        index += 0 if simulated else 1
    return fig


def _fmt(value: float) -> str:
    if value >= 100 or value == round(value):
        return f"{value:,.0f}"
    return f"{value:.3g}"


def _parse_set(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"--set expects KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip().replace("-", "_"), sim.from_flags(["--x", value])["x"]


def analyse_workload(design: analytic.Design, workload: dict[str, Any], timing_only: bool = True,
                     **options: Any) -> list[dict[str, Any]]:
    """One row per npu-sim call; ``options`` are ``workloads.iter_calls`` options."""
    rows = []
    tiles = {"tile_m": design.tile_m, "tile_n": design.tile_n, "tile_k": design.tile_k}
    for layer, gemm in workloads.iter_calls(workload, timing_only, **{**tiles, **options}):
        call = design.replace(m=gemm.m, n=gemm.n, k=gemm.k, groups=gemm.groups)
        result = analytic.analyze(call)
        rows.append({"layer": layer["name"], "class": gemm.op_class, "gemm": gemm,
                     "analysis": result})
    return rows


@cli.main_guard("roofline")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("roofline", __doc__)
    parser.add_argument("--workload", action="append", required=True,
                        help="workload name in workloads/ or a JSON path (repeatable)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="design parameter (an analytic.Design field, npu-sim flag name "
                             "or config path), e.g. macs_per_cycle=256")
    parser.add_argument("--simulate", action="store_true",
                        help="also run every layer on npu-sim: --timing-only calls when the "
                             "build has it, else functional calls split to its smaller "
                             "per-call limits; grouped layers run as one call per group "
                             "when it has no --groups")
    parser.add_argument("--out", help="write the roofline SVG here")
    parser.add_argument("--label", action="append", default=[],
                        help="layer to label on the plot (default: none; repeatable)")
    cli.add_binary_options(parser)
    cli.add_format_option(parser)
    args = parser.parse_args(argv)
    flags = analytic.Design.normalise(dict(_parse_set(item) for item in args.set))
    design = analytic.Design.from_flags(flags, strict=True)
    binary = cli.simulator(args) if args.simulate else None
    options: dict[str, Any] = {"timing_only": True}
    if binary is not None:
        # Model the design the build runs: options it lacks keep its fixed behaviour
        # (for example one panel buffer per slot).
        probe = sim.run({**flags, "m": 8, "n": 8, "k": 8}, binary, args.timeout)
        design = analytic.Design.from_result(probe)
        options = workloads.call_options(binary)
    timing_only = options["timing_only"]
    rows, points, analyses = [], [], []
    for name in args.workload:
        workload = workloads.load(name)
        for row in analyse_workload(design, workload, **options):
            gemm, result = row["gemm"], row["analysis"]
            analyses.append(result)
            entry = {"workload": workload["name"], "layer": row["layer"], "class": row["class"],
                     "m": gemm.m, "n": gemm.n, "k": gemm.k, "g": gemm.groups, "calls": gemm.calls,
                     "macs": gemm.total_macs, "ai": result.arithmetic_intensity,
                     "bound": result.bound, "estimate_cycles": result.estimate_cycles * gemm.calls}
            points.append(from_analysis(result, row["layer"], row["class"]))
            if binary is not None:
                run_flags = {**flags, **gemm.flags(), "timing_only": timing_only or None}
                simulated = sim.run(run_flags, binary, args.timeout)
                entry["simulated_cycles"] = simulated.cycles * gemm.calls
                points.append(from_result(simulated, row["layer"]))
            rows.append(entry)
    if args.out:
        names = ", ".join(workloads.load(name)["name"] for name in args.workload)
        fig = figure(design, points, title=f"Roofline: {names}",
                     subtitle=_design_line(design), effective_bw=effective_bandwidth(analyses),
                     label_points=args.label)
        fig.save(args.out)
    if args.format == "json":
        print(jsonio.dumps({"design": flags, "layers": rows}), end="")
    else:
        print(table.format_table(rows, formats={"ai": ".2f"}), end="")
    return 0


def _design_line(design: analytic.Design) -> str:
    compute = (f"{design.array_rows}x{design.array_cols} {design.dataflow.upper()} array"
               if design.array else f"{design.macs_per_cycle} MAC/cycle")
    memory = (f"{design.memory_bytes_per_cycle} B/cycle, {design.memory_latency}-cycle latency"
              if design.memory_model == "serial" else "AXI")
    return (f"{compute} · {memory} · tile {design.tile_m}x{design.tile_n}"
            + (f"x{design.tile_k}" if design.tile_k else "") + f" · {design.weight_bits}-bit weights")


if __name__ == "__main__":
    raise SystemExit(main())
