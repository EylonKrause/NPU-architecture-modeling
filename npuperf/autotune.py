"""Search tile sizes and buffer slots for one GEMM under an SRAM budget; report the best
mapping and the cycles/SRAM Pareto front.

SRAM feasibility comes from the analytic reservation formula, so a candidate that cannot fit
is skipped without launching the simulator. A candidate the simulator still rejects is
infeasible only if the formula, with any panel-buffer count up to the configured one,
also says it may not fit; any other rejection is an error.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import analytic, cli, designs, jsonio, sim, table

MAX_CANDIDATES = 256


def design_of(flags: dict[str, Any]) -> analytic.Design:
    """The design the flags run: a --config file's keys, then the other flags over them."""
    config = flags.get("config")
    base = designs.leaves(designs.load(config)) if config else {}
    return analytic.Design.from_flags(
        {**base, **{key: value for key, value in flags.items() if key != "config"}}
    )


def _sram_range(flags: dict[str, Any]) -> tuple[int, int, int]:
    design = design_of(flags)
    low = analytic.scratchpad_bytes(design, panel_buffers=1)
    high = analytic.scratchpad_bytes(design, panel_buffers=max(1, design.panel_buffers))
    return low, high, design.sram_bytes


def search(
    binary: Path,
    dimensions: Sequence[int],
    candidates: Sequence[tuple[int, int, int, int]],
    base_flags: sim.Flags,
    objective: str = "latency",
    timeout: float = 60,
) -> tuple[dict, list[dict]]:
    if not candidates or len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"mapping search requires one through {MAX_CANDIDATES} candidates")
    base = sim.from_flags(sim.to_flags(base_flags))
    rows: list[dict] = []
    runs: list[dict] = []
    feasible: list[dict] = []
    output_hash = None
    for tile_m, tile_n, tile_k, slots in candidates:
        mapping = {"tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k, "buffer_slots": slots}
        flags = {**base, **mapping}
        low, high, budget = _sram_range(flags)
        infeasible = {
            **mapping,
            "feasible": False,
            "cycles": None,
            "scratchpad_bytes": None,
            "output_hash": None,
        }
        if low > budget:
            rows.append(infeasible)
            runs.append(
                {
                    "mapping": mapping,
                    "feasible": False,
                    "reason": "SRAM capacity",
                    "predicted_scratchpad_bytes": low,
                }
            )
            continue
        try:
            result = sim.run(flags, binary, timeout)
        except sim.SimError as error:
            if error.returncode == 2 and high > budget:
                rows.append(infeasible)
                runs.append(
                    {
                        "mapping": mapping,
                        "feasible": False,
                        "reason": "SRAM capacity",
                        "predicted_scratchpad_bytes": high,
                    }
                )
                continue
            raise
        if output_hash is None:
            output_hash = result["output_hash"]
        if output_hash != result["output_hash"]:
            raise sim.CheckError("a mapping changed the numerical result")
        entry = {
            **mapping,
            "feasible": True,
            "cycles": result.cycles,
            "scratchpad_bytes": result["scratchpad_bytes"],
            "output_hash": result["output_hash"],
        }
        rows.append(entry)
        feasible.append(entry)
        runs.append({"mapping": mapping, "feasible": True, **result.as_run()})
    if not feasible:
        raise ValueError("no feasible mapping; increase SRAM or choose smaller tiles")

    def rank(entry: dict) -> tuple[int, int]:
        return (
            (entry["cycles"], entry["scratchpad_bytes"])
            if objective == "latency"
            else (entry["scratchpad_bytes"], entry["cycles"])
        )

    best = min(feasible, key=rank)
    pareto = [
        a
        for a in feasible
        if not any(
            b["cycles"] <= a["cycles"]
            and b["scratchpad_bytes"] <= a["scratchpad_bytes"]
            and (b["cycles"] < a["cycles"] or b["scratchpad_bytes"] < a["scratchpad_bytes"])
            for b in feasible
        )
    ]
    pareto.sort(key=lambda e: (e["scratchpad_bytes"], e["cycles"]))
    report = {
        "schema_version": 1,
        "experiment": "exhaustive_mapping_search",
        "objective": objective,
        "shape": list(dimensions),
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
        "best": best,
        "pareto": pareto,
        "runs": runs,
    }
    return report, rows


def summary(report: dict) -> str:
    columns = ["tile_m", "tile_n", "tile_k", "buffer_slots", "cycles", "scratchpad_bytes"]
    best = report["best"]
    head = (
        f"{report['feasible_count']} of {report['candidate_count']} mappings fit; best for "
        f"{report['objective']}: tile {best['tile_m']}x{best['tile_n']}x{best['tile_k']}, "
        f"{best['buffer_slots']} slot(s), {best['cycles']:,} cycles, "
        f"{best['scratchpad_bytes']:,} B SRAM\n\nPareto front (cycles vs SRAM):\n"
    )
    return head + table.format_table(report["pareto"], columns)


@cli.main_guard("autotune")
def main(argv: Sequence[str] | None = None) -> int:
    p = cli.Parser("autotune", __doc__)
    cli.add_binary_options(p)
    cli.add_format_option(p)
    p.add_argument("--m", type=cli.positive_int, default=16, help="output rows")
    p.add_argument("--n", type=cli.positive_int, default=16, help="output columns")
    p.add_argument("--k", type=cli.positive_int, default=32, help="reduction depth")
    p.add_argument(
        "--tile-m-options",
        type=cli.int_list(1, 4096),
        default=[2, 4, 8],
        help="comma-separated tile_m candidates",
    )
    p.add_argument(
        "--tile-n-options",
        type=cli.int_list(1, 4096),
        default=[2, 4, 8],
        help="comma-separated tile_n candidates",
    )
    p.add_argument(
        "--tile-k-options",
        type=cli.int_list(0, 4096),
        default=[0, 8, 16],
        help="comma-separated tile_k candidates (0 = full K)",
    )
    p.add_argument(
        "--slot-options",
        type=cli.int_list(1, 64),
        default=[1, 2],
        help="comma-separated buffer-slot candidates",
    )
    p.add_argument(
        "--sram-bytes", type=cli.positive_int, default=262144, help="SRAM budget in bytes"
    )
    p.add_argument(
        "--macs-per-cycle", type=cli.positive_int, default=64, help="peak INT8 MACs per cycle"
    )
    p.add_argument(
        "--memory-bytes-per-cycle", type=cli.positive_int, default=16, help="DRAM bytes per cycle"
    )
    p.add_argument(
        "--memory-latency",
        type=cli.nonnegative_int,
        default=12,
        help="cycles of latency per DMA request",
    )
    p.add_argument(
        "--pipeline-cycles",
        type=cli.nonnegative_int,
        default=4,
        help="pipeline fill/drain cycles per reduction panel",
    )
    p.add_argument("--weight-bits", type=int, choices=[4, 8], default=8, help="weight precision")
    p.add_argument(
        "--objective",
        choices=["latency", "sram"],
        default="latency",
        help="rank by cycles, or by SRAM then cycles",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        help="also write autotune.json and autotune.csv here (replaces earlier files)",
    )
    args = p.parse_args(argv)
    grid = (args.tile_m_options, args.tile_n_options, args.tile_k_options, args.slot_options)
    if math.prod(map(len, grid)) > MAX_CANDIDATES:
        p.error(f"mapping search requires one through {MAX_CANDIDATES} candidates")
    candidates = list(itertools.product(*grid))
    flags = {
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "sram_bytes": args.sram_bytes,
        "macs_per_cycle": args.macs_per_cycle,
        "memory_bytes_per_cycle": args.memory_bytes_per_cycle,
        "memory_latency": args.memory_latency,
        "pipeline_cycles": args.pipeline_cycles,
        "weight_bits": args.weight_bits,
        "timing": "event",
    }
    report, rows = search(
        cli.simulator(args),
        [args.m, args.n, args.k],
        candidates,
        flags,
        args.objective,
        args.timeout,
    )
    if args.output_dir:
        jsonio.write_report(args.output_dir, "autotune", report, rows)
    print(jsonio.dumps(report) if args.format == "json" else summary(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
