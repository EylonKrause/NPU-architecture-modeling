"""Check that event- and cycle-driven timing give identical results, and compare their host
run time."""
from __future__ import annotations

import statistics
from collections.abc import Sequence
from pathlib import Path

from . import cli, jsonio, sim, table
from .sweep import geomean

SHAPES = ((17, 13, 31), (64, 64, 128), (128, 128, 256))
ARCHITECTURE_FIELDS = (
    "cycles", "macs", "read_bytes", "write_bytes", "compute_busy_cycles",
    "dma_busy_cycles", "peak_inflight_tiles", "slot_wait_cycles", "scratchpad_bytes",
    "output_hash", "numerical_match", "compute_utilization", "memory_utilization",
    "modeled_latency_ns",
)


def compare(binary: Path, repeats: int, timeout: float = 60) -> tuple[dict, list[dict]]:
    cases, rows, wall_ratios, simulation_ratios = [], [], [], []
    for m, n, k in SHAPES:
        samples: dict[str, list[dict]] = {"event": [], "cycle": []}
        reference = None
        base = {"m": m, "n": n, "k": k, "tile_m": 8, "tile_n": 8, "seed": 1, "buffer_slots": 2}
        for repetition in range(repeats):
            # Alternate the order so warm-up does not favour one mode.
            for mode in (("event", "cycle") if repetition % 2 == 0 else ("cycle", "event")):
                result = sim.run({**base, "timing": mode}, binary, timeout)
                current = {key: result.get(key) for key in ARCHITECTURE_FIELDS}
                if reference is None:
                    reference = current
                elif current != reference:
                    different = [key for key in ARCHITECTURE_FIELDS
                                 if current[key] != reference[key]]
                    raise sim.CheckError("timing modes or repeats disagree: " + ", ".join(different))
                samples[mode].append(result.as_run())
        assert reference is not None
        medians = {mode: {
            "subprocess_wall_seconds": statistics.median(r["subprocess_wall_seconds"]
                                                         for r in samples[mode]),
            "simulation_wall_seconds": statistics.median(r["result"]["simulation_wall_seconds"]
                                                         for r in samples[mode])}
            for mode in ("event", "cycle")}
        wall_ratio = (medians["cycle"]["subprocess_wall_seconds"]
                      / medians["event"]["subprocess_wall_seconds"])
        event_sim = medians["event"]["simulation_wall_seconds"]
        cycle_sim = medians["cycle"]["simulation_wall_seconds"]
        simulation_ratio = cycle_sim / event_sim if event_sim > 0 else None
        wall_ratios.append(wall_ratio)
        if simulation_ratio:
            simulation_ratios.append(simulation_ratio)
        cases.append({"shape": [m, n, k], "architecture_agreement": True,
                      "architecture": reference, "median_seconds": medians,
                      "cycle_over_event_subprocess_wall_ratio": wall_ratio,
                      "cycle_over_event_simulation_wall_ratio": simulation_ratio,
                      "samples": samples})
        rows.append({"m": m, "n": n, "k": k, "repeats": repeats, "cycles": reference["cycles"],
                     "output_hash": reference["output_hash"],
                     "event_subprocess_wall_median_seconds": medians["event"]["subprocess_wall_seconds"],
                     "cycle_subprocess_wall_median_seconds": medians["cycle"]["subprocess_wall_seconds"],
                     "cycle_over_event_subprocess_wall_ratio": wall_ratio,
                     "event_simulation_wall_median_seconds": event_sim,
                     "cycle_simulation_wall_median_seconds": cycle_sim,
                     "cycle_over_event_simulation_wall_ratio": simulation_ratio})
    report = {"schema_version": 1, "experiment": "timing_mode_host_runtime", "repeats": repeats,
              "geomean_cycle_over_event_subprocess_wall_ratio": geomean(wall_ratios),
              "geomean_cycle_over_event_simulation_wall_ratio":
                  geomean(simulation_ratios) if len(simulation_ratios) == len(SHAPES) else None,
              "cases": cases}
    return report, rows


def summary(rows: list[dict]) -> str:
    view = [{"shape": f"{r['m']}x{r['n']}x{r['k']}", "cycles": r["cycles"],
             "event s": r["event_simulation_wall_median_seconds"],
             "cycle s": r["cycle_simulation_wall_median_seconds"],
             "cycle/event": r["cycle_over_event_simulation_wall_ratio"]} for r in rows]
    return ("Identical cycles, traffic and outputs in both modes; host seconds are medians.\n"
            + table.format_table(view, formats={"event s": ".4f", "cycle s": ".4f",
                                                "cycle/event": ".2f"}))


@cli.main_guard("compare_modes")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("compare_modes", __doc__)
    cli.add_binary_options(parser)
    cli.add_format_option(parser)
    parser.add_argument("--repeats", type=cli.positive_int, default=3, help="runs per mode")
    parser.add_argument("--output-dir", type=Path, help="also write compare_modes.json and "
                        ".csv here (replaces earlier files)")
    args = parser.parse_args(argv)
    report, rows = compare(cli.simulator(args), args.repeats, args.timeout)
    if args.output_dir:
        jsonio.write_report(args.output_dir, "compare_modes", report, rows)
    print(jsonio.dumps(report) if args.format == "json" else summary(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
