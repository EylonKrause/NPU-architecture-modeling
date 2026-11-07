"""Sweep buffer slots, MAC throughput and DRAM bandwidth over fixed GEMM shapes and report
cycle speedups. Every variant must reproduce the baseline's output hash."""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from . import analytic, cli, jsonio, sim, table

SHAPES = ((16, 16, 32), (17, 13, 31), (64, 64, 128))
# (name, buffer slots, MACs per cycle, DRAM bytes per cycle)
VARIANTS = (
    ("serial_baseline", 1, 64, 16),
    ("overlap", 2, 64, 16),
    ("more_compute", 2, 128, 16),
    ("more_bandwidth", 2, 64, 32),
    ("balanced", 2, 128, 32),
)
BASELINES = {"serial_baseline": "serial_baseline", "overlap": "serial_baseline",
             "more_compute": "overlap", "more_bandwidth": "overlap", "balanced": "overlap"}


def geomean(ratios: Sequence[float]) -> float:
    if not ratios or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("geometric mean requires nonempty positive finite ratios")
    return math.exp(math.fsum(math.log(value) for value in ratios) / len(ratios))


def run_sweep(binary: Path, timeout: float = 60) -> tuple[dict, list[dict]]:
    runs, rows = [], []
    ratios: dict[str, list[float]] = {name: [] for name, *_ in VARIANTS if name != "serial_baseline"}
    for m, n, k in SHAPES:
        by_variant: dict[str, sim.SimResult] = {}
        for name, slots, macs, bandwidth in VARIANTS:
            flags = {"m": m, "n": n, "k": k, "tile_m": 8, "tile_n": 8, "seed": 1,
                     "timing": "event", "buffer_slots": slots, "macs_per_cycle": macs,
                     "memory_bytes_per_cycle": bandwidth}
            result = sim.run(flags, binary, timeout)
            by_variant[name] = result
            serial = by_variant["serial_baseline"]
            if result["output_hash"] != serial["output_hash"]:
                raise sim.CheckError(f"{name} changed the arithmetic output of {m}x{n}x{k}")
            baseline_name = BASELINES[name]
            baseline = by_variant[baseline_name]
            ratio = baseline.cycles / result.cycles
            if name in ratios:
                ratios[name].append(ratio)
            model = analytic.analyze(analytic.Design.from_result(result))
            runs.append({"variant": name, "baseline_variant": baseline_name, "shape": [m, n, k],
                         **result.as_run()})
            rows.append({"m": m, "n": n, "k": k, "variant": name,
                         "baseline_variant": baseline_name, "buffer_slots": slots,
                         "macs_per_cycle": macs, "memory_bytes_per_cycle": bandwidth,
                         "cycles": result.cycles, "baseline_cycles": baseline.cycles,
                         "baseline_over_candidate_cycles": ratio,
                         "serial_over_candidate_cycles": serial.cycles / result.cycles,
                         "read_bytes": result["read_bytes"], "write_bytes": result["write_bytes"],
                         "compute_busy_cycles": result["compute_busy_cycles"],
                         "dma_busy_cycles": result["dma_busy_cycles"], "bound": model.bound,
                         "output_hash": result["output_hash"],
                         "numerical_match": result["numerical_match"]})
    report = {"schema_version": 1, "experiment": "paired_architecture_cycles",
              "pair_count_per_variant": len(SHAPES),
              "geomean_baseline_over_candidate_cycles": {name: geomean(values)
                                                         for name, values in ratios.items()},
              "runs": runs}
    return report, rows


def summary(report: dict, rows: list[dict]) -> str:
    view = [{"shape": f"{r['m']}x{r['n']}x{r['k']}", "variant": r["variant"],
             "vs": r["baseline_variant"], "slots": r["buffer_slots"],
             "MAC/cycle": r["macs_per_cycle"], "B/cycle": r["memory_bytes_per_cycle"],
             "cycles": r["cycles"], "speedup": r["baseline_over_candidate_cycles"],
             "bound": r["bound"]} for r in rows]
    means = [{"variant": name, "geomean speedup": value}
             for name, value in report["geomean_baseline_over_candidate_cycles"].items()]
    return (table.format_table(view, formats={"speedup": ".3f"}) + "\n"
            + table.format_table(means, formats={"geomean speedup": ".3f"}))


@cli.main_guard("sweep")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("sweep", __doc__)
    cli.add_binary_options(parser)
    cli.add_format_option(parser)
    parser.add_argument("--output-dir", type=Path,
                        help="also write sweep.json and sweep.csv here (replaces earlier files)")
    args = parser.parse_args(argv)
    report, rows = run_sweep(cli.simulator(args), args.timeout)
    if args.output_dir:
        jsonio.write_report(args.output_dir, "sweep", report, rows)
    print(jsonio.dumps(report) if args.format == "json" else summary(report, rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
