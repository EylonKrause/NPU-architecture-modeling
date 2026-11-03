#!/usr/bin/env python3
"""Check identical modeled results and measure event/cycle simulator host runtime."""

import argparse
import json
import statistics
import subprocess
import sys

from sweep import geomean, positive_int, resolve_binary, run_model, write_reports


ARCHITECTURE_FIELDS = (
    "cycles", "macs", "read_bytes", "write_bytes", "compute_busy_cycles",
    "dma_busy_cycles", "peak_inflight_tiles", "slot_wait_cycles", "scratchpad_bytes",
    "output_hash", "numerical_match", "compute_utilization", "memory_utilization",
    "modeled_latency_ns",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/npu-sim")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--timeout", type=positive_int, default=60, help="seconds per subprocess")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    binary = resolve_binary(args.binary)
    workloads = [(17, 13, 31), (64, 64, 128), (128, 128, 256)]
    cases, rows, wall_ratios, simulation_ratios = [], [], [], []
    for m, n, k in workloads:
        samples = {"event": [], "cycle": []}
        reference = None
        base = ["--m", m, "--n", n, "--k", k, "--tile-m", 8, "--tile-n", 8,
                "--seed", 1, "--buffer-slots", 2]
        for repetition in range(args.repeats):
            # Alternate order to reduce a systematic warmup/order bias.
            modes = ("event", "cycle") if repetition % 2 == 0 else ("cycle", "event")
            for mode in modes:
                run = run_model(binary, base + ["--timing", mode], args.timeout)
                result = run["result"]
                current = {key: result[key] for key in ARCHITECTURE_FIELDS}
                if reference is None:
                    reference = current
                elif current != reference:
                    different = [key for key in ARCHITECTURE_FIELDS if current[key] != reference[key]]
                    raise ValueError("timing modes or repeats disagree: " + ", ".join(different))
                samples[mode].append(run)
        medians = {}
        for mode in ("event", "cycle"):
            medians[mode] = {
                "subprocess_wall_seconds": statistics.median(run["subprocess_wall_seconds"] for run in samples[mode]),
                "simulation_wall_seconds": statistics.median(run["result"]["simulation_wall_seconds"] for run in samples[mode]),
            }
        wall_ratio = medians["cycle"]["subprocess_wall_seconds"] / medians["event"]["subprocess_wall_seconds"]
        event_simulation = medians["event"]["simulation_wall_seconds"]
        cycle_simulation = medians["cycle"]["simulation_wall_seconds"]
        simulation_ratio = cycle_simulation / event_simulation if event_simulation > 0 else None
        wall_ratios.append(wall_ratio)
        if simulation_ratio is not None and simulation_ratio > 0:
            simulation_ratios.append(simulation_ratio)
        cases.append({"shape": [m, n, k], "architecture_agreement": True,
                      "architecture": reference, "median_seconds": medians,
                      "cycle_over_event_subprocess_wall_ratio": wall_ratio,
                      "cycle_over_event_simulation_wall_ratio": simulation_ratio,
                      "samples": samples})
        rows.append({"m": m, "n": n, "k": k, "repeats": args.repeats,
                     "cycles": reference["cycles"], "output_hash": reference["output_hash"],
                     "event_subprocess_wall_median_seconds": medians["event"]["subprocess_wall_seconds"],
                     "cycle_subprocess_wall_median_seconds": medians["cycle"]["subprocess_wall_seconds"],
                     "cycle_over_event_subprocess_wall_ratio": wall_ratio,
                     "event_simulation_wall_median_seconds": event_simulation,
                     "cycle_simulation_wall_median_seconds": cycle_simulation,
                     "cycle_over_event_simulation_wall_ratio": simulation_ratio})
    report = {"schema_version": 1, "experiment": "timing_mode_host_runtime",
              "interpretation": "Same modeled architecture; subprocess time includes startup. Ratios describe this host run, not accelerator speed.",
              "repeats": args.repeats,
              "geomean_cycle_over_event_subprocess_wall_ratio": geomean(wall_ratios),
              "geomean_cycle_over_event_simulation_wall_ratio": geomean(simulation_ratios) if len(simulation_ratios) == len(workloads) else None,
              "cases": cases}
    write_reports(args.output_dir, "compare_modes", report, rows)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print("compare_modes: " + str(error), file=sys.stderr)
        sys.exit(1)
