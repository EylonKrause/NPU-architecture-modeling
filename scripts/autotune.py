#!/usr/bin/env python3
"""Exhaustive bounded mapping search with explicit SRAM constraints."""

import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

from sweep import positive_int, resolve_binary, run_model, write_reports


def options(text, allow_zero=False):
    try:
        values = sorted(set(int(item) for item in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(x < (0 if allow_zero else 1) or x > 4096 for x in values):
        raise argparse.ArgumentTypeError("tile options must be bounded dimensions")
    return values


def search(binary, dimensions, candidates, base_flags, objective="latency", timeout=60):
    if len(candidates) > 256 or not candidates:
        raise ValueError("mapping search requires one through 256 candidates")
    rows, runs, feasible, output_hash = [], [], [], None
    for tile_m, tile_n, tile_k, slots in candidates:
        flags = list(base_flags) + ["--tile-m", tile_m, "--tile-n", tile_n,
                                   "--tile-k", tile_k, "--buffer-slots", slots]
        mapping = {"tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k, "buffer_slots": slots}
        try:
            run = run_model(binary, flags, timeout)
        except RuntimeError as error:
            if "tile buffers exceed configured SRAM capacity" not in str(error):
                raise
            rows.append({**mapping, "feasible": False, "cycles": None,
                         "scratchpad_bytes": None, "output_hash": None})
            runs.append({"mapping": mapping, "feasible": False, "reason": "SRAM capacity"})
            continue
        result = run["result"]
        if output_hash is None:
            output_hash = result["output_hash"]
        if output_hash != result["output_hash"]:
            raise ValueError("a mapping changed the numerical result")
        entry = {**mapping, "feasible": True, "cycles": result["cycles"],
                 "scratchpad_bytes": result["scratchpad_bytes"], "output_hash": result["output_hash"]}
        rows.append(entry)
        feasible.append(entry)
        runs.append({"mapping": mapping, "feasible": True, **run})
    if not feasible:
        raise ValueError("no feasible mapping; increase SRAM or choose smaller tiles")
    def rank(entry):
        return ((entry["cycles"], entry["scratchpad_bytes"]) if objective == "latency" else
                (entry["scratchpad_bytes"], entry["cycles"]))
    best = min(feasible, key=rank)
    pareto = [a for a in feasible if not any(
        b["cycles"] <= a["cycles"] and b["scratchpad_bytes"] <= a["scratchpad_bytes"]
        and (b["cycles"] < a["cycles"] or b["scratchpad_bytes"] < a["scratchpad_bytes"])
        for b in feasible)]
    report = {"schema_version": 1, "experiment": "exhaustive_mapping_search",
              "objective": objective, "shape": dimensions, "candidate_count": len(candidates),
              "feasible_count": len(feasible), "best": best, "pareto": pareto,
              "interpretation": "Optimal only within this enumerated mapping set and timing contract.",
              "runs": runs}
    return report, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="build/npu-sim")
    p.add_argument("--m", type=positive_int, default=16)
    p.add_argument("--n", type=positive_int, default=16)
    p.add_argument("--k", type=positive_int, default=32)
    p.add_argument("--tile-m-options", type=options, default=[2, 4, 8])
    p.add_argument("--tile-n-options", type=options, default=[2, 4, 8])
    p.add_argument("--tile-k-options", type=lambda s: options(s, True), default=[0, 8, 16])
    p.add_argument("--slot-options", type=options, default=[1, 2])
    p.add_argument("--sram-bytes", type=positive_int, default=262144)
    p.add_argument("--macs-per-cycle", type=positive_int, default=64)
    p.add_argument("--memory-bytes-per-cycle", type=positive_int, default=16)
    p.add_argument("--memory-latency", type=int, default=12)
    p.add_argument("--pipeline-cycles", type=int, default=4)
    p.add_argument("--weight-bits", type=int, choices=[4, 8], default=8)
    p.add_argument("--objective", choices=["latency", "sram"], default="latency")
    p.add_argument("--timeout", type=positive_int, default=60)
    p.add_argument("--output-dir")
    args = p.parse_args()
    if args.memory_latency < 0 or args.pipeline_cycles < 0:
        p.error("latencies must be nonnegative")
    if any(x > 64 for x in args.slot_options):
        p.error("slots must be at most 64")
    if math.prod(map(len, (args.tile_m_options, args.tile_n_options,
                          args.tile_k_options, args.slot_options))) > 256:
        p.error("mapping search requires one through 256 candidates")
    candidates = list(itertools.product(args.tile_m_options, args.tile_n_options,
                                        args.tile_k_options, args.slot_options))
    flags = ["--m", args.m, "--n", args.n, "--k", args.k, "--sram-bytes", args.sram_bytes,
             "--macs-per-cycle", args.macs_per_cycle,
             "--memory-bytes-per-cycle", args.memory_bytes_per_cycle,
             "--memory-latency", args.memory_latency, "--pipeline-cycles", args.pipeline_cycles,
             "--weight-bits", args.weight_bits, "--timing", "event"]
    report, rows = search(resolve_binary(args.binary), [args.m, args.n, args.k], candidates,
                          flags, args.objective, args.timeout)
    write_reports(args.output_dir, "autotune", report, rows)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print("autotune: " + str(error), file=sys.stderr)
        sys.exit(1)
