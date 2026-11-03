#!/usr/bin/env python3
"""Reproducible paired architecture experiments, using only the standard library."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def resolve_binary(name):
    binary = Path(name).expanduser().resolve()
    if not binary.is_file():
        raise ValueError("binary does not exist: " + str(binary))
    return binary


def validate_result(result):
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError("expected a JSON object with schema_version 1")
    if result.get("numerical_match") is not True:
        raise ValueError("simulator did not report full numerical agreement")
    for key in ("cycles", "macs", "read_bytes", "write_bytes", "compute_busy_cycles",
                "dma_busy_cycles", "peak_inflight_tiles", "slot_wait_cycles",
                "scratchpad_bytes"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("invalid nonnegative integer result: " + key)
    if result["cycles"] == 0:
        raise ValueError("cycle count must be positive")
    for key in ("compute_busy_cycles", "dma_busy_cycles"):
        if result[key] > result["cycles"]:
            raise ValueError("resource busy cycles exceed elapsed cycles: " + key)
    for key in ("simulation_wall_seconds", "compute_utilization", "memory_utilization",
                "modeled_latency_ns"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid finite nonnegative result: " + key)
    for key in ("compute_utilization", "memory_utilization"):
        if result[key] > 1:
            raise ValueError("utilization must be a fraction: " + key)
    if not isinstance(result.get("config"), dict):
        raise ValueError("missing configuration object")
    output_hash = result.get("output_hash")
    if isinstance(output_hash, bool) or not isinstance(output_hash, (str, int)):
        raise ValueError("missing deterministic output hash")
    return result


def run_model(binary, flags, timeout=60):
    command = [str(binary)] + [str(flag) for flag in flags]
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError("simulator exited with status {}: {}".format(
            process.returncode, process.stderr.strip()[:2000]))
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("stdout must contain exactly one JSON result") from error
    validate_result(result)
    return {"command": command, "subprocess_wall_seconds": elapsed, "result": result}


def geomean(ratios):
    if not ratios or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("geometric mean requires nonempty positive finite ratios")
    return math.exp(math.fsum(math.log(value) for value in ratios) / len(ratios))


def write_reports(directory, name, report, rows):
    if directory is None:
        return
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / (name + ".json")
    csv_path = destination / (name + ".csv")
    # Refuse to silently overwrite a previous experiment.
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("report already exists; choose a new output directory")
    if not rows:
        raise ValueError("no report rows")
    json_text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(json_text)
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(csv_stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/npu-sim")
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout", type=positive_int, default=60, help="seconds per subprocess")
    args = parser.parse_args()
    binary = resolve_binary(args.binary)
    workloads = [(16, 16, 32), (17, 13, 31), (64, 64, 128)]
    variants = [
        ("serial_baseline", 1, 64, 16),
        ("overlap", 2, 64, 16),
        ("more_compute", 2, 128, 16),
        ("more_bandwidth", 2, 64, 32),
        ("balanced", 2, 128, 32),
    ]
    runs, rows = [], []
    ratios = {name: [] for name, *_ in variants if name != "serial_baseline"}
    baseline_names = {"serial_baseline": "serial_baseline", "overlap": "serial_baseline",
                      "more_compute": "overlap", "more_bandwidth": "overlap", "balanced": "overlap"}
    for m, n, k in workloads:
        results_by_variant = {}
        for name, slots, macs, bandwidth in variants:
            flags = ["--m", m, "--n", n, "--k", k, "--tile-m", 8, "--tile-n", 8,
                     "--seed", 1, "--timing", "event", "--buffer-slots", slots,
                     "--macs-per-cycle", macs, "--memory-bytes-per-cycle", bandwidth]
            run = run_model(binary, flags, args.timeout)
            result = run["result"]
            results_by_variant[name] = result
            serial = results_by_variant["serial_baseline"]
            if result["output_hash"] != serial["output_hash"]:
                raise ValueError("architecture variant changed the arithmetic output")
            baseline_name = baseline_names[name]
            baseline = results_by_variant[baseline_name]
            ratio = baseline["cycles"] / result["cycles"]
            if name in ratios:
                ratios[name].append(ratio)
            runs.append({"variant": name, "baseline_variant": baseline_name, "shape": [m, n, k], **run})
            rows.append({"m": m, "n": n, "k": k, "variant": name,
                         "baseline_variant": baseline_name,
                         "buffer_slots": slots, "macs_per_cycle": macs,
                         "memory_bytes_per_cycle": bandwidth, "cycles": result["cycles"],
                         "baseline_cycles": baseline["cycles"], "baseline_over_candidate_cycles": ratio,
                         "serial_over_candidate_cycles": serial["cycles"] / result["cycles"],
                         "read_bytes": result["read_bytes"], "write_bytes": result["write_bytes"],
                         "compute_busy_cycles": result["compute_busy_cycles"],
                         "dma_busy_cycles": result["dma_busy_cycles"],
                         "output_hash": result["output_hash"], "numerical_match": True})
    report = {"schema_version": 1, "experiment": "paired_architecture_cycles",
              "interpretation": "Modeled cycle ratios under one timing contract; not hardware measurements.",
              "pair_count_per_variant": len(workloads),
              "geomean_baseline_over_candidate_cycles": {name: geomean(values) for name, values in ratios.items()},
              "runs": runs}
    write_reports(args.output_dir, "sweep", report, rows)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print("sweep: " + str(error), file=sys.stderr)
        sys.exit(1)
