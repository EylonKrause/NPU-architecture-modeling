#!/usr/bin/env python3
"""Run reproducible mobile-NPU examples through the actual C++/SystemC engines.

Compression pairs use identical tensors and a separate unbounded-integer oracle.
Reported ratios describe these declared model assumptions, never Apple silicon.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from sweep import positive_int, resolve_binary, run_model


def run_json(binary, flags, timeout):
    process = subprocess.run([str(binary), *map(str, flags)], capture_output=True,
                             text=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(binary.name + ": " + process.stderr[-2000:])
    result = json.loads(process.stdout)
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError("executable did not return a schema-version-one result")
    return result


def compression_case(binary, directory, name, timeout):
    m, n, k = 17, 13, 31
    a = [(index * 17 + 11) % 19 - 9 for index in range(m * k)]
    if name == "palette":
        w = [(-3, -1, 1, 3)[(index * 7 + index // n) % 4] for index in range(k * n)]
        encoding = ["--weight-encoding", "palette", "--palette-bits", "2"]
    else:
        w = [0 if (index // 8) % 4 else (index % 7) - 3 for index in range(k * n)]
        encoding = ["--weight-encoding", "sparse", "--sparse-block", "4"]
    bias = [index - 6 for index in range(n)]
    path = directory / (name + ".npu")
    path.write_text("NPU1 17 13 31\n" + " ".join(map(str, a + w + bias)) + "\n", encoding="ascii")
    flags = ["--input", path, "--input-zero", 0, "--weight-zero", 0, "--shift", 5,
             "--tile-m", 8, "--tile-n", 8, "--tile-k", 8, "--dump-output"]
    baseline = run_model(binary, flags, timeout)["result"]
    compressed = run_model(binary, flags + encoding, timeout)["result"]
    accumulators = [bias[col] + sum(a[row * k + i] * w[i * n + col] for i in range(k))
                    for row in range(m) for col in range(n)]
    def rounded(value):
        return max(-128, min(127, (-1 if value < 0 else 1) * ((abs(value) + 16) // 32)))
    outputs = list(map(rounded, accumulators))
    for result in (baseline, compressed):
        if result["accumulators"] != accumulators or result["outputs"] != outputs:
            raise ValueError(name + " disagrees with independent integer arithmetic")
    if compressed["executed_macs"] + compressed["skipped_macs"] != m * n * k:
        raise ValueError("compression work accounting disagrees with logical shape")
    for variant, result in (("dense", baseline), (name, compressed)):
        (directory / (name + "-" + variant + ".json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    keys = ("cycles", "read_bytes", "write_bytes", "executed_macs", "skipped_macs",
            "weight_decode_cycles", "encoded_weight_storage_bytes", "dense_weight_storage_bytes")
    return {"independent_outputs_match": True,
            "baseline": {key: baseline[key] for key in keys},
            "compressed": {key: compressed[key] for key in keys},
            "modeled_cycle_ratio": baseline["cycles"] / compressed["cycles"],
            "note": "Decode/metadata can outweigh compression gains; this ratio is specific to the declared fixture and model."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/mobile"))
    parser.add_argument("--timeout", type=positive_int, default=60)
    args = parser.parse_args()
    suffix = ".exe" if sys.platform == "win32" else ""
    binaries = {name: resolve_binary(args.build_dir / (name + suffix))
                for name in ("npu-sim", "npu-transformer-demo", "npu-mobile-demo")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "silicon_calibrated": False,
              "compression": {name: compression_case(binaries["npu-sim"], args.output_dir, name, args.timeout)
                              for name in ("palette", "sparse")}, "transformer": {}, "multicore": {}}
    for precision in ("fp32", "fp16", "bf16"):
        result = run_json(binaries["npu-transformer-demo"],
                          ["--precision", precision, "--prefill", 4, "--decode", 3,
                           "--verify-lifecycle", "--dump-output"], args.timeout)
        if not result.get("partition_match") or result.get("tokens_completed") != 7:
            raise ValueError("transformer partition validation failed")
        report["transformer"][precision] = result
    for cores in (1, 2):
        result = run_json(binaries["npu-mobile-demo"], ["--cores", cores, "--jobs", 6,
                          "--opps", "200:800", "--opp", 0, "--dump-output"], args.timeout)
        if not result.get("numerical_match") or result.get("jobs_completed") != 6:
            raise ValueError("multicore validation failed")
        report["multicore"][str(cores)] = result
    serial, parallel = (report["multicore"][str(cores)] for cores in (1, 2))
    if [job["outputs"] for job in serial["jobs"]] != [job["outputs"] for job in parallel["jobs"]]:
        raise ValueError("multicore job outputs differ from serial execution")
    report["multicore_modeled_makespan_ratio"] = serial["simulation_ns"] / parallel["simulation_ns"]
    report["validation_boundary"] = "Compression has a separate Python integer oracle. Other results run executable self-checks; the regression suite additionally supplies independent numerical and scheduling oracles."
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir / "summary.json"), "completed": True,
                      "compression": report["compression"],
                      "transformer_precisions": list(report["transformer"]),
                      "multicore_modeled_makespan_ratio": report["multicore_modeled_makespan_ratio"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        print("mobile_suite: " + str(error), file=sys.stderr)
        sys.exit(1)
