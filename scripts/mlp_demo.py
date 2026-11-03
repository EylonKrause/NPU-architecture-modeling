#!/usr/bin/env python3
"""Run a deterministic two-layer INT8 network using the file-input interface."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from sweep import positive_int, resolve_binary, run_model


def write_problem(path, m, n, k, activations, weights, bias):
    if len(activations) != m * k or len(weights) != k * n or len(bias) != n:
        raise ValueError("invalid layer shape")
    values = ["NPU1 {} {} {}".format(m, n, k),
              " ".join(map(str, activations)), " ".join(map(str, weights)),
              " ".join(map(str, bias))]
    path.write_text("\n".join(values) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/npu-sim")
    parser.add_argument("--timeout", type=positive_int, default=60)
    args = parser.parse_args()
    binary = resolve_binary(args.binary)
    common = ["--input-zero", 0, "--weight-zero", 0, "--output-zero", 0,
              "--multiplier", 1, "--dump-output", "--timing", "event"]
    layers = []
    with tempfile.TemporaryDirectory(prefix="npu-mlp-") as directory:
        first = Path(directory) / "first.npu"
        write_problem(first, 2, 3, 4, [4, -3, 2, 1, -2, 5, 0, 3],
                      [2, -1, 1, 1, 3, -2, -2, 1, 2, 0, -1, 1], [1, -2, 3])
        first_run = run_model(binary, ["--input", first, "--shift", 2, "--relu"] + common, args.timeout)
        first_result = first_run["result"]
        if first_result.get("outputs") != [1, 0, 5, 1, 3, 0]:
            raise ValueError("first layer differs from independently calculated expected output")
        second = Path(directory) / "second.npu"
        write_problem(second, 2, 2, 3, first_result["outputs"], [2, -1, -1, 2, 1, 1], [0, 1])
        second_run = run_model(binary, ["--input", second, "--shift", 1] + common, args.timeout)
        second_result = second_run["result"]
        if second_result.get("outputs") != [4, 3, -1, 3]:
            raise ValueError("second layer differs from independently calculated expected output")
        for name, result in (("dense_relu", first_result), ("dense_output", second_result)):
            layers.append({"name": name, "numerical_match": result["numerical_match"],
                           "cycles": result["cycles"], "read_bytes": result["read_bytes"],
                           "write_bytes": result["write_bytes"], "output_hash": result["output_hash"],
                           "outputs": result["outputs"]})
    print(json.dumps({"schema_version": 1, "workload": "two_layer_int8_mlp",
                      "layers": layers, "outputs": second_result["outputs"],
                      "sum_layer_cycles": sum(layer["cycles"] for layer in layers),
                      "boundary": "Separate simulator invocations; no cross-layer overlap or retained SRAM is modeled."},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print("mlp_demo: " + str(error), file=sys.stderr)
        sys.exit(1)
