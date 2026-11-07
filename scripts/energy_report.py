#!/usr/bin/env python3
"""Estimate selected energy components from a validated simulator result.

Coefficients are mandatory user inputs, not measured silicon defaults. Only
MACs, external-memory bytes and an optional constant static-power term are
included. The standard library is the only dependency.
"""

import argparse
import json
import math
from pathlib import Path
import sys


UINT64_MAX = (1 << 64) - 1
MAX_INPUT_BYTES = 64 * 1024 * 1024
LIMITATIONS = [
    "User-supplied coefficients are assumptions unless independently calibrated.",
    "Counts come from the simulator's architecture contract, not hardware counters.",
    "Excludes SRAM/register accesses, clock trees, interconnect, DMA control and decompression energy.",
    "Excludes host execution, process launches and work outside the reported accelerator invocation.",
    "Static power is constant; no voltage, frequency-scaling, temperature or thermal feedback model.",
    "Average power is an estimate over modeled latency, not a measured device power reading.",
]


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a finite nonnegative number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " is outside finite floating-point range") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(name + " must be a finite nonnegative number")
    return number


def uint64(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT64_MAX:
        raise ValueError(name + " must be a nonnegative uint64 integer")
    if positive and value == 0:
        raise ValueError(name + " must be positive")
    return value


def estimate(result, mac_pj, dram_byte_pj, static_mw=0.0):
    if not isinstance(result, dict) or type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise ValueError("expected a simulator JSON object with schema_version 1")
    if result.get("numerical_match") is not True:
        raise ValueError("simulator must report numerical_match true")
    cycles = uint64(result.get("cycles"), "cycles", positive=True)
    macs = uint64(result.get("macs"), "macs")
    executed_macs = uint64(result.get("executed_macs", macs), "executed_macs")
    skipped_macs = uint64(result.get("skipped_macs", macs - executed_macs), "skipped_macs")
    if executed_macs > macs or skipped_macs != macs - executed_macs:
        raise ValueError("executed_macs plus skipped_macs must equal logical macs")
    read_bytes = uint64(result.get("read_bytes"), "read_bytes")
    write_bytes = uint64(result.get("write_bytes"), "write_bytes")
    uint64(result.get("output_hash"), "output_hash")
    if read_bytes > UINT64_MAX - write_bytes:
        raise ValueError("total external-memory bytes exceed uint64")
    config = result.get("config")
    if not isinstance(config, dict):
        raise ValueError("simulator result requires a config object")
    # npu-sim nests the clock under "arch"; hand-written results may keep it flat.
    clock = config.get("arch") if isinstance(config.get("arch"), dict) else config
    frequency = finite_nonnegative(clock.get("frequency_mhz"), "frequency_mhz")
    latency = finite_nonnegative(result.get("modeled_latency_ns"), "modeled_latency_ns")
    if frequency == 0 or latency == 0:
        raise ValueError("frequency_mhz and modeled_latency_ns must be positive")
    expected_latency = cycles * 1000.0 / frequency
    if not math.isfinite(expected_latency) or not math.isclose(latency, expected_latency, rel_tol=1e-12):
        raise ValueError("modeled_latency_ns disagrees with cycles and frequency_mhz")
    mac_cost = finite_nonnegative(mac_pj, "mac_pj")
    dram_cost = finite_nonnegative(dram_byte_pj, "dram_byte_pj")
    static_power = finite_nonnegative(static_mw, "static_mw")
    components = {
        "mac": executed_macs * mac_cost,
        "dram_read": read_bytes * dram_cost,
        "dram_write": write_bytes * dram_cost,
        "static": latency * static_power,
    }
    if any(not math.isfinite(value) for value in components.values()):
        raise ValueError("energy component overflows finite floating-point range")
    try:
        total = math.fsum(components.values())
    except OverflowError as error:
        raise ValueError("total energy overflows finite floating-point range") from error
    power = total / latency
    if not math.isfinite(total) or not math.isfinite(power):
        raise ValueError("total energy or average power overflows finite floating-point range")
    return {
        "schema_version": 1,
        "model": "user_coefficients_mac_external_bytes_constant_static",
        "numerical_match": True,
        "source_output_hash": result["output_hash"],
        "source_counts": {"cycles": cycles, "macs": macs, "executed_macs": executed_macs,
                          "skipped_macs": skipped_macs, "read_bytes": read_bytes,
                          "write_bytes": write_bytes},
        "coefficients": {"mac_pj": mac_cost, "dram_byte_pj": dram_cost,
                         "static_mw": static_power},
        "component_pj": components,
        "total_pj": total,
        "total_nj": total / 1000.0,
        "modeled_latency_ns": latency,
        "average_power_mw": power,
        "limitations": LIMITATIONS,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("non-finite JSON constant is not permitted: " + value)


def finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON floating-point value is outside finite range")
    return number


def load_result(path):
    path = Path(path)
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("simulator JSON exceeds the 64 MiB input limit")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant, parse_float=finite_json_float,
                         object_pairs_hook=unique_object)


def coefficient(text):
    try:
        return finite_nonnegative(float(text), "coefficient")
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="npu-sim schema-version-1 JSON result")
    parser.add_argument("--mac-pj", type=coefficient, required=True, help="user-supplied pJ per modeled MAC")
    parser.add_argument("--dram-byte-pj", type=coefficient, required=True, help="user-supplied pJ per external-memory byte")
    parser.add_argument("--static-mw", type=coefficient, default=0.0, help="constant static power in mW (default 0, excluded)")
    parser.add_argument("--output", type=Path, help="optional new output file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = estimate(load_result(args.result), args.mac_pj, args.dram_byte_pj, args.static_mw)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        sys.stdout.write(encoded)
    except (OSError, ValueError, OverflowError, RecursionError) as error:
        parser.exit(2, "energy_report: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
