"""Estimate MAC, DRAM and static energy for one npu-sim result.

    E [pJ] = executed MACs x pJ/MAC + (read + write bytes) x pJ/byte
             + static mW x modeled latency ns            (1 mW x 1 ns = 1 pJ)

The coefficients are inputs. The report's "scope" lists the included and excluded terms.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import cli, jsonio, sim

UINT64_MAX = (1 << 64) - 1
SCOPE = {
    "included": ["mac", "dram_read", "dram_write", "static"],
    "excluded": [
        "sram",
        "register_file",
        "clock_tree",
        "interconnect",
        "dma_control",
        "weight_decode",
        "host",
    ],
}


def finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a finite nonnegative number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " is outside finite floating-point range") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(name + " must be a finite nonnegative number")
    return number


def uint64(value: Any, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT64_MAX:
        raise ValueError(name + " must be a nonnegative uint64 integer")
    if positive and value == 0:
        raise ValueError(name + " must be positive")
    return value


def estimate(result: Any, mac_pj: float, dram_byte_pj: float, static_mw: float = 0.0) -> dict:
    if (
        not isinstance(result, dict)
        or type(result.get("schema_version")) is not int
        or result["schema_version"] not in sim.RESULT_SCHEMA_VERSIONS
    ):
        raise ValueError("expected a simulator JSON object with schema_version 1 or 2")
    numerics = sim.result_numerics(result)
    if numerics is None:
        raise ValueError(
            'simulator must report numerical_match true, or numerics "skipped" '
            "with numerical_match null"
        )
    skipped = numerics == "skipped"
    cycles = uint64(result.get("cycles"), "cycles", positive=True)
    macs = uint64(result.get("macs"), "macs")
    executed_macs = uint64(result.get("executed_macs", macs), "executed_macs")
    skipped_macs = uint64(result.get("skipped_macs", macs - executed_macs), "skipped_macs")
    if executed_macs > macs or skipped_macs != macs - executed_macs:
        raise ValueError("executed_macs plus skipped_macs must equal logical macs")
    read_bytes = uint64(result.get("read_bytes"), "read_bytes")
    write_bytes = uint64(result.get("write_bytes"), "write_bytes")
    if not skipped:
        uint64(result.get("output_hash"), "output_hash")
    if read_bytes > UINT64_MAX - write_bytes:
        raise ValueError("total external-memory bytes exceed uint64")
    config = result.get("config")
    if not isinstance(config, dict):
        raise ValueError("simulator result requires a config object")
    frequency = finite_nonnegative(sim.config_value(config, "frequency_mhz", None), "frequency_mhz")
    latency = finite_nonnegative(result.get("modeled_latency_ns"), "modeled_latency_ns")
    if frequency == 0 or latency == 0:
        raise ValueError("frequency_mhz and modeled_latency_ns must be positive")
    expected_latency = cycles * 1000.0 / frequency
    if not math.isfinite(expected_latency) or not math.isclose(
        latency, expected_latency, rel_tol=1e-12
    ):
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
        "model": "mac_dram_static",
        "scope": SCOPE,
        "numerics": "skipped" if skipped else "bit-exact",
        "source_output_hash": result.get("output_hash"),
        "source_counts": {
            "cycles": cycles,
            "macs": macs,
            "executed_macs": executed_macs,
            "skipped_macs": skipped_macs,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
        },
        "coefficients": {"mac_pj": mac_cost, "dram_byte_pj": dram_cost, "static_mw": static_power},
        "component_pj": components,
        "component_share": {
            key: (value / total if total else 0.0) for key, value in components.items()
        },
        "total_pj": total,
        "total_nj": total / 1000.0,
        "modeled_latency_ns": latency,
        "average_power_mw": power,
    }


def load_result(path: Path | str) -> Any:
    return jsonio.load_strict(path)


def coefficient(text: str) -> float:
    try:
        return finite_nonnegative(float(text), "coefficient")
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


@cli.main_guard("energy_report")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("energy_report", __doc__)
    parser.add_argument("--result", type=Path, required=True, help="npu-sim JSON result")
    parser.add_argument("--mac-pj", type=coefficient, required=True, help="pJ per executed MAC")
    parser.add_argument(
        "--dram-byte-pj", type=coefficient, required=True, help="pJ per DRAM byte read or written"
    )
    parser.add_argument(
        "--static-mw", type=coefficient, default=0.0, help="constant static power in mW"
    )
    parser.add_argument(
        "--output", type=Path, help="also write the report here; an existing file needs --force"
    )
    parser.add_argument("--force", action="store_true", help="overwrite --output if it exists")
    args = parser.parse_args(argv)
    report = estimate(load_result(args.result), args.mac_pj, args.dram_byte_pj, args.static_mw)
    encoded = jsonio.dumps(report)
    if args.output is not None:
        jsonio.write_text(args.output, encoded, overwrite=args.force)
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
