"""Run the weight-compression, transformer and multicore examples and write a summary.

Compression cases run identical tensors dense and compressed and check both against a
Python integer reference. Results go to --output-dir (default results/mobile), replacing
the files of an earlier run. The transformer and mobile demos come from the build directory
that holds the chosen npu-sim ($NPU_TRANSFORMER_DEMO / $NPU_MOBILE_DEMO override them).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from . import binary as executables
from . import cli, jsonio, quant, sim, table
from .binary import ROOT


def compression_case(binary: Path, directory: Path, name: str, timeout: float) -> dict:
    m, n, k = 17, 13, 31
    a = [(index * 17 + 11) % 19 - 9 for index in range(m * k)]
    if name == "palette":
        w = [(-3, -1, 1, 3)[(index * 7 + index // n) % 4] for index in range(k * n)]
        encoding = {"weight_encoding": "palette", "palette_bits": 2}
    else:
        w = [0 if (index // 8) % 4 else (index % 7) - 3 for index in range(k * n)]
        encoding = {"weight_encoding": "sparse", "sparse_block": 4}
    bias = [index - 6 for index in range(n)]
    path = directory / (name + ".npu")
    path.write_text(
        f"NPU1 {m} {n} {k}\n" + " ".join(map(str, a + w + bias)) + "\n", encoding="ascii"
    )
    flags = {
        "input": path,
        "input_zero": 0,
        "weight_zero": 0,
        "shift": 5,
        "tile_m": 8,
        "tile_n": 8,
        "tile_k": 8,
        "dump_output": True,
    }
    baseline = sim.run(flags, binary, timeout)
    compressed = sim.run({**flags, **encoding}, binary, timeout)
    accumulators, outputs = quant.gemm_reference(
        a, w, bias, m, n, k, quant={"multiplier": 1, "shift": 5, "output_zero": 0}
    )
    for result in (baseline, compressed):
        if result["accumulators"] != accumulators or result["outputs"] != outputs:
            raise sim.CheckError(name + " disagrees with the Python integer reference")
    if compressed["executed_macs"] + compressed["skipped_macs"] != m * n * k:
        raise sim.CheckError("compression work accounting disagrees with the logical shape")
    for variant, result in (("dense", baseline), (name, compressed)):
        jsonio.write_json(directory / f"{name}-{variant}.json", result.raw)
    keys = (
        "cycles",
        "read_bytes",
        "write_bytes",
        "executed_macs",
        "skipped_macs",
        "weight_decode_cycles",
        "encoded_weight_storage_bytes",
        "dense_weight_storage_bytes",
    )
    return {
        "reference_match": True,
        "baseline": {key: baseline[key] for key in keys},
        "compressed": {key: compressed[key] for key in keys},
        "modeled_cycle_ratio": baseline.cycles / compressed.cycles,
    }


def run_suite(binaries: dict[str, Path], output_dir: Path, timeout: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1,
        "compression": {
            name: compression_case(binaries["npu-sim"], output_dir, name, timeout)
            for name in ("palette", "sparse")
        },
        "transformer": {},
        "multicore": {},
    }
    for precision in ("fp32", "fp16", "bf16"):
        result, _, _ = sim.execute(
            binaries["npu-transformer-demo"],
            {
                "precision": precision,
                "prefill": 4,
                "decode": 3,
                "verify_lifecycle": True,
                "dump_output": True,
            },
            timeout,
        )
        if not result.get("partition_match") or result.get("tokens_completed") != 7:
            raise sim.CheckError("transformer partition validation failed")
        report["transformer"][precision] = result
    for cores in (1, 2):
        result, _, _ = sim.execute(
            binaries["npu-mobile-demo"],
            {"cores": cores, "jobs": 6, "opps": "200:800", "opp": 0, "dump_output": True},
            timeout,
        )
        if not result.get("numerical_match") or result.get("jobs_completed") != 6:
            raise sim.CheckError("multicore validation failed")
        report["multicore"][str(cores)] = result
    serial, parallel = (report["multicore"][str(cores)] for cores in (1, 2))
    if [job["outputs"] for job in serial["jobs"]] != [job["outputs"] for job in parallel["jobs"]]:
        raise sim.CheckError("multicore job outputs differ from serial execution")
    report["multicore_modeled_makespan_ratio"] = serial["simulation_ns"] / parallel["simulation_ns"]
    jsonio.write_json(output_dir / "summary.json", report)
    return report


def summary(report: dict, path: Path) -> str:
    rows = [
        {
            "case": f"{name} weights",
            "dense cycles": case["baseline"]["cycles"],
            "compressed cycles": case["compressed"]["cycles"],
            "dense/compressed": case["modeled_cycle_ratio"],
        }
        for name, case in report["compression"].items()
    ]
    return (
        table.format_table(rows, formats={"dense/compressed": ".3f"})
        + f"\ntransformer precisions: {', '.join(report['transformer'])}; "
        f"1-core/2-core makespan: {report['multicore_modeled_makespan_ratio']:.3f}\n"
        f"summary: {path}\n"
    )


@cli.main_guard("mobile_suite")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("mobile_suite", __doc__)
    cli.add_binary_options(parser)
    cli.add_format_option(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "mobile",
        help="directory for the per-case JSON files and summary.json",
    )
    args = parser.parse_args(argv)
    binaries = {"npu-sim": cli.simulator(args)}
    build = binaries["npu-sim"].parent
    if build.name == "Release":
        build = build.parent
    for name in ("npu-transformer-demo", "npu-mobile-demo"):
        binaries[name] = (
            executables.find_simulator(name)
            if os.environ.get(executables.environment_variable(name))
            else executables.find_in(build, name)
        )
    report = run_suite(binaries, args.output_dir, args.timeout)
    path = args.output_dir / "summary.json"
    if args.format == "json":
        print(
            jsonio.dumps(
                {
                    "output": str(path),
                    "compression": report["compression"],
                    "transformer_precisions": list(report["transformer"]),
                    "multicore_modeled_makespan_ratio": report["multicore_modeled_makespan_ratio"],
                }
            ),
            end="",
        )
    else:
        print(summary(report, path), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
