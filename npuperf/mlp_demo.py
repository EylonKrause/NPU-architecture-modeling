"""Run a two-layer INT8 MLP through npu-sim with tensors from NPU1 files and check both
layers against hand-computed outputs."""
from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from . import cli, jsonio, npufile, sim

COMMON = {"input_zero": 0, "weight_zero": 0, "output_zero": 0, "multiplier": 1,
          "dump_output": True, "timing": "event"}


@cli.main_guard("mlp_demo")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("mlp_demo", __doc__)
    cli.add_binary_options(parser)
    args = parser.parse_args(argv)
    binary = cli.simulator(args)
    layers = []
    with tempfile.TemporaryDirectory(prefix="npu-mlp-") as directory:
        first = npufile.write_problem(Path(directory) / "first.npu", 2, 3, 4,
                                      [4, -3, 2, 1, -2, 5, 0, 3],
                                      [2, -1, 1, 1, 3, -2, -2, 1, 2, 0, -1, 1], [1, -2, 3])
        first_result = sim.run({"input": first, "shift": 2, "relu": True, **COMMON}, binary,
                               args.timeout)
        if first_result.get("outputs") != [1, 0, 5, 1, 3, 0]:
            raise sim.CheckError("first layer differs from the hand-computed output")
        second = npufile.write_problem(Path(directory) / "second.npu", 2, 2, 3,
                                       first_result["outputs"], [2, -1, -1, 2, 1, 1], [0, 1])
        second_result = sim.run({"input": second, "shift": 1, **COMMON}, binary, args.timeout)
        if second_result.get("outputs") != [4, 3, -1, 3]:
            raise sim.CheckError("second layer differs from the hand-computed output")
        for name, result in (("dense_relu", first_result), ("dense_output", second_result)):
            layers.append({"name": name, "numerics": result.numerics,
                           "cycles": result.cycles, "read_bytes": result["read_bytes"],
                           "write_bytes": result["write_bytes"],
                           "output_hash": result["output_hash"], "outputs": result["outputs"]})
    print(jsonio.dumps({"schema_version": 1, "workload": "two_layer_int8_mlp", "layers": layers,
                        "outputs": second_result["outputs"],
                        "sum_layer_cycles": sum(layer["cycles"] for layer in layers),
                        "layer_overlap": False}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
