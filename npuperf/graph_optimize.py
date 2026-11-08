"""Fuse a ReLU into the preceding accelerator op when the intermediate tensor has no other
consumer and is not a graph output. ``--verify`` runs both graphs on npu-sim and compares
every output."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from . import cli, jsonio, sim
from .graph import load_graph, run_graph, validate_graph


ACCELERATED = {"Dense", "MatMul", "Conv2D", "DepthwiseConv2D"}


def optimize_graph(document):
    """Fold each ReLU into its accelerator producer when the producer's output has no other
    consumer and is not a graph output.

    ReLU after INT8 requantization equals the engine's fused clamp at the output zero point,
    with saturation and under either rounding rule, so every output is unchanged. The pass
    repeats until no ReLU folds, which collapses chains of ReLUs.
    """
    validate_graph(document)
    compiled = deepcopy(document)
    fusions = []
    while True:
        _, nodes, order, outputs, _ = validate_graph(compiled)
        uses = Counter(name for node in nodes for name in node["inputs"])
        producers = {node["output"]: index for index, node in enumerate(nodes)}
        removed = set()
        for index in order:
            node = nodes[index]
            if node["op"] != "ReLU":
                continue
            intermediate = node["inputs"][0]
            producer_index = producers.get(intermediate)
            if producer_index is None or producer_index in removed:
                continue
            producer = nodes[producer_index]
            if producer["op"] not in ACCELERATED or uses[intermediate] != 1 or intermediate in outputs:
                continue
            quant = producer.get("quant")
            if not isinstance(quant, dict):
                raise ValueError("accelerated nodes require a quant object")
            if "relu" in quant and not isinstance(quant["relu"], bool):
                raise ValueError("relu must be boolean")
            quant["relu"] = True
            producer["output"] = node["output"]
            fusions.append({"producer": producer["name"], "removed_node": node["name"],
                            "removed_tensor": intermediate, "output": node["output"],
                            "rule": "quantized-output-relu"})
            removed.add(index)
        if not removed:
            break
        compiled["nodes"] = [nodes[index] for index in order if index not in removed]
    validate_graph(compiled)
    return compiled, {
        "schema_version": 1, "pass": "single-consumer-relu-fusion",
        "original_nodes": len(document["nodes"]), "compiled_nodes": len(compiled["nodes"]),
        "fused_nodes": len(fusions), "fusions": fusions,
    }


def verify_graphs(original, compiled, binary, timeout=60):
    before = run_graph(original, binary, timeout=timeout)
    after = run_graph(compiled, binary, timeout=timeout)
    if before["outputs"] != after["outputs"] or before["external_outputs"] != after["external_outputs"]:
        raise sim.CheckError("compiled graph disagrees with the original graph")
    def counters(report):
        return {"npu_cycles": report["sum_npu_cycles"],
                "npu_read_bytes": report["npu_read_bytes"],
                "npu_write_bytes": report["npu_write_bytes"],
                "host_read_bytes": sum(layer.get("host_logical_read_bytes", 0) for layer in report["layers"]),
                "host_write_bytes": sum(layer.get("host_logical_write_bytes", 0) for layer in report["layers"]),
                "peak_live_tensor_bytes": report["peak_live_tensor_bytes"],
                "cpu_fallback_nodes": report["cpu_fallback_nodes"]}
    return {"outputs_match": True, "outputs": after["outputs"],
            "original": counters(before), "compiled": counters(after)}


@cli.main_guard("graph_optimize")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("graph_optimize", __doc__)
    parser.add_argument("graph", type=Path, help="graph JSON")
    parser.add_argument("--output", type=Path,
                        help="write the compiled graph here; an existing file needs --force")
    parser.add_argument("--force", action="store_true", help="overwrite --output if it exists")
    parser.add_argument("--verify", action="store_true",
                        help="run original and compiled graphs and compare every output")
    cli.add_binary_options(parser)
    args = parser.parse_args(argv)
    original = load_graph(args.graph)
    compiled, report = optimize_graph(original)
    if args.verify:
        report["verification"] = verify_graphs(original, compiled, cli.simulator(args),
                                               args.timeout)
    if args.output:
        jsonio.write_text(args.output, jsonio.dumps(compiled, sort_keys=False),
                          overwrite=args.force)
        report["compiled_graph"] = str(args.output)
    else:
        report["graph"] = compiled
    print(jsonio.dumps(report, sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
