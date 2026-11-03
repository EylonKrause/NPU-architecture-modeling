#!/usr/bin/env python3
"""Compile safe single-consumer activation fusions in the quantized graph IR.

This is a bounded rewrite pass, not a hardware placement compiler. Optional
verification executes both graphs through the actual SystemC binary.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from graph_runtime import load_graph, run_graph, validate_graph
from sweep import positive_int, resolve_binary


ACCELERATED = {"Dense", "MatMul", "Conv2D", "DepthwiseConv2D"}


def optimize_graph(document):
    """Fuse producer->ReLU only if the unfused producer has no observer.

    ReLU after INT8 requantization is exactly the engine's existing fused
    output-zero clamp, including saturation and either rounding rule. The
    producer's pre-activation output must not be externally visible or shared.
    Repeating this pass handles redundant chains of ReLUs without changing IR.
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
                raise ValueError("accelerated nodes require explicit quantization")
            # Never conceal a malformed original quantization attribute.
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
        "boundary": "Fuses the output-zero clamp into requantization. No general cross-layer SRAM residency or hardware speedup is implied.",
    }


def verify_graphs(original, compiled, binary, timeout=60):
    before = run_graph(original, binary, timeout=timeout)
    after = run_graph(compiled, binary, timeout=timeout)
    if before["outputs"] != after["outputs"] or before["external_outputs"] != after["external_outputs"]:
        raise ValueError("compiled graph disagrees with the original graph")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output", type=Path, help="Create the compiled graph; existing files are preserved")
    parser.add_argument("--verify", action="store_true", help="Execute original and compiled graphs and compare every output")
    parser.add_argument("--binary", default="build/npu-sim")
    parser.add_argument("--timeout", type=positive_int, default=60)
    args = parser.parse_args()
    original = load_graph(args.graph)
    compiled, report = optimize_graph(original)
    if args.verify:
        report["verification"] = verify_graphs(original, compiled, resolve_binary(args.binary), args.timeout)
    if args.output:
        # Exclusive creation preserves source documents and earlier experiments.
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(compiled, stream, indent=2, allow_nan=False)
            stream.write("\n")
        report["compiled_graph"] = str(args.output)
    else:
        report["graph"] = compiled
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print("graph_optimize: " + str(error), file=sys.stderr)
        sys.exit(1)
