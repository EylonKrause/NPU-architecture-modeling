"""Writers for the simulator's text tensor formats.

NPU1:  ``NPU1 M N K`` then the activations (row-major, M x G*K), the weights
       (G blocks of K x N, each row-major) and the INT32 bias (G*N), whitespace separated.
       G is the --groups count given on the command line; G = 1 is a plain GEMM.
NPUQ1: ``NPUQ1 C`` then one ``weight_zero multiplier shift`` line per output channel
       (C = G*N). Shifts are signed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def format_problem(
    m: int,
    n: int,
    k: int,
    activations: Sequence[int],
    weights: Sequence[int],
    bias: Sequence[int],
    groups: int = 1,
) -> str:
    if min(m, n, k, groups) < 1:
        raise ValueError("NPU1 dimensions and groups must be positive")
    if (
        len(activations) != m * groups * k
        or len(weights) != groups * k * n
        or len(bias) != groups * n
    ):
        raise ValueError(f"tensor sizes do not match NPU1 {m} {n} {k} with {groups} group(s)")
    lines = [
        f"NPU1 {m} {n} {k}",
        " ".join(map(str, activations)),
        " ".join(map(str, weights)),
        " ".join(map(str, bias)),
    ]
    return "\n".join(lines) + "\n"


def write_problem(
    path: Path | str,
    m: int,
    n: int,
    k: int,
    activations: Sequence[int],
    weights: Sequence[int],
    bias: Sequence[int],
    groups: int = 1,
) -> Path:
    path = Path(path)
    path.write_text(format_problem(m, n, k, activations, weights, bias, groups), encoding="ascii")
    return path


def format_quant(lanes: Iterable[Mapping[str, int] | Sequence[int]]) -> str:
    rows = []
    for lane in lanes:
        if isinstance(lane, Mapping):
            rows.append((lane["weight_zero"], lane["multiplier"], lane["shift"]))
        else:
            weight_zero, multiplier, shift = lane
            rows.append((weight_zero, multiplier, shift))
    if not rows:
        raise ValueError("NPUQ1 needs at least one channel")
    return f"NPUQ1 {len(rows)}\n" + "".join(f"{z} {m} {s}\n" for z, m, s in rows)


def write_quant(path: Path | str, lanes: Iterable[Mapping[str, int] | Sequence[int]]) -> Path:
    path = Path(path)
    path.write_text(format_quant(lanes), encoding="ascii")
    return path
