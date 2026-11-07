"""Design points: the configs/*.json files, their headline figures and the rules that size them.

    python -m npuperf.designs                    # the table in docs/design-points.md
    python -m npuperf.designs --format json      # the same rows as JSON

A design point is an npu-sim --config file (option-table paths, plus "name" and
"description"). Each one is a hypothetical configuration sized from public data;
docs/design-points.md derives every field and cites the sources. Peak operations count one
MAC as two operations, the convention of the vendor figures the points are sized from.

Sizing rules, shared by the points and checked against the files by
tests/design_point_tests.py:
  tile_rule    square output tile T x T, T the largest power of two that is a multiple of
               both array dimensions and still fits the slots with tile_k >= min_depth;
               tile_k the largest power of two that then fits
  burst_rule   the shortest power-of-two burst with which the read outstanding limit covers
               the read latency, so a long run streams one beat per cycle
  weight cache the SRAM the slots leave free (cache_rule)

Two fields have no public figure and are sized by simulation, each study failing when a
config disagrees with its rule: the depthwise datapath rate (npuperf.studies S9) and the
cluster cores' SRAM (S10).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import analytic, cli, jsonio
from .binary import ROOT

CONFIG_DIR = ROOT / "configs"
# Smallest to largest; the order of the table.
DESIGN_POINTS = ("edge-256", "mobile-2k", "client-16c", "workstation-32c")
OPS_PER_MAC = 2
MAX_DIMENSION = 1 << 20
MAX_BURST_BEATS = 256
TEXT_KEYS = ("name", "description")
# Minimum efficient burst of the memory behind each point's port, in bytes: Vela 5.2.0
# Dram_burst_length, as every point streams its tensors from DRAM (docs/design-points.md).
# tile_rule and npuperf.mapper keep panels at least this deep.
MIN_PANEL_DEPTH = {"edge-256": 128, "mobile-2k": 128, "client-16c": 128, "workstation-32c": 128}
# The memory behind each point's AXI port.
EXTERNAL_MEMORY = {"edge-256": "DRAM", "mobile-2k": "DRAM", "client-16c": "DRAM",
                   "workstation-32c": "DRAM"}


def config_path(name: str | Path) -> Path:
    """configs/<name>.json, or ``name`` itself when it is a JSON file."""
    path = Path(name)
    if path.suffix == ".json" and path.is_file():
        return path
    path = CONFIG_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no design point {name!r} (known: {', '.join(names())})")
    return path


def min_panel_depth(name: str) -> int:
    """MIN_PANEL_DEPTH of a design point; 32 B (the smaller Vela burst) for any other."""
    return MIN_PANEL_DEPTH.get(name, 32)


def external_memory(name: str) -> str:
    """EXTERNAL_MEMORY of a design point; DRAM for any other."""
    return EXTERNAL_MEMORY.get(name, "DRAM")


def names() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.json"))


def load(name: str | Path) -> dict[str, Any]:
    """The config document: strict JSON, one object with string "name" and "description"."""
    path = config_path(name)
    document = jsonio.load_strict(path)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: a config file holds one JSON object")
    for key in TEXT_KEYS:
        if not isinstance(document.get(key), str) or not document[key]:
            raise ValueError(f"{path}: a design point needs a nonempty string {key!r}")
    return document


def leaves(node: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Option-table path -> value for every leaf except "name" and "description"."""
    result: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            result.update(leaves(value, path))
        elif path not in TEXT_KEYS:
            result[path] = value
    return result


def design(document: Mapping[str, Any]) -> analytic.Design:
    """The analytic.Design of a config document. Keys the analytic model does not read
    (quant, sim) are skipped; every other field takes the simulator default."""
    return analytic.Design.from_flags(leaves(document))


# ------------------------------------------------------------------ rules --

def slot_bytes(d: analytic.Design) -> int:
    """SRAM the tile slots reserve when a tile is full size and spans more than one panel:
    the reservation of every layer at most, since edge tiles and single panels need less."""
    if not d.tile_k:
        raise ValueError("a design point needs tile_k > 0; a full-K panel has no SRAM bound")
    full = d.replace(m=d.tile_m, n=d.tile_n, k=2 * d.tile_k, weight_cache_bytes=0)
    return analytic.scratchpad_bytes(full)


def cache_rule(d: analytic.Design) -> int:
    """Weight-cache bytes: the SRAM the slots leave free."""
    free = d.sram_bytes - slot_bytes(d)
    if free < 0:
        raise ValueError(f"tile slots need {slot_bytes(d)} B, above the {d.sram_bytes} B SRAM")
    return free


def _fits(d: analytic.Design, tile: int, depth: int) -> bool:
    return slot_bytes(d.replace(tile_m=tile, tile_n=tile, tile_k=depth)) <= d.sram_bytes


def _deepest(d: analytic.Design, tile: int, min_depth: int) -> int | None:
    depth = 1 << max(0, math.ceil(math.log2(min_depth)))
    if depth > MAX_DIMENSION or not _fits(d, tile, depth):
        return None
    while depth * 2 <= MAX_DIMENSION and _fits(d, tile, depth * 2):
        depth *= 2
    return depth


def tile_rule(d: analytic.Design, min_depth: int) -> tuple[int, int]:
    """(T, tile_k) for ``d``'s array, SRAM, slots and panel buffers; see the module text.

    ``min_depth`` is in bytes: each activation-panel row is one tile_k-byte run, so a
    shallower panel issues runs shorter than the memory's minimum efficient burst."""
    if not d.array:
        raise ValueError("the tile rule sizes tiles for a systolic array")
    step = math.lcm(d.array_rows, d.array_cols)
    tile = 1 << math.ceil(math.log2(step))
    if tile % step:
        raise ValueError(f"no power of two is a multiple of the {d.array_rows}x{d.array_cols} "
                         "array")
    best = None
    while tile <= MAX_DIMENSION:
        depth = _deepest(d, tile, min_depth)
        if depth is None:
            break
        best = (tile, depth)
        tile *= 2
    if best is None:
        raise ValueError(f"no {step}-multiple tile fits {d.sram_bytes} B with tile_k >= "
                         f"{min_depth}")
    return best


def burst_rule(read_outstanding: int, read_latency: int) -> int:
    """Shortest power-of-two burst B (beats) with outstanding * B >= latency + B.

    A read burst holds an outstanding slot from issue to its last beat, latency + B cycles,
    so N slots move N * B beats in that time: one beat per cycle once N * B >= latency + B."""
    if read_outstanding < 2:
        raise ValueError("one outstanding read cannot cover any latency")
    beats = 1
    while read_outstanding * beats < read_latency + beats:
        beats *= 2
        if beats > MAX_BURST_BEATS:
            raise ValueError(f"{read_outstanding} outstanding reads cannot cover "
                             f"{read_latency} cycles with {MAX_BURST_BEATS}-beat bursts")
    return beats


# ---------------------------------------------------------------- summary --

@dataclass(frozen=True)
class Summary:
    name: str
    macs_per_cycle: int          # every core
    array: str
    cores: int
    frequency_mhz: float
    peak_tops: float
    sram_bytes: int              # per core
    weight_cache_bytes: int      # per core, inside sram_bytes
    depthwise_macs_per_cycle: int  # per core
    axi_bits: int                # per core port
    dram_bytes_per_cycle: int
    dram_gbps: float
    ridge_macs_per_byte: float


def summary(document: Mapping[str, Any]) -> Summary:
    d = design(document)
    if d.memory_model != "axi" or not d.dram_bytes_per_cycle or not d.axi_data_bytes:
        raise ValueError(f"{document['name']}: a design point uses the axi memory model with "
                         "a DRAM rate")
    peak = d.peak_macs_per_cycle
    return Summary(
        name=document["name"], macs_per_cycle=peak,
        array=f"{d.array_rows}x{d.array_cols}" if d.array else "vector",
        cores=d.cores, frequency_mhz=d.frequency_mhz,
        peak_tops=OPS_PER_MAC * peak * d.frequency_mhz / 1e6,
        sram_bytes=d.sram_bytes, weight_cache_bytes=d.weight_cache_bytes,
        depthwise_macs_per_cycle=d.depthwise_macs_per_cycle, axi_bits=8 * d.axi_data_bytes,
        dram_bytes_per_cycle=d.dram_bytes_per_cycle,
        dram_gbps=d.dram_bytes_per_cycle * d.frequency_mhz / 1e3,
        ridge_macs_per_byte=d.ridge)


def _per_core(cores: int, text: str) -> str:
    return text if cores == 1 else f"{cores} x {text}"


def markdown(rows: Sequence[Summary]) -> str:
    """The design-point table of docs/design-points.md."""
    header = ("Design point", "MAC/cycle", "Array", "Cores", "Clock", "Peak TOP/s", "SRAM",
              "Depthwise MAC/cycle", "AXI", "DRAM B/cycle", "DRAM GB/s", "Ridge MAC/B")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = (f"`{row.name}`", str(row.macs_per_cycle), row.array, str(row.cores),
                 f"{row.frequency_mhz:g} MHz", f"{row.peak_tops:.6g}",
                 _per_core(row.cores, f"{row.sram_bytes / 1024:g} KiB"),
                 _per_core(row.cores, str(row.depthwise_macs_per_cycle)),
                 _per_core(row.cores, f"{row.axi_bits}-bit"), str(row.dram_bytes_per_cycle),
                 f"{row.dram_gbps:.6g}", f"{row.ridge_macs_per_byte:.4g}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


@cli.main_guard("designs")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("designs", __doc__)
    parser.add_argument("names", nargs="*", metavar="NAME",
                        help="design points (configs/<NAME>.json or a JSON path); default: "
                             + ", ".join(DESIGN_POINTS))
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown",
                        help="stdout format")
    args = parser.parse_args(argv)
    rows = [summary(load(name)) for name in args.names or DESIGN_POINTS]
    if args.format == "json":
        print(jsonio.dumps([asdict(row) for row in rows], sort_keys=False), end="")
    else:
        print(markdown(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
