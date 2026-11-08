"""npu-sim --trace CSV files: stage spans, busy cycles, compute idle gaps and Gantt figures."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import svg

HEADER = ["stage", "tile_id", "start_cycle", "end_cycle", "bytes", "block_id"]
STAGES = ("load", "compute", "store")


@dataclass(frozen=True)
class Span:
    stage: str
    tile_id: int
    start: int
    end: int
    bytes: int
    block_id: int

    @property
    def cycles(self) -> int:
        return self.end - self.start


def read_trace(path: Path | str) -> list[Span]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0][:6] != HEADER:
        raise ValueError(f"{path}: expected a trace CSV with header {','.join(HEADER)}")
    spans = []
    for row in rows[1:]:
        if not row:
            continue
        stage, *numbers = row[:6]
        tile, start, end, nbytes, block = (int(value) for value in numbers)
        if end < start:
            raise ValueError(f"{path}: span ends before it starts: {row}")
        spans.append(Span(stage, tile, start, end, nbytes, block))
    return spans


def busy_cycles(spans: Iterable[Span]) -> dict[str, int]:
    totals = {stage: 0 for stage in STAGES}
    for span in spans:
        totals[span.stage] = totals.get(span.stage, 0) + span.cycles
    return totals


def compute_idle_gaps(spans: Sequence[Span]) -> list[tuple[int, int]]:
    """Intervals between compute spans (compute waiting on data or a slot)."""
    compute = sorted((s for s in spans if s.stage == "compute"), key=lambda s: s.start)
    return [(a.end, b.start) for a, b in zip(compute, compute[1:], strict=False) if b.start > a.end]


def gantt(
    spans: Sequence[Span],
    title: str = "Pipeline",
    subtitle: str | None = None,
    max_tiles: int | None = 16,
    width: int = 760,
    height: int = 300,
) -> svg.Figure:
    """One row per engine, spans colored by stage. Loads and stores share the DMA row."""
    if max_tiles is not None:
        spans = [s for s in spans if s.tile_id < max_tiles]
    rows = ["DMA", "MAC array"]
    fig = svg.Figure(
        width,
        height,
        title=title,
        subtitle=subtitle,
        x_label="cycle",
        y_categories=rows,
        legend="right",
        x_range=(0, None),
    )
    colors = {"load": svg.PALETTE[0], "compute": svg.PALETTE[1], "store": svg.PALETTE[2]}
    labeled: set[str] = set()
    for span in sorted(spans, key=lambda s: (s.start, s.stage)):
        row = 1 if span.stage == "compute" else 0
        label = (
            None
            if span.stage in labeled
            else {"load": "load", "compute": "compute", "store": "store"}.get(span.stage)
        )
        labeled.add(span.stage)
        fig.span(row, span.start, span.end, colors.get(span.stage), label)
    return fig
