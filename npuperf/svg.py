"""Dependency-free SVG plots for engineering figures.

``Figure`` draws on one linear or log10 x/y axis pair (or categorical bands for bars and
Gantt rows): lines, scatter, steps, bars, rectangles, reference lines, text annotations and
a legend. Output is deterministic: coordinates use fixed two-decimal formatting, element
order follows call order, and nothing depends on time or randomness, so committed figures
diff cleanly. The background is opaque white, so a figure reads the same on GitHub's
light and dark themes.

    fig = Figure(title="Roofline", x_label="MAC per DRAM byte", y_label="MAC per cycle",
                 x_scale="log", y_scale="log")
    fig.line([0.1, 4, 1000], [1.6, 64, 64], label="roof")
    fig.scatter([3.8], [26.8], label="layer", point_labels=["conv1"])
    svg_text = fig.to_svg()
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Categorical order validated for colour-vision deficiency on white (adjacent pairs).
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("circle", "square", "triangle", "diamond", "down", "circle", "square", "triangle")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#6f6e69"
GRID = "#e6e5df"
AXIS = "#9c9a93"
SURFACE = "#ffffff"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# Average advance widths (em) for the sans faces above; used to size margins and legends.
_NARROW = set("il.,:;'|!()[]{}/\\ ")
_WIDE = set("mwMW@%")


def text_width(text: str, size: float) -> float:
    width = 0.0
    for char in text:
        if char in _NARROW:
            width += 0.30
        elif char in _WIDE:
            width += 0.86
        elif char.isupper():
            width += 0.66
        elif char.isdigit():
            width += 0.56
        else:
            width += 0.53
    return width * size


def num(value: float) -> str:
    """Fixed two-decimal coordinate formatting without trailing zeros."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _compact(value: float) -> str:
    magnitude = abs(value)
    for limit, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if magnitude >= limit:
            scaled = value / limit
            return (f"{scaled:.0f}" if scaled == round(scaled) else f"{scaled:.1f}") + suffix
    return _plain(value)


def _plain(value: float, decimals: int | None = None) -> str:
    if decimals is None:
        if value == round(value):
            return f"{int(round(value)):,}"
        decimals = max(0, min(6, -math.floor(math.log10(abs(value))) + 1)) if value else 0
        text = f"{value:,.{decimals}f}".rstrip("0").rstrip(".")
        return text
    return f"{value:,.{decimals}f}"


def nice_step(span: float, target: int = 6) -> float:
    raw = span / max(1, target)
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude * 1.0000001:
            return factor * magnitude
    return 10 * magnitude


@dataclass
class Axis:
    scale: str = "linear"
    lo: float | None = None
    hi: float | None = None
    label: str | None = None
    categories: Sequence[str] | None = None
    ticks: Sequence[float] | None = None
    tick_format: Any = None
    include_zero: bool = False

    def fit(self, values: Sequence[float]) -> None:
        if self.categories is not None:
            self.lo, self.hi = -0.5, len(self.categories) - 0.5
            return
        finite = [v for v in values if math.isfinite(v) and (self.scale != "log" or v > 0)]
        lo = self.lo if self.lo is not None else (min(finite) if finite else 0.0)
        hi = self.hi if self.hi is not None else (max(finite) if finite else 1.0)
        if self.scale == "log":
            lo = lo if lo > 0 else 1.0
            if hi <= lo:
                hi = lo * 10
            if self.lo is None:
                lo = 10 ** math.floor(math.log10(lo) + 1e-9)
            if self.hi is None:
                hi = 10 ** math.ceil(math.log10(hi) - 1e-9)
        else:
            if self.include_zero:
                lo, hi = min(lo, 0.0), max(hi, 0.0)
            if hi == lo:
                hi = lo + 1.0
            step = nice_step(hi - lo)
            if self.lo is None:
                lo = math.floor(lo / step + 1e-9) * step
            if self.hi is None:
                hi = math.ceil(hi / step - 1e-9) * step
        self.lo, self.hi = float(lo), float(hi)

    def unit(self, value: float) -> float:
        """Position in 0..1 along the axis."""
        assert self.lo is not None and self.hi is not None
        if self.scale == "log":
            value = max(value, 1e-300)
            return (math.log10(value) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
        return (value - self.lo) / (self.hi - self.lo)

    def major_ticks(self) -> list[float]:
        assert self.lo is not None and self.hi is not None
        if self.categories is not None:
            return list(range(len(self.categories)))
        if self.ticks is not None:
            return [t for t in self.ticks if self.lo <= t <= self.hi]
        if self.scale == "log":
            first, last = math.ceil(math.log10(self.lo) - 1e-9), math.floor(math.log10(self.hi) + 1e-9)
            decades = [10.0 ** e for e in range(first, last + 1)]
            if len(decades) >= 2:
                return decades
            ticks = [m * 10.0 ** e for e in range(first - 1, last + 1) for m in (1, 2, 5)]
            return [t for t in ticks if self.lo * 0.999999 <= t <= self.hi * 1.000001]
        step = nice_step(self.hi - self.lo)
        start = math.ceil(self.lo / step - 1e-9)
        ticks, index = [], start
        while index * step <= self.hi + step * 1e-9:
            ticks.append(round(index * step, 12))
            index += 1
        return ticks

    def minor_ticks(self) -> list[float]:
        if self.scale != "log" or self.categories is not None or self.ticks is not None:
            return []
        assert self.lo is not None and self.hi is not None
        out = []
        for exponent in range(math.floor(math.log10(self.lo)), math.ceil(math.log10(self.hi)) + 1):
            for mantissa in range(2, 10):
                value = mantissa * 10.0 ** exponent
                if self.lo < value < self.hi:
                    out.append(value)
        return out

    def label_for(self, value: float, ticks: Sequence[float]) -> str:
        if self.categories is not None:
            return str(self.categories[int(value)])
        if self.tick_format is not None:
            return self.tick_format(value)
        top = max((abs(t) for t in ticks), default=0)
        if top >= 10000:
            return _compact(value)
        if self.scale == "log":
            return _plain(value)
        decimals = next((d for d in range(7) if all(
            abs(t * 10 ** d - round(t * 10 ** d)) < 1e-6 for t in ticks)), 6)
        return _plain(value, decimals) if decimals else f"{int(round(value)):,}"


@dataclass
class _Series:
    kind: str
    label: str | None
    color: str
    marker: str | None
    dash: str | None
    width: float
    data: Any
    extra: dict = field(default_factory=dict)


class Figure:
    def __init__(self, width: int = 720, height: int = 440, *, title: str | None = None,
                 subtitle: str | None = None, x_label: str | None = None,
                 y_label: str | None = None, x_scale: str = "linear", y_scale: str = "linear",
                 x_range: tuple[float | None, float | None] | None = None,
                 y_range: tuple[float | None, float | None] | None = None,
                 x_categories: Sequence[str] | None = None,
                 y_categories: Sequence[str] | None = None, font_size: float = 13,
                 legend: str = "auto", id_prefix: str = "fig", y_include_zero: bool = False,
                 x_ticks: Sequence[float] | None = None, y_ticks: Sequence[float] | None = None):
        self.width, self.height = width, height
        self.title, self.subtitle = title, subtitle
        self.font_size = font_size
        self.legend_position = legend
        self.id_prefix = "".join(c for c in id_prefix if c.isalnum() or c in "-_") or "fig"
        x_lo, x_hi = x_range or (None, None)
        y_lo, y_hi = y_range or (None, None)
        self._ranges = ((x_lo, x_hi), (y_lo, y_hi))
        self.x = Axis(x_scale, x_lo, x_hi, x_label, x_categories, ticks=x_ticks)
        self.y = Axis(y_scale, y_lo, y_hi, y_label, y_categories, ticks=y_ticks,
                      include_zero=y_include_zero)
        self.series: list[_Series] = []
        self.notes: list[dict] = []
        self._color_index = 0

    # ----------------------------------------------------------------- data
    def _style(self, color: str | None, marker: str | None) -> tuple[str, str | None]:
        if color is None:
            color = PALETTE[self._color_index % len(PALETTE)]
            if marker == "auto":
                marker = MARKERS[self._color_index % len(MARKERS)]
            self._color_index += 1
        elif marker == "auto":
            marker = "circle"
        return color, marker

    def line(self, xs: Sequence[float], ys: Sequence[float], label: str | None = None,
             color: str | None = None, width: float = 2, dash: str | None = None,
             marker: str | None = None) -> "Figure":
        color, marker = self._style(color, marker)
        self.series.append(_Series("line", label, color, marker, dash, width,
                                   list(zip(map(float, xs), map(float, ys)))))
        return self

    def step(self, xs: Sequence[float], ys: Sequence[float], label: str | None = None,
             color: str | None = None, width: float = 2, where: str = "post") -> "Figure":
        color, _ = self._style(color, None)
        self.series.append(_Series("step", label, color, None, None, width,
                                   list(zip(map(float, xs), map(float, ys))), {"where": where}))
        return self

    def scatter(self, xs: Sequence[float], ys: Sequence[float], label: str | None = None,
                color: str | None = None, marker: str | None = "auto", size: float = 4.5,
                point_labels: Sequence[str | None] | None = None,
                hollow: bool = False) -> "Figure":
        """Markers at (x, y). ``hollow`` draws outlines, so a second population (for example
        measured next to modelled points) stays visible where the two coincide."""
        color, marker = self._style(color, marker)
        self.series.append(_Series("scatter", label, color, marker or "circle", None, size,
                                   list(zip(map(float, xs), map(float, ys))),
                                   {"labels": list(point_labels) if point_labels else None,
                                    "hollow": hollow}))
        return self

    def bars(self, categories: Sequence[str], values: dict[str, Sequence[float]] | Sequence[float],
             label: str | None = None, baseline: float = 0.0, value_labels: bool = False,
             value_format: str = "{:.2f}", colors: Sequence[str] | None = None) -> "Figure":
        """Vertical bars on categorical x; a dict gives one grouped series per key."""
        if self.x.categories is None:
            self.x.categories = list(categories)
        groups = values if isinstance(values, dict) else {label or "": values}
        names = list(groups)
        for index, name in enumerate(names):
            color = colors[index] if colors else PALETTE[(self._color_index + index) % len(PALETTE)]
            self.series.append(_Series("bar", name if isinstance(values, dict) or label else None,
                                       color, "square", None, 0,
                                       [(float(i), float(v)) for i, v in enumerate(groups[name])],
                                       {"group": index, "groups": len(names), "baseline": baseline,
                                        "value_labels": value_labels, "format": value_format}))
        self._color_index += len(names)
        return self

    def rect(self, x0: float, x1: float, y0: float, y1: float, color: str | None = None,
             label: str | None = None, opacity: float = 1.0, gap: float = 0.0) -> "Figure":
        """Filled rectangle in data coordinates; ``gap`` trims that many pixels off its right
        edge so back-to-back rectangles stay visibly separate."""
        color = color or PALETTE[0]
        self.series.append(_Series("rect", label, color, "square", None, 0,
                                   [(float(x0), float(y0)), (float(x1), float(y1))],
                                   {"opacity": opacity, "gap": gap}))
        return self

    def span(self, row: int, x0: float, x1: float, color: str | None = None,
             label: str | None = None, height: float = 0.62) -> "Figure":
        """Horizontal bar in categorical row ``row`` (Gantt charts)."""
        return self.rect(x0, x1, row - height / 2, row + height / 2, color, label, gap=1.0)

    def hline(self, y: float, label: str | None = None, color: str = INK_SECONDARY,
              dash: str | None = "5 4", width: float = 1.25, text_anchor: str = "end") -> "Figure":
        self.series.append(_Series("hline", None, color, None, dash, width, [(0.0, float(y))],
                                   {"text": label, "anchor": text_anchor}))
        return self

    def vline(self, x: float, label: str | None = None, color: str = INK_SECONDARY,
              dash: str | None = "5 4", width: float = 1.25) -> "Figure":
        self.series.append(_Series("vline", None, color, None, dash, width, [(float(x), 0.0)],
                                   {"text": label}))
        return self

    def annotate(self, x: float, y: float, text: str, dx: float = 6, dy: float = -6,
                 anchor: str = "start", color: str = INK_SECONDARY, size: float | None = None,
                 rotate: float | str = 0, weight: str = "normal") -> "Figure":
        """Text at data point (x, y), offset by (dx, dy) pixels. ``rotate="along"`` with
        ``along=(x2, y2)`` is handled by ``annotate_along``."""
        self.notes.append({"x": float(x), "y": float(y), "text": text, "dx": dx, "dy": dy,
                           "anchor": anchor, "color": color, "size": size or self.font_size - 1,
                           "rotate": rotate, "weight": weight})
        return self

    def annotate_along(self, p0: tuple[float, float], p1: tuple[float, float], text: str,
                       at: float = 0.5, offset: float = -7, color: str = INK_SECONDARY,
                       size: float | None = None, anchor: str = "middle") -> "Figure":
        """Text parallel to the segment p0-p1 (in screen space), ``at`` of the way along it."""
        self.notes.append({"along": (p0, p1), "at": at, "text": text, "offset": offset,
                           "color": color, "size": size or self.font_size - 1, "anchor": anchor})
        return self

    # --------------------------------------------------------------- layout
    def _fit_axes(self) -> None:
        xs: list[float] = []
        ys: list[float] = []
        for series in self.series:
            if series.kind == "hline":
                ys.append(series.data[0][1])
                continue
            if series.kind == "vline":
                xs.append(series.data[0][0])
                continue
            for x, y in series.data:
                xs.append(x)
                ys.append(y)
            if series.kind == "bar":
                ys.append(series.extra["baseline"])
        (self.x.lo, self.x.hi), (self.y.lo, self.y.hi) = self._ranges
        self.x.fit(xs)
        self.y.fit(ys)

    def _layout(self) -> None:
        size = self.font_size
        self._fit_axes()
        self.y_ticks = self.y.major_ticks()
        self.x_ticks = self.x.major_ticks()
        y_labels = [self.y.label_for(t, self.y_ticks) for t in self.y_ticks]
        tick_width = max((text_width(t, size - 1) for t in y_labels), default=0)
        left = 14 + tick_width + 8 + (size + 10 if self.y.label else 0)
        top = 16 + (size + 6 + 8 if self.title else 0) + (size + 4 if self.subtitle else 0)
        bottom = 10 + size + 8 + (size + 10 if self.x.label else 0)
        right = 24
        self.legend_items = [s for s in self.series if s.label]
        if self.legend_position == "right" and self.legend_items:
            right = 18 + 44 + max(text_width(s.label or "", size - 1) for s in self.legend_items) + 12
        x_labels = [self.x.label_for(t, self.x_ticks) for t in self.x_ticks]
        if x_labels:
            right = max(right, text_width(x_labels[-1], size - 1) / 2 + 8)
        self.plot = (left, top, self.width - right, self.height - bottom)
        self.y_labels, self.x_labels = y_labels, x_labels
        self.boxes: list[tuple[float, float, float, float]] = []

    def px(self, x: float) -> float:
        x0, _, x1, _ = self.plot
        return x0 + self.x.unit(x) * (x1 - x0)

    def py(self, y: float) -> float:
        _, y0, _, y1 = self.plot
        return y1 - self.y.unit(y) * (y1 - y0)

    # -------------------------------------------------------------- drawing
    def _text(self, x: float, y: float, text: str, size: float, color: str = INK,
              anchor: str = "start", weight: str = "normal", rotate: float = 0,
              baseline: str | None = None, track: bool = True) -> str:
        attrs = [f'x="{num(x)}"', f'y="{num(y)}"', f'font-size="{num(size)}"', f'fill="{color}"']
        if anchor != "start":
            attrs.append(f'text-anchor="{anchor}"')
        if weight != "normal":
            attrs.append(f'font-weight="{weight}"')
        if baseline:
            attrs.append(f'dominant-baseline="{baseline}"')
        if rotate:
            attrs.append(f'transform="rotate({num(rotate)} {num(x)} {num(y)})"')
        if track:
            width = text_width(text, size)
            start = -(width if anchor == "end" else width / 2 if anchor == "middle" else 0)
            if not rotate:
                self.boxes.append((x + start, y - size * 0.8, x + start + width, y + size * 0.25))
            else:  # cover the rotated run with small boxes along its baseline
                cos, sin = math.cos(math.radians(rotate)), math.sin(math.radians(rotate))
                steps = max(1, int(width / (size * 0.6)))
                for i in range(steps + 1):
                    along = start + width * i / steps
                    cx, cy = x + along * cos + size * 0.3 * sin, y + along * sin - size * 0.3 * cos
                    self.boxes.append((cx - size * 0.45, cy - size * 0.45,
                                       cx + size * 0.45, cy + size * 0.45))
        return f'<text {" ".join(attrs)}>{escape(text)}</text>'

    def _marker(self, kind: str, x: float, y: float, r: float, color: str,
                hollow: bool = False) -> str:
        ring = f'stroke="{SURFACE}" stroke-width="1.5"'
        if hollow:
            ring = f'stroke="{color}" stroke-width="1.75"'
            color = "none"
        if kind == "square":
            s = r * 0.9
            return (f'<rect x="{num(x - s)}" y="{num(y - s)}" width="{num(2 * s)}" '
                    f'height="{num(2 * s)}" fill="{color}" {ring}/>')
        if kind in ("triangle", "down"):
            h = r * 1.15
            sign = 1 if kind == "triangle" else -1
            pts = [(x, y - sign * h), (x + h, y + sign * h * 0.75), (x - h, y + sign * h * 0.75)]
            return (f'<polygon points="{" ".join(f"{num(a)},{num(b)}" for a, b in pts)}" '
                    f'fill="{color}" {ring}/>')
        if kind == "diamond":
            h = r * 1.25
            pts = [(x, y - h), (x + h, y), (x, y + h), (x - h, y)]
            return (f'<polygon points="{" ".join(f"{num(a)},{num(b)}" for a, b in pts)}" '
                    f'fill="{color}" {ring}/>')
        return f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r)}" fill="{color}" {ring}/>'

    def _path(self, points: list[tuple[float, float]]) -> str:
        return " ".join(("M" if i == 0 else "L") + f"{num(x)},{num(y)}"
                        for i, (x, y) in enumerate(points))

    def _grid_and_axes(self) -> list[str]:
        x0, y0, x1, y1 = self.plot
        size = self.font_size - 1
        out = [f'<g stroke="{GRID}" stroke-width="1" fill="none">']
        if self.y.categories is None:
            for t in self.y_ticks:
                y = self.py(t)
                out.append(f'<line x1="{num(x0)}" y1="{num(y)}" x2="{num(x1)}" y2="{num(y)}"/>')
        if self.x.categories is None:
            for t in self.x_ticks:
                x = self.px(t)
                out.append(f'<line x1="{num(x)}" y1="{num(y0)}" x2="{num(x)}" y2="{num(y1)}"/>')
        out.append("</g>")
        out.append(f'<g stroke="{AXIS}" stroke-width="1" fill="none">')
        out.append(f'<line x1="{num(x0)}" y1="{num(y1)}" x2="{num(x1)}" y2="{num(y1)}"/>')
        out.append(f'<line x1="{num(x0)}" y1="{num(y0)}" x2="{num(x0)}" y2="{num(y1)}"/>')
        for t in self.x.minor_ticks():
            x = self.px(t)
            out.append(f'<line x1="{num(x)}" y1="{num(y1)}" x2="{num(x)}" y2="{num(y1 + 3)}"/>')
        for t in self.y.minor_ticks():
            y = self.py(t)
            out.append(f'<line x1="{num(x0 - 3)}" y1="{num(y)}" x2="{num(x0)}" y2="{num(y)}"/>')
        for t in self.x_ticks:
            x = self.px(t)
            out.append(f'<line x1="{num(x)}" y1="{num(y1)}" x2="{num(x)}" y2="{num(y1 + 5)}"/>')
        for t in self.y_ticks:
            y = self.py(t)
            out.append(f'<line x1="{num(x0 - 5)}" y1="{num(y)}" x2="{num(x0)}" y2="{num(y)}"/>')
        out.append("</g>")
        out.append('<g style="font-variant-numeric: tabular-nums">')
        for t, label in zip(self.x_ticks, self.x_labels):
            out.append(self._text(self.px(t), y1 + 7 + size, label, size, INK_SECONDARY, "middle"))
        for t, label in zip(self.y_ticks, self.y_labels):
            out.append(self._text(x0 - 9, self.py(t) + size * 0.35, label, size, INK_SECONDARY, "end"))
        out.append("</g>")
        if self.x.label:
            out.append(self._text((x0 + x1) / 2, self.height - 12, self.x.label, self.font_size,
                                  INK, "middle"))
        if self.y.label:
            cx, cy = 16 + self.font_size * 0.4, (y0 + y1) / 2
            out.append(self._text(cx, cy, self.y.label, self.font_size, INK, "middle",
                                  rotate=-90, baseline="middle"))
        return out

    def _bar_geometry(self, series: _Series, index: float, value: float) -> tuple[float, float, float, float]:
        x0, _, x1, _ = self.plot
        band = (x1 - x0) / len(self.x.categories or [0])
        groups, group = series.extra["groups"], series.extra["group"]
        gap = 2.0
        width = min(28.0, (band * 0.72 - gap * (groups - 1)) / groups)
        total = groups * width + gap * (groups - 1)
        left = self.px(index) - total / 2 + group * (width + gap)
        base, top = self.py(series.extra["baseline"]), self.py(value)
        return left, width, base, top

    def _bar(self, series: _Series, index: float, value: float) -> str:
        left, width, base, top = self._bar_geometry(series, index, value)
        r = min(4.0, width / 2, abs(base - top))
        if top <= base:  # grows upward; rounded top, square base
            path = (f"M{num(left)},{num(base)} L{num(left)},{num(top + r)} "
                    f"Q{num(left)},{num(top)} {num(left + r)},{num(top)} "
                    f"L{num(left + width - r)},{num(top)} "
                    f"Q{num(left + width)},{num(top)} {num(left + width)},{num(top + r)} "
                    f"L{num(left + width)},{num(base)} Z")
        else:
            path = (f"M{num(left)},{num(base)} L{num(left)},{num(top - r)} "
                    f"Q{num(left)},{num(top)} {num(left + r)},{num(top)} "
                    f"L{num(left + width - r)},{num(top)} "
                    f"Q{num(left + width)},{num(top)} {num(left + width)},{num(top - r)} "
                    f"L{num(left + width)},{num(base)} Z")
        return f'<path d="{path}" fill="{series.color}"/>'

    def _series_svg(self, series: _Series) -> list[str]:
        out: list[str] = []
        x0, y0, x1, y1 = self.plot
        dash = f' stroke-dasharray="{series.dash}"' if series.dash else ""
        if series.kind in ("line", "step"):
            points = [(self.px(x), self.py(y)) for x, y in series.data]
            if series.kind == "step" and len(points) > 1:
                stepped = [points[0]]
                for (ax, ay), (bx, by) in zip(points, points[1:]):
                    stepped.append((bx, ay) if series.extra.get("where") == "post" else (ax, by))
                    stepped.append((bx, by))
                points = stepped
            out.append(f'<path d="{self._path(points)}" fill="none" stroke="{series.color}" '
                       f'stroke-width="{num(series.width)}" stroke-linejoin="round" '
                       f'stroke-linecap="round"{dash}/>')
            if series.marker:
                out += [self._marker(series.marker, x, y, 4, series.color)
                        for x, y in ((self.px(a), self.py(b)) for a, b in series.data)]
        elif series.kind == "scatter":
            out += [self._marker(series.marker or "circle", self.px(x), self.py(y), series.width,
                                 series.color, series.extra.get("hollow", False))
                    for x, y in series.data]
        elif series.kind == "bar":
            out += [self._bar(series, x, y) for x, y in series.data]
        elif series.kind == "rect":
            (ax, ay), (bx, by) = series.data
            left, right = sorted((self.px(ax), self.px(bx)))
            upper, lower = sorted((self.py(ay), self.py(by)))
            opacity = series.extra.get("opacity", 1.0)
            fade = f' fill-opacity="{num(opacity)}"' if opacity < 1 else ""
            right -= min(series.extra.get("gap", 0.0), (right - left) / 2)
            out.append(f'<rect x="{num(left)}" y="{num(upper)}" width="{num(max(right - left, 0.6))}" '
                       f'height="{num(lower - upper)}" fill="{series.color}"{fade}/>')
        elif series.kind == "hline":
            y = self.py(series.data[0][1])
            out.append(f'<line x1="{num(x0)}" y1="{num(y)}" x2="{num(x1)}" y2="{num(y)}" '
                       f'stroke="{series.color}" stroke-width="{num(series.width)}"{dash}/>')
        elif series.kind == "vline":
            x = self.px(series.data[0][0])
            out.append(f'<line x1="{num(x)}" y1="{num(y0)}" x2="{num(x)}" y2="{num(y1)}" '
                       f'stroke="{series.color}" stroke-width="{num(series.width)}"{dash}/>')
        return out

    def _overlaps(self, box: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = self.plot
        if box[0] < x0 + 1 or box[2] > x1 - 1 or box[1] < y0 + 1 or box[3] > y1 - 1:
            return True
        return any(not (box[2] <= b[0] or box[0] >= b[2] or box[3] <= b[1] or box[1] >= b[3])
                   for b in self.boxes)

    def _point_labels(self, series: _Series, marks: list[tuple[float, float, float]]) -> list[str]:
        out = []
        labels = series.extra.get("labels") or []
        size = self.font_size - 2
        for (x, y), label in zip(series.data, labels):
            if not label:
                continue
            px, py = self.px(x), self.py(y)
            width = text_width(label, size)
            options = [(8, size * 0.35, "start"), (-8, size * 0.35, "end"), (0, -9, "middle"),
                       (0, size + 6, "middle"), (7, -7, "start"), (-7, -7, "end"),
                       (7, size + 3, "start"), (-7, size + 3, "end")]
            for dx, dy, anchor in options:
                left = px + dx - (width if anchor == "end" else width / 2 if anchor == "middle" else 0)
                box = (left - 1, py + dy - size * 0.8, left + width + 1, py + dy + size * 0.25)
                hits_mark = any(abs(mx - min(max(mx, box[0]), box[2])) < r and
                                abs(my - min(max(my, box[1]), box[3])) < r for mx, my, r in marks)
                if not self._overlaps(box) and not hits_mark:
                    out.append(self._text(px + dx, py + dy, label, size, INK_SECONDARY, anchor))
                    break
        return out

    def _notes(self) -> list[str]:
        out = []
        for note in self.notes:
            if "along" in note:
                (ax, ay), (bx, by) = note["along"]
                p0, p1 = (self.px(ax), self.py(ay)), (self.px(bx), self.py(by))
                angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                t = note["at"]
                cx, cy = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
                normal = math.radians(angle - 90)
                cx += math.cos(normal) * -note["offset"]
                cy += math.sin(normal) * -note["offset"]
                out.append(self._text(cx, cy, note["text"], note["size"], note["color"],
                                      note["anchor"], rotate=angle if abs(angle) > 0.01 else 0))
                continue
            x, y = self.px(note["x"]) + note["dx"], self.py(note["y"]) + note["dy"]
            out.append(self._text(x, y, note["text"], note["size"], note["color"], note["anchor"],
                                  note["weight"], rotate=float(note["rotate"] or 0)))
        return out

    def _legend(self, marks: list[tuple[float, float, float]]) -> list[str]:
        items = self.legend_items
        if not items or self.legend_position == "none":
            return []
        size = self.font_size - 1
        row = size + 7
        width = 34 + max(text_width(s.label or "", size) for s in items) + 10
        height = 10 + row * len(items)
        x0, y0, x1, y1 = self.plot
        corners = {"upper left": (x0 + 10, y0 + 8), "upper right": (x1 - width - 10, y0 + 8),
                   "lower right": (x1 - width - 10, y1 - height - 8),
                   "lower left": (x0 + 10, y1 - height - 8),
                   "right": (x1 + 18, y0 + 8)}
        position = self.legend_position
        if position == "auto":
            def cost(corner: str) -> float:
                lx, ly = corners[corner]
                pad = 14  # keep a clear margin between the legend and any mark
                inside = sum(1 for mx, my, r in marks if lx - pad - r <= mx <= lx + width + pad + r
                             and ly - pad - r <= my <= ly + height + pad + r)
                text = sum(1 for b in self.boxes if not (b[2] <= lx or b[0] >= lx + width
                                                          or b[3] <= ly or b[1] >= ly + height))
                return inside + 3 * text
            position = min(("upper left", "upper right", "lower right", "lower left"), key=cost)
        lx, ly = corners[position]
        out = [f'<g transform="translate({num(lx)},{num(ly)})">',
               f'<rect width="{num(width)}" height="{num(height)}" rx="3" fill="{SURFACE}" '
               f'fill-opacity="0.92" stroke="{GRID}"/>']
        for index, series in enumerate(items):
            cy = 5 + row * index + row / 2
            if series.kind in ("line", "step"):
                dash = f' stroke-dasharray="{series.dash}"' if series.dash else ""
                out.append(f'<line x1="8" y1="{num(cy)}" x2="28" y2="{num(cy)}" '
                           f'stroke="{series.color}" stroke-width="{num(series.width)}" '
                           f'stroke-linecap="round"{dash}/>')
                if series.marker:
                    out.append(self._marker(series.marker, 18, cy, 3.5, series.color))
            elif series.kind in ("bar", "rect"):
                out.append(f'<rect x="12" y="{num(cy - 5)}" width="10" height="10" rx="2" '
                           f'fill="{series.color}"/>')
            else:
                out.append(self._marker(series.marker or "circle", 18, cy, 4, series.color,
                                        series.extra.get("hollow", False)))
            out.append(self._text(34, cy + size * 0.35, series.label or "", size, INK,
                                  track=False))
        out.append("</g>")
        self.boxes.append((lx, ly, lx + width, ly + height))
        return out

    def to_svg(self) -> str:
        self._layout()
        x0, y0, x1, y1 = self.plot
        clip = f"{self.id_prefix}-plot"
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
               f'viewBox="0 0 {self.width} {self.height}" font-family="{escape(FONT)}">']
        if self.title:
            out.append(f"<title>{escape(self.title)}</title>")
        out.append(f'<rect width="{self.width}" height="{self.height}" fill="{SURFACE}"/>')
        out.append(f'<defs><clipPath id="{clip}"><rect x="{num(x0)}" y="{num(y0 - 2)}" '
                   f'width="{num(x1 - x0 + 2)}" height="{num(y1 - y0 + 2)}"/></clipPath></defs>')
        if self.title:
            out.append(self._text(14, 16 + self.font_size + 2, self.title, self.font_size + 2,
                                  INK, weight="600"))
        if self.subtitle:
            out.append(self._text(14, 16 + 2 * self.font_size + 10, self.subtitle,
                                  self.font_size - 1, INK_SECONDARY))
        out += self._grid_and_axes()
        marks: list[tuple[float, float, float]] = []   # markers: labels must not cover them
        ink: list[tuple[float, float, float]] = []     # markers and line samples: legend avoids
        out.append(f'<g clip-path="url(#{clip})">')
        for series in self.series:
            out += self._series_svg(series)
            if series.kind == "scatter" or (series.kind == "line" and series.marker):
                r = series.width if series.kind == "scatter" else 4
                marks += [(self.px(x), self.py(y), r + 2) for x, y in series.data]
            if series.kind in ("line", "step"):
                pts = [(self.px(x), self.py(y)) for x, y in series.data]
                for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                    steps = max(1, int(math.hypot(bx - ax, by - ay) / 6))
                    ink += [(ax + (bx - ax) * i / steps, ay + (by - ay) * i / steps, 3)
                            for i in range(steps + 1)]
            elif series.kind in ("bar", "rect"):
                for x, y in series.data:
                    ink.append((self.px(x), self.py(y), 3))
        ink += marks
        out.append("</g>")
        for series in self.series:
            if series.kind == "hline" and series.extra.get("text"):
                anchor = series.extra["anchor"]
                x = x1 - 6 if anchor == "end" else x0 + 6
                out.append(self._text(x, self.py(series.data[0][1]) - 6, series.extra["text"],
                                      self.font_size - 2, series.color, anchor))
            if series.kind == "vline" and series.extra.get("text"):
                out.append(self._text(self.px(series.data[0][0]) + 5, y0 + self.font_size,
                                      series.extra["text"], self.font_size - 2, series.color))
            if series.kind == "bar" and series.extra.get("value_labels"):
                for x, y in series.data:
                    left, width, base, top = self._bar_geometry(series, x, y)
                    above = top <= base
                    out.append(self._text(left + width / 2, top - 4 if above else top + 12,
                                          series.extra["format"].format(y), self.font_size - 3,
                                          INK_SECONDARY, "middle"))
        out += self._notes()
        for series in self.series:
            if series.kind == "scatter":
                out += self._point_labels(series, marks)
        out += self._legend(ink)
        out.append("</svg>")
        return "\n".join(out) + "\n"

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_svg(), encoding="utf-8", newline="\n")
        return path
