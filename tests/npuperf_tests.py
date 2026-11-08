#!/usr/bin/env python3
"""npuperf unit tests, and the analytic model and quantization oracle checked against npu-sim.

Tests of simulator options the npu-sim build lacks (probed with --help) are skipped.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf import (  # noqa: E402
    analytic,
    cli,
    jsonio,
    npufile,
    quant,
    roofline,
    sim,
    svg,
    table,
    trace,
    workloads,
)
from npuperf import binary as binaries  # noqa: E402

BINARY: Path


def supports(flag: str, value: str | None = None) -> bool:
    return sim.supports(BINARY, flag, value)


def run(flags: dict) -> sim.SimResult:
    return sim.run(flags, BINARY)


# ----------------------------------------------------------------------------------- svg
def sample_figure() -> svg.Figure:
    fig = svg.Figure(
        title="Sample <&>",
        subtitle="every mark kind",
        x_label="x",
        y_label="y",
        x_scale="log",
        y_scale="log",
    )
    fig.line([0.1, 1, 100], [1, 10, 10], label="roof")
    fig.scatter([0.5, 3, 30], [2, 8, 9], label="points", point_labels=["a", "b", "c"])
    fig.step([0.2, 2, 20], [1.5, 3, 6], label="step")
    fig.hline(5, label="reference")
    fig.vline(2, label="ridge")
    fig.annotate(1, 1, "note")
    fig.annotate_along((0.1, 1), (1, 10), "along")
    return fig


class SvgTests(unittest.TestCase):
    def test_output_is_deterministic(self):
        self.assertEqual(sample_figure().to_svg(), sample_figure().to_svg())
        fig = sample_figure()
        self.assertEqual(fig.to_svg(), fig.to_svg())

    def test_well_formed_white_background_fixed_number_format(self):
        text = sample_figure().to_svg()
        root = ET.fromstring(text)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        background = root.find("{http://www.w3.org/2000/svg}rect")
        self.assertEqual(background.get("fill"), "#ffffff")
        self.assertIn("Sample &lt;&amp;&gt;", text)
        numbers = re.findall(r' (?:x|y|x1|x2|y1|y2|cx|cy|r|width|height)="([^"]+)"', text)
        self.assertTrue(numbers)
        for value in numbers:
            self.assertRegex(value, r"^-?\d+(\.\d{1,2})?$")
        self.assertNotRegex(text, r'="-?[0-9.]+[eE][+-]?[0-9]')

    def test_inside_annotation_mirrors_at_the_plot_side(self):
        def anchors(**options) -> list[str | None]:
            fig = svg.Figure(x_range=(0, 1), y_range=(0, 1), legend="none")
            fig.annotate(0.97, 0.5, "a label wider than the room left", dx=5, **options)
            fig.annotate(
                0.03, 0.2, "another label wider than the room", dx=-5, anchor="end", **options
            )
            root = ET.fromstring(fig.to_svg())
            texts = [
                t
                for t in root.iter("{http://www.w3.org/2000/svg}text")
                if "label wider" in (t.text or "")
            ]
            return [t.get("text-anchor") for t in texts]

        self.assertEqual(anchors(), [None, "end"])
        self.assertEqual(anchors(inside=True), ["end", None])

    def test_log_axis_has_decade_ticks_and_minor_ticks(self):
        axis = svg.Axis("log")
        axis.fit([0.15, 70])
        self.assertEqual((axis.lo, axis.hi), (0.1, 100))
        ticks = axis.major_ticks()
        self.assertEqual(ticks, [0.1, 1, 10, 100])
        self.assertEqual([axis.label_for(t, ticks) for t in ticks], ["0.1", "1", "10", "100"])
        self.assertIn(2, axis.minor_ticks())

    def test_linear_axis_nice_ticks_and_compact_labels(self):
        axis = svg.Axis("linear", include_zero=True)
        axis.fit([30, 2300])
        ticks = axis.major_ticks()
        self.assertEqual(ticks, [0, 500, 1000, 1500, 2000, 2500])
        self.assertEqual(axis.label_for(1500, ticks), "1,500")
        big = svg.Axis("linear")
        big.fit([0, 240000])
        ticks = big.major_ticks()
        self.assertEqual(big.label_for(ticks[1], ticks), "50k")
        frac = svg.Axis("linear", lo=0, hi=1)
        frac.fit([0.1])
        ticks = frac.major_ticks()
        self.assertEqual([frac.label_for(t, ticks) for t in ticks][:3], ["0.0", "0.2", "0.4"])

    def test_every_mark_kind_renders(self):
        fig = svg.Figure(x_categories=None)
        fig.bars(["a", "b"], {"one": [1, 2], "two": [2, 0.5]}, value_labels=True)
        fig.hline(1)
        gantt = svg.Figure(y_categories=["dma", "mac"], x_range=(0, None), legend="right")
        gantt.span(0, 0, 10, label="load")
        gantt.span(1, 10, 20, label="compute")
        for figure in (fig, gantt, sample_figure()):
            root = ET.fromstring(figure.to_svg())
            self.assertGreater(len(list(root.iter())), 10)
        self.assertEqual(fig.to_svg().count('<path d="M'), 4)

    def test_legend_keeps_clear_of_the_data(self):
        fig = svg.Figure()
        xs = [i / 10 for i in range(11)]
        fig.scatter(xs[:6], [1.0] * 6, label="top left")
        fig.scatter(xs[6:], [0.0] * 5, label="bottom right")
        fig.to_svg()
        x0, y0, x1, y1 = fig.plot
        legend = fig.boxes[-1]
        self.assertGreater(legend[0], (x0 + x1) / 2)  # right half
        self.assertLess(legend[3], (y0 + y1) / 2)  # top half


class TableAndJsonTests(unittest.TestCase):
    def test_table_alignment(self):
        text = table.format_table([{"name": "a", "cycles": 1234}, {"name": "bb", "cycles": 5}])
        lines = text.splitlines()
        self.assertEqual(lines[0], "name  cycles")
        self.assertEqual(lines[2], "a      1,234")
        self.assertEqual(lines[3], "bb         5")

    def test_strict_json(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '{"a":'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                jsonio.loads_strict(text)
        self.assertEqual(jsonio.loads_strict('{"a":[1,2.5]}'), {"a": [1, 2.5]})
        self.assertEqual(jsonio.dumps({"b": 1, "a": 2}), '{\n  "a": 2,\n  "b": 1\n}\n')

    def test_write_text_overwrite_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sub" / "out.txt"
            jsonio.write_text(path, "one")
            jsonio.write_text(path, "two")
            self.assertEqual(path.read_text(), "two")
            with self.assertRaises(FileExistsError):
                jsonio.write_text(path, "three", overwrite=False)


class BinaryDiscoveryTests(unittest.TestCase):
    def test_search_order_and_variants(self):
        name = "npu-sim.exe" if os.name == "nt" else "npu-sim"
        with tempfile.TemporaryDirectory() as folder:
            build = Path(folder)
            (build / "Release").mkdir()
            release = build / "Release" / name
            release.write_bytes(b"")
            self.assertEqual(binaries.find_in(build), release.resolve())
            self.assertEqual(
                binaries.find_simulator("npu-sim", build, environ={}), release.resolve()
            )
            other = build / "elsewhere"
            other.write_bytes(b"")
            self.assertEqual(
                binaries.find_simulator("npu-sim", None, {"NPU_SIM": str(other)}), other.resolve()
            )
            with self.assertRaises(binaries.BinaryNotFound):
                binaries.find_simulator("npu-sim", None, {"NPU_SIM": str(build / "missing")})
            with self.assertRaises(binaries.BinaryNotFound):
                binaries.find_in(build / "Release" / "nothing-here")
            self.assertEqual(binaries.resolve(build), release.resolve())

    def test_environment_variable_names(self):
        self.assertEqual(
            binaries.environment_variable("npu-transformer-demo"), "NPU_TRANSFORMER_DEMO"
        )


class FlagTests(unittest.TestCase):
    def test_dict_to_argv_and_back(self):
        flags = sim.to_flags(
            {
                "tile_m": 8,
                "relu": True,
                "trace": None,
                "dump_output": False,
                "frequency_mhz": 200.5,
                "--memory-model": "axi",
                "opps": [1, 2],
            }
        )
        self.assertEqual(
            flags,
            [
                "--tile-m",
                "8",
                "--relu",
                "--frequency-mhz",
                "200.5",
                "--memory-model",
                "axi",
                "--opps",
                "1,2",
            ],
        )
        self.assertEqual(
            sim.from_flags(["--tile-m", 8, "--relu", "--output-zero", "-3"]),
            {"tile_m": 8, "relu": True, "output_zero": -3},
        )
        self.assertEqual(sim.to_flags(["--m", 4, Path("x")]), ["--m", "4", "x"])
        with self.assertRaises(ValueError):
            sim.to_flags({"x": float("nan")})

    def test_config_lookup_flat_dotted_and_nested(self):
        nested = {"shape": {"m": 4}, "arch": {"memory": {"axi": {"data_bytes": 16}}}}
        self.assertEqual(sim.config_value({"m": 3}, "m"), 3)
        self.assertEqual(sim.config_value(nested, "shape.m"), 4)
        self.assertEqual(sim.config_value(nested, "data_bytes"), 16)
        self.assertIsNone(sim.config_value(nested, "missing", None))
        with self.assertRaises(KeyError):
            sim.config_value(nested, "missing")

    def test_config_lookup_renamed_option_table_leaves(self):
        # Option-table leaves differ from the flat names (weight_bits is weights.bits);
        # both resolve.
        nested = {
            "weights": {"bits": 4, "encoding": "palette", "palette_bits": 2},
            "arch": {
                "memory": {"bytes_per_cycle": 16, "latency": 12, "burst_bytes": 32},
                "frequency_mhz": 400,
            },
        }
        flat = {
            "weight_bits": 4,
            "weight_encoding": "palette",
            "dma_burst_bytes": 32,
            "memory_latency": 12,
            "frequency_mhz": 400,
        }
        for config in (nested, flat):
            with self.subTest(config=config):
                self.assertEqual(sim.config_value(config, "weight_bits"), 4)
                self.assertEqual(sim.config_value(config, "weights.bits"), 4)
                self.assertEqual(sim.config_value(config, "weight_encoding"), "palette")
                self.assertEqual(sim.config_value(config, "dma_burst_bytes"), 32)
                self.assertEqual(sim.config_value(config, "arch.memory.burst_bytes"), 32)
                self.assertEqual(sim.config_value(config, "memory_latency"), 12)
                self.assertEqual(sim.config_value(config, "frequency_mhz"), 400)
                self.assertIsNone(sim.config_value(config, "cores", None))
        self.assertEqual(sim.config_value(nested, "memory_bytes_per_cycle"), 16)
        self.assertEqual(sim.config_value(nested, "palette_bits"), 2)

    def test_result_validation(self):
        good = {
            "schema_version": 1,
            "numerical_match": True,
            "cycles": 10,
            "macs": 1,
            "read_bytes": 1,
            "write_bytes": 1,
            "compute_busy_cycles": 2,
            "dma_busy_cycles": 3,
            "peak_inflight_tiles": 1,
            "slot_wait_cycles": 0,
            "scratchpad_bytes": 64,
            "simulation_wall_seconds": 0.001,
            "compute_utilization": 0.2,
            "memory_utilization": 0.3,
            "modeled_latency_ns": 50,
            "config": {},
            "output_hash": 7,
        }
        sim.validate_result(dict(good))
        sim.validate_result({**good, "schema_version": 2})
        for version in (0, 3, True, 2.0, "2"):
            with self.subTest(schema_version=version), self.assertRaises(sim.CheckError):
                sim.validate_result({**good, "schema_version": version})
        sim.validate_result({**good, "numerics": "bit-exact"})
        skipped = {**good, "numerics": "skipped"}
        del skipped["numerical_match"], skipped["output_hash"]
        sim.validate_result(skipped)
        # npu-sim --timing-only prints null for both.
        sim.validate_result({**skipped, "numerical_match": None, "output_hash": None})
        for change in (
            {"numerical_match": False},
            {"cycles": 0},
            {"dma_busy_cycles": 11},
            {"numerics": "approximate"},
            {"macs": -1},
            {"config": None},
            {"numerics": "bit-exact", "numerical_match": None},
            {"numerics": "skipped", "numerical_match": True},
            {"compute_utilization": 1.5},
            {"modeled_latency_ns": float("nan")},
        ):
            with self.subTest(change=change), self.assertRaises(sim.CheckError):
                sim.validate_result({**good, **change})
        for key in (
            "peak_inflight_tiles",
            "slot_wait_cycles",
            "scratchpad_bytes",
            "simulation_wall_seconds",
            "compute_utilization",
            "memory_utilization",
            "modeled_latency_ns",
            "dma_busy_cycles",
        ):
            truncated = dict(good)
            del truncated[key]
            with self.subTest(missing=key), self.assertRaises(sim.CheckError):
                sim.validate_result(truncated)


class CliTests(unittest.TestCase):
    def test_guard_maps_errors_to_status_and_one_line(self):
        @cli.main_guard("tool")
        def failing(argv):
            raise argv

        import contextlib
        import io

        for error, status in (
            (ValueError("bad\ninput"), 2),
            (FileExistsError("x"), 2),
            (RuntimeError("failed"), 1),
            (OSError("disk"), 1),
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(failing(error), status)
            self.assertTrue(stderr.getvalue().startswith("tool: "))
            self.assertEqual(len(stderr.getvalue().splitlines()), 1)

    def test_parser_error_is_one_line_with_status_two(self):
        command = [sys.executable, "-m", "npuperf.sweep", "--timeout", "0"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.startswith("sweep: argument --timeout"), result.stderr)
        self.assertEqual(len(result.stderr.strip().splitlines()), 1)


# ----------------------------------------------------------------------------- quantization
def tosa_apply_scale_32(value, multiplier, shift, double_round=False):
    """TOSA 1.0.2 scale_helpers.tosac apply_scale_32, transcribed for the cross-check."""
    assert multiplier >= 0 and 2 <= shift <= 62
    assert -(1 << (shift - 1)) <= value < (1 << (shift - 1))
    rounding = 1 << (shift - 1)
    if double_round and shift > 31:
        rounding += (1 << 30) if value >= 0 else -(1 << 30)
    return (value * multiplier + rounding) >> shift


class QuantOracleTests(unittest.TestCase):
    def test_golden_vectors(self):
        # Golden vectors of docs/numerics.md: (x, multiplier, TFLite shift, single, double).
        for x, multiplier, tfl_shift, single, double in (
            (3, 1861152495, -1, 1, 2),
            (-6, 1073741824, -1, -1, -2),
            (-3, 1073741824, 0, -1, -1),
            (5038971, 2062684434, -2, 1209998, 1209999),
        ):
            shift = 31 - tfl_shift
            with self.subTest(x=x):
                self.assertEqual(quant.rescale_tosa(x, multiplier, shift, False), single)
                self.assertEqual(quant.rescale_tosa(x, multiplier, shift, True), double)
                self.assertEqual(quant.tflite_single_rounding(x, multiplier, tfl_shift), single)
                self.assertEqual(quant.tflite_double_rounding(x, multiplier, tfl_shift), double)

    def test_native_rounding_is_sign_magnitude(self):
        # -1.5: native rounds the magnitude (-2 either tie rule); TOSA single gives -1.
        self.assertEqual(quant.rescale_native(-3, 1, 1, "away"), -2)
        self.assertEqual(quant.rescale_native(-3, 1, 1, "even"), -2)
        self.assertEqual(quant.rescale_native(-5, 1, 1, "even"), -2)
        self.assertEqual(quant.rescale_native(5, 1, 1, "away"), 3)
        self.assertEqual(quant.rescale_tosa(-3, 1 << 30, 31), -1)

    def test_tflite_formulation_equals_tosa_pseudocode(self):
        rng = random.Random(20261107)
        cases = []
        for shift in range(2, 63):
            half = 1 << (shift - 1)
            low, high = max(-half, quant.INT32_MIN), min(half - 1, quant.INT32_MAX)
            for value in {low, high, 0, 1, -1, min(high, 1 << 20), max(low, -(1 << 20))}:
                for multiplier in (0, 1, 1 << 30, (1 << 31) - 1, 1518500250):
                    cases.append((value, multiplier, shift))
        for _ in range(20000):
            shift = rng.randint(2, 62)
            half = 1 << (shift - 1)
            value = rng.randint(max(-half, quant.INT32_MIN), min(half - 1, quant.INT32_MAX))
            cases.append((value, rng.randint(1 << 30, (1 << 31) - 1), shift))
        for value, multiplier, shift in cases:
            for double in (False, True):
                self.assertEqual(
                    quant.rescale_tosa(value, multiplier, shift, double),
                    tosa_apply_scale_32(value, multiplier, shift, double),
                    (value, multiplier, shift, double),
                )

    def test_tosa_requirements_are_enforced(self):
        for args in ((1 << 20, 1, 20), (0, 1, 1), (0, 1, 63), (0, -1, 31), (1 << 31, 1, 40)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                quant.rescale_tosa(*args)

    def test_quantize_multiplier_round_trip(self):
        for scale in (1.0, 0.5, 0.3, 2e-3, 1.7e-6, 12.25, 1e-9):
            multiplier, shift = quant.quantize_multiplier(scale)
            with self.subTest(scale=scale):
                self.assertTrue((1 << 30) <= multiplier < (1 << 31))
                self.assertTrue(2 <= shift <= 62)
                self.assertLess(abs(multiplier / 2**shift - scale) / scale, 2**-30)

    def test_requantize_adds_zero_point_relu_and_saturates(self):
        q = {"multiplier": 1, "shift": 0, "output_zero": 10, "relu": True}
        self.assertEqual(quant.requantize(-50, q), 10)
        self.assertEqual(quant.requantize(500, q), 127)
        self.assertEqual(quant.requantize(-500, {**q, "relu": False}), -128)

    def test_grouped_reference_equals_per_group_gemms(self):
        rng = random.Random(3)
        m, n, k, g = 3, 2, 4, 3
        a = [rng.randint(-128, 127) for _ in range(m * g * k)]
        w = [rng.randint(-128, 127) for _ in range(g * k * n)]
        b = [rng.randint(-1000, 1000) for _ in range(g * n)]
        acc, _ = quant.gemm_reference(a, w, b, m, n, k, groups=g, input_zero=3)
        for group in range(g):
            ag = [a[r * g * k + group * k + i] for r in range(m) for i in range(k)]
            single, _ = quant.gemm_reference(
                ag,
                w[group * k * n : (group + 1) * k * n],
                b[group * n : (group + 1) * n],
                m,
                n,
                k,
                input_zero=3,
            )
            for r in range(m):
                self.assertEqual(
                    acc[r * g * n + group * n : r * g * n + (group + 1) * n],
                    single[r * n : (r + 1) * n],
                )


class QuantSimulatorTests(unittest.TestCase):
    """The simulator's requantized outputs against the Python oracle, per channel."""

    def check(self, lanes, rounding="away", rescale=None, small=False, seed=5):
        rng = random.Random(seed)
        m, k, n = 32, 3, len(lanes)
        span, bias_span = (3, 20) if small else (20, 2000)
        a = [rng.randint(-span, span) for _ in range(m * k)]
        w = [rng.randint(-span, span) for _ in range(k * n)]
        bias = [rng.randint(-bias_span, bias_span) for _ in range(n)]
        params = [
            {
                "weight_zero": lane[0],
                "multiplier": lane[1],
                "shift": lane[2],
                "rounding": rounding,
                "rescale": rescale or "native",
            }
            for lane in lanes
        ]
        accumulators, outputs = quant.gemm_reference(
            a,
            w,
            bias,
            m,
            n,
            k,
            input_zero=-3,
            lanes=params,
            quant={"output_zero": 4, "relu": False},
        )
        with tempfile.TemporaryDirectory() as folder:
            problem = npufile.write_problem(Path(folder) / "p.npu", m, n, k, a, w, bias)
            lanes_file = npufile.write_quant(Path(folder) / "p.quant", lanes)
            flags = {
                "input": problem,
                "quantization_file": lanes_file,
                "input_zero": -3,
                "output_zero": 4,
                "rounding": rounding,
                "dump_output": True,
                "rescale": rescale,
            }
            result = run(flags)
        self.assertEqual(result["accumulators"], accumulators)
        self.assertEqual(result["outputs"], outputs)
        unsaturated = sum(-128 < value < 127 for value in outputs)
        self.assertGreater(unsaturated, len(outputs) // 2)

    def test_native_per_channel(self):
        rng = random.Random(11)
        lanes = []
        for _ in range(16):
            shift = rng.randint(16, 31)
            lanes.append(
                (rng.randint(-2, 2), max(1, round(rng.uniform(0.005, 0.05) * 2**shift)), shift)
            )
        ties = [(0, 1, 1), (0, 1, 2), (0, 3, 2), (1, 5, 3), (0, 1, 0)]
        for rounding in ("away", "even"):
            with self.subTest(rounding=rounding):
                self.check(lanes, rounding)
                self.check(ties, rounding, small=True)

    def test_tosa_per_channel(self):
        if not supports("--rescale"):
            self.skipTest("npu-sim has no --rescale")
        rng = random.Random(12)
        lanes = [
            (rng.randint(-2, 2), rng.randint(1 << 30, quant.INT32_MAX), rng.randint(36, 40))
            for _ in range(16)
        ]
        ties = [(0, 1 << 30, 31), (0, 1 << 30, 32), (0, 1 << 30, 33), (0, 1 << 30, 34)]
        for mode in ("tosa-single", "tosa-double"):
            with self.subTest(mode=mode):
                self.check(lanes, rescale=mode)
                self.check(ties, rescale=mode, small=True)


# --------------------------------------------------------------------------- analytic model
SHAPES = ((16, 16, 32), (17, 13, 31), (64, 64, 128), (33, 70, 45))
TILES = ((8, 8), (4, 16), (16, 4))


def dense_configs():
    for (m, n, k), (tm, tn), tk, macs, bw, bits in itertools.product(
        SHAPES, TILES, (0, 8, 16), (16, 64, 256), (4, 16, 64), (8, 4)
    ):
        yield {
            "m": m,
            "n": n,
            "k": k,
            "tile_m": tm,
            "tile_n": tn,
            "tile_k": tk,
            "macs_per_cycle": macs,
            "memory_bytes_per_cycle": bw,
            "weight_bits": bits,
        }


class AnalyticSimulatorTests(unittest.TestCase):
    def one_panel_buffer(self) -> dict:
        return {"panel_buffers": 1} if supports("--panel-buffers") else {}

    def test_closed_form_matches_serial_dense_simulator(self):
        """Traffic, requests and busy cycles are exact on 648 dense configurations."""
        count = 0
        for flags in dense_configs():
            result = run({**flags, "buffer_slots": 2})
            model = analytic.analyze(analytic.Design.from_result(result))
            got = (
                result["read_bytes"],
                result["write_bytes"],
                result["compute_busy_cycles"],
                result["dma_busy_cycles"],
                result["dma_requests"],
                result["macs"],
            )
            want = (
                model.read_bytes,
                model.write_bytes,
                model.compute_busy_cycles,
                model.dma_busy_cycles,
                model.dma_requests,
                model.macs,
            )
            self.assertEqual(got, want, flags)
            self.assertLessEqual(model.lower_bound_cycles, result.cycles, flags)
            self.assertLessEqual(result.cycles, model.serial_cycles, flags)
            count += 1
        self.assertEqual(count, 648)

    def test_closed_form_matches_array_simulator(self):
        """The same on arrays, with the build's default dataflow (os-per-wave on builds
        without --dataflow)."""
        for (rows, cols), flags in itertools.product(
            ((4, 4), (8, 4), (3, 5)), itertools.islice(dense_configs(), 0, 648, 9)
        ):
            flags = {key: value for key, value in flags.items() if key != "macs_per_cycle"}
            result = run({**flags, "array_rows": rows, "array_cols": cols, "buffer_slots": 2})
            design = analytic.Design.from_result(result)
            if not supports("--dataflow"):
                self.assertEqual(design.dataflow, "os-per-wave")
            model = analytic.analyze(design)
            got = (
                result["read_bytes"],
                result["write_bytes"],
                result["compute_busy_cycles"],
                result["dma_busy_cycles"],
                result["dma_requests"],
            )
            want = (
                model.read_bytes,
                model.write_bytes,
                model.compute_busy_cycles,
                model.dma_busy_cycles,
                model.dma_requests,
            )
            self.assertEqual(got, want, (rows, cols, flags))
            self.assertLessEqual(model.lower_bound_cycles, result.cycles, flags)
            self.assertLessEqual(result.cycles, model.serial_cycles, flags)
            single = run(
                {
                    **flags,
                    "array_rows": rows,
                    "array_cols": cols,
                    "buffer_slots": 1,
                    **self.one_panel_buffer(),
                }
            )
            self.assertEqual(
                analytic.analyze(analytic.Design.from_result(single)).exact_cycles,
                single.cycles,
                (rows, cols, flags),
            )

    def test_schedules_without_overlap_are_exact_and_estimates_are_close(self):
        ratios = []
        for flags in dense_configs():
            single = run({**flags, "buffer_slots": 1, **self.one_panel_buffer()})
            model = analytic.analyze(analytic.Design.from_result(single))
            self.assertIsNotNone(model.exact_cycles)
            self.assertEqual(model.exact_cycles, single.cycles, flags)
            double = run({**flags, "buffer_slots": 2, **self.one_panel_buffer()})
            estimate = analytic.analyze(analytic.Design.from_result(double)).estimate_cycles
            ratios.append(estimate / double.cycles)
        self.assertGreater(min(ratios), 0.8)
        self.assertLess(max(ratios), 1.15)

    def test_scratchpad_reservation(self):
        for flags in itertools.islice(dense_configs(), 0, 648, 7):
            result = run({**flags, **self.one_panel_buffer()})
            design = analytic.Design.from_result(result)
            self.assertEqual(
                result["scratchpad_bytes"],
                analytic.scratchpad_bytes(design, panel_buffers=1),
                flags,
            )
        if not supports("--panel-buffers"):
            return
        # Two panel buffers: the parameters are held once per slot, and a tile with one
        # panel reserves one buffer.
        for flags in itertools.islice(dense_configs(), 3, 648, 7):
            result = run({**flags, "panel_buffers": 2})
            design = analytic.Design.from_result(result)
            self.assertEqual(result["scratchpad_bytes"], analytic.scratchpad_bytes(design), flags)

    def test_roofline_invariant(self):
        for flags in itertools.islice(dense_configs(), 0, 648, 5):
            result = run(flags)
            design = analytic.Design.from_result(result)
            point = roofline.from_result(result, "x")
            self.assertLessEqual(
                point.macs_per_cycle, analytic.attainable(design, point.intensity) * (1 + 1e-12)
            )

    def test_dma_bursts(self):
        for burst in (16, 50, 256):
            result = run({"m": 33, "n": 70, "k": 45, "dma_burst_bytes": burst})
            model = analytic.analyze(analytic.Design.from_result(result))
            self.assertEqual(
                (result["dma_busy_cycles"], result["dma_bursts"]),
                (model.dma_busy_cycles, model.dma_bursts),
            )

    def test_grouped_gemm(self):
        if not supports("--groups"):
            self.skipTest("npu-sim has no --groups")
        for groups, (m, n, k) in itertools.product((2, 3), ((16, 1, 9), (17, 13, 31))):
            result = run({"m": m, "n": n, "k": k, "groups": groups, "tile_k": 8})
            model = analytic.analyze(analytic.Design.from_result(result))
            self.assertEqual(model.design.groups, groups)
            self.assertEqual(
                (
                    result["macs"],
                    result["read_bytes"],
                    result["write_bytes"],
                    result["compute_busy_cycles"],
                    result["dma_busy_cycles"],
                ),
                (
                    model.macs,
                    model.read_bytes,
                    model.write_bytes,
                    model.compute_busy_cycles,
                    model.dma_busy_cycles,
                ),
            )

    def test_array_dataflows(self):
        if not supports("--dataflow"):
            self.skipTest("npu-sim has no --dataflow")
        for dataflow, (rows, cols) in itertools.product(("os", "ws"), ((4, 4), (8, 4), (3, 5))):
            for flags in itertools.islice(dense_configs(), 0, 648, 37):
                flags = {k: v for k, v in flags.items() if k != "macs_per_cycle"}
                result = run(
                    {**flags, "array_rows": rows, "array_cols": cols, "dataflow": dataflow}
                )
                model = analytic.analyze(analytic.Design.from_result(result))
                self.assertEqual(model.design.dataflow, dataflow)
                self.assertEqual(
                    result["compute_busy_cycles"],
                    model.compute_busy_cycles,
                    (dataflow, rows, cols, flags),
                )
                self.assertEqual(result["read_bytes"], model.read_bytes)

    def test_panel_buffers(self):
        if not supports("--panel-buffers"):
            self.skipTest("npu-sim has no --panel-buffers")
        for flags in itertools.islice(dense_configs(), 1, 648, 11):
            one = run({**flags, "panel_buffers": 1})
            two = run({**flags, "panel_buffers": 2})
            for key in ("read_bytes", "write_bytes", "compute_busy_cycles", "dma_busy_cycles"):
                self.assertEqual(one[key], two[key], (key, flags))
            self.assertLessEqual(two.cycles, one.cycles, flags)
            model = analytic.analyze(analytic.Design.from_result(two))
            self.assertLessEqual(model.lower_bound_cycles, two.cycles, flags)

    def test_axi_memory_model(self):
        if not supports("--memory-model", "axi"):
            self.skipTest("npu-sim has no AXI memory model")
        for flags in itertools.islice(dense_configs(), 0, 648, 13):
            # npu-sim rejects the serial DMA's bandwidth flag under the AXI model.
            flags = {k: v for k, v in flags.items() if k != "memory_bytes_per_cycle"}
            result = run({**flags, "memory_model": "axi"})
            try:
                design = analytic.Design.from_result(result)
                model = analytic.analyze(design)
            except ValueError as error:
                self.skipTest(f"result config lacks AXI parameters: {error}")
            self.assertEqual(design.memory_model, "axi")
            self.assertEqual(
                (result["read_bytes"], result["write_bytes"]),
                (model.read_bytes, model.write_bytes),
                flags,
            )
            self.assertLessEqual(model.lower_bound_cycles, result.cycles, flags)

    def test_cluster(self):
        if not supports("--cores") or not supports("--memory-model", "axi"):
            self.skipTest("npu-sim has no --cores")
        for cores, partition in itertools.product((2, 4), ("m", "n")):
            flags = {
                "m": 64,
                "n": 64,
                "k": 128,
                "cores": cores,
                "partition": partition,
                "memory_model": "axi",
            }
            result = run(flags)
            design = analytic.Design.from_result(result)
            model = analytic.analyze(design)
            self.assertEqual(result["macs"], model.macs)
            self.assertLessEqual(
                analytic.ceil_div(model.compute_busy_cycles, cores), result.cycles, flags
            )

    def test_timing_only(self):
        if not supports("--timing-only"):
            self.skipTest("npu-sim has no --timing-only")
        flags = {"m": 3136, "n": 64, "k": 576, "tile_m": 64, "tile_n": 64, "timing_only": True}
        result = run(flags)
        self.assertEqual(result["numerics"], "skipped")
        model = analytic.analyze(analytic.Design.from_result(result))
        self.assertEqual(
            (result["read_bytes"], result["write_bytes"], result["compute_busy_cycles"]),
            (model.read_bytes, model.write_bytes, model.compute_busy_cycles),
        )


class AnalyticUnitTests(unittest.TestCase):
    def test_array_dataflow_formulas(self):
        os_design = analytic.Design(array_rows=4, array_cols=8, pipeline_cycles=3, dataflow="os")
        # OS: ceil(10/4)*ceil(12/8) waves x K + (A_r + A_c - 2) + pipeline
        self.assertEqual(analytic.panel_compute_cycles(os_design, 10, 12, 20), 3 * 2 * 20 + 10 + 3)
        ws = os_design.replace(dataflow="ws")
        # WS: ceil(20/4)*ceil(12/8) folds x M + A_r + (A_r + A_c - 2) + pipeline
        self.assertEqual(analytic.panel_compute_cycles(ws, 10, 12, 20), 5 * 2 * 10 + 4 + 10 + 3)
        # A tile shorter than the array cannot hide the next A_r-cycle weight preload: every
        # fold but the last occupies A_r cycles.
        self.assertEqual(analytic.panel_compute_cycles(ws, 1, 12, 20), 4 + 9 * 4 + 1 + 10 + 3)
        self.assertEqual(os_design.peak_macs_per_cycle, 32)

    def test_grouped_traffic_is_groups_times_one_group(self):
        one = analytic.analyze(analytic.Design(m=20, n=3, k=9, tile_k=4))
        many = analytic.analyze(analytic.Design(m=20, n=3, k=9, tile_k=4, groups=5))
        self.assertEqual(many.read_bytes, 5 * one.read_bytes)
        self.assertEqual(many.compute_busy_cycles, 5 * one.compute_busy_cycles)

    def test_axi_bounds(self):
        design = analytic.Design(
            memory_model="axi",
            axi_data_bytes=16,
            axi_max_burst_beats=16,
            axi_read_outstanding=4,
            axi_write_outstanding=2,
            axi_read_latency=40,
            axi_write_latency=10,
            dram_bytes_per_cycle=8,
        )
        bounds = analytic.axi_bounds(design)
        model = analytic.analyze(design)
        self.assertEqual(
            bounds["dram_cycles"], analytic.ceil_div(model.read_bytes + model.write_bytes, 8)
        )
        self.assertGreaterEqual(bounds["lower_bound_cycles"], bounds["dram_cycles"])
        self.assertGreaterEqual(bounds["read_beats"], model.read_bytes // 16)
        self.assertEqual(design.bandwidth, 8.0)
        with self.assertRaises(ValueError):
            analytic.axi_bounds(analytic.Design(memory_model="axi"))

    # The same result config in the nested option-table layout and the flat layout.
    NESTED = {
        "name": "",
        "description": "",
        "shape": {"m": 16, "n": 16, "k": 32, "groups": 1, "seed": 1},
        "arch": {
            "compute": {
                "macs_per_cycle": 64,
                "array_rows": 0,
                "array_cols": 0,
                "pipeline_cycles": 4,
                "weight_decode_values_per_cycle": 16,
            },
            "memory": {"bytes_per_cycle": 16, "latency": 12, "burst_bytes": 16},
            "sram_bytes": 262144,
            "weight_cache_bytes": 0,
            "frequency_mhz": 200,
        },
        "mapping": {"tile_m": 8, "tile_n": 8, "tile_k": 0, "buffer_slots": 2},
        "weights": {"bits": 4, "encoding": "dense", "palette_bits": 4, "sparse_block": 8},
        "quant": {"weight_zero_points": [], "multipliers": [], "shifts": []},
        "sim": {"timing": "event", "timing_only": False, "trace_path": ""},
    }
    FLAT = {
        "m": 16,
        "n": 16,
        "k": 32,
        "tile_m": 8,
        "tile_n": 8,
        "tile_k": 0,
        "weight_bits": 4,
        "weight_encoding": "dense",
        "palette_bits": 4,
        "sparse_block": 8,
        "weight_decode_values_per_cycle": 16,
        "array_rows": 0,
        "array_cols": 0,
        "macs_per_cycle": 64,
        "memory_bytes_per_cycle": 16,
        "memory_latency": 12,
        "dma_burst_bytes": 16,
        "weight_cache_bytes": 0,
        "pipeline_cycles": 4,
        "buffer_slots": 2,
        "sram_bytes": 262144,
        "seed": 1,
        "frequency_mhz": 200,
        "timing": "event",
        "quant": {"weight_zero_points": []},
    }

    def test_design_from_nested_and_flat_result_configs(self):
        nested = analytic.Design.from_result({"config": self.NESTED})
        flat = analytic.Design.from_result({"config": self.FLAT})
        self.assertEqual(nested, flat)
        self.assertEqual(
            (nested.weight_bits, nested.weight_encoding, nested.dma_burst_bytes), (4, "dense", 16)
        )
        self.assertEqual(
            (nested.panel_buffers, nested.dataflow, nested.memory_model), (1, "os", "serial")
        )
        model = analytic.analyze(nested)
        # 2 x 2 tiles of 8 x 8, one panel each: A 8*32 + W 32*8*4/8 + bias 4*8 bytes.
        self.assertEqual(model.read_bytes, 4 * (8 * 32 + 128 + 32))
        self.assertEqual(model.dma_bursts, 4 * analytic.ceil_div(416, 16) + 4 * 4)
        palette = json.loads(json.dumps(self.NESTED))
        palette["weights"]["encoding"] = "palette"
        # 32 x 8 values, 4-bit indices: 4-byte header, 16-entry LUT, 128 index bytes.
        self.assertEqual(
            analytic.analyze(analytic.Design.from_result({"config": palette})).weight_bytes,
            4 * (4 + 16 + 128),
        )
        sparse = json.loads(json.dumps(self.NESTED))
        sparse["weights"]["encoding"] = "sparse"
        with self.assertRaisesRegex(ValueError, "dense and palette weights"):
            analytic.analyze(analytic.Design.from_result({"config": sparse}))
        array = json.loads(json.dumps(self.NESTED))
        array["arch"]["compute"].update(array_rows=4, array_cols=4)
        self.assertEqual(analytic.Design.from_result({"config": array}).dataflow, "os-per-wave")
        array["arch"]["compute"]["dataflow"] = "ws"
        self.assertEqual(analytic.Design.from_result({"config": array}).dataflow, "ws")

    def test_design_from_result_requires_the_keys_it_models(self):
        for path in (
            "weights.bits",
            "weights.encoding",
            "arch.memory.burst_bytes",
            "mapping.tile_k",
            "arch.compute.pipeline_cycles",
        ):
            config = json.loads(json.dumps(self.NESTED))
            group, *middle, leaf = path.split(".")
            node = config[group]
            for part in middle:
                node = node[part]
            del node[leaf]
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, path):
                analytic.Design.from_result({"config": config})

    def test_os_per_wave_formula(self):
        design = analytic.Design(
            array_rows=4, array_cols=8, pipeline_cycles=3, dataflow="os-per-wave"
        )
        # ceil(10/4)*ceil(12/8) waves x (K + A_r + A_c - 2) + pipeline
        self.assertEqual(analytic.panel_compute_cycles(design, 10, 12, 20), 3 * 2 * 30 + 3)

    def test_from_flags_strict_rejects_unknown_keys(self):
        design = analytic.Design.from_flags(
            {"--tile-m": 4, "mapping.tile_n": 2, "macs_per_cycle": 32}, strict=True
        )
        self.assertEqual((design.tile_m, design.tile_n, design.macs_per_cycle), (4, 2, 32))
        with self.assertRaisesRegex(ValueError, "did you mean 'macs_per_cycle'"):
            analytic.Design.from_flags({"macs_per_cyle": 256}, strict=True)
        self.assertEqual(analytic.Design.from_flags({"input": "x.npu", "tile_m": 4}).tile_m, 4)

    def test_rejected_designs(self):
        for design in (
            analytic.Design(weight_encoding="sparse"),
            analytic.Design(cores=2),
            analytic.Design(memory_model="tlm"),
        ):
            with self.subTest(design=design), self.assertRaises(ValueError):
                analytic.analyze(design)
            with self.subTest(design=design), self.assertRaises(ValueError):
                analytic.schedule(design)


# ------------------------------------------------------------------------------- workloads
class WorkloadTests(unittest.TestCase):
    TOTALS = {
        "mobilenet_v2_1.0_224": 300_774_272,
        "resnet50_v1.5_224": 4_089_184_256,
        "llama3.2_1b_prefill_s128": 125_890_461_696,
        "llama3.2_1b_decode_kv129": 1_244_200_960,
        "bert_base_encoder_layer_s128": 931_135_488,
    }

    def test_published_parameter_counts_and_mac_totals(self):
        self.assertEqual(sorted(workloads.names()), sorted(self.TOTALS))
        params = {
            "mobilenet_v2_1.0_224": ("computed_torchvision_style_params", 3_504_872),
            "resnet50_v1.5_224": ("computed_torchvision_style_params", 25_557_032),
            "llama3.2_1b_prefill_s128": ("computed_total_params", 1_235_814_400),
            "bert_base_encoder_layer_s128": ("computed_bert_base_params", 109_482_240),
        }
        for name, macs in self.TOTALS.items():
            document = workloads.load(name)
            with self.subTest(name=name):
                self.assertEqual(document["totals"]["macs"], macs)
                self.assertEqual(sum(g.total_macs for g in workloads.gemms(document)), macs)
                self.assertTrue(all(url.startswith("https://") for url in document["source"]))
                if name in params:
                    key, count = params[name]
                    self.assertEqual(document["totals"]["param_check"][key], count)

    def test_generator_matches_committed_files(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "gen_workloads.py"), "--check"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lowering(self):
        mobilenet = workloads.load("mobilenet_v2_1.0_224")
        depthwise = [
            workloads.lower(x) for x in mobilenet["layers"] if x["op"] == "depthwise_conv2d"
        ]
        self.assertEqual(len(depthwise), 17)
        first = depthwise[0]
        self.assertEqual(
            (first.m, first.n, first.k, first.groups, first.calls), (112 * 112, 1, 9, 32, 1)
        )
        llama = workloads.load("llama3.2_1b_decode_kv129")
        scores = workloads.lower(next(x for x in llama["layers"] if x["name"] == "attn_scores_QKt"))
        self.assertEqual(
            (scores.m, scores.n, scores.k, scores.groups, scores.calls), (4, 129, 64, 8, 16)
        )

    def test_split_only_where_needed(self):
        gemm = workloads.Gemm("fc", 1, 128256, 2048)
        self.assertEqual(workloads.split(gemm, timing_only=True), [gemm])
        pieces = workloads.split(gemm, timing_only=False)
        self.assertGreater(sum(p.calls for p in pieces), 1)
        self.assertTrue(all(workloads.fits(p, timing_only=False) for p in pieces))
        self.assertEqual(sum(p.total_macs for p in pieces), gemm.total_macs)
        legacy = workloads.split(
            workloads.Gemm("conv", 12544, 32, 27), True, max_dimension=4096, max_macs=100_000_000
        )
        self.assertEqual(sum(p.total_macs for p in legacy), 12544 * 32 * 27)
        self.assertTrue(all(p.m <= 4096 for p in legacy))
        with self.assertRaisesRegex(ValueError, "limit of 4096; a split K needs partial-sum"):
            workloads.split(workloads.Gemm("deep", 1, 1, 8192), True, max_dimension=4096)

    def test_grouped_rows_obey_the_dimension_limit(self):
        # Activation rows hold G*K values and output rows G*N.
        wide = workloads.Gemm("dw", 100, 1, 9, groups=1000)
        self.assertFalse(workloads.fits(wide, True, max_dimension=4096))
        pieces = workloads.split(wide, True, max_dimension=4096)
        self.assertTrue(all(p.groups * p.k <= 4096 for p in pieces))
        self.assertEqual(sum(p.total_macs for p in pieces), wide.total_macs)

    def test_calls_follow_the_binary(self):
        options = workloads.call_options(BINARY)
        self.assertEqual(options["timing_only"], supports("--timing-only"))
        self.assertEqual(options["grouped"], supports("--groups"))
        if not options["timing_only"]:
            self.assertEqual(options["max_dimension"], 4096)
        mobilenet = workloads.load("mobilenet_v2_1.0_224")
        calls = [gemm for _, gemm in workloads.iter_calls(mobilenet, **options)]
        self.assertEqual(sum(g.total_macs for g in calls), self.TOTALS["mobilenet_v2_1.0_224"])
        limit = options.get("max_dimension", workloads.MAX_DIMENSION)
        for gemm in calls:
            self.assertLessEqual(max(gemm.m, gemm.groups * gemm.n, gemm.groups * gemm.k), limit)
            if not options["grouped"]:
                self.assertEqual(gemm.groups, 1)
        ungrouped = [g for _, g in workloads.iter_calls(mobilenet, grouped=False)]
        self.assertTrue(all(g.groups == 1 for g in ungrouped))
        self.assertEqual(sum(g.total_macs for g in ungrouped), self.TOTALS["mobilenet_v2_1.0_224"])


class ExperimentScriptTests(unittest.TestCase):
    def tool(self, *arguments: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

    def build_dir(self) -> Path:
        return BINARY.parent.parent if BINARY.parent.name == "Release" else BINARY.parent

    def test_sweep_rows_carry_the_numerical_match_column(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.tool("npuperf.sweep", "--binary", str(BINARY), "--output-dir", folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(Path(folder) / "sweep.csv", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["numerical_match"] == "True" for row in rows))

    def test_mobile_suite_takes_the_demos_from_the_npu_sim_build(self):
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in ("NPU_SIM", "NPU_TRANSFORMER_DEMO", "NPU_MOBILE_DEMO")
        }
        try:
            binaries.find_in(self.build_dir(), "npu-transformer-demo")
        except binaries.BinaryNotFound:
            self.skipTest("this build directory has npu-sim only")
        for given in (BINARY, self.build_dir()):
            with self.subTest(binary=given), tempfile.TemporaryDirectory() as folder:
                result = self.tool(
                    "npuperf.mobile_suite",
                    "--binary",
                    str(given),
                    "--output-dir",
                    folder,
                    "--format",
                    "json",
                    env=environment,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((Path(folder) / "summary.json").is_file())
        # A build directory without the demos fails, even when another build has them.
        with tempfile.TemporaryDirectory() as folder:
            # resolve() expands Windows 8.3 short names (RUNNER~1) to match the tool's message.
            lone = Path(folder).resolve() / "build"
            lone.mkdir()
            shutil.copy2(BINARY, lone / BINARY.name)
            result = self.tool(
                "npuperf.mobile_suite",
                "--binary",
                str(lone),
                "--output-dir",
                str(Path(folder) / "out"),
                env=environment,
            )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertTrue(
            result.stderr.startswith("mobile_suite: npu-transformer-demo"), result.stderr
        )
        self.assertIn(str(lone), result.stderr)


class RooflineAndTraceTests(unittest.TestCase):
    def test_roofline_figure_and_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "roof.svg"
            command = [
                sys.executable,
                "-m",
                "npuperf.roofline",
                "--workload",
                "resnet50_v1.5_224",
                "--set",
                "macs_per_cycle=256",
                "--set",
                "memory_bytes_per_cycle=32",
                "--out",
                str(out),
                "--format",
                "json",
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            layers = json.loads(result.stdout)["layers"]
            self.assertEqual(sum(x["macs"] for x in layers), 4_089_184_256)
            self.assertTrue(all(x["bound"] in ("compute", "bandwidth", "latency") for x in layers))
            ET.fromstring(out.read_text(encoding="utf-8"))
            bad = subprocess.run(
                command[:4] + ["no-such-network"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(bad.returncode, 2)
            self.assertTrue(bad.stderr.startswith("roofline: no workload"), bad.stderr)
            typo = subprocess.run(
                command[:6] + ["macs_per_cyle=256"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(typo.returncode, 2)
            self.assertEqual(
                typo.stderr,
                "roofline: unknown design parameter 'macs_per_cyle'; "
                "did you mean 'macs_per_cycle'?\n",
            )

    def test_roofline_simulate_on_a_small_workload(self):
        # A depthwise layer (grouped) and a dense layer with M = 5000, above the 4096-row
        # limit of builds without --timing-only.
        layers = [
            {
                "name": "dw",
                "op": "depthwise_conv2d",
                "in_c": 8,
                "out_c": 8,
                "groups": 8,
                "kernel_h": 3,
                "kernel_w": 3,
                "out_h": 6,
                "out_w": 6,
                "repeat": 1,
                "macs": 6 * 6 * 8 * 9,
            },
            {
                "name": "fc",
                "op": "dense",
                "M": 5000,
                "N": 8,
                "K": 16,
                "repeat": 2,
                "macs": 5000 * 8 * 16,
            },
        ]
        workload = {"schema_version": 1, "name": "tiny", "layers": layers}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.json"
            path.write_text(json.dumps(workload), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "npuperf.roofline",
                    "--workload",
                    str(path),
                    "--simulate",
                    "--binary",
                    str(BINARY),
                    "--set",
                    "tile_k=8",
                    "--format",
                    "json",
                    "--out",
                    str(Path(folder) / "roof.svg"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads(result.stdout)["layers"]
            ET.fromstring((Path(folder) / "roof.svg").read_text(encoding="utf-8"))
        self.assertEqual(sum(row["macs"] for row in rows), 6 * 6 * 8 * 9 + 2 * 5000 * 8 * 16)
        limit = 4096 if not supports("--timing-only") else workloads.MAX_DIMENSION
        for row in rows:
            self.assertGreater(row["simulated_cycles"], 0, row)
            self.assertLessEqual(row["m"], limit)
            self.assertEqual(row["g"] > 1, row["layer"] == "dw" and supports("--groups"))
        # Each call: npu-sim cycles at or above the analytic lower bound of the same call.
        for row in rows:
            design = analytic.Design.from_result(
                run(
                    {
                        "m": row["m"],
                        "n": row["n"],
                        "k": row["k"],
                        "tile_k": 8,
                        **({"groups": row["g"]} if row["g"] > 1 else {}),
                    }
                )
            )
            model = analytic.analyze(design)
            self.assertLessEqual(
                model.lower_bound_cycles * row["calls"], row["simulated_cycles"], row
            )

    def test_trace_spans_match_counters(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.csv"
            result = run({"m": 16, "n": 16, "k": 32, "tile_k": 16, "trace": path})
            spans = trace.read_trace(path)
        busy = trace.busy_cycles(spans)
        self.assertEqual(busy["compute"], result["compute_busy_cycles"])
        self.assertEqual(busy["load"] + busy["store"], result["dma_busy_cycles"])
        self.assertEqual(sum(s.bytes for s in spans), result["read_bytes"] + result["write_bytes"])
        ET.fromstring(trace.gantt(spans).to_svg())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve(strict=True)
    unittest.main(argv=[sys.argv[0], *remaining])
