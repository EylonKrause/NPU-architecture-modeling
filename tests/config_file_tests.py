#!/usr/bin/env python3
"""npu-sim --config, --describe-config, --version and option precedence, end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "example.json"
BINARY: Path


def run(*arguments, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(BINARY), *map(str, arguments)], input=stdin, capture_output=True, text=True, timeout=60
    )


def leaves(node: dict, prefix: str = ""):
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from leaves(value, path)
        else:
            yield path, value


def lookup(node: dict, path: str):
    for key in path.split("."):
        node = node[key]
    return node


class ConfigFile(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="npu-config-")
        self.addCleanup(self.directory.cleanup)

    def write(self, name: str, text: str) -> Path:
        path = Path(self.directory.name) / name
        path.write_text(text, encoding="utf-8")
        return path

    def result(self, *arguments) -> dict:
        proc = run(*arguments)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def describe(self, *arguments) -> dict:
        return self.result(*arguments, "--describe-config")

    def assert_rejected(self, proc: subprocess.CompletedProcess, fragment: str) -> None:
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertEqual(proc.stdout, "")
        lines = proc.stderr.strip().splitlines()
        self.assertTrue(lines and lines[-1].startswith("npu-sim: "), proc.stderr)
        self.assertIn(fragment, lines[-1])

    def test_example_file_is_echoed(self):
        document = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        result = self.result("--config", EXAMPLE)
        echoed = result["config"]
        given = dict(leaves(document))
        for path, value in given.items():
            self.assertEqual(lookup(echoed, path), value, path)
        defaults = self.describe()
        self.assertEqual(sorted(dict(leaves(defaults))), sorted(dict(leaves(echoed))))
        for path, value in leaves(echoed):
            if path not in given:
                self.assertEqual(value, lookup(defaults, path), path)
        self.assertIs(result["numerical_match"], True)
        self.assertEqual(result["effective_macs_per_cycle"], 256)
        self.assertEqual(result["macs"], 64 * 64 * 128)
        self.assertEqual(result["reduction_blocks"], 2)

    def test_echoed_config_reproduces_the_run(self):
        first = self.result("--config", EXAMPLE, "--dump-output")
        path = self.write("echo.json", json.dumps(first["config"]))
        second = self.result("--config", path, "--dump-output")
        for value in (first, second):
            value.pop("simulation_wall_seconds")
        self.assertEqual(first, second)
        self.assertNotEqual(first["cycles"], self.result()["cycles"])

    def test_describe_config_equals_the_result_echo(self):
        described = self.describe("--config", EXAMPLE, "--m", 32)
        result = self.result("--config", EXAMPLE, "--m", 32)
        self.assertEqual(described, result["config"])
        self.assertEqual(described["shape"]["m"], 32)
        # Version 2 is the nested layout; version 1 results had flat keys.
        self.assertEqual(result["schema_version"], 2)
        self.assertNotIn("m", result["config"])

    def test_later_assignments_win(self):
        later = self.describe("--config", EXAMPLE, "--m", 32, "--rounding", "away")
        self.assertEqual(later["shape"]["m"], 32)
        self.assertEqual(later["quant"]["rounding"], "away")
        earlier = self.describe("--m", 32, "--config", EXAMPLE)
        self.assertEqual(earlier["shape"]["m"], 64)
        extra = self.write("extra.json", '{"shape": {"k": 64}, "name": "second"}')
        both = self.describe("--config", EXAMPLE, "--config", extra)
        self.assertEqual((both["shape"]["m"], both["shape"]["k"], both["name"]), (64, 64, "second"))
        flag = self.describe(
            "--config", self.write("relu.json", '{"quant": {"relu": false}}'), "--relu"
        )
        self.assertIs(flag["quant"]["relu"], True)

    def test_snapshots_carry_the_complete_config(self):
        proc = run("--config", EXAMPLE, "--live-stdio", stdin="stop\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        snapshots = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual([s["type"] for s in snapshots], ["snapshot", "stopped"])
        for snapshot in snapshots:
            self.assertEqual(snapshot["schema_version"], 2)
            self.assertEqual(snapshot["config"], self.describe("--config", EXAMPLE))
        progress = Path(self.directory.name) / "progress.jsonl"
        self.result("--config", EXAMPLE, "--progress-jsonl", progress)
        lines = progress.read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines)
        self.assertEqual({json.loads(line)["schema_version"] for line in lines}, {2})

    def test_rejected_config_files(self):
        cases = [
            ("{", "expected a string member name"),
            ('{"shape": {"m": 16,}}', "expected a string member name"),
            ('{"shape": {"rows": 16}}', 'unknown config key "shape.rows"'),
            ('{"m": 16}', 'unknown config key "m"'),
            ('{"shape": {"m": "16"}}', 'shape.m expects an unsigned 64-bit integer, got "16"'),
            ('{"shape": {"m": 1.5}}', "shape.m expects an unsigned 64-bit integer, got 1.5"),
            ('{"shape": {"m": 1}, "shape": {"n": 1}}', 'duplicate member "shape"'),
            (
                '{"arch": {"sram_bytes": 1048576}, "arch.sram_bytes": 2097152}',
                """config key "arch.sram_bytes" contains '.'; nest one object per path segment""",
            ),
            ('{"shape": {"m": 4}, "shape.m": 8}', """config key "shape.m" contains '.'"""),
            (
                '{"weights": {"encoding": "zip"}}',
                "weights.encoding expects one of dense|palette|sparse",
            ),
            ('{"quant": {"relu": "yes"}}', "quant.relu expects true or false"),
            ("[]", "a config file holds one JSON object"),
            ('{"shape": 16}', "shape expects an object, got 16"),
        ]
        for index, (text, fragment) in enumerate(cases):
            with self.subTest(text=text):
                path = self.write(f"bad{index}.json", text)
                proc = run("--config", path)
                self.assert_rejected(proc, fragment)
                self.assertIn(f"{path}:", proc.stderr)
        # A well-formed file still goes through validate() before the run.
        sram = self.write("sram.json", '{"arch": {"sram_bytes": 1}}')
        self.assert_rejected(run("--config", sram), "tile buffers exceed configured SRAM capacity")
        self.assert_rejected(
            run("--config", Path(self.directory.name) / "absent.json"), "cannot open config file"
        )
        self.assert_rejected(run("--config"), "missing value for --config")
        self.assert_rejected(run("--config", ""), "--config: expected a nonempty path")

    def test_conflicts_are_checked_after_merging(self):
        array = self.write(
            "array.json", '{"arch": {"compute": {"array_rows": 4, "array_cols": 4}}}'
        )
        self.assert_rejected(run("--config", array, "--macs-per-cycle", 16), "not both")
        # The echoed form names macs_per_cycle beside the array; the array wins.
        echoed = self.write(
            "echoed.json",
            json.dumps(
                {"arch": {"compute": {"macs_per_cycle": 64, "array_rows": 4, "array_cols": 4}}}
            ),
        )
        self.assertEqual(self.result("--config", echoed)["effective_macs_per_cycle"], 16)
        per_channel = self.write("channels.json", json.dumps({"quant": {"shifts": [1] * 16}}))
        quantization = self.write("q.txt", "NPUQ1 16\n" + "0 1 1\n" * 16)
        self.assert_rejected(
            run("--config", per_channel, "--quantization-file", quantization), "not both"
        )
        self.assertEqual(self.describe("--config", per_channel)["quant"]["shifts"], [1] * 16)

    def test_input_header_and_config_shape(self):
        tensors = self.write("t.npu", "NPU1 1 2 1\n3\n1 -1\n5 7\n")
        matching = self.write("match.json", '{"shape": {"m": 1, "n": 2, "k": 1}}')
        first = self.result("--input", tensors, "--config", matching, "--dump-output")
        plain_gemm = {
            "input_h": 0,
            "input_w": 0,
            "input_c": 0,
            "kernel_h": 1,
            "kernel_w": 1,
            "stride_h": 1,
            "stride_w": 1,
            "dilation_h": 1,
            "dilation_w": 1,
            "pad_top": 0,
            "pad_left": 0,
            "pad_bottom": 0,
            "pad_right": 0,
        }
        self.assertEqual(
            first["config"]["shape"],
            {"m": 1, "n": 2, "k": 1, "groups": 1, "seed": 1, "conv": plain_gemm},
        )
        echo = self.write("echo.json", json.dumps(first["config"]))
        second = self.result("--input", tensors, "--config", echo, "--dump-output")
        self.assertEqual(first["accumulators"], second["accumulators"])
        self.assertEqual(first["outputs"], second["outputs"])
        self.assert_rejected(
            run("--input", tensors, "--config", self.write("wrong.json", '{"shape": {"n": 3}}')),
            "--input header gives shape.n = 2 but the config file sets 3",
        )
        self.assert_rejected(run("--input", tensors, "--m", 1), "omit --m/--n/--k")

    def test_numerics_keys_round_trip(self):
        path = self.write(
            "numerics.json",
            json.dumps(
                {
                    "shape": {"m": 6, "n": 1, "k": 9, "groups": 5},
                    "quant": {
                        "rescale": "tosa-double",
                        "shift": 40,
                        "multiplier": 1518500250,
                        "shifts": [40, 41, 42, 43, 44],
                    },
                }
            ),
        )
        first = self.result("--config", path, "--dump-output")
        self.assertEqual(first["config"]["shape"]["groups"], 5)
        self.assertEqual(first["config"]["quant"]["rescale"], "tosa-double")
        self.assertEqual(len(first["outputs"]), 6 * 5)
        echo = self.write("echo.json", json.dumps(first["config"]))
        second = self.result("--config", echo, "--dump-output")
        for value in (first, second):
            value.pop("simulation_wall_seconds")
        self.assertEqual(first, second)
        timed = self.result("--config", echo, "--timing-only")
        self.assertIs(timed["config"]["sim"]["timing_only"], True)
        self.assertEqual(timed["numerics"], "skipped")
        self.assertEqual(timed["cycles"], first["cycles"])
        self.assert_rejected(
            run(
                "--config",
                self.write(
                    "short.json",
                    json.dumps({"shape": {"groups": 2}, "quant": {"shifts": [1] * 16}}),
                ),
            ),
            "per-channel shift count must equal G*N",
        )
        self.assert_rejected(
            run(
                "--config",
                self.write(
                    "tosa.json", json.dumps({"quant": {"rescale": "tosa-single", "shift": 1}})
                ),
            ),
            "requantization shift must be in 2..62 for tosa-single rescale",
        )

    def test_frequency_errors_name_the_flag(self):
        for value in (
            "abc",
            "1e999",
            " 200",
            "200 ",
            "0x10",
            "",
            "200MHz",
            "inf",
            "nan",
            ".",
            "+",
            "5e",
            "1.2.3",
        ):
            with self.subTest(value=value):
                self.assert_rejected(run("--frequency-mhz", value), "--frequency-mhz: ")
        # The flag accepts every strtod decimal form.
        for value, mhz in (
            ("1.5e3", 1500),
            (".5", 0.5),
            ("5.", 5),
            ("+5", 5),
            ("007", 7),
            ("+.25E+3", 250),
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    self.describe("--frequency-mhz", value)["arch"]["frequency_mhz"], mhz
                )
        # Config files keep the JSON number grammar.
        for text in ('{"arch": {"frequency_mhz": .5}}', '{"arch": {"frequency_mhz": +5}}'):
            with self.subTest(text=text):
                path = self.write("frequency.json", text)
                proc = run("--config", path)
                self.assert_rejected(proc, f"{path}:1:")

    def test_live_stdio_rejects_batch_outputs(self):
        trace = Path(self.directory.name) / "trace.csv"
        traced = self.write("traced.json", json.dumps({"sim": {"trace_path": str(trace)}}))
        for extra in (["--dump-output"], ["--trace", trace], ["--config", traced]):
            with self.subTest(extra=extra):
                self.assert_rejected(run("--live-stdio", *extra, stdin="stop\n"), "--live-stdio")
        self.assertFalse(trace.exists())

    def test_outputs_never_overwrite_a_config_file(self):
        directory = Path(self.directory.name)
        plain = self.write("plain.json", json.dumps({"shape": {"m": 4}}))
        waveform = self.write("waveform.vcd", json.dumps({"shape": {"m": 4}}))
        # A file whose own trace path names itself, and one aimed at another
        # config file loaded before it.
        own = directory / "own.json"
        own.write_text(json.dumps({"sim": {"trace_path": str(own)}}), encoding="utf-8")
        other = self.write("other.json", json.dumps({"sim": {"trace_path": str(plain)}}))
        cases = [
            (["--config", plain, "--trace", plain], "CSV trace"),
            (["--trace", plain, "--config", plain], "CSV trace"),
            (["--config", plain, "--progress-jsonl", plain], "progress JSONL"),
            (["--config", waveform, "--vcd", waveform], "VCD"),
            (["--config", own], "CSV trace"),
            (["--config", plain, "--config", other], "CSV trace"),
            (["--config", plain, "--config", plain, "--trace", plain], "CSV trace"),
        ]
        files = {path: path.read_text(encoding="utf-8") for path in (plain, waveform, own, other)}
        for arguments, output in cases:
            with self.subTest(arguments=arguments):
                self.assert_rejected(
                    run(*arguments), "output path conflicts with an input file: " + output
                )
        for path, text in files.items():
            self.assertEqual(path.read_text(encoding="utf-8"), text, path)
        # A trace beside the config file is accepted.
        trace = directory / "trace.csv"
        self.result("--config", plain, "--trace", trace)
        self.assertTrue(trace.read_text(encoding="utf-8").startswith("stage,"))

    def test_version_and_help(self):
        proc = run("--version")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertRegex(proc.stdout, r"^npu-sim \d+\.\d+\.\d+ \(\S+\)\n$")
        self.assertEqual(run("--unknown-flag", "--version").stdout, proc.stdout)
        text = run("--help").stdout
        for path, _ in leaves(self.describe()):
            self.assertIn(path, text)
        for flag in (
            "--config FILE.json",
            "--describe-config",
            "--version",
            "--m N",
            "--rounding away|even",
            "--weight-encoding dense|palette|sparse",
        ):
            self.assertIn(flag, text)
        self.assertNotRegex(text, r"per tile")


def main() -> int:
    global BINARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    BINARY = args.binary.resolve()
    if not BINARY.is_file():
        parser.error(f"simulator binary not found: {BINARY}")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConfigFile)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
