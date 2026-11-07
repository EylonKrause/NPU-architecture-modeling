#!/usr/bin/env python3
"""Independent unit arithmetic, malformed-input and CLI energy checks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import energy_report as energy

BINARY: Path


def fixture():
    """A schema_version 1 result: flat config keys."""
    return {"schema_version": 1, "numerical_match": True, "cycles": 20,
            "macs": 10, "read_bytes": 7, "write_bytes": 3,
            "modeled_latency_ns": 100, "output_hash": 123,
            "config": {"frequency_mhz": 200}}


def nested_fixture():
    """fixture() at schema_version 2: config nested by option-table path."""
    return {**fixture(), "schema_version": 2,
            "config": {"shape": {"m": 1, "n": 1, "k": 10}, "arch": {"frequency_mhz": 200}}}


class EnergyTests(unittest.TestCase):
    def test_hand_calculated_units(self):
        # Illustrative arithmetic coefficients, not hardware energy data.
        result = energy.estimate(fixture(), mac_pj=2, dram_byte_pj=3, static_mw=4)
        self.assertEqual(result["component_pj"], {"mac": 20, "dram_read": 21,
                                                "dram_write": 9, "static": 400})
        self.assertEqual(result["total_pj"], 450)
        self.assertEqual(result["total_nj"], 0.45)
        self.assertEqual(result["average_power_mw"], 4.5)
        self.assertEqual(result["coefficients"]["static_mw"], 4)

    def test_zero_coefficients_and_default_excluded_static(self):
        result = energy.estimate(fixture(), 0, 0)
        self.assertEqual(result["total_pj"], 0)
        self.assertEqual(result["average_power_mw"], 0)
        self.assertEqual(result["coefficients"]["static_mw"], 0)

    def test_sparse_energy_counts_executed_macs_and_retains_logical_work(self):
        source = {**fixture(), "executed_macs": 4, "skipped_macs": 6}
        report = energy.estimate(source, mac_pj=2, dram_byte_pj=0)
        self.assertEqual(report["component_pj"]["mac"], 8)
        self.assertEqual(report["source_counts"]["macs"], 10)
        self.assertEqual(report["source_counts"]["executed_macs"], 4)
        self.assertTrue(any("decompression" in item for item in report["limitations"]))

    def test_inconsistent_or_invalid_sparse_counters_rejected(self):
        for executed, skipped in ((11, 0), (4, 5), (True, 9), (-1, 11), (0, 11)):
            with self.subTest(executed=executed, skipped=skipped), self.assertRaises(ValueError):
                energy.estimate({**fixture(), "executed_macs": executed, "skipped_macs": skipped}, 1, 1)

    def test_schema_and_numerical_agreement_required(self):
        for key, value in (("schema_version", True), ("schema_version", 1.0),
                           ("schema_version", 0), ("schema_version", 3),
                           ("schema_version", "2"), ("numerical_match", False),
                           ("numerical_match", 1), ("numerical_match", "true")):
            with self.subTest(key=key, value=value):
                result = fixture()
                result[key] = value
                with self.assertRaises(ValueError):
                    energy.estimate(result, 1, 1)

    def test_config_layout_follows_the_schema_version(self):
        self.assertEqual(energy.estimate(nested_fixture(), 2, 3, 4),
                         energy.estimate(fixture(), 2, 3, 4))
        # Each version reads the clock from its own layout only.
        flat_v2 = {**fixture(), "schema_version": 2}
        nested_v1 = {**nested_fixture(), "schema_version": 1}
        arch_not_object = {**nested_fixture(), "config": {"arch": 200, "frequency_mhz": 200}}
        for result in (flat_v2, nested_v1, arch_not_object):
            with self.subTest(result=result), self.assertRaises(ValueError):
                energy.estimate(result, 1, 1)

    def test_npu_sim_result_feeds_the_estimate(self):
        process = subprocess.run([str(BINARY), "--m", "8", "--n", "8", "--k", "16",
                                  "--frequency-mhz", "400"],
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["config"]["arch"]["frequency_mhz"], 400)
        self.assertNotIn("frequency_mhz", result["config"])
        report = energy.estimate(result, mac_pj=2, dram_byte_pj=3, static_mw=4)
        self.assertEqual(report["component_pj"]["mac"], 2 * result["executed_macs"])
        self.assertEqual(report["component_pj"]["dram_read"], 3 * result["read_bytes"])
        self.assertEqual(report["component_pj"]["dram_write"], 3 * result["write_bytes"])
        # mW times ns is pJ; 400 MHz is 2.5 ns per cycle.
        self.assertAlmostEqual(report["component_pj"]["static"], result["cycles"] * 2.5 * 4)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "result.json"
            source.write_text(process.stdout, encoding="utf-8")
            cli = subprocess.run([sys.executable, str(ROOT / "scripts" / "energy_report.py"),
                                  "--result", str(source), "--mac-pj", "2", "--dram-byte-pj", "3",
                                  "--static-mw", "4"],
                                 text=True, capture_output=True, timeout=10)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertEqual(json.loads(cli.stdout), report)

    def test_missing_fields_and_graph_report_rejected(self):
        for key in fixture():
            result = fixture()
            del result[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                energy.estimate(result, 1, 1)
        with self.assertRaises(ValueError):
            energy.estimate({"schema_version": 1, "sum_npu_cycles": 100}, 1, 1)

    def test_counts_reject_boolean_float_negative_and_uint64_overflow(self):
        for key in ("cycles", "macs", "read_bytes", "write_bytes", "output_hash"):
            for value in (True, 1.0, -1, 1 << 64, float("nan"), float("inf")):
                result = fixture()
                result[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    energy.estimate(result, 1, 1)

    def test_latency_and_frequency_reject_invalid_values(self):
        for value in (True, 0, -1, float("nan"), float("inf"), 10 ** 1000):
            for field in ("modeled_latency_ns", "frequency_mhz"):
                result = fixture()
                (result["config"] if field == "frequency_mhz" else result)[field] = value
                with self.subTest(field=field, value=str(value)[:20]), self.assertRaises(ValueError):
                    energy.estimate(result, 1, 1)

    def test_latency_must_agree_with_cycles_and_frequency(self):
        result = fixture()
        result["modeled_latency_ns"] = 99
        with self.assertRaises(ValueError):
            energy.estimate(result, 1, 1)

    def test_bad_coefficients_and_overflow_rejected(self):
        for name in ("mac_pj", "dram_byte_pj", "static_mw"):
            for value in (True, -1, float("nan"), float("inf"), 10 ** 1000):
                arguments = {"mac_pj": 1, "dram_byte_pj": 1, "static_mw": 0}
                arguments[name] = value
                with self.subTest(name=name, value=str(value)[:20]), self.assertRaises(ValueError):
                    energy.estimate(fixture(), **arguments)
        for coefficients in ((1e308, 1, 0), (1, 1e308, 0), (1, 1, 1e308),
                             (1e307, 1e307, 0)):
            with self.subTest(coefficients=coefficients), self.assertRaises(ValueError):
                energy.estimate(fixture(), *coefficients)

    def test_total_byte_count_overflow(self):
        result = fixture()
        result["read_bytes"], result["write_bytes"] = (1 << 64) - 1, 1
        with self.assertRaises(ValueError):
            energy.estimate(result, 0, 0)

    def test_cli_json_and_create_only_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "result.json", Path(folder) / "energy.json"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts" / "energy_report.py"),
                       "--result", str(source), "--mac-pj", "2", "--dram-byte-pj", "3",
                       "--static-mw", "4", "--output", str(output)]
            first = subprocess.run(command, text=True, capture_output=True, timeout=10)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads(first.stdout)["total_pj"], 450)
            self.assertEqual(json.loads(output.read_text()), json.loads(first.stdout))
            before = output.read_bytes()
            second = subprocess.run(command, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(second.stdout, "")
            self.assertEqual(output.read_bytes(), before)

    def test_cli_required_coefficients_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "result.json"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            base = [sys.executable, str(ROOT / "scripts" / "energy_report.py"), "--result", str(source)]
            for arguments in ([], ["--mac-pj", "1"], ["--dram-byte-pj", "1"],
                              ["--mac-pj", "nan", "--dram-byte-pj", "1"],
                              ["--mac-pj", "1", "--dram-byte-pj", "1e999"],
                              ["--mac-pj", "-1", "--dram-byte-pj", "1"]):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(base + arguments, text=True, capture_output=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")

    def test_json_nan_duplicate_keys_and_nonobject_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "invalid.json"
            for text in ('{"value":NaN}', '{"value":1e999}',
                         '{"schema_version":1,"schema_version":1}', '[]'):
                source.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    energy.estimate(energy.load_result(source), 1, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=os.environ.get("NPU_SIM"),
                        help="npu-sim executable (default: $NPU_SIM)")
    arguments, remaining = parser.parse_known_args()
    if arguments.binary is None:
        parser.error("pass --binary or set NPU_SIM")
    BINARY = arguments.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
