#!/usr/bin/env python3
"""Independent unit arithmetic, malformed-input and CLI energy checks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import energy_report as energy


def fixture():
    return {"schema_version": 1, "numerical_match": True, "cycles": 20,
            "macs": 10, "read_bytes": 7, "write_bytes": 3,
            "modeled_latency_ns": 100, "output_hash": 123,
            "config": {"frequency_mhz": 200}}


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

    def test_schema_and_numerical_agreement_required(self):
        for key, value in (("schema_version", True), ("schema_version", 1.0),
                           ("schema_version", 2), ("numerical_match", False),
                           ("numerical_match", 1), ("numerical_match", "true")):
            with self.subTest(key=key, value=value):
                result = fixture()
                result[key] = value
                with self.assertRaises(ValueError):
                    energy.estimate(result, 1, 1)

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
    unittest.main()
