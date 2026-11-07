#!/usr/bin/env python3
"""The generated-results machinery: npuperf.studies and npuperf.reproduce.

The pipeline must be deterministic, rewrite only the text between its markers, and fail
--check on a table that differs from the committed one without touching the docs.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npuperf import reproduce, studies  # noqa: E402

BINARY: Path


class ReproduceTests(unittest.TestCase):
    TEXT = ("# Results\n\nprose\n\n<!-- generated:a -->\nold a\n<!-- end:a -->\n\nmore prose\n"
            "<!-- generated:b -->\n<!-- end:b -->\n")

    def test_sections_round_trip_and_leave_prose_alone(self):
        self.assertEqual(reproduce.sections(self.TEXT), {"a": "old a\n", "b": ""})
        text = reproduce.replace_sections(self.TEXT, {"a": "| new |\n", "b": "x"})
        self.assertEqual(reproduce.sections(text), {"a": "| new |\n", "b": "x\n"})
        self.assertIn("prose\n\n<!-- generated:a -->", text)
        self.assertIn("\nmore prose\n", text)
        self.assertEqual(reproduce.replace_sections(text, {}), text)
        with self.assertRaisesRegex(ValueError, "markers of: c"):
            reproduce.replace_sections(self.TEXT, {"c": "x"})

    def test_check_reports_a_differing_table(self):
        self.assertEqual(reproduce.check({"a": "old a"}, self.TEXT), [])
        problems = reproduce.check({"a": "new a\n", "c": "x"}, self.TEXT)
        self.assertEqual(len(problems), 2)
        self.assertIn("-old a", problems[0])
        self.assertIn("+new a", problems[0])
        self.assertIn("no generated:c section", problems[1])

    def test_quick_selection(self):
        quick = [study.id for study in reproduce.select(None, True)]
        self.assertEqual(quick, [s.id for s in studies.STUDIES.values() if s.quick])
        self.assertEqual(len(reproduce.select(None, False)), len(studies.STUDIES))
        self.assertEqual([s.id for s in reproduce.select("s3,S1", True)], ["S3"])
        with self.assertRaisesRegex(ValueError, "unknown study S9"):
            reproduce.select("S9", False)

    def test_study_output_is_deterministic(self):
        first = studies.s3_outstanding(studies.Context(BINARY, jobs=4))
        second = studies.s3_outstanding(studies.Context(BINARY, jobs=1))
        self.assertEqual(first.tables, second.tables)
        self.assertEqual({name: fig.to_svg() for name, fig in first.figures.items()},
                         {name: fig.to_svg() for name, fig in second.figures.items()})
        self.assertEqual(studies.s6_configs(), studies.s6_configs())
        self.assertEqual(len(studies.s6_configs()), sum(studies.S6_COUNT.values()))

    def test_check_mode_fails_on_drift_and_writes_nothing(self):
        table = studies.s3_outstanding(studies.Context(BINARY, jobs=4)).tables
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.md"
            figures = Path(directory) / "figures"
            work = Path(directory) / "studies"
            args = ["--only", "S3", "--check", "--binary", str(BINARY), "--jobs", "4"]
            with mock.patch.object(reproduce, "RESULTS_MD", results), \
                    mock.patch.object(studies, "FIGURE_DIR", figures), \
                    mock.patch.object(studies, "RESULTS_DIR", work), \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                good = reproduce.replace_sections(
                    "<!-- generated:s3-outstanding -->\n<!-- end:s3-outstanding -->\n", table)
                results.write_text(good, encoding="utf-8")
                self.assertEqual(reproduce.main(args), 0)
                stale = good.replace("| 1 |", "| one |")
                results.write_text(stale, encoding="utf-8")
                self.assertEqual(reproduce.main(args), 1)
                self.assertEqual(results.read_text(encoding="utf-8"), stale)
                self.assertFalse(figures.exists())
                self.assertIn("differ from", stderr.getvalue())
                self.assertEqual(reproduce.main(args[:2] + args[3:]), 0)
                self.assertEqual(results.read_text(encoding="utf-8"), good)
                self.assertTrue((figures / "s3-outstanding.svg").is_file())

    def test_formatting(self):
        self.assertEqual(studies.fmt(1234.5678), "1,235")
        self.assertEqual(studies.fmt(0.0239123), "0.0239")
        self.assertEqual(studies.fmt(25.24), "25.2")
        self.assertEqual(studies.pct(0.4567), "45.7%")
        table = studies.markdown_table(["A", "B"], [["x", "1"]])
        self.assertEqual(table, "| A | B |\n|---|---:|\n| x | 1 |\n")

    def test_sanity_checks_reject_impossible_results(self):
        result = {"cycles": 10, "macs": 1000, "effective_macs_per_cycle": 64,
                  "peak_memory_bytes_per_cycle": 16, "read_bytes": 10, "write_bytes": 10,
                  "config": {"arch": {"memory": {"model": "serial"}}}}
        with self.assertRaisesRegex(studies.SanityError, "MACs exceed"):
            studies.check_result(result, "x")
        result.update(macs=100, read_bytes=200)
        with self.assertRaisesRegex(studies.SanityError, "DRAM bytes exceed"):
            studies.check_result(result, "x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve(strict=True)
    unittest.main(argv=[sys.argv[0], *remaining])
