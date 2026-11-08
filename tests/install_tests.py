#!/usr/bin/env python3
"""Install layout: the npu component installs only the tools, and a fetched SystemC installs its
documentation under its own docdir."""

import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

# Install components of the fetched SystemC 3.0.2 (library, headers + CMake package, docs).
SYSTEMC_COMPONENTS = ("lib", "dev", "doc")

ARGS = argparse.Namespace()


def install(prefix, component=None):
    """Runs `cmake --install` into prefix and returns the installed files, relative, as posix."""
    command = [ARGS.cmake, "--install", ARGS.build_dir, "--prefix", str(prefix)]
    if ARGS.config:
        command += ["--config", ARGS.config]
    if component:
        command += ["--component", component]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise AssertionError(f"{' '.join(command)} failed:\n{result.stdout}{result.stderr}")
    return {p.relative_to(prefix).as_posix() for p in Path(prefix).rglob("*") if not p.is_dir()}


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="npu-install-")
        self.root = Path(self.scratch.name)

    def tearDown(self):
        self.scratch.cleanup()

    def test_npu_component_is_the_tools(self):
        files = install(self.root / "npu", "npu")
        self.assertEqual(files, {f"{ARGS.bindir}/{tool}" for tool in ARGS.tools})

    def test_every_installed_file_has_a_known_component(self):
        # A project install rule without COMPONENT npu would land in "Unspecified" and be
        # missing from a tools-only install.
        components = ["npu", *(SYSTEMC_COMPONENTS if ARGS.systemc_docdir else ())]
        by_component = set()
        for component in components:
            by_component |= install(self.root / component, component)
        self.assertEqual(install(self.root / "all"), by_component)

    def test_project_docdir_holds_no_dependency_files(self):
        files = install(self.root / "all")
        docdir = ARGS.docdir.rstrip("/") + "/"
        self.assertEqual(sorted(f for f in files if f.startswith(docdir)), [])

    def test_systemc_documentation_installs_under_its_own_docdir(self):
        if not ARGS.systemc_docdir:
            self.skipTest("SystemC is a prebuilt package; this build installs none of it")
        files = install(self.root / "doc", "doc")
        docdir = ARGS.systemc_docdir.rstrip("/") + "/"
        self.assertIn(docdir + "LICENSE", files)
        self.assertIn(docdir + "NOTICE", files)
        self.assertEqual(sorted(f for f in files if not f.startswith(docdir)), [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--bindir", required=True)
    parser.add_argument("--docdir", required=True)
    parser.add_argument("--tools", nargs="+", required=True)
    parser.add_argument("--systemc-docdir", default="", help="set when SystemC is fetched")
    ARGS, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__, *remaining])
