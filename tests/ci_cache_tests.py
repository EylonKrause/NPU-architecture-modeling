#!/usr/bin/env python3
"""The CI SystemC prefix is cached as soon as it is built, whatever the job's later steps do.

actions/cache saves in a post step that runs only when the whole job succeeds, so a leg whose
build or tests fail would rebuild SystemC on every run. The composite action therefore restores
with actions/cache/restore and saves with actions/cache/save in the step after the build.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / ".github" / "actions" / "systemc" / "action.yml"
COMBINED_CACHE = "lines using actions/cache@, which saves only when the job succeeds"


def steps(text):
    """Splits the `steps:` list of a composite action into one text block per step."""
    body = text.split("\n  steps:\n", 1)[1]
    return [block for block in re.split(r"\n(?=    - )", body) if block.strip()]


def field(step, name):
    match = re.search(rf"^\s*(?:- )?{name}: *(.+)$", step, re.MULTILINE)
    return match.group(1).strip() if match else None


def position(blocks, what, predicate):
    for i, block in enumerate(blocks):
        if predicate(block):
            return i
    raise AssertionError(f"{ACTION.relative_to(ROOT).as_posix()} has no {what} step")


class SystemCCacheTests(unittest.TestCase):
    def test_no_job_uses_the_success_only_cache_action(self):
        for path in sorted((ROOT / ".github").rglob("*.yml")):
            lines = path.read_text(encoding="utf-8").splitlines()
            offending = [n for n, line in enumerate(lines, 1) if "actions/cache@" in line]
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                self.assertEqual(offending, [], COMBINED_CACHE)

    def test_prefix_is_saved_right_after_it_is_built(self):
        blocks = steps(ACTION.read_text(encoding="utf-8"))

        def uses(prefix):
            return lambda block: (field(block, "uses") or "").startswith(prefix)

        restore = position(blocks, "cache restore", uses("actions/cache/restore@"))
        build = position(blocks, "SystemC build", lambda b: "cmake/BuildSystemC.cmake" in b)
        save = position(blocks, "cache save", uses("actions/cache/save@"))
        self.assertEqual((build, save), (restore + 1, restore + 2))

        restore_id = field(blocks[restore], "id")
        self.assertTrue(restore_id, "the restore step needs an id")
        miss = f"steps.{restore_id}.outputs.cache-hit != 'true'"
        self.assertEqual(field(blocks[build], "if"), miss)
        self.assertEqual(field(blocks[save], "if"), miss)
        self.assertEqual(field(blocks[save], "path"), field(blocks[restore], "path"))
        self.assertEqual(
            field(blocks[save], "key"),
            f"${{{{ steps.{restore_id}.outputs.cache-primary-key }}}}",
        )


if __name__ == "__main__":
    unittest.main()
